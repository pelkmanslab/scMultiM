import logging
import threading, time, logging
import numpy as np
import cupy as cp
import cupyx.scipy.special as cp_special
import anndata as ad
import zarr
import dask.array as da
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import convert_ROI_table_to_indices
from fractal_tasks_core.tables import write_table
from pydantic import BaseModel, validate_call
import pandas as pd
from typing import List, Tuple, Dict
import gc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CuPy intensity helpers (GPU-only)
# ----------------------------------------------------------------------
def intensity_gaussian3D(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz):
    xp_center = cp.ceil(xp) - 0.5
    yp_center = cp.ceil(yp) - 0.5
    zp_center = cp.round(zp)
    gx = cp.exp(-((xp_center * dx - x) ** 2) / (2 * sxy ** 2))
    gy = cp.exp(-((yp_center * dy - y) ** 2) / (2 * sxy ** 2))
    gz = cp.exp(-((zp_center * dz - z) ** 2) / (2 * sz ** 2))
    return gx * gy * gz


def intensity_integrated_gaussian3D(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz):
    xp_center = cp.ceil(xp) - 0.5
    yp_center = cp.ceil(yp) - 0.5
    zp_center = cp.round(zp)
    diffx1 = (xp_center - 0.5) * dx - x
    diffx2 = (xp_center + 0.5) * dx - x
    diffy1 = (yp_center - 0.5) * dy - y
    diffy2 = (yp_center + 0.5) * dy - y
    diffz1 = (zp_center - 0.5) * dz - z
    diffz2 = (zp_center + 0.5) * dz - z
    diffx1 /= cp.sqrt(2) * sxy
    diffx2 /= cp.sqrt(2) * sxy
    diffy1 /= cp.sqrt(2) * sxy
    diffy2 /= cp.sqrt(2) * sxy
    diffz1 /= cp.sqrt(2) * sz
    diffz2 /= cp.sqrt(2) * sz
    return (
        cp.abs(cp_special.erf(diffx1) - cp_special.erf(diffx2))
        * cp.abs(cp_special.erf(diffy1) - cp_special.erf(diffy2))
        * cp.abs(cp_special.erf(diffz1) - cp_special.erf(diffz2))
    )


def intensity_integrated_gaussian3D_stdZ(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz):
    xp_center = cp.ceil(xp) - 0.5
    yp_center = cp.ceil(yp) - 0.5
    zp_center = cp.round(zp)
    diffx1 = (xp_center - 0.5) * dx - x
    diffx2 = (xp_center + 0.5) * dx - x
    diffy1 = (yp_center - 0.5) * dy - y
    diffy2 = (yp_center + 0.5) * dy - y
    diffx1 /= cp.sqrt(2) * sxy
    diffx2 /= cp.sqrt(2) * sxy
    diffy1 /= cp.sqrt(2) * sxy
    diffy2 /= cp.sqrt(2) * sxy
    gz = cp.exp(-((zp_center * dz - z) ** 2) / (2 * sz ** 2))
    return (
        cp.abs(cp_special.erf(diffx1) - cp_special.erf(diffx2))
        * cp.abs(cp_special.erf(diffy1) - cp_special.erf(diffy2))
        * gz
    )


# ----------------------------------------------------------------------
# SUB-BATCHED GPU FITTER (500–1000 spots at a time)
# ----------------------------------------------------------------------
def _gaussian_mask_gpu_subbatch(
    crops_np: List[np.ndarray],
    centers: List[Tuple[float, float, float]],
    params: dict
) -> List[Tuple[float, float, float, float, float, float, int]]:
    N = len(crops_np)
    if N == 0:
        return []

    with cp.cuda.Device(0):
        crops_gpu = [cp.asarray(crop, dtype=cp.float32) for crop in crops_np]
        sxy, sz = params["psfSigma"]
        tol = params["tol"]
        maxcount = params["maxIterations"]
        psf_type = params["psfType"]
        cutSize = params["fittedRegionSize"]
        cutwidth = [cutSize * sxy, cutSize * sxy, cutSize * sz]
        dx = dy = dz = 1.0
        tol = tol * (dx + dy) / 2.0

        results = []

        for i in range(N):
            data = crops_gpu[i]
            x0_prev = centers[i][0]
            y0_prev = centers[i][1]
            z0_prev = centers[i][2]

            # Initialize with NaN
            x0_final = y0_final = z0_final = N0_final = cp.nan
            err0 = cp.nan
            dist_final = cp.nan
            iterations = 0

            for it in range(1, maxcount):
                xmin = max(0, int(x0_prev - cutwidth[0] / 2))
                xmax = min(data.shape[2], int(x0_prev + cutwidth[0] / 2))
                ymin = max(0, int(y0_prev - cutwidth[1] / 2))
                ymax = min(data.shape[1], int(y0_prev + cutwidth[1] / 2))
                zmin = max(0, int(z0_prev - cutwidth[2] / 2))
                zmax = min(data.shape[0], int(z0_prev + cutwidth[2] / 2))

                if xmax <= xmin or ymax <= ymin or zmax <= zmin:
                    break

                xp = cp.arange(xmin, xmax, dtype=cp.float32)
                yp = cp.arange(ymin, ymax, dtype=cp.float32)
                zp = cp.arange(zmin, zmax, dtype=cp.float32)
                xp, yp, zp = cp.meshgrid(xp, yp, zp, indexing="ij")
                sdata = data[zmin:zmax, ymin:ymax, xmin:xmax]

                x, y, z = x0_prev, y0_prev, z0_prev

                if psf_type == "gaussian":
                    intensity = intensity_gaussian3D(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz)
                elif psf_type == "integratedGaussian":
                    intensity = intensity_integrated_gaussian3D(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz)
                else:
                    intensity = intensity_integrated_gaussian3D_stdZ(xp, yp, zp, x, y, z, sxy, sz, dx, dy, dz)

                intsum = cp.sum(intensity * sdata)
                sumsum = cp.sum(intensity ** 2)
                sumx = cp.sum((xp - 0.5) * intensity * sdata)
                sumy = cp.sum((yp - 0.5) * intensity * sdata)
                sumz = cp.sum(zp * intensity * sdata)

                if intsum <= 0 or sumsum == 0:
                    break

                x0_new = sumx / intsum
                y0_new = sumy / intsum
                z0_new = sumz / intsum
                N0_new = intsum / sumsum  # ← defined here

                if (x0_new < xmin - 1 or x0_new >= xmax or
                    y0_new < ymin - 1 or y0_new >= ymax or
                    z0_new < zmin - 1 or z0_new >= zmax):
                    break

                dist = cp.sqrt((x0_new - x0_prev)**2 + (y0_new - y0_prev)**2 + (z0_new - z0_prev)**2)
                if dist < tol:
                    x0_final, y0_final, z0_final = x0_new, y0_new, z0_new
                    N0_final = N0_new
                    dist_final = dist
                    iterations = it
                    break

                x0_prev, y0_prev, z0_prev = x0_new, y0_new, z0_new
                iterations = it

            # --- FINAL ERROR CALCULATION (always safe) ---
            if not cp.isnan(N0_final):
                xmin = max(0, int(x0_final - cutwidth[0] / 2))
                xmax = min(data.shape[2], int(x0_final + cutwidth[0] / 2))
                ymin = max(0, int(y0_final - cutwidth[1] / 2))
                ymax = min(data.shape[1], int(y0_final + cutwidth[1] / 2))
                zmin = max(0, int(z0_final - cutwidth[2] / 2))
                zmax = min(data.shape[0], int(z0_final + cutwidth[2] / 2))

                if xmax > xmin and ymax > ymin and zmax > zmin:
                    xp = cp.arange(xmin, xmax, dtype=cp.float32)
                    yp = cp.arange(ymin, ymax, dtype=cp.float32)
                    zp = cp.arange(zmin, zmax, dtype=cp.float32)
                    xp, yp, zp = cp.meshgrid(xp, yp, zp, indexing="ij")
                    sdata = data[zmin:zmax, ymin:ymax, xmin:xmax]
                    if psf_type == "gaussian":
                        intensity = intensity_gaussian3D(xp, yp, zp, x0_final, y0_final, z0_final, sxy, sz, dx, dy, dz)
                    elif psf_type == "integratedGaussian":
                        intensity = intensity_integrated_gaussian3D(xp, yp, zp, x0_final, y0_final, z0_final, sxy, sz, dx, dy, dz)
                    else:
                        intensity = intensity_integrated_gaussian3D_stdZ(xp, yp, zp, x0_final, y0_final, z0_final, sxy, sz, dx, dy, dz)
                    err0 = float(cp.sqrt(cp.sum((N0_final * intensity - sdata) ** 2)))
                else:
                    err0 = cp.nan
            else:
                err0 = cp.nan

            # --- SCALING ---
            if not cp.isnan(N0_final) and psf_type == "integratedGaussian":
                N0_final *= 8

            results.append((
                float(x0_final), float(y0_final), float(z0_final),
                float(N0_final), err0, float(dist_final), iterations
            ))

        cp.get_default_memory_pool().free_all_blocks()
        return results


# ----------------------------------------------------------------------
# Helper models
# ----------------------------------------------------------------------
class InitArgsMeasureSpotIntensities(BaseModel):
    zarr_urls: list[str]
    wavelengths: list[str]


def get_channel_index(zarr_group, wavelength_id: str) -> int:
    channels = zarr_group.attrs.get("omero", {}).get("channels", [])
    for idx, channel in enumerate(channels):
        if channel.get("wavelength_id") == wavelength_id:
            return idx
    raise ValueError(f"Channel with wavelength_id '{wavelength_id}' not found in {zarr_group.store.path}")

# ----------------------------------------------------------------------
# 1. Helper – estimate free GPU memory (once at import time)
# ----------------------------------------------------------------------
def _get_gpu_free_mem() -> int:
    """Return free memory in bytes (≈ 80 % of total)."""
    try:
        mem_info = cp.get_default_memory_pool().mem_info()
        free = int(mem_info[0] * 0.8)          # keep 20 % head-room
        logger.debug(f"GPU free memory ≈ {free/1e9:.2f} GB")
        return free
    except Exception:
        logger.warning("Could not query GPU memory – falling back to 8 GB")
        return 8_000_000_000


GPU_FREE_MEM = _get_gpu_free_mem()

# ----------------------------------------------------------------------
# 2. Helper – build spot list (unchanged logic, just a function)
# ----------------------------------------------------------------------
def _build_spot_coords(label_roi_gpu, unique_labels, s_z, s_y, s_x):
    """Return [(lbl, gz, gy, gx, z_rel, y_rel, x_rel), …]"""
    spot_coords = []
    for lbl in unique_labels:
        coords = cp.where(label_roi_gpu == lbl)
        if coords[0].size == 0:
            continue
        z_rel, y_rel, x_rel = int(coords[0][0]), int(coords[1][0]), int(coords[2][0])
        spot_coords.append(
            (lbl, s_z + z_rel, s_y + y_rel, s_x + x_rel, z_rel, y_rel, x_rel)
        )
    return spot_coords


# ----------------------------------------------------------------------
# 1. Tiny heartbeat logger (prints every 2 s while GPU works)
# ----------------------------------------------------------------------


class Heartbeat:
    def __init__(self, name: str):
        self.name = name
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            logger.info(f"HEARTBEAT [{self.name}] – still alive")
            time.sleep(2)

    def end(self):
        self.stop.set()
        self.thread.join(timeout=1)


# ----------------------------------------------------------------------
# 2. MAIN TASK – ULTRA-VERBOSE + HEARTBEAT + SAFE BATCH
# ----------------------------------------------------------------------
@validate_call
def measure_spot_intensities(
    *,
    zarr_url: str,
    init_args: InitArgsMeasureSpotIntensities,
    ROI_table_name: str = "FOV_ROI_table",
    label_name: str = "candidate_spots",
    output_table_name: str = "spot_intensities",
    level: int = 0,
    crop_size: int = 5,
    sub_batch_size: int = 800,
    overwrite: bool = True,
) -> dict:
    # --------------------------------------------------------------
    # sanity
    # --------------------------------------------------------------
    if not init_args.wavelengths:
        raise ValueError("At least one wavelength must be given.")
    if zarr_url != init_args.zarr_urls[0]:
        logger.warning(f"Using {init_args.zarr_urls[0]} as label source.")

    # --------------------------------------------------------------
    # NGFF metadata
    # --------------------------------------------------------------
    ngff_meta = load_NgffImageMeta(init_args.zarr_urls[0])
    full_res_pxl_sizes_zyx = ngff_meta.get_pixel_sizes_zyx(level=0)
    coarsening_xy = ngff_meta.coarsening_xy

    label_path = f"{init_args.zarr_urls[0]}/labels/{label_name}/{level}"
    labels = da.from_zarr(label_path)

    ROI_table = ad.read_zarr(f"{init_args.zarr_urls[0]}/tables/{ROI_table_name}")
    list_indices = convert_ROI_table_to_indices(
        ROI_table, level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )

    channel_labels = [
        f"cycle_{i+1}_{wl}"
        for i, cycle_url in enumerate(init_args.zarr_urls)
        for wl in init_args.wavelengths
    ]

    spot_data = []
    intensity_matrix = []

    # --------------------------------------------------------------
    # cache full image arrays (once per task)
    # --------------------------------------------------------------
    logger.info("Pre-loading full image arrays …")
    data_arrays = [da.from_zarr(f"{url}/0") for url in init_args.zarr_urls]

    # --------------------------------------------------------------
    # outer progress bar
    # --------------------------------------------------------------
    roi_iter = tqdm(list_indices, desc="Processing ROIs", unit="ROI",
                    position=0, leave=True) if tqdm else list_indices

    # --------------------------------------------------------------
    # ROI loop
    # --------------------------------------------------------------
    for i_roi, indices in enumerate(roi_iter):
        s_z, e_z, s_y, e_y, s_x, e_x = indices
        ROI_id = ROI_table.obs.index[i_roi]

        logger.info(f"--- START ROI {i_roi+1}/{len(list_indices)} [{ROI_id}] ---")

        # ---------- 1. load label mask ----------
        logger.info("  → loading label mask …")
        region = (slice(s_z, e_z), slice(s_y, e_y), slice(s_x, e_x))
        label_roi_np = labels[region].compute()
        logger.info(f"  → label mask shape {label_roi_np.shape}")

        label_roi_gpu = cp.asarray(label_roi_np)
        unique_labels_gpu = cp.unique(label_roi_gpu)
        unique_labels_gpu = unique_labels_gpu[unique_labels_gpu > 0]
        unique_labels = unique_labels_gpu.get().tolist()
        logger.info(f"  → {len(unique_labels)} unique spots")

        if not unique_labels:
            logger.warning("  → no spots – skipping")
            continue

        if tqdm:
            roi_iter.set_postfix({"ROI": ROI_id, "spots": len(unique_labels)})

        # ---------- 2. build spot coordinate list ----------
        logger.info("  → building spot coordinates …")
        spot_coords = _build_spot_coords(label_roi_gpu, unique_labels, s_z, s_y, s_x)
        logger.info(f"  → {len(spot_coords)} coordinates ready")

        # ---------- 3. PRE-LOAD ALL CHANNELS ----------
        logger.info("  → pre-loading all channels …")
        channel_caches = {}
        for cycle_i, data_zyx in enumerate(data_arrays):
            zarr_group = zarr.open(init_args.zarr_urls[cycle_i], mode="r")
            for wl in init_args.wavelengths:
                try:
                    ch_idx = get_channel_index(zarr_group, wl)
                    key = (cycle_i, wl)
                    logger.debug(f"    caching {key} …")
                    channel_caches[key] = data_zyx[
                        ch_idx, s_z:e_z, s_y:e_y, s_x:e_x
                    ].compute()
                except Exception as exc:
                    logger.warning(f"    cache failed {key}: {exc}")
                    channel_caches[key] = None
        logger.info("  → all channels cached")

        # ---------- 4. dynamic batch size ----------
        crop_bytes = 7 * 7 * 7 * 4                     # ~1.4 kB per crop
        max_spots = max(1, GPU_FREE_MEM // crop_bytes)
        effective_batch = min(sub_batch_size, max_spots)
        if effective_batch != sub_batch_size:
            logger.info(f"  → GPU-limited batch size = {effective_batch}")
        total_sub_batches = (len(spot_coords) + effective_batch - 1) // effective_batch

        # ---------- 5. sub-batch progress ----------
        sub_iter = (
            tqdm(range(total_sub_batches),
                 desc=f"Sub-batches ROI {ROI_id}",
                 leave=False, position=1)
            if tqdm else range(total_sub_batches)
        )

        # ---------- 6. HEARTBEAT ----------
        hb = Heartbeat(f"ROI {ROI_id}")
        hb.start()

        try:
            # ---------- 7. cycle / wavelength loop ----------
            for cycle_i in range(len(init_args.zarr_urls)):
                for wl in init_args.wavelengths:
                    key = (cycle_i, wl)
                    cache = channel_caches.get(key)
                    if cache is None:
                        continue

                    offset = cycle_i * len(init_args.wavelengths) + init_args.wavelengths.index(wl)

                    # ---------- 8. sub-batch loop ----------
                    for batch_idx in sub_iter:
                        start = batch_idx * effective_batch
                        end = min(start + effective_batch, len(spot_coords))
                        batch = spot_coords[start:end]

                        crops, centers, info = [], [], []
                        half = 3
                        for lbl, gz, gy, gx, z_rel, y_rel, x_rel in batch:
                            z0 = max(0, z_rel - half)
                            z1 = min(cache.shape[0], z_rel + half + 1)
                            y0 = max(0, y_rel - half)
                            y1 = min(cache.shape[1], y_rel + half + 1)
                            x0 = max(0, x_rel - half)
                            x1 = min(cache.shape[2], x_rel + half + 1)

                            crop = cache[z0:z1, y0:y1, x0:x1]
                            if crop.size == 0 or not np.all(np.isfinite(crop)):
                                continue

                            center_local = [
                                float(x_rel - x0),
                                float(y_rel - y0),
                                float(z_rel - z0),
                            ]
                            crops.append(crop)
                            centers.append(center_local)
                            info.append((lbl, gz, gy, gx))

                        if not crops:
                            continue

                        logger.debug(f"    → fitting {len(crops)} spots (sub-batch {batch_idx+1}/{total_sub_batches})")
                        params = {
                            "psfType": "integratedGaussian",
                            "psfSigma": [1.5, 2.0],
                            "fittedRegionSize": crop_size / 1.5,
                            "maxIterations": 20,
                            "tol": 0.1,
                        }

                        fits = _gaussian_mask_gpu_subbatch(crops, centers, params)

                        for (lbl, gz, gy, gx), (x0, y0, z0, N0, err0, dist, it) in zip(info, fits):
                            spot_entry = next((r for r in spot_data if r["spot_id"] == lbl), None)
                            if spot_entry is None:
                                intensities = [np.nan] * len(channel_labels)
                                intensities[offset] = N0 if np.isfinite(N0) else np.nan
                                spot_data.append({
                                    "spot_id": lbl,
                                    "ROI": ROI_id,
                                    "x": float(gx), "y": float(gy), "z": float(gz),
                                })
                                intensity_matrix.append(intensities)
                            else:
                                idx = spot_data.index(spot_entry)
                                intensity_matrix[idx][offset] = N0 if np.isfinite(N0) else np.nan

                        if tqdm:
                            sub_iter.set_postfix({"spots": end})

        finally:
            hb.end()
            if tqdm:
                sub_iter.close()
            logger.info(f"--- FINISHED ROI {ROI_id} – {len(unique_labels)} spots ---")

    # --------------------------------------------------------------
    # write table
    # --------------------------------------------------------------
    if not intensity_matrix:
        logger.warning("No valid spots found.")
        return dict(image_list_updates=[])

    intensity_matrix = np.array(intensity_matrix)
    spot_table = ad.AnnData(
        X=intensity_matrix,
        obs=pd.DataFrame(spot_data),
        var=pd.DataFrame(index=channel_labels),
    )

    grp = zarr.open(init_args.zarr_urls[0], mode="rw")
    write_table(
        grp, output_table_name, spot_table, overwrite=overwrite,
        table_attrs={
            "type": "spot_intensity_table",
            "region": {"path": f"../labels/{label_name}"},
            "fractal_table_version": "1",
        },
    )
    logger.info(f"Wrote table {output_table_name}")

    return dict(image_list_updates=[])
if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task
    run_fractal_task(task_function=measure_spot_intensities, logger_name=logger.name)
