import logging
import numpy as np
import anndata as ad
import zarr
import dask.array as da
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import convert_ROI_table_to_indices
from fractal_tasks_core.tables import write_table
from pydantic import BaseModel, validate_call
import pandas as pd
from typing import List, Tuple
from scipy import ndimage
from skimage.measure import regionprops
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. Spherical kernel
# ----------------------------------------------------------------------
def make_spherical_kernel(radius: float, shape=(15, 15, 15)):
    center = np.array(shape) // 2
    zz, yy, xx = np.ogrid[
        -center[0]:shape[0]-center[0],
        -center[1]:shape[1]-center[1],
        -center[2]:shape[2]-center[2]
    ]
    dist2 = zz**2 + yy**2 + xx**2
    kernel = np.zeros(shape, dtype=np.float32)
    kernel[dist2 <= radius**2] = 1.0
    return kernel

# ----------------------------------------------------------------------
# 2. Init args model
# ----------------------------------------------------------------------
class InitArgsMeasureSpotIntensities(BaseModel):
    zarr_urls: list[str]
    wavelengths: list[str]

def get_channel_index(zarr_group, wavelength_id: str) -> int:
    channels = zarr_group.attrs.get("omero", {}).get("channels", [])
    for idx, channel in enumerate(channels):
        if channel.get("wavelength_id") == wavelength_id:
            return idx
    raise ValueError(f"Channel '{wavelength_id}' not found")

# ----------------------------------------------------------------------
# 3. Worker: process one ROI → returns sum, mean, max
# ----------------------------------------------------------------------
def process_single_roi(args):
    i_roi, indices, ROI_id, init_args, kernel, channel_labels, label_name, level = args
    s_z, e_z, s_y, e_y, s_x, e_x = indices
    region = (slice(s_z, e_z), slice(s_y, e_y), slice(s_x, e_x))

    # Reload data
    label_path = f"{init_args.zarr_urls[0]}/labels/{label_name}/{level}"
    labels = da.from_zarr(label_path)
    label_roi = labels[region].compute().astype(np.int32)

    if np.max(label_roi) == 0:
        return [], []

    data_arrays = [da.from_zarr(f"{url}/0") for url in init_args.zarr_urls]

    local_spot_data = []
    local_intensity_dict = {}

    # Precompute sphere coordinates (relative to center)
    k_size = kernel.shape[0]
    center = k_size // 2
    zz, yy, xx = np.ogrid[-center:center+1, -center:center+1, -center:center+1]
    sphere_mask = (zz**2 + yy**2 + xx**2 <= (kernel.shape[0]//2)**2)

    for cycle_i, url in enumerate(init_args.zarr_urls):
        zarr_group = zarr.open(url, mode="r")
        for wl in init_args.wavelengths:
            try:
                ch_idx = get_channel_index(zarr_group, wl)
                key = (cycle_i, wl)
                img = data_arrays[cycle_i][ch_idx, s_z:e_z, s_y:e_y, s_x:e_x].compute().astype(np.float32)
            except Exception:
                img = None

            if img is None:
                continue

            base_name = f"cycle_{cycle_i+1}_{wl}"
            props = regionprops(label_roi)

            for prop in props:
                lbl = prop.label
                cz, cy, cx = prop.centroid
                z0, y0, x0 = int(round(cz)), int(round(cy)), int(round(cx))

                # Extract sphere region
                z_start = max(z0 - center, 0)
                z_end = min(z0 - center + k_size, img.shape[0])
                y_start = max(y0 - center, 0)
                y_end = min(y0 - center + k_size, img.shape[1])
                x_start = max(x0 - center, 0)
                x_end = min(x0 - center + k_size, img.shape[2])

                if (z_end <= z_start or y_end <= y_start or x_end <= x_start):
                    vals = np.array([np.nan])
                else:
                    crop = img[z_start:z_end, y_start:y_end, x_start:x_end]
                    mask_crop = sphere_mask[
                        z_start - (z0 - center):z_end - (z0 - center),
                        y_start - (y0 - center):y_end - (y0 - center),
                        x_start - (x0 - center):x_end - (x0 - center)
                    ]
                    vals = crop[mask_crop]

                if len(vals) == 0:
                    s, m, mx = np.nan, np.nan, np.nan
                else:
                    s = float(vals.sum())
                    m = float(vals.mean())
                    mx = float(vals.max())

                if lbl not in local_intensity_dict:
                    local_intensity_dict[lbl] = {}
                    local_spot_data.append({
                        "spot_id": str(lbl),
                        "ROI": ROI_id,
                        "x": float(cx + s_x),
                        "y": float(cy + s_y),
                        "z": float(cz + s_z),
                    })

                local_intensity_dict[lbl][f"{base_name}_sum"] = s
                local_intensity_dict[lbl][f"{base_name}_mean"] = m
                local_intensity_dict[lbl][f"{base_name}_max"] = mx

    # Build matrix
    local_intensity_matrix = []
    for spot in local_spot_data:
        lbl = int(spot["spot_id"])
        row = [local_intensity_dict[lbl].get(col, np.nan) for col in channel_labels]
        local_intensity_matrix.append(row)

    return local_spot_data, local_intensity_matrix

# ----------------------------------------------------------------------
# 4. MAIN TASK – SUM, MEAN, MAX IN SPHERE
# ----------------------------------------------------------------------
@validate_call
def measure_spot_intensities_sphere(
    *,
    zarr_url: str,
    init_args: InitArgsMeasureSpotIntensities,
    ROI_table_name: str = "FOV_ROI_table",
    label_name: str = "candidate_spots",
    output_table_name: str = "spot_intensities_sphere",
    level: int = 0,
    radius_pixels: float = 3.0,
    overwrite: bool = True,
) -> dict:
    if not init_args.wavelengths:
        raise ValueError("At least one wavelength must be given.")
    if zarr_url != init_args.zarr_urls[0]:
        logger.warning(f"Using {init_args.zarr_urls[0]} as label source.")

    # --- NGFF metadata ---
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

    # --- Channel labels: sum, mean, max ---
    base_labels = [f"cycle_{i+1}_{wl}" for i in range(len(init_args.zarr_urls)) for wl in init_args.wavelengths]
    channel_labels = []
    for base in base_labels:
        channel_labels.extend([f"{base}_sum", f"{base}_mean", f"{base}_max"])

    # --- Pre-load image arrays ---
    logger.info("Pre-loading image arrays …")
    data_arrays = [da.from_zarr(f"{url}/0") for url in init_args.zarr_urls]

    # --- Precompute kernel ---
    kernel_size = int(np.ceil(radius_pixels)) * 2 + 1
    kernel = make_spherical_kernel(radius_pixels, shape=(kernel_size, kernel_size, kernel_size))

    # --- Parallel execution ---
    max_workers = min(8, mp.cpu_count())
    logger.info(f"Using {max_workers} CPU cores")

    spot_data = []
    intensity_matrix = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i_roi, indices in enumerate(list_indices):
            ROI_id = ROI_table.obs.index[i_roi]
            args = (
                i_roi, indices, ROI_id,
                init_args, kernel, channel_labels,
                label_name, level  # pass as strings
            )
            futures.append(executor.submit(process_single_roi, args))

        for future in tqdm(as_completed(futures), total=len(futures), desc="ROIs"):
            try:
                local_spot_data, local_intensity_matrix = future.result()
                spot_data.extend(local_spot_data)
                intensity_matrix.extend(local_intensity_matrix)
            except Exception as e:
                logger.error(f"ROI {i_roi} failed: {e}")
                continue  # don't crash entire job

    # --- Write table ---
    if not intensity_matrix:
        logger.warning("No valid spots found.")
        return dict(image_list_updates=[])

    intensity_matrix = np.array(intensity_matrix, dtype=np.float32)
    spot_table = ad.AnnData(
        X=intensity_matrix,
        obs=pd.DataFrame(spot_data),
        var=pd.DataFrame(index=channel_labels),
    )

    grp = zarr.open(init_args.zarr_urls[0], mode="rw")
    write_table(
        grp, output_table_name, spot_table, overwrite=overwrite,
        table_attrs={
            "type": "measurement_table",
            "region": {"path": f"../labels/{label_name}"},
            "fractal_table_version": "1",
        },
    )
    logger.info(f"Wrote table {output_table_name} with sum, mean, max per channel")

    return dict(image_list_updates=[])


# ----------------------------------------------------------------------
# Run as Fractal task
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task
    run_fractal_task(task_function=measure_spot_intensities_sphere, logger_name=logger.name)
