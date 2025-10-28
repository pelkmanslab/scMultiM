import logging
import os
from pathlib import Path
import dask.array as da
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndi
import zarr
import anndata as ad
from bigfish.detection import local_maximum_detection
from kneed import KneeLocator
from skimage.morphology import label
from skimage.exposure import rescale_intensity
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.utils import rescale_datasets, _split_well_path_image_path
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import check_valid_ROI_indices, convert_ROI_table_to_indices
from fractal_tasks_core.tasks._zarr_utils import _update_well_metadata
from pydantic import validate_call, BaseModel, Field
import matplotlib.pyplot as plt
import scipy.signal as signal
import json
from typing import Tuple

__OME_NGFF_VERSION__ = "0.4"

logger = logging.getLogger(__name__)

class InitArgsDetectSpotsBigfish(BaseModel):
    """
    Initialization arguments for detect_spots_bigfish.

    Attributes:
        zarr_urls: List of OME-Zarr URLs for different cycles.
        wavelengths: List of wavelength IDs for processing (e.g., ['A01_C03', 'A02_C02']).
    """
    zarr_urls: list[str] = Field(..., description="List of OME-Zarr URLs for different cycles.")
    wavelengths: list[str] = Field(
        ..., description="List of wavelength IDs for processing (e.g., ['A01_C03', 'A02_C02'])."
    )

def get_channel_index(zarr_group, wavelength_id: str) -> int:
    """
    Get the channel index for a given wavelength_id from OME-Zarr metadata.
    """
    channels = zarr_group.attrs.get("omero", {}).get("channels", [])
    for idx, channel in enumerate(channels):
        if channel.get("wavelength_id") == wavelength_id:
            return idx
    raise ValueError(f"Channel with wavelength_id '{wavelength_id}' not found in {zarr_group.store.path}")

def gpu_local_maximum_detection(image, min_distance=(1, 1, 1)):
    ndim = image.ndim
    if isinstance(min_distance, int):
        min_distance = (min_distance,) * ndim
    size = tuple(2 * d + 1 for d in min_distance)
    filtered = ndi.maximum_filter(image, size=size)
    mask = (image == filtered)
    del filtered  # Free memory
    cp.get_default_memory_pool().free_all_blocks()
    return mask

def gpu_rescale_intensity(image, out_range=(0, 1000)):
    in_min = image.min()
    in_max = image.max()
    out_min, out_max = out_range
    scaled = (image - in_min) / (in_max - in_min + 1e-10) * (out_max - out_min) + out_min
    return scaled

def bigfish_spot_detection(
    intensity_image: np.ndarray,
    min_distance: tuple = (1, 1, 1),
    threshold_range: tuple = (0, 1500),
    knee_sensitivity: float = 1.0,
    use_threshold: bool = False,
    arbitrary_threshold: float = None,
    plot_output_dir: str = None,
    plot_filename: str = None,
    refine_spots: bool = True,
    crop_size: int = 4
) -> tuple[np.ndarray, str, float]:
    """
    Detect spots in a 3D intensity image using BigFish and either KneeLocator or a fixed threshold.
    Optionally refine spot positions using radial symmetry.
    """
    intensity_image = cp.array(intensity_image, dtype=cp.float16)  # Use float16 to reduce memory
    logger.info(f"Input image shape: {intensity_image.shape}, dtype: {intensity_image.dtype}, "
                f"min: {intensity_image.min().get()}, max: {intensity_image.max().get()}")

    labels = cp.zeros(intensity_image.shape, dtype=cp.uint32)

    localmax = gpu_local_maximum_detection(image=intensity_image, min_distance=min_distance)
    coordinates = localmax.nonzero()
    peak_intensities = intensity_image[coordinates]
    del localmax  # Free memory
    logger.info(f"Detected {len(coordinates[0].get())} local maxima, "
                f"peak intensity range: [{peak_intensities.min().get() if len(peak_intensities) > 0 else 'N/A'}, "
                f"{peak_intensities.max().get() if len(peak_intensities) > 0 else 'N/A'}]")

    if use_threshold:
        if arbitrary_threshold is None:
            raise ValueError("arbitrary_threshold must be provided when use_threshold=True")
        threshold = float(arbitrary_threshold)
        logger.info(f"Using fixed threshold: {threshold}")
    else:
        if len(peak_intensities) == 0:
            logger.warning("No local maxima detected, setting threshold to threshold_range[1]")
            threshold = float(threshold_range[1])
        else:
            thresholds = range(threshold_range[0], threshold_range[1])
            counts = [cp.sum(peak_intensities > n).get() for n in thresholds]
            kneedle = KneeLocator(
                thresholds, counts, S=knee_sensitivity, curve='convex', direction='decreasing', interp_method='polynomial'
            )
            threshold = round(kneedle.elbow, 3) if kneedle.elbow is not None else threshold_range[1]
            logger.info(f"Computed threshold with KneeLocator: {threshold}")

    plot_path = None
    if plot_output_dir and plot_filename:
        os.makedirs(plot_output_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        thresholds = range(threshold_range[0], threshold_range[1])
        counts = [cp.sum(peak_intensities > n).get() for n in thresholds] if len(peak_intensities) > 0 else [0] * len(thresholds)
        plt.plot(thresholds, counts, label='Spot Counts')
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
        plt.xlabel('Threshold (Intensity)')
        plt.ylabel('Number of Spots')
        plt.title(f'Spot Detection Threshold Curve ({plot_filename})')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(plot_output_dir, plot_filename)
        plt.savefig(plot_path, format='png', dpi=300)
        plt.close()
        logger.info(f"Saved threshold plot to {plot_path}")

    keep = peak_intensities >= threshold
    trans_coord_merge = cp.stack(coordinates).T
    trans_coord_merge = trans_coord_merge[keep]
    del peak_intensities, keep  # Free memory
    logger.info(f"After thresholding, {len(trans_coord_merge)} spots remain")

    final_coords = trans_coord_merge
    del trans_coord_merge

    if len(final_coords) > 0:
        z, y, x = final_coords.T
        z = z.astype(cp.int32)
        y = y.astype(cp.int32)
        x = x.astype(cp.int32)
        valid = (0 <= y) & (y < intensity_image.shape[1]) & (0 <= x) & (x < intensity_image.shape[2])
        if valid.any():
            labels[z[valid], y[valid], x[valid]] = cp.arange(1, valid.sum() + 1, dtype=cp.uint32)
        else:
            logger.warning(f"No valid spots within image bounds")
        del z, y, x, valid, final_coords
        labels = ndi.label(labels)[0]
        logger.info(f"Assigned {labels.max().get()} spots to labels")

    result = labels.get()  # Transfer back to CPU
    del labels, intensity_image
    cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory
    return result, plot_path, threshold

@validate_call
def detect_spots_bigfish(
    *,
    zarr_url: str,
    init_args: InitArgsDetectSpotsBigfish,
    ROI_table_name: str = "FOV_ROI_table",
    min_distance: tuple = (1, 1, 1),
    threshold_range: tuple = (0, 1500),
    knee_sensitivity: float = 1.0,
    use_threshold: bool = False,
    arbitrary_threshold: float = None,
    output_label_name: str = "candidate_spots",
    level: int = 0,
    relabeling: bool = True,
    overwrite: bool = True,
    refine_spots: bool = True,
    crop_size: int = 4
) -> dict:
    """
    Detects spots in merged channels across multiple cycles of OME-Zarr images using BigFish.
    Saves labels to the first zarr_url and outputs threshold values with QC plots.
    """
    if not init_args.wavelengths:
        raise ValueError("At least one wavelength ID must be specified in wavelengths.")
    if use_threshold and arbitrary_threshold is None:
        raise ValueError("arbitrary_threshold must be provided when use_threshold=True")
    if zarr_url != init_args.zarr_urls[0]:
        logger.warning(
            f"zarr_url ({zarr_url}) differs from first init_args.zarr_urls ({init_args.zarr_urls[0]}). "
            f"Using {init_args.zarr_urls[0]} for label output."
        )

    well_url, old_img_path = _split_well_path_image_path(init_args.zarr_urls[0])
    zarr_group = zarr.open(init_args.zarr_urls[0], mode="r")

    channel_indices = []
    for zarr_url_iter in init_args.zarr_urls:
        zarr_group_iter = zarr.open(zarr_url_iter, mode="r")
        for wavelength_id in init_args.wavelengths:
            try:
                idx = get_channel_index(zarr_group_iter,wavelength_id)
                if zarr_url_iter == init_args.zarr_urls[0]:
                    channel_indices.append(idx)
                logger.info(f"Channel: index={idx}, wavelength_id={wavelength_id} in {zarr_url_iter}")
            except ValueError as e:
                raise ValueError(f"Channel {wavelength_id} not found in {zarr_url_iter}: {e}")
    logger.info(f"Using channel indices: {channel_indices}")

    ngff_image_meta = load_NgffImageMeta(init_args.zarr_urls[0])
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)

    if ngff_image_meta.axes_names[0] != "c":
        raise ValueError(
            f"Cannot set `remove_channel_axis=True` for multiscale metadata with axes={ngff_image_meta.axes_names}. "
            "First axis should have name 'c'."
        )
    new_datasets = rescale_datasets(
        datasets=[ds.dict() for ds in ngff_image_meta.datasets],
        coarsening_xy=coarsening_xy,
        reference_level=level,
        remove_channel_axis=True,
    )

    label_attrs = {
        "image-label": {
            "version": __OME_NGFF_VERSION__,
            "source": {"image": "../"},
        },
        "multiscales": [
            {
                "name": output_label_name,
                "version": __OME_NGFF_VERSION__,
                "axes": [
                    ax.dict()
                    for ax in ngff_image_meta.multiscale.axes
                    if ax.type != "channel"
                ],
                "datasets": new_datasets,
            }
        ],
    }
    logger.info(f"Label attributes: {label_attrs}")

    image_group = zarr.group(init_args.zarr_urls[0])
    label_group = prepare_label_group(
        image_group,
        output_label_name,
        overwrite=overwrite,
        label_attrs=label_attrs,
        logger=logger,
    )
    logger.info(f"Prepared label group for {init_args.zarr_urls[0]}: {label_group=}")

    out = f"{init_args.zarr_urls[0]}/labels/{output_label_name}/0"
    logger.info(f"Output label path: {out}")
    store = zarr.storage.FSStore(str(out))
    label_dtype = np.uint32

    first_data_zyx = da.from_zarr(f"{init_args.zarr_urls[0]}/0")[0]
    shape = first_data_zyx.shape
    if len(shape) == 2:
        shape = (1, *shape)
    chunks = first_data_zyx.chunksize
    if len(chunks) == 2:
        chunks = (1, *chunks)
    mask_zarr = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=overwrite,
        dimension_separator="/",
    )
    logger.info(f"Created mask for {init_args.zarr_urls[0]} with shape {shape} and chunks {chunks}")

    plot_output_dir = f"{init_args.zarr_urls[0]}/tables/spot_detection_qc"
    os.makedirs(plot_output_dir, exist_ok=True)

    ROI_table = ad.read_zarr(f"{init_args.zarr_urls[0]}/tables/{ROI_table_name}")
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    num_ROIs = len(list_indices)
    logger.info(f"Loaded {num_ROIs} ROIs from {ROI_table_name}")

    plot_metadata = []
    spot_counts = {}
    if relabeling:
        num_labels_tot = 0

    for i_ROI, indices in enumerate(list_indices):
        s_z, e_z, s_y, e_y, s_x, e_x = indices
        ROI_id = ROI_table.obs.index[i_ROI]
        logger.info(f"Processing {ROI_id} ({i_ROI + 1}/{num_ROIs})")

        region = (slice(s_z, e_z), slice(s_y, e_y), slice(s_x, e_x))
        logger.info(f"ROI {ROI_id} region: {region}")

        try:
            check_valid_ROI_indices([s_z, e_z, s_y, e_y, s_x, e_x], shape)
        except ValueError as e:
            logger.warning(f"Invalid ROI indices for {ROI_id}: {e}. Skipping.")
            continue

        merged_image = None
        for cycle_zarr_url in init_args.zarr_urls:
            data_zyx = da.from_zarr(f"{cycle_zarr_url}/0")
            for idx in channel_indices:
                tmp_channel = data_zyx[idx, s_z:e_z, s_y:e_y, s_x:e_x].compute()
                logger.info(f"FOV {ROI_id}, Channel index {idx} from {cycle_zarr_url}, "
                            f"shape: {tmp_channel.shape}, min: {tmp_channel.min()}, max: {tmp_channel.max()}")

                tmp_channel = cp.array(tmp_channel, dtype=cp.float16)  # Use float16 to reduce memory
                tmp_channel_normalized = gpu_rescale_intensity(
                    tmp_channel,
                    out_range=(0, 1000)
                )

                if merged_image is None:
                    merged_image = tmp_channel_normalized
                else:
                    merged_image += tmp_channel_normalized

                del tmp_channel, tmp_channel_normalized  # Free memory
                cp.get_default_memory_pool().free_all_blocks()

                logger.info(f"FOV {ROI_id}, Merged channel index {idx} from cycle {cycle_zarr_url}, "
                            f"max after normalization: {merged_image.max().get()}")

        if merged_image is None:
            logger.warning(f"FOV {ROI_id}: No channels merged, skipping spot detection.")
            continue

        plot_filename = f"{ROI_id}.png"
        new_label_image, plot_path, threshold = bigfish_spot_detection(
            intensity_image=merged_image.get(),
            min_distance=min_distance,
            threshold_range=threshold_range,
            knee_sensitivity=knee_sensitivity,
            use_threshold=use_threshold,
            arbitrary_threshold=arbitrary_threshold,
            plot_output_dir=plot_output_dir,
            plot_filename=plot_filename,
            refine_spots=refine_spots,
            crop_size=crop_size
        )
        del merged_image
        cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory

        num_spots = int(np.max(new_label_image))
        spot_counts[ROI_id] = num_spots
        if plot_path:
            plot_metadata.append({
                "fov_name": ROI_id,
                "plot_path": plot_path,
                "num_spots": int(num_spots),
                "threshold": float(threshold),
                "threshold_method": "fixed" if use_threshold else "knee"
            })

        if relabeling:
            num_labels_roi = np.max(new_label_image)
            new_label_image[new_label_image > 0] += num_labels_tot
            num_labels_tot += num_labels_roi
            logger.info(f"FOV {ROI_id}, num_spots={num_spots}, threshold={threshold}, "
                        f"threshold_method={'fixed' if use_threshold else 'knee'}, num_labels_tot={num_labels_tot}")

            if num_labels_tot > np.iinfo(label_dtype).max:
                raise ValueError(
                    f"ERROR in re-labeling: Reached {num_labels_tot} labels, but dtype={label_dtype}"
                )

        da.array(new_label_image).to_zarr(
            url=mask_zarr,
            region=region,
            compute=True,
        )
        logger.info(f"Wrote label image for FOV {ROI_id} to {init_args.zarr_urls[0]}/labels/{output_label_name}/0")

    metadata_path = f"{plot_output_dir}/plot_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(plot_metadata, f, indent=2)
    logger.info(f"Saved plot metadata to {metadata_path}")

    logger.info("Summary of spots detected and thresholds per FOV:")
    for ROI_id, num_spots in spot_counts.items():
        meta = next((item for item in plot_metadata if item["fov_name"] == ROI_id), {})
        threshold = meta.get("threshold")
        method = meta.get("threshold_method", "unknown")
        logger.info(f"FOV {ROI_id}: {num_spots} spots, threshold={threshold}, method={method}")

    label_path = f"{init_args.zarr_urls[0]}/labels/{output_label_name}"
    logger.info(f"Building pyramid for {label_path}")
    build_pyramid(
        zarrurl=label_path,
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.max,
    )

    image_list_updates = []
    well_url, img_path = _split_well_path_image_path(init_args.zarr_urls[0])
    image_list_updates.append(dict(zarr_url=init_args.zarr_urls[0]))
    try:
        _update_well_metadata(
            well_url=well_url,
            old_image_path=img_path,
            new_image_path=img_path,
        )
    except ValueError as e:
        logger.warning(f"Could not update well metadata for {init_args.zarr_urls[0]}: {e}")

    return dict(image_list_updates=image_list_updates)

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task
    run_fractal_task(
        task_function=detect_spots_bigfish,
        logger_name=logger.name,
    )
