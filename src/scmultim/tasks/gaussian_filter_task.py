import logging
from pathlib import Path
import numpy as np
import cv2
import dask.array as da
from scipy.ndimage import gaussian_filter
from fractal_tasks_core.utils import _split_well_path_image_path
from fractal_tasks_core.tasks._zarr_utils import _update_well_metadata
from ngio import open_ome_zarr_container, open_ome_zarr_well
from pydantic import validate_call

logger = logging.getLogger(__name__)

@validate_call
def gaussian_filter_task(
    *,
    zarr_url: str,
    output_image_suffix: str = "high_pass",
    sigma: float = 2.0,
    overwrite_input: bool = True,
    roi_table: str = "well_ROI_table",
    exclude_channel_ids: list[str] = [],
    chunk_size: tuple[int, int] = (1024, 1024),  # New parameter for Dask chunking
):
    """Apply a 2D Gaussian high-pass filter to each slice of 3D OME-Zarr images.

    Args:
        zarr_url: Path to the OME-Zarr image to be processed.
        output_image_suffix: Suffix for the output image (e.g., 'high_pass').
        sigma: Standard deviation for the 2D Gaussian filter.
        overwrite_input: Whether to replace the original image with the filtered one.
        roi_table: Name of the ROI table to use for processing ROIs.
        exclude_channel_ids: List of channel IDs to exclude from filtering (e.g., ['C1_DAPI']).
        chunk_size: Tuple of (Y, X) dimensions for Dask chunking to control memory usage.
    """
    logger.info(
        f"Running gaussian high-pass filter on {zarr_url}, "
        f"output_suffix={output_image_suffix}, sigma={sigma}, overwrite={overwrite_input}, "
        f"roi_table={roi_table}, exclude_channels={exclude_channel_ids}, chunk_size={chunk_size}"
    )

    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    new_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}_{output_image_suffix}" if not overwrite_input else zarr_url

    # Open OME-Zarr containers
    ome_zarr = open_ome_zarr_container(zarr_url)
    if not ome_zarr.is_3d:
        raise ValueError("Input image must be 3D.")

    # Process images
    logger.info("Applying 2D Gaussian high-pass filter to images...")
    write_filtered_zarr(
        zarr_url=zarr_url,
        new_zarr_url=new_zarr_url,
        roi_table_name=roi_table,
        ome_zarr=ome_zarr,
        sigma=sigma,
        exclude_channel_ids=exclude_channel_ids,
        overwrite=overwrite_input,
        chunk_size=chunk_size,
    )
    logger.info("Finished applying filter.")

    # Copy ROI table
    new_ome_zarr = open_ome_zarr_container(new_zarr_url)
    roi_table_data = ome_zarr.get_table(roi_table)
    new_ome_zarr.add_table(roi_table, roi_table_data, overwrite=True)
    logger.info(f"Copied ROI table {roi_table} to new Zarr.")

    # Update well metadata if not overwriting
    image_list_updates = dict(image_list_updates=[dict(zarr_url=new_zarr_url, origin=zarr_url)])
    if not overwrite_input:
        well_url, new_img_path = _split_well_path_image_path(new_zarr_url)
        try:
            _update_well_metadata(
                well_url=well_url,
                old_image_path=old_img_path,
                new_image_path=new_img_path,
            )
        except ValueError as e:
            logger.warning(f"Could not update well metadata for {new_zarr_url}: {e}")

    return image_list_updates

def write_filtered_zarr(
    zarr_url: str,
    new_zarr_url: str,
    roi_table_name: str,
    ome_zarr,
    sigma: float,
    exclude_channel_ids: list[str],
    overwrite: bool,
    chunk_size: tuple[int, int],
):
    """Apply 2D Gaussian high-pass filter to a 3D Zarr image.

    Args:
        zarr_url: Path to the input OME-Zarr image.
        new_zarr_url: Path to the output OME-Zarr image.
        roi_table_name: Name of the ROI table.
        ome_zarr: OME-Zarr container of the input image.
        sigma: Standard deviation for the 2D Gaussian filter.
        exclude_channel_ids: List of channel IDs to exclude.
        overwrite: Whether to overwrite the input Zarr.
        chunk_size: Tuple of (Y, X) dimensions for Dask chunking.
    """
    # Create new OME-Zarr
    ome_zarr_new = ome_zarr.derive_image(
        store=new_zarr_url,
        ref_path="0",
        copy_labels=False,
        copy_tables=False,
        overwrite=True,
    )

    images = ome_zarr.get_image()
    new_images = ome_zarr_new.get_image()
    num_channels = len(images.meta.channel_labels)
    omero_channels = images.meta.channel_labels
    data_shape = images.get_array(c=0).shape[1:]  # Shape: (Z, Y, X)

    # Load ROI table
    roi_table = ome_zarr.get_table(roi_table_name)
    rois = list(roi_table.rois())

    # Process each channel and ROI
    for ind_ch in range(num_channels):
        channel_id = omero_channels[ind_ch]
        logger.info(f"Processing channel {ind_ch} ({channel_id})")

        for roi in rois:
            roi_data = images.get_roi(
                roi = roi,
                c=ind_ch,
            ).squeeze()
            logger.info(f"Processing roi {roi})")
            if channel_id in exclude_channel_ids:
                logger.info(f"Skipping filter for excluded channel {channel_id}")
                data_filtered = roi_data
            else:
                # Apply 2D Gaussian high-pass filter to all slices in ROI
                data_low_pass = gaussian_filter(roi_data, sigma=(0, sigma, sigma), mode='reflect')
                data_filtered = cv2.subtract(roi_data, data_low_pass)
                data_filtered = gaussian_filter(data_filtered, sigma=(0, 0.9, 0.9), mode='reflect')

            # Write to new Zarr
            new_images.set_roi(
                roi=roi,
                c=ind_ch,
                patch=np.expand_dims(data_filtered, axis=0),
            )
        new_images.consolidate()

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task
    run_fractal_task(
        task_function=gaussian_filter_task,
        logger_name=logger.name,
    )
