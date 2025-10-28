# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Cheng-Han Yang <cheng-han.yang@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Calculates registration for image-based registration."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from ngio import open_ome_zarr_container
from pydantic import validate_call
from skimage.exposure import rescale_intensity

from scmultim.tasks.conversions import to_itk
from scmultim.registration.fractal_helper_tasks import pad_to_max_shape
from scmultim.registration.itk_elastix import register_transform_only

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s; %(levelname)s; %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("elastix_registration.log"),
    ],
)
logger = logging.getLogger(__name__)

@validate_call
def compute_3D_2Dto2D_registration_elastix(
    *,
    zarr_url: str,
    ref_zarr_url: str,
    level: int = 0,
    ref_wavelength_id: str,
    mov_wavelength_id: Optional[str] = None,#make it optional, if not go back to ref
    parameter_files: list[str],
    lower_rescale_quantile: float = 0.0,
    upper_rescale_quantile: float = 0.99,
    roi_table: str = "FOV_ROI_table",
    use_masks: bool = False,
    masking_label_name: Optional[str] = None,
) -> None:
    """Calculate Elastix registration for 3D or 2D moving images to a 2D reference image.

    For 3D moving images, a maximum intensity projection (MIP) is applied along the Z-axis
    to create a 2D image before registration. The reference image is ensured to be 2D.
    Supports different wavelength IDs for reference and moving images (e.g., A03_C01 to
    A03_C01 or A03_C01 to A03_C04). Loops over ROIs in the specified ROI table.

    Args:
        zarr_url: Path to the moving OME-Zarr image to be processed.
        ref_zarr_url: Path to the reference OME-Zarr image.
        level: Pyramid level of the image for registration (0 for full resolution).
        ref_wavelength_id: Wavelength ID for the reference image (e.g., 'A03_C01').
        mov_wavelength_id: Wavelength ID for the moving image (e.g., 'A03_C01', 'A03_C04').
        parameter_files: List of paths to Elastix parameter files (e.g., rigid, affine).
        lower_rescale_quantile: Lower quantile for rescaling image intensities (default: 0.0).
        upper_rescale_quantile: Upper quantile for rescaling image intensities (default: 0.99).
        roi_table: Name of the ROI table to loop over (e.g., 'FOV_ROI_table').
        use_masks: If True, use masked loading (falls back to False if not supported).
        masking_label_name: Optional label for masked loading (e.g., 'embryo').
    """
    logger.info(
        f"Running for {zarr_url=}, {ref_zarr_url=}, {roi_table=}, "
        f"{ref_wavelength_id=}, {mov_wavelength_id=}."
    )

    # Validate parameter files
    for param_file in parameter_files:
        if not Path(param_file).exists():
            raise FileNotFoundError(f"Parameter file not found: {param_file}")

    # Load channels
    ome_zarr_ref = open_ome_zarr_container(ref_zarr_url)
    try:
        channel_index_ref = ome_zarr_ref.image_meta._get_channel_idx_by_wavelength_id(
            ref_wavelength_id
        )
    except ValueError as e:
        raise ValueError(f"Reference wavelength {ref_wavelength_id} not found in {ref_zarr_url}") from e
    logger.info(f"Reference channel: index={channel_index_ref}, wavelength_id={ref_wavelength_id}")

    ome_zarr_mov = open_ome_zarr_container(zarr_url)
    try:
        channel_index_align = ome_zarr_mov.image_meta._get_channel_idx_by_wavelength_id(
            mov_wavelength_id
        )
    except ValueError as e:
        raise ValueError(f"Moving wavelength {mov_wavelength_id} not found in {zarr_url}") from e
    logger.info(f"Moving channel: index={channel_index_align}, wavelength_id={mov_wavelength_id}")

    ref_images = ome_zarr_ref.get_image(path=str(level))
    mov_images = ome_zarr_mov.get_image(path=str(level))

    # Read ROIs
    ref_roi_table = ome_zarr_ref.get_table(roi_table)
    mov_roi_table = ome_zarr_mov.get_table(roi_table)
    ref_roi_names = [roi.name for roi in ref_roi_table.rois()]
    mov_roi_names = [roi.name for roi in mov_roi_table.rois()]
    logger.info(f"Found {len(ref_roi_names)} ROIs in {roi_table=}. ROI names: {ref_roi_names}")

    # Validate ROI matching
    if set(ref_roi_names) != set(mov_roi_names):
        raise ValueError(f"ROI names mismatch. Reference: {ref_roi_names}, Moving: {mov_roi_names}")

    # Masked loading checks
    if use_masks:
        if ref_roi_table.type() != "masking_roi_table":
            logger.warning(
                f"ROI table {roi_table} in reference OME-Zarr is not a masking ROI table. "
                "Falling back to use_masks=False."
            )
            use_masks = False
        if masking_label_name is None:
            logger.warning(
                "No masking label provided, but use_masks is True. "
                "Falling back to use_masks=False."
            )
            use_masks = False
        if use_masks:
            ref_images = ome_zarr_ref.get_masked_image(
                masking_label_name=masking_label_name,
                masking_table_name=roi_table,
                path=str(level),
            )
            mov_images = ome_zarr_mov.get_masked_image(
                masking_label_name=masking_label_name,
                masking_table_name=roi_table,
                path=str(level),
            )

    # Read pixel sizes
    pxl_sizes_zyx_ref = ome_zarr_ref.get_image(path=str(level)).pixel_size.zyx
    pxl_sizes_zyx_mov = ome_zarr_mov.get_image(path=str(level)).pixel_size.zyx
    logger.info(f"pxl_sizes_zyx_ref: {pxl_sizes_zyx_ref}")
    logger.info(f"pxl_sizes_zyx_mov: {pxl_sizes_zyx_mov}")
    if pxl_sizes_zyx_ref != pxl_sizes_zyx_mov:
        logger.warning(
            f"Pixel sizes differ between acquisitions: ref={pxl_sizes_zyx_ref}, mov={pxl_sizes_zyx_mov}. "
            "Proceeding with registration, but results may be affected."
        )

    num_ROIs = len(ref_roi_table.rois())
    for i_ROI, ref_roi in enumerate(ref_roi_table.rois()):
        ROI_id = ref_roi.name
        logger.info(
            f"Processing ROI {i_ROI+1}/{num_ROIs} (ID: {ROI_id}) for {mov_wavelength_id=}."
        )

        # Load images
        if use_masks:
            img_ref = ref_images.get_roi_masked(
                label=int(ROI_id),
                c=channel_index_ref,
            ).squeeze()
            img_mov = mov_images.get_roi_masked(
                label=int(ROI_id),
                c=channel_index_align,
            ).squeeze()
        else:
            img_ref = ref_images.get_roi(
                roi=ref_roi,
                c=channel_index_ref,
            ).squeeze()
            mov_roi = mov_roi_table.get(ROI_id)
            if mov_roi is None:
                raise ValueError(f"ROI {ROI_id} not found in moving ROI table")
            img_mov = mov_images.get_roi(
                roi=mov_roi,
                c=channel_index_align,
            ).squeeze()

        # Ensure reference image is 2D
        logger.info(f"Loaded img_ref shape: {img_ref.shape}")
        if img_ref.ndim == 3:
            img_ref = img_ref[0]  # Select first Z-slice
        logger.info(f"img_ref shape after processing: {img_ref.shape}")
        if img_ref.ndim != 2:
            raise ValueError(
                f"Reference image must be 2D after processing, but got ndim={img_ref.ndim}."
            )

        # Apply maximum intensity projection for 3D moving image
        if img_mov.ndim == 3:
            logger.info(f"Applying maximum intensity projection to 3D moving image: {img_mov.shape}")
            img_mov = np.max(img_mov, axis=0)  # MIP along Z-axis
        logger.info(f"img_mov shape after processing: {img_mov.shape}")

        # Pad images to match shapes
        max_shape = tuple(max(r, m) for r, m in zip(img_ref.shape, img_mov.shape))
        img_ref = pad_to_max_shape(img_ref, max_shape)
        img_mov = pad_to_max_shape(img_mov, max_shape)
        logger.info(f"Padded shapes: img_ref={img_ref.shape}, img_mov={img_mov.shape}")

        # Rescale intensities
        img_ref = rescale_intensity(
            img_ref,
            in_range=(
                np.quantile(img_ref, lower_rescale_quantile),
                np.quantile(img_ref, upper_rescale_quantile),
            ),
        )
        img_mov = rescale_intensity(
            img_mov,
            in_range=(
                np.quantile(img_mov, lower_rescale_quantile),
                np.quantile(img_mov, upper_rescale_quantile),
            ),
        )

        # Calculate transformation
        logger.info(f"img_ref ndim: {img_ref.ndim}, img_mov ndim: {img_mov.ndim}")
        pxl_sizes_yx_ref = pxl_sizes_zyx_ref[1:]
        pxl_sizes_yx_mov = pxl_sizes_zyx_mov[1:]
        ref = to_itk(img_ref, scale=pxl_sizes_yx_ref)
        move = to_itk(img_mov, scale=pxl_sizes_yx_mov)
        trans = register_transform_only(ref, move, parameter_files)

        # Write transformation files
        for i in range(trans.GetNumberOfParameterMaps()):
            trans_map = trans.GetParameterMap(i)
            fn = Path(zarr_url) / "registration" / f"{roi_table}_roi_{ROI_id}_t{i}_2Dto2D.txt"
            fn.parent.mkdir(exist_ok=True, parents=True)
            if fn.exists():
                logger.warning(f"Overwriting existing transformation file: {fn}")
            trans.WriteParameterFile(trans_map, fn.as_posix())
            logger.info(f"Saved transformation file: {fn}")

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task
    run_fractal_task(
        task_function=compute_3D_2Dto2D_registration_elastix,
        logger_name=logger.name,
    )
