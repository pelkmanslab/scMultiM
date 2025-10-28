import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import dask.array as da
from fractal_tasks_core.tasks._zarr_utils import (
    _get_matching_ref_acquisition_path_heuristic,
    _update_well_metadata,
)
from fractal_tasks_core.utils import (
    _split_well_path_image_path,
)
from ngio import open_ome_zarr_container, open_ome_zarr_well
from ngio.images.ome_zarr_container import OmeZarrContainer
from pydantic import validate_call
from scmultim.registration.fractal_helper_tasks import (
    get_acquisition_paths,
)

logger = logging.getLogger(__name__)

@validate_call
def apply_z_registration_merged(
    *,
    zarr_url: str,
    ref_zarr_url: str,
    output_image_suffix: str = "z_registered",
    roi_table: str = "well_ROI_table",
    overwrite_input: bool = True,
):
    """Apply z-registration to 3D images using z-shift transformation parameters.

    Both reference and moving images are assumed to be 3D.

    Args:
        zarr_url: Path to the OME-Zarr moving image to be processed.
        ref_zarr_url: Path to the reference OME-Zarr image.
        output_image_suffix: Suffix for the output image (e.g., 'z_registered').
        roi_table: Name of the ROI table used for registration (e.g., 'well_ROI_table').
        overwrite_input: Whether to replace the original image with the registered one.
    """
    logger.info(
        f"Running `apply_z_registration` on {zarr_url=}, "
        f"{roi_table=}, {ref_zarr_url=}, "
        f"Using {overwrite_input=}, {output_image_suffix=}"
    )

    registration_dir = Path(zarr_url) / "registration_z"
    if not registration_dir.exists():
        raise FileNotFoundError(f"Registration directory {registration_dir} does not exist.")

    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    new_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}_{output_image_suffix}"

    # Open OME-Zarr containers
    ome_zarr_well = open_ome_zarr_well(well_url)
    ome_zarr_mov = open_ome_zarr_container(zarr_url)
    ome_zarr_ref = open_ome_zarr_container(ref_zarr_url)

    if not ome_zarr_mov.is_3d or not ome_zarr_ref.is_3d:
        raise ValueError("Both moving and reference images must be 3D.")

    # Get acquisition paths
    acq_dict = get_acquisition_paths(ome_zarr_well)
    logger.info(f"{acq_dict=}")

    # Process images
    logger.info("Starting to apply z-registration to images...")
    write_registered_zarr(
        zarr_url=zarr_url,
        reference_zarr_url=ref_zarr_url,
        new_zarr_url=new_zarr_url,
        roi_table_name=roi_table,
        ome_zarr_mov=ome_zarr_mov,
        registration_dir=registration_dir,
    )
    logger.info("Finished applying z-registration to images.")

    # Copy tables from reference
    logger.info("Copying tables from the reference acquisition to the new acquisition.")
    new_ome_zarr = open_ome_zarr_container(new_zarr_url)
    table_names = ome_zarr_ref.list_tables()

    for table_name in table_names:
        table = ome_zarr_ref.get_table(table_name)
        if table.type() in ["roi_table", "masking_roi_table"]:
            new_ome_zarr.add_table(table_name, table, overwrite=True)
        else:
            logger.warning(
                f"Table {table_name} is not a standard ROI table. Copying without modification."
            )
            new_ome_zarr.add_table(table_name, table, overwrite=True)

    logger.info("Finished copying tables from the reference acquisition.")

    # Clean up Zarr file
    if overwrite_input:
        logger.info("Replacing original zarr image with the new zarr image")
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(new_zarr_url, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
        image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=new_zarr_url, origin=zarr_url)]
        )
        well_url, new_img_path = _split_well_path_image_path(new_zarr_url)
        try:
            _update_well_metadata(
                well_url=well_url,
                old_image_path=old_img_path,
                new_image_path=new_img_path,
            )
        except ValueError as e:
            logger.warning(
                f"Could not update the well metadata for {zarr_url=} and {new_img_path}: {e}"
            )

    return image_list_updates

def write_registered_zarr(
    zarr_url: str,
    reference_zarr_url: str,
    new_zarr_url: str,
    roi_table_name: str,
    ome_zarr_mov: OmeZarrContainer,
    registration_dir: Path,
):
    """Apply z-shift registration to a 3D Zarr image.

    Args:
        zarr_url: Path to the OME-Zarr moving image to be processed.
        reference_zarr_url: Path to the reference OME-Zarr image.
        new_zarr_url: Path to the new OME-Zarr image to be written.
        roi_table_name: Name of the ROI table used for registration.
        ome_zarr_mov: OME-Zarr container of the moving image.
        registration_dir: Directory containing z-registration files.
    """
    ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
    ref_roi_table = ome_zarr_ref.get_table(roi_table_name)

    # Create new OME-Zarr
    ome_zarr_new = ome_zarr_mov.derive_image(
        store=new_zarr_url,
        ref_path="0",
        copy_labels=False,
        copy_tables=False,
        overwrite=True,
    )
    ome_zarr_new.add_table(roi_table_name, table=ref_roi_table)

    mov_images = ome_zarr_mov.get_image()
    new_images = ome_zarr_new.get_image()
    roi_table_mov = ome_zarr_mov.get_table(roi_table_name)

    if len(ref_roi_table.rois()) != len(roi_table_mov.rois()):
        raise ValueError(
            f"Number of ROIs in reference ({len(ref_roi_table.rois())}) "
            f"and moving ({len(roi_table_mov.rois())}) tables do not match."
        )

    axes_list = mov_images.meta.axes_mapper.on_disk_axes_names
    if axes_list != ["c", "z", "y", "x"]:
        raise NotImplementedError(
            f"`write_registered_zarr` only supports axes ['c', 'z', 'y', 'x'], got {axes_list}"
        )

    for ref_roi in ref_roi_table.rois():
        mov_roi = roi_table_mov.get(ref_roi.name)
        if not mov_roi:
            raise ValueError(f"ROI {ref_roi.name} not found in moving ROI table {roi_table_name}.")

        ROI_id = mov_roi.name.split('_')[1]
        ROI_id = int(ROI_id) - 1
        fn_pattern = f"{roi_table_name}_roi_{ROI_id}_z.npy"
        parameter_file = registration_dir / fn_pattern

        if not parameter_file.exists():
            raise FileNotFoundError(f"Registration parameter file {parameter_file} not found for ROI {ROI_id}.")

        # Load z-shift transformation
        tform = np.load(parameter_file, allow_pickle=True).item()
        z_shift = int(tform.get('translation_z', 0))

        # Load all channels at once
        data_mov = mov_images.get_roi(roi=mov_roi)
        # Convert to NumPy if Dask array
        if isinstance(data_mov, da.Array):
            data_mov = data_mov.compute()
        else:
            data_mov = np.asarray(data_mov)  # Ensure NumPy array
        num_channels = data_mov.shape[0]
        data_mov_reg = np.zeros_like(data_mov, dtype=data_mov.dtype)

        # Apply z-shift to all channels
        if z_shift > 0:
            data_mov_reg[:, z_shift:] = data_mov[:, :-z_shift]
        elif z_shift < 0:
            data_mov_reg[:, :z_shift] = data_mov[:, -z_shift:]
        else:
            data_mov_reg = data_mov  # Avoid copying if no shift

        # Write all channels to new Zarr
        for ind_ch in range(num_channels):
            new_images.set_roi(
                roi=mov_roi,
                c=ind_ch,
                patch=np.expand_dims(data_mov_reg[ind_ch], axis=0),
            )
        new_images.consolidate()

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=apply_z_registration_merged_between_modality,
        logger_name=logger.name,
    )
