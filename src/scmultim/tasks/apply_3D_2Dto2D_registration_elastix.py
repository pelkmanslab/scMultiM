# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
# Cheng-Han Yang <cheng-han.yang@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Computes and applies elastix registration."""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
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

from scmultim.tasks.conversions import to_itk, to_numpy
from scmultim.registration.fractal_helper_tasks import (
    get_acquisition_paths,
    get_pad_width,
    pad_to_max_shape,
    unpad_array,
)
from scmultim.registration.itk_elastix import (
    adapt_itk_params,
    apply_transform,
    load_parameter_files,
)

logger = logging.getLogger(__name__)


@validate_call
def apply_3D_2Dto2D_registration_elastix(
    *,
    # Fractal parameters
    zarr_url: str,
    reference_zarr_url: str,
    # Core parameters
    output_image_suffix: str = "registered",
    roi_table: str,
    use_masks: bool = False,
    masking_label_name: Optional[str] = None,
    overwrite_input: bool = True,
):
    """Apply elastix registration to images

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_zarr_url: Path or url to the reference OME-Zarr image to be processed.
        output_image_suffix: Name of the output image suffix. E.g. "registered".
        roi_table: Name of the ROI table which has been used during computation of
            registration.
            Examples: `FOV_ROI_table` => loop over the field of views,
            `well_ROI_table` => process the whole well as one image.
        use_masks: If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            be loaded.
        masking_label_name: Name of the label that will be used for masking.
            If `use_masks=True`, the label image will be used to mask the
            bounding box of the ROI table. If `use_masks=False`, the whole
            bounding box will be loaded.
        overwrite_input: Whether the old image data should be replaced with the
            newly registered image data.

    """
    logger.info(
        f"Running `apply_registration_elastix` on {zarr_url=}, "
        f"{roi_table=} and {reference_zarr_url=}. "
        f", {use_masks=}, {masking_label_name=}, "
        f"Using {overwrite_input=} and {output_image_suffix=}"
    )

    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    new_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}_{output_image_suffix}"

    # Get the zarr_url for the reference acquisition
    ome_zarr_well = open_ome_zarr_well(well_url)
    acquisition_ids = ome_zarr_well.acquisition_ids
    acq_dict = get_acquisition_paths(ome_zarr_well)
    logger.info(f"{acq_dict=}")


    # Open the OME-Zarr containers for both the reference and moving images
    ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
    ome_zarr_mov = open_ome_zarr_container(zarr_url)
    
    # Masked loading checks
    ref_roi_table = ome_zarr_ref.get_table(roi_table)
    if use_masks:
        if ref_roi_table.type() != "masking_roi_table":
            logger.warning(
                f"ROI table {roi_table} in reference OME-Zarr is not "
                "a masking ROI table. Falling back to use_masks=False."
            )
            use_masks = False
        if masking_label_name is None:
            logger.warning(
                "No masking label provided, but use_masks is True. "
                "Falling back to use_masks=False."
            )
            use_masks = False

    ####################
    # Process images
    ####################
    logger.info("Starting to apply elastix registration to images...")
    write_registered_zarr(
        zarr_url=zarr_url,
        reference_zarr_url=reference_zarr_url,
        new_zarr_url=new_zarr_url,
        roi_table_name=roi_table,
        ome_zarr_mov=ome_zarr_mov,
        use_masks=use_masks,
        masking_label_name=masking_label_name,
    )
    logger.info("Finished applying elastix registration to images.")

    ####################
    # Process labels
    ####################

    label_list = ome_zarr_mov.list_labels()

    if label_list:
        logger.warning(
            "Skipping registration of labels ... Label registration "
            "has not been implemented."
        )

    ####################
    # Copy tables
    # 1. Copy and transform all standard ROI tables from the reference acquisition into 3D.
    # 2. Give a warning to tables that aren't standard ROI tables from the given
    # acquisition.
    ####################
    logger.info("Copying tables from the reference acquisition to the new acquisition.")

    new_ome_zarr = open_ome_zarr_container(new_zarr_url)

    table_names = ome_zarr_ref.list_tables()
    for table_name in table_names:
        table = ome_zarr_ref.get_table(table_name)
        if table.type() == "roi_table" or table.type() == "masking_roi_table":
            # Extend ROI table to 3D if reference is 2D and new is 3D
            if not ome_zarr_ref.is_3d and new_ome_zarr.is_3d:
                new_image = new_ome_zarr.get_image()
                try:
                    z_index = new_image.meta.axes_mapper.get_index("z")
                    nb_z_planes = new_image.shape[z_index]
                    pixel_size_z = new_image.pixel_size.z
                    z_extent = nb_z_planes * pixel_size_z
                    for roi in table.rois():
                        roi.z_length = z_extent
                except KeyError:
                    pass  # No z dimension in new_image, no extension needed
            # Copy ROI tables from the reference acquisition
            new_ome_zarr.add_table(table_name, table, overwrite=True)
        else:
            logger.warning(
                f"{zarr_url} contained a table that is not a standard "
                "ROI table. The `Apply Registration (elastix)` task is "
                "best used before additional e.g. feature tables are generated."
            )
            new_ome_zarr.add_table(
                table_name,
                table,
                overwrite=True,
            )

    logger.info(
        "Finished copying tables from the reference acquisition to the new acquisition."
    )

    ####################
    # Clean up Zarr file
    ####################
    if overwrite_input:
        logger.info("Replace original zarr image with the newly created Zarr image")
        # Potential for race conditions: Every acquisition reads the
        # reference acquisition, but the reference acquisition also gets
        # modified
        # See issue #516 for the details
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(new_zarr_url, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
        image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=new_zarr_url, origin=zarr_url)]
        )
        # Update the metadata of the the well
        well_url, new_img_path = _split_well_path_image_path(new_zarr_url)
        try:
            _update_well_metadata(
                well_url=well_url,
                old_image_path=old_img_path,
                new_image_path=new_img_path,
            )
        except ValueError as e:
            logger.warning(
                f"Could not update the well metadata for {zarr_url=} and "
                f"{new_img_path}: {e}"
            )

    return image_list_updates


def write_registered_zarr(
    zarr_url: str,
    reference_zarr_url: str,
    new_zarr_url: str,
    roi_table_name: str,
    ome_zarr_mov: OmeZarrContainer,
    use_masks: bool = False,
    masking_label_name: Optional[str] = None,
):
    """Apply elastix registration to a Zarr image, applying 2D transformations to each z-slice of a 3D moving image.

    This function loads the image or label data from a zarr array based on the
    ROI bounding-box coordinates and stores them into a new zarr array.
    The new Zarr array has the same shape as the original array, but will have
    0s where the ROI tables don't specify loading of the image data.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be used as
            the basis for the new OME-Zarr image.
        reference_zarr_url: Path or url to the reference OME-Zarr image (2D).
        new_zarr_url: Path or url to the new OME-Zarr image to be written.
        roi_table_name: Name of the ROI table in reference_zarr_url used for registration.
        ome_zarr_mov: OME-Zarr container of the moving image to be registered.
        use_masks: If `True` applies masked image loading, otherwise loads the
            whole bounding box of the ROI table.
        masking_label_name: Name of the label that will be used for masking.
    """
    ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
    ref_roi_table = ome_zarr_ref.get_table(roi_table_name)

    ome_zarr_new = ome_zarr_mov.derive_image(
        store=new_zarr_url,
        ref_path="0",
        copy_labels=False,
        copy_tables=False,
        overwrite=True,
    )
    if not ome_zarr_ref.is_3d and ome_zarr_new.is_3d:
        new_image = ome_zarr_new.get_image()
        try:
            z_index = new_image.meta.axes_mapper.get_index("z")
            nb_z_planes = new_image.shape[z_index]
            pixel_size_z = new_image.pixel_size.z
            z_extent = nb_z_planes * pixel_size_z
            for roi in ref_roi_table.rois():
                roi.z_length = z_extent
        except KeyError:
            pass
    ome_zarr_new.add_table(roi_table_name, table=ref_roi_table)

    if use_masks:
        new_label = ome_zarr_new.derive_label(masking_label_name, overwrite=True)
        ref_masking_label = ome_zarr_ref.get_label(masking_label_name, path="0")
        ref_masking_label_array = ref_masking_label.get_array(mode="dask")
        if not ome_zarr_ref.is_3d and ome_zarr_new.is_3d:
            ref_masking_label_array = ref_masking_label_array.squeeze()
            if len(ref_masking_label_array.shape) == 2:
                try:
                    z_index = new_label.meta.axes_mapper.get_index("z")
                    nb_z = new_label.shape[z_index]
                    ref_masking_label_array = da.stack([ref_masking_label_array] * nb_z, axis=0)
                except KeyError:
                    pass
        new_label.set_array(ref_masking_label_array)
        new_label.consolidate()

        ref_images = ome_zarr_ref.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table_name,
            path="0",
        )
        mov_images = ome_zarr_mov.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table_name,
            path="0",
        )
        new_images = ome_zarr_new.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table_name,
            path="0",
        )
    else:
        ref_images = ome_zarr_ref.get_image()
        mov_images = ome_zarr_mov.get_image()
        new_images = ome_zarr_new.get_image()

    roi_table_mov = ome_zarr_mov.get_table(roi_table_name)
    roi_table_ref = ome_zarr_ref.get_table(roi_table_name)

    if len(roi_table_ref.rois()) != len(roi_table_mov.rois()):
        raise ValueError(
            f"Number of ROIs in reference ({len(roi_table_ref.rois())}) "
            f"and moving ({len(roi_table_mov.rois())}) tables do not match."
        )

    for ref_roi in roi_table_ref.rois():
        mov_roi = roi_table_mov.get(ref_roi.name)
        if not mov_roi:
            raise ValueError(f"ROI {ref_roi.name} not found in moving ROI table {roi_table_name}.")

        ROI_id = mov_roi.name
        fn_pattern = f"{roi_table_name}_roi_{ROI_id}_t*.txt"
        parameter_path = Path(zarr_url) / "registration"
        parameter_files = sorted(parameter_path.glob(fn_pattern))
        if not parameter_files:
            raise FileNotFoundError(f"No registration parameter files found for ROI {ROI_id} at {parameter_path}.")
        parameter_object = load_parameter_files([str(x) for x in parameter_files])

        pxl_sizes_zyx = mov_images.pixel_size.zyx
        axes_list = mov_images.meta.axes_mapper.on_disk_axes_names

        if axes_list == ["c", "z", "y", "x"]:
            num_channels = len(mov_images.meta.channel_labels)
            for ind_ch in range(num_channels):
                if use_masks:
                    data_ref = ref_images.get_roi_masked(
                        label=int(ROI_id),
                        c=0,
                    ).squeeze()
                    data_mov = mov_images.get_roi_masked(
                        label=int(ROI_id),
                        c=ind_ch,
                    ).squeeze()
                else:
                    data_ref = ref_images.get_roi(
                        roi=ref_roi,
                        c=0,
                    ).squeeze()
                    data_mov = mov_images.get_roi(
                        roi=mov_roi,
                        c=ind_ch,
                    ).squeeze()

                # Force load to numpy for faster slicing
                data_ref = np.asarray(data_ref)
                data_mov = np.asarray(data_mov)

                # Ensure reference is 2D
                if len(data_ref.shape) != 2:
                    raise ValueError(
                        f"Reference ROI {ROI_id} is not 2D, shape: {data_ref.shape}"
                    )

                # Pad to the same shape in y, x
                max_shape_2d = tuple(
                    max(r, m) for r, m in zip(data_ref.shape, data_mov.shape[-2:], strict=True)
                )
                pad_width_2d = get_pad_width(data_mov.shape[-2:], max_shape_2d)

                # Handle 3D moving image by processing each z-slice
                if len(data_mov.shape) == 3:  # 3D case (z, y, x)
                    z_slices = data_mov.shape[0]

                    # Create template for parameter adaptation
                    data_mov_slice0 = data_mov[0]
                    data_mov_slice0 = pad_to_max_shape(data_mov_slice0, max_shape_2d)
                    itk_template = to_itk(data_mov_slice0, scale=pxl_sizes_zyx[1:])
                    try:
                        parameter_object_adapted = adapt_itk_params(
                            parameter_object=parameter_object, itk_img=itk_template
                        )
                    except Exception as e:
                        logger.error(f"Failed to adapt parameters for ROI {ROI_id}, channel {ind_ch}: {e}")
                        raise

                    transformed_slices = []
                    for z in range(z_slices):
                        data_mov_slice = data_mov[z]
                        data_mov_slice = pad_to_max_shape(data_mov_slice, max_shape_2d)
                        itk_img = to_itk(data_mov_slice, scale=pxl_sizes_zyx[1:])
                        try:
                            transformed_slice = to_numpy(apply_transform(itk_img, parameter_object_adapted))
                            transformed_slice = unpad_array(transformed_slice, pad_width_2d)
                            transformed_slices.append(transformed_slice)
                        except Exception as e:
                            logger.warning(
                                f"Invalid transformation for ROI {ROI_id}, channel {ind_ch}, z-slice {z}: {e}. "
                                "Using untransformed slice."
                            )
                            transformed_slices.append(unpad_array(data_mov_slice, pad_width_2d))
                    data_mov_reg = np.stack(transformed_slices, axis=0)
                else:  # 2D case (y, x)
                    data_mov = pad_to_max_shape(data_mov, max_shape_2d)
                    itk_img = to_itk(data_mov, scale=pxl_sizes_zyx[1:])
                    try:
                        parameter_object_adapted = adapt_itk_params(
                            parameter_object=parameter_object, itk_img=itk_img
                        )
                        data_mov_reg = to_numpy(apply_transform(itk_img, parameter_object_adapted))
                        data_mov_reg = unpad_array(data_mov_reg, pad_width_2d)
                        data_mov_reg = np.expand_dims(data_mov_reg, axis=0)
                    except Exception as e:
                        logger.error(f"Failed to adapt parameters for ROI {ROI_id}, channel {ind_ch}: {e}")
                        raise

                if use_masks:
                    new_images.set_roi_masked(
                        label=int(ROI_id),
                        c=ind_ch,
                        patch=np.expand_dims(data_mov_reg, axis=0),
                    )
                else:
                    new_images.set_roi(
                        roi=mov_roi,
                        c=ind_ch,
                        patch=np.expand_dims(data_mov_reg, axis=0),
                    )
            new_images.consolidate()

        elif axes_list == ["z", "y", "x"]:
            if use_masks:
                data_ref = ref_images.get_roi_masked(
                    label=int(ROI_id),
                )
                data_mov = mov_images.get_roi_masked(
                    label=int(ROI_id),
                )
            else:
                data_ref = ref_images.get_roi(
                    roi=ref_roi,
                )
                data_mov = mov_images.get_roi(
                    roi=mov_roi,
                )

            # Force load to numpy for faster slicing
            data_ref = np.asarray(data_ref)
            data_mov = np.asarray(data_mov)

            if len(data_ref.shape) != 2:
                raise ValueError(
                    f"Reference ROI {ROI_id} is not 2D, shape: {data_ref.shape}"
                )

            max_shape_2d = tuple(
                max(r, m) for r, m in zip(data_ref.shape, data_mov.shape[-2:], strict=True)
            )
            pad_width_2d = get_pad_width(data_mov.shape[-2:], max_shape_2d)

            if len(data_mov.shape) == 3:
                z_slices = data_mov.shape[0]

                # Create template for parameter adaptation
                data_mov_slice0 = data_mov[0]
                data_mov_slice0 = pad_to_max_shape(data_mov_slice0, max_shape_2d)
                itk_template = to_itk(data_mov_slice0, scale=pxl_sizes_zyx[1:])
                try:
                    parameter_object_adapted = adapt_itk_params(
                        parameter_object=parameter_object, itk_img=itk_template
                    )
                except Exception as e:
                    logger.error(f"Failed to adapt parameters for ROI {ROI_id}: {e}")
                    raise

                transformed_slices = []
                for z in range(z_slices):
                    data_mov_slice = data_mov[z]
                    data_mov_slice = pad_to_max_shape(data_mov_slice, max_shape_2d)
                    itk_img = to_itk(data_mov_slice, scale=pxl_sizes_zyx[1:])
                    try:
                        transformix_object = itk.TransformixFilter.New(itk_img)
                        transformix_object.SetTransformParameterObject(parameter_object_adapted)
                        transformix_object.UpdateLargestPossibleRegion()
                        transformed_slice = to_numpy(transformix_object.GetOutput())
                        transformed_slice = unpad_array(transformed_slice, pad_width_2d)
                        transformed_slices.append(transformed_slice)
                    except RuntimeError as e:
                        logger.warning(
                            f"Invalid transformation for ROI {ROI_id}, z-slice {z}: {e}. "
                            "Using untransformed slice."
                        )
                        transformed_slices.append(unpad_array(data_mov_slice, pad_width_2d))
                    except Exception as e:
                        logger.error(f"Failed to apply transform for ROI {ROI_id}, z-slice {z}: {e}")
                        raise
                data_mov_reg = np.stack(transformed_slices, axis=0)
            else:
                data_mov = pad_to_max_shape(data_mov, max_shape_2d)
                itk_img = to_itk(data_mov, scale=pxl_sizes_zyx[1:])
                try:
                    parameter_object_adapted = adapt_itk_params(
                        parameter_object=parameter_object, itk_img=itk_img
                    )
                    transformix_object = itk.TransformixFilter.New(itk_img)
                    transformix_object.SetTransformParameterObject(parameter_object_adapted)
                    transformix_object.UpdateLargestPossibleRegion()
                    data_mov_reg = to_numpy(transformix_object.GetOutput())
                    data_mov_reg = unpad_array(data_mov_reg, pad_width_2d)
                except RuntimeError as e:
                    logger.warning(
                        f"Invalid transformation for ROI {ROI_id}: {e}. "
                        "Using untransformed data."
                    )
                    data_mov_reg = unpad_array(data_mov, pad_width_2d)
                except Exception as e:
                    logger.error(f"Failed to adapt parameters for ROI {ROI_id}: {e}")
                    raise

            if use_masks:
                new_images.set_roi_masked(
                    label=int(ROI_id),
                    patch=data_mov_reg,
                )
            else:
                new_images.set_roi(
                    roi=ref_roi,
                    patch=data_mov_reg,
                )
            new_images.consolidate()

        elif axes_list == ["c", "y", "x"]:
            raise NotImplementedError(
                f"`write_registered_zarr` has not been implemented for a zarr with {axes_list=}"
            )
        elif axes_list == ["y", "x"]:
            raise NotImplementedError(
                f"`write_registered_zarr` has not been implemented for a zarr with {axes_list=}"
            )
        else:
            raise NotImplementedError(
                f"`write_registered_zarr` has not been implemented for a zarr with {axes_list=}"
            )

    shutil.rmtree(f"{new_zarr_url}/tables", ignore_errors=True)
    if use_masks:
        shutil.rmtree(f"{new_zarr_url}/labels", ignore_errors=True)

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=apply_registration_elastix,
        logger_name=logger.name,
    )
