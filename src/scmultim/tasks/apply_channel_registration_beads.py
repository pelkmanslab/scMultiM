import logging
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.channels import (
    OmeroChannel,
    get_channel_from_image_zarr,
    get_omero_channel_list,
)
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    is_standard_roi_table,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks._zarr_utils import (
    _update_well_metadata,
)
from fractal_tasks_core.utils import (
    _get_table_path_dict,
    _split_well_path_image_path,
)
from pydantic import validate_call
from skimage.transform import AffineTransform, warp
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@validate_call
def apply_channel_registration_beads(
    *,
    zarr_url: str,
    registration_dir: str = None,
    roi_table: str = "FOV_ROI_table",
    reference_wavelength: str,
    level: int = 0,
    overwrite_input: bool = True,
    overwrite_output: bool = True,
):
    """Apply registration to images using transformation parameters from compute_registration_beads.
    
    Supports both 3D and 2D images. For 3D, uses channel_registration_3D; for 2D, uses channel_registration_2D if specified.
    """
    if registration_dir is None:
        registration_dir = str(Path(zarr_url) / "channel_registration")
    
    # Validate directory existence
    registration_dir = Path(registration_dir)
    if not registration_dir.exists():
        logger.warning(f"Registration directory {registration_dir} does not exist.")
        
    logger.info(
        f"Running `apply_channel_registration_beads` on {zarr_url=}, "
        f"{roi_table=}, {level=}, {reference_wavelength=}, {registration_dir=}."
        f"Using {overwrite_input=}"
    )
    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    suffix = "channels_registered"
    new_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}_{suffix}"
    acq_dict = load_NgffWellMeta(well_url).get_acquisition_paths()
    logger.info(f"Well metadata: {load_NgffWellMeta(well_url)}")
    logger.info(f"Acquisitions: {acq_dict}")
    ROI_table = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    coarsening_xy = ngff_image_meta.coarsening_xy
    num_levels = ngff_image_meta.num_levels
    
    logger.info("Write the registered Zarr image to disk")
    write_registered_zarr(
        zarr_url=zarr_url,
        new_zarr_url=new_zarr_url,
        registration_dir=registration_dir,
        reference_wavelength=reference_wavelength,
        ROI_table=ROI_table,
        roi_table_name=roi_table,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.mean,
        overwrite=overwrite_output,
    )
    
    try:
        labels_group = zarr.open_group(f"{zarr_url}/labels", "r")
        label_list = labels_group.attrs["labels"]
        if label_list:
            logger.warning(
                "Skipping registration of labels ... Label registration "
                "has not been implemented."
            )
    except (zarr.errors.GroupNotFoundError, KeyError):
        logger.info("No labels found in the zarr file ... Continuing ...")
    
    table_dict_component = _get_table_path_dict(zarr_url)
    table_dict = {}
    for table in table_dict_component:
        if is_standard_roi_table(table):
            table_dict[table] = table_dict_component[table]
        else:
            logger.warning(
                f"{zarr_url} contained a table that is not a standard "
                "ROI table. The `Apply Registration To Image task` is "
                "best used before additional tables are generated. It "
                f"will copy the {table} from this acquisition without "
                "applying any transformations."
            )
            table_dict[table] = table_dict_component[table]
    
    if table_dict:
        logger.info(f"Processing the tables: {table_dict}")
        new_image_group = zarr.group(new_zarr_url)
        for table in table_dict.keys():
            logger.info(f"Copying table: {table}")
            max_retries = 20
            sleep_time = 5
            current_round = 0
            while current_round < max_retries:
                try:
                    old_table_group = zarr.open_group(table_dict[table], mode="r")
                    current_round = max_retries
                except zarr.errors.GroupNotFoundError:
                    logger.debug(
                        f"Table {table} not found in attempt {current_round}. "
                        f"Waiting {sleep_time} seconds before trying again."
                    )
                    current_round += 1
                    time.sleep(sleep_time)
            curr_table = ad.read_zarr(table_dict[table])
            write_table(
                new_image_group,
                table,
                curr_table,
                table_attrs=old_table_group.attrs.asdict(),
                overwrite=True,
            )
    ####################
    # Clean up Zarr file
    ####################
    if overwrite_input:
        logger.info("Replace original zarr image with the newly created Zarr image")
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
        except ValueError:
            logger.warning(f"{new_zarr_url} was already listed in well metadata")
    
    return image_list_updates

def write_registered_zarr(
    zarr_url: str,
    new_zarr_url: str,
    registration_dir: str,
    reference_wavelength: str,
    ROI_table: ad.AnnData,
    roi_table_name: str,
    num_levels: int,
    coarsening_xy: int = 2,
    aggregation_function: Callable = np.mean,
    overwrite: bool = True,
):
    """Write registered zarr array using transformation parameters from compute_registration_beads.
    
    This function loads the image or label data from a zarr array based on the
    ROI bounding-box coordinates and stores them into a new zarr array.
    The new Zarr array has the same shape as the original array, but will have
    0s where the ROI tables don't specify loading of the image data.
    The ROIs loaded from `list_indices` will be written into the
    `list_indices_ref` position, thus performing translational registration if
    the two lists of ROI indices vary.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be used as
            the basis for the new OME-Zarr image.
        new_zarr_url: Path or url to the new OME-Zarr image to be written
        registration_dir: Path to registration files.
        reference_wavelength: Wavelength to register against.
        ROI_table: Fractal ROI table for the component
        roi_table_name: Name of the ROI table that the registration was
            calculated on. Used to load the correct registration files.
        num_levels: Number of pyramid layers to be created (argument of
            `build_pyramid`).
        coarsening_xy: Coarsening factor between pyramid levels
        aggregation_function: Function to be used when downsampling (argument
            of `build_pyramid`).
        overwrite: Whether an existing zarr at new_zarr_url should be
            overwritten.

    """
    level = 0
    # Read pixel sizes from Zarr attributes
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    
    #Create list of indices for ROIs at full resolution
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    
    old_image_group = zarr.open_group(zarr_url, mode="r")
    old_ngff_image_meta = load_NgffImageMeta(zarr_url)
    new_image_group = zarr.group(new_zarr_url)
    new_image_group.attrs.put(old_image_group.attrs.asdict())
    
    # Loop over all channels. For each channel, write full-res image data.
    data_array = da.from_zarr(old_image_group[str(level)])
    
    new_zarr_array = zarr.create(
        shape=data_array.shape,
        chunks=data_array.chunksize,
        dtype=data_array.dtype,
        store=zarr.storage.FSStore(f"{new_zarr_url}/0"),
        overwrite=overwrite,
        dimension_separator="/",
    )
    axes_list = old_ngff_image_meta.axes_names
    
    # List available transformation files
    registration_dir = Path(registration_dir)
    transform_files = list(registration_dir.glob('*.npy')) if registration_dir.exists() else []
    logger.info(f"Available transform files: {[f.name for f in transform_files]}")
    
    # Get channels once (outside ROI loop for efficiency)
    channels_align = get_omero_channel_list(image_zarr_path=zarr_url)
    logger.info(f"Channels to align: {[ch.wavelength_id for ch in channels_align]}")
    
    for i_ROI, roi_indices in enumerate(list_indices):
        logger.info(f"Now processing ROI {i_ROI+1}/{len(list_indices)}.")
        region = convert_indices_to_regions(roi_indices)
        
        # Copy reference channel (unchanged) for this ROI region
        ref_channel = next((ch for ch in channels_align if ch.wavelength_id == reference_wavelength), None)
        if ref_channel:
            generate_copy_of_reference_wavelength(
                zarr_url=zarr_url,
                data_array=data_array,
                new_zarr_array=new_zarr_array,
                region=region,
                reference_wavelength=reference_wavelength,
            )
        
        # Apply transformation to each non-reference channel
        for channel in channels_align:
            if channel.wavelength_id == reference_wavelength:
                continue  # Already copied above
            channel_wavelength_acq_x = channel.wavelength_id
            logger.info(
                f"Processing ROI index {i_ROI}, wavelength_id {channel_wavelength_acq_x}"
            )
            channel_align: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=channel_wavelength_acq_x,
            )
            ind_ch = channel_align.index
            # Define region
            channel_region = (slice(ind_ch, ind_ch + 1), *region)
            
            image_np = np.squeeze(
                load_region(data_zyx=data_array[ind_ch], region=region, compute=True)
            )
            
            # Try multiple channel ID formats
            channel_id_variants = [channel_wavelength_acq_x]
            tform = None
            tform_type = None
            for variant in channel_id_variants:
                if image_np.ndim == 3:
                    fn_3D = registration_dir / f"{roi_table_name}_roi_{i_ROI}_{variant}_3D.npy"
                    if fn_3D.exists():
                        tform = np.load(fn_3D, allow_pickle=True).item()
                        tform_type = '3D'
                        logger.info(f"Found 3D transformation file: {fn_3D}")
                        break  # Found it, no need for more variants
                elif image_np.ndim == 2:
                    fn_2D = registration_dir / f"{roi_table_name}_roi_{i_ROI}_{variant}_2D.npy"  # Fixed: registration_dir_2d -> registration_dir
                    if fn_2D.exists():
                        tform = np.load(fn_2D, allow_pickle=True).item()
                        tform_type = '2D'
                        logger.info(f"Found 2D transformation file: {fn_2D}")
                        break  # Found it
            if tform is None:
                logger.warning(
                    f"No transformation file found for ROI {i_ROI}, channel {channel_wavelength_acq_x}. "
                    f"Copying untransformed."
                )
                registered_roi = image_np
            else:
                # Apply transformation
                if tform_type == '3D':
                    # --- Apply XY affine (rotation + scale + translation) FIRST ---
                    theta = tform['rotation']
                    scale = tform['scale']
                    ty, tx = tform['translation_xy']  # Note: (dy, dx) -> (y, x) for AffineTransform
                    tform_sk = AffineTransform(scale=(scale, scale),
                                               rotation=np.deg2rad(theta),
                                               translation=(tx, ty))
                    im_mov_xy = np.zeros_like(image_np, dtype=np.float32)
                    for z in range(image_np.shape[0]):
                        im_mov_xy[z] = warp(image_np[z], tform_sk.inverse, preserve_range=True)
                    
                    # --- Apply Z translation SECOND ---
                    z_shift = int(tform['translation_z'])
                    registered_roi = np.zeros_like(im_mov_xy, dtype=np.float32)
                    if z_shift > 0:
                        registered_roi[z_shift:] = im_mov_xy[:-z_shift]
                    elif z_shift < 0:
                        registered_roi[:z_shift] = im_mov_xy[-z_shift:]
                    else:
                        registered_roi = im_mov_xy.copy()
                elif tform_type == '2D':
                    tform_sk = AffineTransform(
                        rotation=np.deg2rad(tform['rotation']),
                        scale=tform['scale'],
                        translation=tform['translation'][::-1] # (dy, dx) -> (x, y)
                    )
                    registered_roi = warp(image_np, tform_sk.inverse, preserve_range=True)
                # Cast back to original dtype (warp returns float64)
                registered_roi = registered_roi.astype(image_np.dtype)
            
            # Write to disk
            if image_np.ndim == 3:
                img = np.expand_dims(registered_roi, axis=0)
            else:  # 2D
                img = np.expand_dims(registered_roi, axis=(0, 1))
            logger.info(
                f"registered_roi shape {registered_roi.shape} img shape {img.shape}"
            )
            chunks = tuple(int(s) for s in img.shape)
            dask_img = da.from_array(img, chunks=chunks)
            dask_img.to_zarr(
                url=new_zarr_array,
                region=channel_region,
                compute=True,
            )
    
    build_pyramid(
        zarrurl=new_zarr_url,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_array.chunksize,
        aggregation_function=aggregation_function,
    )

def generate_copy_of_reference_wavelength(
    zarr_url: str,
    data_array: da.Array,
    new_zarr_array: zarr.Array,
    region: tuple,
    reference_wavelength: str,
):
    """Write a copy of the reference wavelength to the new zarr."""
    channel_reference = get_channel_from_image_zarr(
        image_zarr_path=zarr_url,
        wavelength_id=reference_wavelength,
    )
    ind_ch = channel_reference.index
    channel_region = (slice(ind_ch, ind_ch + 1), *region)
    img = load_region(data_zyx=data_array[ind_ch], region=region, compute=True)
    img = np.expand_dims(img, 0)
    da.array(img).to_zarr(
        url=new_zarr_array,
        region=channel_region,
        compute=True,
    )

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task
    run_fractal_task(
        task_function=apply_channel_registration_beads,
        logger_name=logger.name,
    )
