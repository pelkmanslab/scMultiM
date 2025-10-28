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
def apply_channel_registration_beads_seq(
    *,
    zarr_url: str,
    registration_dir: str = None,
    roi_table: str = "FOV_ROI_table",
    reference_wavelength: str,
    level: int = 0,
    overwrite_input: bool = True,
    overwrite_output: bool = True,
):
    """Apply registration to images using transformation parameters from compute_registration_beads_seq.
    
    Supports both 3D and 2D images. For 3D, uses tform_2d (2x3 affine matrix) and translation_z from channel_registration_3D.
    For 2D, uses only tform_2d from channel_registration_3D. Saves debug MIPs (grayscale and overlay) for XY and XZ projections for 3D images.
    
    Args:
        zarr_url: Path or URL to the input OME-Zarr image.
        registration_dir: Directory containing transformation files (default: zarr_url/channel_registration_3D).
        roi_table: Name of the ROI table (default: 'FOV_ROI_table').
        reference_wavelength: Wavelength of the reference channel (e.g., 'A01_C01').
        level: Pyramid level to process (default: 0 for full resolution).
        overwrite_input: If True, replace input Zarr with registered Zarr.
        overwrite_output: If True, overwrite existing output Zarr.
    
    Returns:
        dict: Image list updates with new or updated Zarr URLs.
    """
    if registration_dir is None:
        registration_dir = str(Path(zarr_url) / "channel_registration_3D")
    
    # Validate directory existence
    registration_dir = Path(registration_dir)
    if not registration_dir.exists():
        raise ValueError(f"Registration directory {registration_dir} does not exist.")
    
    logger.info(
        f"Running `apply_channel_registration_beads_seq` on {zarr_url=}, "
        f"{roi_table=}, {level=}, {reference_wavelength=}, {registration_dir=}, "
        f"{overwrite_input=}, {overwrite_output=}."
    )
    
    # Define new Zarr URL
    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    suffix = "channels_registered"
    new_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}_{suffix}"
    
    # Load metadata
    acq_dict = load_NgffWellMeta(well_url).get_acquisition_paths()
    logger.info(f"Well metadata: {load_NgffWellMeta(well_url)}")
    logger.info(f"Acquisitions: {acq_dict}")
    ROI_table = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    coarsening_xy = ngff_image_meta.coarsening_xy
    num_levels = ngff_image_meta.num_levels
    
    # Write registered Zarr
    logger.info("Writing the registered Zarr image to disk")
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
    
    # Handle labels (if any)
    try:
        labels_group = zarr.open_group(f"{zarr_url}/labels", "r")
        label_list = labels_group.attrs["labels"]
        if label_list:
            logger.warning(
                "Skipping registration of labels. Label registration "
                "has not been implemented."
            )
    except (zarr.errors.GroupNotFoundError, KeyError):
        logger.info("No labels found in the Zarr file. Continuing...")
    
    # Copy non-ROI tables
    table_dict_component = _get_table_path_dict(zarr_url)
    table_dict = {}
    for table in table_dict_component:
        if is_standard_roi_table(table):
            table_dict[table] = table_dict_component[table]
        else:
            logger.warning(
                f"{zarr_url} contains a non-standard ROI table {table}. "
                "Copying without transformation."
            )
            table_dict[table] = table_dict_component[table]
    
    if table_dict:
        logger.info(f"Copying tables: {table_dict}")
        new_image_group = zarr.group(new_zarr_url)
        for table in table_dict.keys():
            logger.info(f"Copying table: {table}")
            max_retries = 20
            sleep_time = 5
            current_round = 0
            while current_round < max_retries:
                try:
                    old_table_group = zarr.open_group(table_dict[table], mode="r")
                    break
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
    
    # Clean up Zarr file
    if overwrite_input:
        logger.info("Replacing original Zarr with registered Zarr")
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
    """Write registered Zarr array using transformation parameters from compute_registration_beads_seq.
    
    For 3D images, applies tform_2d (2x3 affine matrix) to each Z-slice and translation_z for Z-shift.
    For 2D images, applies only tform_2d. Saves debug MIPs (grayscale and overlay) for XY and XZ projections for 3D images.
    
    Args:
        zarr_url: Path to the input OME-Zarr image.
        new_zarr_url: Path to the new registered OME-Zarr image.
        registration_dir: Path to transformation files.
        reference_wavelength: Wavelength to register against.
        ROI_table: Fractal ROI table for the component.
        roi_table_name: Name of the ROI table used for registration.
        num_levels: Number of pyramid levels to create.
        coarsening_xy: Coarsening factor between pyramid levels.
        aggregation_function: Function for pyramid downsampling.
        overwrite: Whether to overwrite existing Zarr at new_zarr_url.
    """
    level = 0
    # Read pixel sizes
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    
    # Create ROI indices
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx,
    )
    
    # Initialize Zarr groups
    old_image_group = zarr.open_group(zarr_url, mode="r")
    new_image_group = zarr.group(new_zarr_url)
    new_image_group.attrs.put(old_image_group.attrs.asdict())
    
    # Load data array
    data_array = da.from_zarr(old_image_group[str(level)])
    
    # Create new Zarr array
    new_zarr_array = zarr.create(
        shape=data_array.shape,
        chunks=data_array.chunksize,
        dtype=data_array.dtype,
        store=zarr.storage.FSStore(f"{new_zarr_url}/0"),
        overwrite=overwrite,
        dimension_separator="/",
    )
    
    # Get channels
    channels_align = get_omero_channel_list(image_zarr_path=zarr_url)
    logger.info(f"Channels to align: {[ch.wavelength_id for ch in channels_align]}")
    
    for i_ROI, roi_indices in enumerate(list_indices):
        logger.info(f"Processing ROI {i_ROI+1}/{len(list_indices)}")
        region = convert_indices_to_regions(roi_indices)
        
        # Load reference image for overlays (for 3D debug MIPs)
        ref_channel = next((ch for ch in channels_align if ch.wavelength_id == reference_wavelength), None)
        img_ref = None
        # Copy reference channel unchanged
        if ref_channel:
            generate_copy_of_reference_wavelength(
                zarr_url=zarr_url,
                data_array=data_array,
                new_zarr_array=new_zarr_array,
                region=region,
                reference_wavelength=reference_wavelength,
            )
        
        # Apply transformations to non-reference channels
        for channel in channels_align:
            if channel.wavelength_id == reference_wavelength:
                continue
            channel_wavelength_acq_x = channel.wavelength_id
            logger.info(f"Applying transformation for ROI {i_ROI}, channel {channel_wavelength_acq_x}")
            
            channel_align = get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=channel_wavelength_acq_x,
            )
            ind_ch = channel_align.index
            channel_region = (slice(ind_ch, ind_ch + 1), *region)
            
            # Load image
            image_np = np.squeeze(
                load_region(data_zyx=data_array[ind_ch], region=region, compute=True)
            )
            logger.info(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}")
            logger.info(f"Pre-transform intensity range: {np.min(image_np)}, {np.max(image_np)}")
            
            # Load transformation
            registration_dir = Path(registration_dir)
            fn_3D = registration_dir / "channel_registration_3D" / f"{roi_table_name}_roi_{i_ROI}_{channel_wavelength_acq_x}_3D.npy"
            
            tform = None
            if fn_3D.exists():
                tform = np.load(fn_3D, allow_pickle=True).item()
                logger.info(f"Loaded transformation: {fn_3D}")
            else:
                logger.warning(
                    f"No transformation file for ROI {i_ROI}, channel {channel_wavelength_acq_x}. "
                    f"File {fn_3D} does not exist. Copying untransformed."
                )
                tform = {
                    'tform_2d': np.array([[1, 0, 0], [0, 1, 0]]),  # Identity 2x3 affine matrix
                    'translation_z': 0
                }
            
            # Apply transformation
            if image_np.ndim == 3:
                # Apply 3D transformation (2D affine per slice + Z-shift)
                tform_2d = tform['tform_2d']
                tform_skimage = AffineTransform(matrix=np.vstack([tform_2d, [0, 0, 1]]))
                im_mov_xy = np.zeros_like(image_np, dtype=np.float32)
                for z in range(image_np.shape[0]):
                    im_mov_xy[z] = warp(image_np[z], tform_skimage.inverse, preserve_range=True)
                
                z_shift = int(tform['translation_z'])
                registered_roi = np.zeros_like(im_mov_xy, dtype=np.float32)
                if z_shift > 0:
                    registered_roi[z_shift:] = im_mov_xy[:-z_shift]
                elif z_shift < 0:
                    registered_roi[:z_shift] = im_mov_xy[-z_shift:]
                else:
                    registered_roi = im_mov_xy.copy()
            else:
                # Apply 2D transformation (only tform_2d)
                tform_2d = tform['tform_2d']
                tform_skimage = AffineTransform(matrix=np.vstack([tform_2d, [0, 0, 1]]))
                registered_roi = warp(image_np, tform_skimage.inverse, preserve_range=True)
            
            registered_roi = registered_roi.astype(image_np.dtype)
            logger.info(f"Post-transform intensity range: {np.min(registered_roi)}, {np.max(registered_roi)}")
            
            # Save debug MIPs for 3D images (grayscale and overlay for XY and XZ)
            if image_np.ndim == 3 and img_ref is not None:
                debug_dir = Path(new_zarr_url) / "debug_mips" / f"roi_{i_ROI}"
                debug_dir.mkdir(parents=True, exist_ok=True)
                
                def normalize(im):
                    im = im.astype(np.float32)
                    im -= np.min(im)
                    if np.max(im) > 0:
                        im /= np.max(im)
                    return im
                
                # Compute XY MIPs
                mip_xy_ref = np.max(normalize(img_ref), axis=0)
                mip_xy_mov_before = np.max(normalize(image_np), axis=0)
                mip_xy_mov_after = np.max(normalize(registered_roi), axis=0)
                
                # Save XY grayscale MIPs
                plt.imsave(debug_dir / f"{channel_wavelength_acq_x}_before_mip_xy.png", 
                          mip_xy_mov_before, cmap='gray')
                plt.close()
                plt.imsave(debug_dir / f"{channel_wavelength_acq_x}_after_mip_xy.png", 
                          mip_xy_mov_after, cmap='gray')
                plt.close()
                
                # Save XY overlay MIPs (ref: green, mov: magenta)
                overlay_xy_before = np.zeros((*mip_xy_ref.shape, 3), dtype=np.float32)
                overlay_xy_before[..., 1] = mip_xy_ref
                overlay_xy_before[..., 0] = mip_xy_mov_before
                overlay_xy_before[..., 2] = mip_xy_mov_before
                plt.imsave(debug_dir / f"{channel_wavelength_acq_x}_overlay_before_mip_xy.png", 
                          overlay_xy_before)
                plt.close()
                
                overlay_xy_after = np.zeros((*mip_xy_ref.shape, 3), dtype=np.float32)
                overlay_xy_after[..., 1] = mip_xy_ref
                overlay_xy_after[..., 0] = mip_xy_mov_after
                overlay_xy_after[..., 2] = mip_xy_mov_after
                plt.imsave(debug_dir / f"{channel_wavelength_acq_x}_overlay_after_mip_xy.png", 
                          overlay_xy_after)
                plt.close()
                
                # Compute XZ MIPs
                mip_xz_ref = np.max(normalize(img_ref), axis=1)
                mip_xz_mov_before = np.max(normalize(image_np), axis=1)
                mip_xz_mov_after = np.max(normalize(registered_roi), axis=1)
                
                # Save XZ grayscale MIPs
                plt.imsave(debug_dir / f"{channel_wavelength_acq_x}_before_mip_xz.png", 
                          mip_xz_mov_before, cmap='gray')
                plt.close()
                plt.imsave(debug_dir / f"{channel_wavelength_acq_x}_after_mip_xz.png", 
                          mip_xz_mov_after, cmap='gray')
                plt.close()
                
                # Save XZ overlay MIPs (ref: green, mov: magenta)
                overlay_xz_before = np.zeros((*mip_xz_ref.shape, 3), dtype=np.float32)
                overlay_xz_before[..., 1] = mip_xz_ref
                overlay_xz_before[..., 0] = mip_xz_mov_before
                overlay_xz_before[..., 2] = mip_xz_mov_before
                plt.imsave(debug_dir / f"{channel_wavelength_acq_x}_overlay_before_mip_xz.png", 
                          overlay_xz_before)
                plt.close()
                
                overlay_xz_after = np.zeros((*mip_xz_ref.shape, 3), dtype=np.float32)
                overlay_xz_after[..., 1] = mip_xz_ref
                overlay_xz_after[..., 0] = mip_xz_mov_after
                overlay_xz_after[..., 2] = mip_xz_mov_after
                plt.imsave(debug_dir / f"{channel_wavelength_acq_x}_overlay_after_mip_xz.png", 
                          overlay_xz_after)
                plt.close()
            
            # Write to Zarr
            if image_np.ndim == 3:
                img = np.expand_dims(registered_roi, axis=0)
            else:
                img = np.expand_dims(registered_roi, axis=(0, 1))
            logger.info(f"Registered ROI shape: {registered_roi.shape}, Zarr write shape: {img.shape}")
            chunks = tuple(int(s) for s in img.shape)
            dask_img = da.from_array(img, chunks=chunks)
            dask_img.to_zarr(
                url=new_zarr_array,
                region=channel_region,
                compute=True,
            )
    
    # Build pyramid
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
    """Write a copy of the reference wavelength to the new Zarr."""
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
        task_function=apply_channel_registration_beads_seq,
        logger_name=logger.name,
    )
