import logging
from pathlib import Path
import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from fractal_tasks_core.channels import (
    OmeroChannel,
    get_channel_from_image_zarr,
    get_omero_channel_list,
)
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region,
)
from pydantic import validate_call
from skimage.registration import phase_cross_correlation
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.transform import AffineTransform, warp
from photutils.detection import DAOStarFinder
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def fft_rigid_scale_z(ref_struct, mov_struct, upsample_factor=10):
    """
    Estimate 2D rigid transform (rotation + scale + translation) using a structure-rich channel.
    
    Parameters
    ----------
    ref_struct, mov_struct : 2D ndarray
        Reference and moving structural images.
    upsample_factor : int
        Upsampling factor for subpixel accuracy
    Returns
    -------
    shift_2d : (dy, dx)
    """
    shift_2d, _, _ = phase_cross_correlation(ref_struct, mov_struct,
                                             upsample_factor=upsample_factor, normalization=None)
    return shift_2d

def daofinder(data, threshold, fwhm=5.0):
    """
    Detect point sources (beads) in a 2D image using DAOStarFinder.
    
    Parameters
    ----------
    data : 2D ndarray
        Input image.
    threshold : float
        Absolute intensity threshold.
    fwhm : float
        Full width at half maximum for detection.
    
    Returns
    -------
    list
        List of [x, y] coordinates of detected beads.
    """
    if threshold <= 0:
        raise ValueError(f"Threshold must be positive, got {threshold}")
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, brightest=None, exclude_border=True)
    sources = daofind(data)
    if sources is None:
        return []
    return sources[['xcentroid', 'ycentroid']].to_pandas().values.tolist()

def nearest_neighbors_transform(ref_points, fit_points, max_dist=2, ransac_threshold=0.5):
    """
    Find corresponding points and estimate an affine transform using RANSAC.
    
    Parameters
    ----------
    ref_points : list
        List of [x, y] coordinates for reference beads.
    fit_points : list
        List of [x, y] coordinates for moving beads.
    max_dist : float
        Maximum distance for nearest neighbor matching.
    ransac_threshold : float
        RANSAC reprojection threshold.
    
    Returns
    -------
    tuple
        (tform, dists, ref_pts_corr, fit_pts_corr, inliers)
        tform: 2x3 affine transformation matrix from cv2.estimateAffine2D
        dists: Distances between matched points
        ref_pts_corr: Reference points used in transform
        fit_pts_corr: Moving points used in transform
        inliers: Boolean mask of inlier points
    """
    ref_points = np.array(ref_points)
    fit_points = np.array(fit_points)
    ref_points = ref_points[~np.isnan(ref_points).any(axis=1)]
    fit_points = fit_points[~np.isnan(fit_points).any(axis=1)]
    
    if len(ref_points) == 0 or len(fit_points) == 0:
        return None, [], [], [], []
    
    ref_neighbors = NearestNeighbors(n_neighbors=1).fit(ref_points)
    dists, ref_indices = ref_neighbors.kneighbors(fit_points)
    dists = dists[:, 0]
    ref_indices = ref_indices.ravel()
    fit_indices = np.arange(len(fit_points))
    
    if max_dist is not None:
        mask = dists <= max_dist
        dists = dists[mask]
        ref_indices = ref_indices[mask]
        fit_indices = fit_indices[mask]
    
    ref_pts_corr = ref_points[ref_indices]
    fit_pts_corr = fit_points[fit_indices]
    
    if len(ref_pts_corr) < 3 or len(fit_pts_corr) < 3:
        return None, dists, ref_pts_corr, fit_pts_corr, []
    
    tform, inliers = cv2.estimateAffine2D(fit_pts_corr, ref_pts_corr, ransacReprojThreshold=ransac_threshold)
    inliers = inliers.ravel().astype(bool)
    
    if tform is None:
        logger.warning("Affine transformation estimation failed, returning None")
        return None, dists, ref_pts_corr, fit_pts_corr, []
    
    return tform, dists, ref_pts_corr, fit_pts_corr, inliers

def bead_chromatic_shift_3d(im_ref, im_mov, threshold_abs=300, max_dist=2, ransac_threshold=0.5):
    """
    Calculate 3D chromatic shift using bead detection and affine transformation.
    
    Parameters
    ----------
    im_ref : ndarray
        Reference 3D image (Z, Y, X).
    im_mov : ndarray
        Moving 3D image (Z, Y, X).
    threshold_abs : float
        Absolute intensity threshold for bead detection.
    max_dist : float
        Maximum distance for nearest neighbor matching.
    ransac_threshold : float
        RANSAC reprojection threshold.
    
    Returns
    -------
    tform_3d : dict
        Dictionary containing:
            - 'tform_2d': 2x3 affine transformation matrix from cv2.estimateAffine2D
            - 'translation_z': integer shift along Z
    """
    # Maximum intensity projections for XY registration
    mip_ref = np.max(im_ref, axis=0)
    mip_mov = np.max(im_mov, axis=0)
    
    # Detect beads in MIPs
    ref_dots = daofinder(mip_ref, threshold=threshold_abs, fwhm=5)
    mov_dots = daofinder(mip_mov, threshold=threshold_abs, fwhm=5)
    
    logger.info(f"Detected {len(ref_dots)} spots in ref, {len(mov_dots)} in mov (threshold={threshold_abs})")
    
    # Adjust max_dist if few beads are detected
    if len(ref_dots) < 10 or len(mov_dots) < 10:
        logger.warning(f"Few beads detected (ref: {len(ref_dots)}, mov: {len(mov_dots)}), increasing max_dist")
        max_dist = max_dist * 1.5
    
    # Estimate 2D affine transform
    tform_2d, _, _, _, _ = nearest_neighbors_transform(ref_dots, mov_dots, max_dist=max_dist, ransac_threshold=ransac_threshold)
    
    if tform_2d is None:
        logger.warning("No valid affine transformation found for XY plane, returning identity transform.")
        tform_2d = np.array([[1, 0, 0], [0, 1, 0]])  # Identity transform (2x3)
    
    # Estimate Z translation
    proj_xz_ref = np.max(im_ref, axis=1)
    proj_xz_mov = np.max(im_mov, axis=1)
    proj_yz_ref = np.max(im_ref, axis=2)
    proj_yz_mov = np.max(im_mov, axis=2)
    
    shift_xz = fft_rigid_scale_z(proj_xz_ref, proj_xz_mov, upsample_factor=10)
    shift_yz = fft_rigid_scale_z(proj_yz_ref, proj_yz_mov, upsample_factor=10)
    shift_z = (shift_xz[0] + shift_yz[0]) // 2
    
    tform_3d = {
        'tform_2d': tform_2d,
        'translation_z': int(shift_z)
    }
    return tform_3d

def save_overlay_qc_3D(im_ref, im_mov, tform_3d, out_dir, roi_name, channel_name):
    """
    Save QC overlays for 3D images before and after applying bead-based affine transformation.
    
    Parameters
    ----------
    im_ref : ndarray
        Reference 3D image (Z,Y,X)
    im_mov : ndarray
        Moving 3D image (Z,Y,X)
    tform_3d : dict
        Dictionary with keys: tform_2d (2x3 affine matrix), translation_z
    out_dir : str or Path
        Directory to save QC images
    roi_name : str
        ROI information
    channel_name: str
        Channel information
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply XY affine transform
    tform_2d = tform_3d['tform_2d']
    tform_skimage = AffineTransform(matrix=np.vstack([tform_2d, [0, 0, 1]]))
    im_mov_xy = np.zeros_like(im_mov, dtype=np.float32)
    for z in range(im_mov.shape[0]):
        im_mov_xy[z] = warp(im_mov[z], tform_skimage.inverse, preserve_range=True)
    
    # Apply Z translation
    z_shift = int(tform_3d['translation_z'])
    im_mov_reg = np.zeros_like(im_mov_xy, dtype=np.float32)
    if z_shift > 0:
        im_mov_reg[z_shift:] = im_mov_xy[:-z_shift]
    elif z_shift < 0:
        im_mov_reg[:z_shift] = im_mov_xy[-z_shift:]
    else:
        im_mov_reg = im_mov_xy.copy()
    
    # Normalize for visualization
    def normalize(im):
        im = im.astype(np.float32)
        im -= np.min(im)
        if np.max(im) > 0:
            im /= np.max(im)
        return im
    
    ref_norm = normalize(im_ref)
    mov_norm_before = normalize(im_mov)
    mov_norm_after = normalize(im_mov_reg)
    
    # XY projection
    mip_xy_ref = np.max(ref_norm, axis=0)
    mip_xy_mov_before = np.max(mov_norm_before, axis=0)
    mip_xy_mov_after = np.max(mov_norm_after, axis=0)
    
    # Save raw grayscale MIPs
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_mip_XY_before.png", mip_xy_mov_before, cmap='gray')
    plt.close()
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_mip_XY_after.png", mip_xy_mov_after, cmap='gray')
    plt.close()
    
    # Before overlay XY (ref: green, mov: magenta)
    overlay_xy_before = np.zeros((*mip_xy_ref.shape, 3), dtype=np.float32)
    overlay_xy_before[..., 1] = mip_xy_ref
    overlay_xy_before[..., 0] = mip_xy_mov_before
    overlay_xy_before[..., 2] = mip_xy_mov_before
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_overlay_XY_before.png", overlay_xy_before)
    plt.close()
    
    # After overlay XY (ref: green, mov: magenta)
    overlay_xy_after = np.zeros((*mip_xy_ref.shape, 3), dtype=np.float32)
    overlay_xy_after[..., 1] = mip_xy_ref
    overlay_xy_after[..., 0] = mip_xy_mov_after
    overlay_xy_after[..., 2] = mip_xy_mov_after
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_overlay_XY_after.png", overlay_xy_after)
    plt.close()
    
    # XZ projection
    mip_xz_ref = np.max(ref_norm, axis=1)
    mip_xz_mov_before = np.max(mov_norm_before, axis=1)
    mip_xz_mov_after = np.max(mov_norm_after, axis=1)
    
    # Save raw grayscale MIPs
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_mip_XZ_before.png", mip_xz_mov_before, cmap='gray')
    plt.close()
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_mip_XZ_after.png", mip_xz_mov_after, cmap='gray')
    plt.close()
    
    # Before overlay XZ (ref: green, mov: magenta)
    overlay_xz_before = np.zeros((*mip_xz_ref.shape, 3), dtype=np.float32)
    overlay_xz_before[..., 1] = mip_xz_ref
    overlay_xz_before[..., 0] = mip_xz_mov_before
    overlay_xz_before[..., 2] = mip_xz_mov_before
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_overlay_XZ_before.png", overlay_xz_before)
    plt.close()
    
    # After overlay XZ (ref: green, mov: magenta)
    overlay_xz_after = np.zeros((*mip_xz_ref.shape, 3), dtype=np.float32)
    overlay_xz_after[..., 1] = mip_xz_ref
    overlay_xz_after[..., 0] = mip_xz_mov_after
    overlay_xz_after[..., 2] = mip_xz_mov_after
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_overlay_XZ_after.png", overlay_xz_after)
    plt.close()
    
    logger.info(f"QC images saved to {out_dir}")
    return im_mov_reg

def save_overlay_qc_2D(img_ref, img_mov, tform_2d, out_dir, roi_name, channel_name):
    """
    Save 2D overlay QC images before and after applying 2D affine transform.
    
    Parameters
    ----------
    img_ref : ndarray
        Reference 2D image.
    img_mov : ndarray
        Moving 2D image to align.
    tform_2d : ndarray
        2x3 affine transformation matrix from cv2.estimateAffine2D
    out_dir : str or Path
        Directory to save QC images.
    roi_name : str
        ROI identifier.
    channel_name : str
        Channel identifier.
    
    Returns
    -------
    img_mov_reg : ndarray
        Registered moving image.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tform_skimage = AffineTransform(matrix=np.vstack([tform_2d, [0, 0, 1]]))
    img_mov_reg = warp(img_mov, tform_skimage.inverse, preserve_range=True)
    
    def normalize(im):
        im = im.astype(np.float32)
        im -= np.min(im)
        if np.max(im) > 0:
            im /= np.max(im)
        return im
    
    ref_norm = normalize(img_ref)
    mov_norm_before = normalize(img_mov)
    mov_norm_after = normalize(img_mov_reg)
    
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_raw_before.png", mov_norm_before, cmap='gray')
    plt.close()
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_raw_after.png", mov_norm_after, cmap='gray')
    plt.close()
    
    overlay_before = np.zeros((*ref_norm.shape, 3), dtype=np.float32)
    overlay_before[..., 1] = ref_norm
    overlay_before[..., 0] = mov_norm_before
    overlay_before[..., 2] = mov_norm_before
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_overlay_2D_before.png", overlay_before)
    plt.close()
    
    overlay_after = np.zeros((*ref_norm.shape, 3), dtype=np.float32)
    overlay_after[..., 1] = ref_norm
    overlay_after[..., 0] = mov_norm_after
    overlay_after[..., 2] = mov_norm_after
    plt.imsave(out_dir / f"{roi_name}_{channel_name}_overlay_2D_after.png", overlay_after)
    plt.close()
    
    return img_mov_reg

@validate_call
def compute_registration_beads_seq(
    *,
    zarr_url: str,
    level: int = 0,
    reference_wavelength: str,
    lower_rescale_quantile: float = 0.0,
    upper_rescale_quantile: float = 0.99,
    roi_table: str = "FOV_ROI_table",
    threshold_abs: dict,
    max_dist: float = 2,
    ransac_threshold: float = 0.5,
    overlap: int = 50
) -> None:
    """
    Calculate chromatic shift registration based on bead detection for the FOV using affine transformation.
    
    Args:
        zarr_url: Path or URL to the OME-Zarr image.
        level: Pyramid level for registration (0 for full resolution).
        reference_wavelength: Wavelength for reference channel (e.g., 'A01_C01').
        lower_rescale_quantile: Lower quantile for intensity rescaling.
        upper_rescale_quantile: Upper quantile for intensity rescaling.
        roi_table: Name of the ROI table to loop over.
        threshold_abs: Dictionary mapping channel wavelength IDs to their absolute intensity thresholds for bead detection.
        max_dist: Maximum distance for nearest neighbor matching.
        ransac_threshold: RANSAC reprojection threshold.
        overlap: Number of pixels for blending (unused in this version).
    """
    logger.info(
        f"Running for {zarr_url=}.\n"
        f"Calculating bead-based chromatic shift per {roi_table=} for "
        f"{reference_wavelength=} with thresholds={threshold_abs}."
    )
    
    # Read metadata
    zarr_img = zarr.open(f"{zarr_url}/{level}", mode='r')
    data_shape = zarr_img.shape[1:]  # (Z, Y, X)
    
    ngff_image_meta = load_NgffImageMeta(str(zarr_url))
    coarsening_xy = ngff_image_meta.coarsening_xy
    
    # Read channels
    channels_align: list[OmeroChannel] = get_omero_channel_list(image_zarr_path=zarr_url)
    if not any(reference_wavelength in channel.wavelength_id for channel in channels_align):
        raise ValueError(f"Reference wavelength {reference_wavelength} not found in {zarr_url}.")
    
    # Remove reference channel from alignment list
    for channel in channels_align[:]:
        if channel.wavelength_id == reference_wavelength:
            channels_align.remove(channel)
    if len(channels_align) == 0:
        raise ValueError("No channels left to register.")
    
    # Validate threshold_abs dictionary
    for channel in channels_align:
        if channel.wavelength_id not in threshold_abs:
            raise ValueError(f"No threshold specified for channel {channel.wavelength_id} in threshold_abs dictionary.")
        if threshold_abs[channel.wavelength_id] <= 0:
            raise ValueError(f"Threshold for channel {channel.wavelength_id} must be positive, got {threshold_abs[channel.wavelength_id]}")
    
    # Get reference channel index
    channel_ref: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=zarr_url, wavelength_id=reference_wavelength)
    channel_index_ref = channel_ref.index
    
    # Load reference data
    data_reference_zyx = da.from_zarr(f"{zarr_url}/{level}")[channel_index_ref]
    
    # Load ROI table
    with zarr.open(f"{zarr_url}/tables/{roi_table}", mode="r") as roi_group:
        ROI_table = ad.read_zarr(roi_group)
    logger.info(f"Found {len(ROI_table)} ROIs in {roi_table=} to be processed.")
    
    # Validate table type
    valid_table_types = ["roi_table", "masking_roi_table", "ngff:region_table", None]
    ref_table_attrs = roi_group.attrs.asdict()
    ref_table_type = ref_table_attrs.get("type")
    if ref_table_type not in valid_table_types:
        raise ValueError(f"Invalid ROI table type {ref_table_type}.")
    
    # Pixel sizes
    pxl_sizes_zyx_full_res = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    
    # Build ROI list
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx_full_res,
    )
    check_valid_ROI_indices(list_indices, roi_table)
    
    # Process ROIs
    num_ROIs = len(list_indices)
    compute = True
    for i_ROI in range(num_ROIs):
        logger.info(f"Now processing ROI {i_ROI+1}/{num_ROIs} with channel {channel_ref.wavelength_id} as reference.")
        roi_indices = list_indices[i_ROI]
        region = convert_indices_to_regions(roi_indices)
        img_ref = load_region(data_zyx=data_reference_zyx, region=region, compute=compute)
        img_ref = rescale_intensity(
            img_ref,
            in_range=(np.quantile(img_ref, lower_rescale_quantile), np.quantile(img_ref, upper_rescale_quantile)),
            out_range=(0, 1000)
        )
        img_ref_smooth = gaussian(img_ref, sigma=1)
        
        for channel in channels_align:
            channel_wavelength_acq_x = channel.wavelength_id
            channel_align: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=zarr_url, wavelength_id=channel_wavelength_acq_x)
            channel_index_acq_x = channel_align.index
            data_alignment_zyx = da.from_zarr(f"{zarr_url}/{level}")[channel_index_acq_x]
            img_acq_x = load_region(data_zyx=data_alignment_zyx, region=region, compute=compute)
            img_acq_x = rescale_intensity(
                img_acq_x,
                in_range=(np.quantile(img_acq_x, lower_rescale_quantile), np.quantile(img_acq_x, upper_rescale_quantile)),
                out_range=(0, 1000)
            )
            img_acq_x_smooth = gaussian(img_acq_x, sigma=1)
            
            # Get channel-specific threshold
            channel_threshold = threshold_abs[channel_wavelength_acq_x]
            
            # Compute global transform
            logger.info(f"Calculating global 3D bead-based chromatic shift for channel {channel_wavelength_acq_x} with threshold {channel_threshold}...")
            global_tform_3d = bead_chromatic_shift_3d(
                img_ref_smooth, img_acq_x_smooth, threshold_abs=channel_threshold,
                max_dist=max_dist, ransac_threshold=ransac_threshold
            )
            
            # Save global QC
            save_overlay_qc_3D(
                img_ref_smooth, img_acq_x_smooth, global_tform_3d,
                out_dir=Path(zarr_url) / "registered_qc",
                roi_name=f"roi_{i_ROI}", channel_name=channel_wavelength_acq_x
            )
            save_overlay_qc_2D(
                np.max(img_ref_smooth, axis=0), np.max(img_acq_x_smooth, axis=0), global_tform_3d['tform_2d'],
                out_dir=Path(zarr_url) / "registered_qc_2D",
                roi_name=f"roi_{i_ROI}", channel_name=channel_wavelength_acq_x
            )
            
            # Save transform
            transform_data = {
                'tform_2d': global_tform_3d['tform_2d'],
                'translation_z': global_tform_3d['translation_z']
            }
            fn_3d = Path(zarr_url) / "channel_registration_3D" / f"{roi_table}_roi_{i_ROI}_{channel_wavelength_acq_x}_3D.npy"
            np.save(fn_3d, transform_data)

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task
    run_fractal_task(
        task_function=compute_registration_beads_seq,
        logger_name=logger.name,
    )
