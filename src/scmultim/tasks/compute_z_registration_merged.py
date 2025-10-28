import logging
from pathlib import Path
import anndata as ad
import dask.array as da
import numpy as np
import zarr

from ngio import open_ome_zarr_container
from pydantic import validate_call
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter, fourier_shift
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.transform import warp, AffineTransform, rotate, rescale, warp_polar
from skimage.registration import phase_cross_correlation
from skimage.filters import difference_of_gaussians
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def fft_rigid_scale_2d(ref_struct, mov_struct, upsample_factor=10):
    """
    Estimate 2D rigid transform (rotation + scale + translation)
    using a structure-rich channel.
   
    Parameters
    ----------
    ref_struct, mov_struct : 2D ndarray
        Reference and moving structural images.
    upsample_factor : int
        Upsampling factor for subpixel accuracy
    Returns
    -------
    tform2d : dict
        'rotation' : angle in degrees
        'scale' : isotropic scaling factor
        'translation' : (dy, dx)
    """
    # --- Step 1. FFT magnitudes with high-pass filter ---
    fft_ref = np.abs(fftshift(fft2(ref_struct)))
    fft_mov = np.abs(fftshift(fft2(mov_struct)))
    fft_ref = difference_of_gaussians(fft_ref, low_sigma=1, high_sigma=20)
    fft_mov = difference_of_gaussians(fft_mov, low_sigma=1, high_sigma=20)
    # --- Step 2. Log-polar transform to estimate rotation & scale ---
    radius = max(ref_struct.shape) // 2  # Use max to allow larger radius if needed
    log_polar_ref = warp_polar(fft_ref, radius=radius, scaling='log')
    log_polar_mov = warp_polar(fft_mov, radius=radius, scaling='log')
    shift, _, _ = phase_cross_correlation(log_polar_ref, log_polar_mov,
                                           upsample_factor=upsample_factor, normalization=None)
    # Correct calculation based on scikit-image example
    rotation = shift[0]  # Shift in theta dimension (degrees)
    klog = radius / np.log(radius)
    scale = 1 / np.exp(shift[1] / klog)  # Shift in rho for scale
    
    # --- Step 4. Estimate translation after correcting rotation+scale ---
    shift_2d, _, _ = phase_cross_correlation(ref_struct, mov_struct,
                                              upsample_factor=upsample_factor, normalization=None)
    return shift_2d #dz, dx or dz, dy

def z_translation_3d(im_ref, im_mov, upsample_factor=10):
    """
    Compute Z-direction translation using FFT-based registration on 3D images.
    
    Parameters
    ----------
    im_ref : ndarray
        Reference 3D image (Z,Y,X)
    im_mov : ndarray
        Moving 3D image (Z,Y,X)
    upsample_factor : int
        Upsampling factor for subpixel accuracy
    
    Returns
    -------
    tform_z : dict
        Dictionary containing:
            - 'translation_z': integer shift along Z
    """
    # Max projection along Y to compute Z translation
    proj_xz_ref = np.max(im_ref, axis=(1))
    proj_xz_mov = np.max(im_mov, axis=(1))
    
    # Use FFT-based translation for XZ
    shift_xz = fft_rigid_scale_2d(proj_xz_ref, proj_xz_mov, upsample_factor=upsample_factor)
    
    # Max projection along Y to compute Z translation
    proj_yz_ref = np.max(im_ref, axis=(2))
    proj_yz_mov = np.max(im_mov, axis=(2))
    
    # Use FFT-based translation for XZ
    shift_yz = fft_rigid_scale_2d(proj_yz_ref, proj_yz_mov, upsample_factor=upsample_factor)
    
    shift_z = (shift_xz[0]+shift_yz[0])//2
    tform_z = {
        'translation_z': int(shift_z)
    }
    
    return tform_z

def save_z_overlay_qc(im_ref, im_mov, tform_z, out_dir, roi_name, channel_names):
    """
    Save QC overlays (XZ projection) for 3D images before and after applying Z translation.
    Also saves raw normalized XZ MIP images (before/after) as grayscale PNGs.
    
    Parameters
    ----------
    im_ref : ndarray
        Reference 3D image (Z,Y,X)
    im_mov : ndarray
        Moving 3D image (Z,Y,X)
    tform_z : dict
        Dictionary with key: translation_z
    out_dir : str or Path
        Directory to save QC images
    roi_name : str
        ROI information
    channel_names : list[str]
        List of channel names used for registration
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply Z translation
    z_shift = int(tform_z['translation_z'])
    im_mov_reg = np.zeros_like(im_mov, dtype=np.float32)
    if z_shift > 0:
        im_mov_reg[z_shift:] = im_mov[:-z_shift]
    elif z_shift < 0:
        im_mov_reg[:z_shift] = im_mov[-z_shift:]
    else:
        im_mov_reg = im_mov.copy()
    
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
    
    # XZ projection
    mip_xz_ref = np.max(ref_norm, axis=1)
    mip_xz_mov_before = np.max(mov_norm_before, axis=1)
    mip_xz_mov_after = np.max(mov_norm_after, axis=1)
    
    # Save raw grayscale MIPs for XZ (before/after)
    channel_str = "_".join(channel_names)
    plt.imsave(out_dir / f"{roi_name}_{channel_str}_mip_XZ_before.png", mip_xz_mov_before, cmap='gray')
    plt.imsave(out_dir / f"{roi_name}_{channel_str}_mip_XZ_after.png", mip_xz_mov_after, cmap='gray')
    
    # Before overlay XZ
    overlay_xz_before = np.zeros((*mip_xz_ref.shape, 3), dtype=np.float32)
    overlay_xz_before[..., 1] = mip_xz_ref  # green
    overlay_xz_before[..., 0] = mip_xz_mov_before  # red
    overlay_xz_before[..., 2] = mip_xz_mov_before  # blue for magenta
    plt.imsave(out_dir / f"{roi_name}_{channel_str}_overlay_XZ_before.png", overlay_xz_before)
    
    # After overlay XZ
    overlay_xz_after = np.zeros((*mip_xz_ref.shape, 3), dtype=np.float32)
    overlay_xz_after[..., 1] = mip_xz_ref  # green
    overlay_xz_after[..., 0] = mip_xz_mov_after  # red
    overlay_xz_after[..., 2] = mip_xz_mov_after  # blue for magenta
    plt.imsave(out_dir / f"{roi_name}_{channel_str}_overlay_XZ_after.png", overlay_xz_after)
    
    print(f"QC images saved to {out_dir}")
    return im_mov_reg

@validate_call
def compute_z_registration_merged(
    *,
    zarr_url: str,
    ref_zarr_url: str,
    level: int = 0,
    wavelengths: list[str],
    lower_rescale_quantile: float = 0.0,
    upper_rescale_quantile: float = 0.99,
    roi_table: str = "FOV_ROI_table",
) -> None:
    """
    Calculate Z-direction translation registration by merging specified channels.
    
    Args:
        zarr_url: Path or url to the OME-Zarr image to be processed.
        ref_zarr_url: Path or url to the reference OME-Zarr image
        level: Pyramid level of the image to be used for registration (0 for full resolution).
        wavelengths: List of wavelength IDs to merge (e.g., ['A01_C01', 'A02_C02']).
        lower_rescale_quantile: Lower quantile for rescaling image intensities (default: 0.0).
        upper_rescale_quantile: Upper quantile for rescaling image intensities (default: 0.99).
        roi_table: Name of the ROI table to loop over (e.g., 'FOV_ROI_table' or 'well_ROI_table').
    """
    logger.info(
        f"Running for {zarr_url=}, {roi_table=}, wavelengths={wavelengths}, "
    )
    
    # Load channels
    ome_zarr_ref = open_ome_zarr_container(ref_zarr_url)
    ome_zarr_mov = open_ome_zarr_container(zarr_url)
    
    # Validate wavelength IDs
    ref_channel_indices = []
    for wavelength_id in wavelengths:
        idx = ome_zarr_ref.image_meta._get_channel_idx_by_wavelength_id(wavelength_id)
        ref_channel_indices.append(idx)
        logger.info(f"Reference channel: index={idx}, wavelength_id={wavelength_id}")
    mov_channel_indices = []
    for wavelength_id in wavelengths:
        idx = ome_zarr_mov.image_meta._get_channel_idx_by_wavelength_id(wavelength_id)
        mov_channel_indices.append(idx)
        logger.info(f"Moving channel: index={idx}, wavelength_id={wavelength_id}")

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

    # Read pixel sizes
    pxl_sizes_zyx_ref = ome_zarr_ref.get_image(path=str(level)).pixel_size.zyx
    pxl_sizes_zyx_mov = ome_zarr_mov.get_image(path=str(level)).pixel_size.zyx
    logger.info(f"pxl_sizes_zyx_ref: {pxl_sizes_zyx_ref}")
    logger.info(f"pxl_sizes_zyx_mov: {pxl_sizes_zyx_mov}")
    if pxl_sizes_zyx_ref != pxl_sizes_zyx_mov:
        logger.warning(
            f"Pixel sizes differ: ref={pxl_sizes_zyx_ref}, mov={pxl_sizes_zyx_mov}. "
            "Proceeding with registration, but results may be affected."
        )
    
    # Process ROIs
    num_ROIs = len(ref_roi_table.rois())
    compute = True
    for i_ROI, ref_roi in enumerate(ref_roi_table.rois()):
        ROI_id = ref_roi.name
        logger.info(
            f"Processing ROI {i_ROI+1}/{num_ROIs} (ID: {ROI_id})."
        )
        
        # Load and merge reference channels
        ref_roi = ref_roi_table.get(ROI_id)
        img_ref_merged = None
        for idx in ref_channel_indices:
            img = ref_images.get_roi(
                roi=ref_roi,
                c=idx,
                ).squeeze()
            img = rescale_intensity(
                img,
                in_range=(
                    np.quantile(img, lower_rescale_quantile),
                    np.quantile(img, upper_rescale_quantile),
                ),
            )
            if img_ref_merged is None:
                img_ref_merged = img
            else:
                img_ref_merged = np.maximum(img_ref_merged, img)
        
        # Smooth reference image
        img_ref_smooth = gaussian_filter(img_ref_merged, sigma=1)
        
        # Load and merge alignment channels
        mov_roi = mov_roi_table.get(ROI_id)
        img_mov_merged = None
        for idx in mov_channel_indices:
            img = mov_images.get_roi(
                roi=mov_roi,
                c=idx,
            ).squeeze()
            img = rescale_intensity(
                img,
                in_range=(
                    np.quantile(img, lower_rescale_quantile),
                    np.quantile(img, upper_rescale_quantile),
                ),
            )
            if img_mov_merged is None:
                img_mov_merged = img
            else:
                img_mov_merged = np.maximum(img_mov_merged, img)
        
        # Smooth moving image
        img_mov_smooth = gaussian_filter(img_mov_merged, sigma=1)
        
        # Calculate Z translation
        logger.info("Calculating Z-direction FFT translation...")
        tform_z = z_translation_3d(img_ref_smooth, img_mov_smooth)
        
        # Save QC images
        qc_dir = Path(zarr_url) / "registered_z_qc"
        img_mov_reg = save_z_overlay_qc(
            img_ref_smooth,
            img_mov_smooth,
            tform_z=tform_z,
            out_dir=qc_dir,
            roi_name=f"roi_{i_ROI}",
            channel_names=wavelengths,
        )
        
        # Save transformation
        fn_z = (
            Path(zarr_url)
            / "registration_z"
            / f"{roi_table}_roi_{i_ROI}_z.npy"
        )
        fn_z.parent.mkdir(exist_ok=True, parents=True)
        np.save(fn_z, tform_z)

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task
    run_fractal_task(
        task_function=compute_z_registration_merged_channels,
        logger_name=logger.name,
    )
