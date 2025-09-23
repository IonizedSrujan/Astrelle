# galaxy.py
# Handles querying 2MASS and rendering Sersic galaxy models.

import numpy as np
import pandas as pd
import pyvo
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import fftpack
from scipy.special import gamma, gammainc 
from scipy.optimize import brentq

# Use relative imports for package structure.
from .utils import add_stamp, gaussian_psf_kernel

# Estimates the Sersic index 'n' from a galaxy's concentration index.
def estimate_n_from_concentration(c_index):
    if pd.isna(c_index) or c_index <= 0:
        return 2.5 # Default value for missing data.
    if c_index < 2.5: n = 0.5 + (c_index - 2.0)
    elif c_index < 4.0: n = 1.0 + (c_index - 2.8) * 1.5
    else: n = 4.0 + (c_index - 4.5) * 2.0
    return np.clip(n, 0.5, 8.0) # Keep n within a reasonable range.

# Queries the IRSA 2MASS Extended Source Catalog (XSC) for galaxies in the FOV.
def query_galaxies_irsa_xsc(center_ra, center_dec, width_arcmin, height_arcmin):
    half_w_deg = width_arcmin / 120.0
    half_h_deg = height_arcmin / 120.0
    ra_min, ra_max = center_ra - half_w_deg, center_ra + half_w_deg
    dec_min, dec_max = center_dec - half_h_deg, center_dec + half_h_deg

    adql = f"""
    SELECT *
    FROM fp_xsc
    WHERE ra BETWEEN {ra_min:.9f} AND {ra_max:.9f}
      AND dec BETWEEN {dec_min:.9f} AND {dec_max:.9f}
    """
    try:
        tap = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
        job = tap.search(adql)
        df = job.to_table().to_pandas()
        return df, f"2MASS XSC query returned {len(df)} galaxies."
    except Exception as e:
        return pd.DataFrame(), f"ERROR querying 2MASS XSC: {e}"

# Calculates the Sersic profile parameter 'bn'.
def bn_of_n(n):
    # This is an approximation related to the incomplete gamma function.
    def f(b):
        return gammainc(2.0 * n, b) - 0.5
    try:
        # Find the root of the function to solve for b_n.
        return brentq(f, 1e-6, 200.0)
    except ValueError:
        # Fallback to a polynomial approximation if the root finding fails.
        return 1.9992 * n - 0.3271

# Calculates the intensity of a Sersic profile at a given radius.
def sersic_intensity(r, Ie, re, n):
    if re <= 0 or n <= 0: return np.zeros_like(r)
    b = bn_of_n(n)
    # Use errstate to prevent overflow warnings with large exponents.
    with np.errstate(over='ignore'):
        x = (r / re)**(1.0 / n)
    return Ie * np.exp(-b * (x - 1.0))

# Generates a 2D Sersic profile stamp.
def sample_sersic_to_image(shape, pixel_scale, Ie, re_arcsec, n, ellip, theta_deg, 
                           x0=None, y0=None, oversample=5, psf_kernel=None):
    ny, nx = shape
    if x0 is None: x0 = nx / 2.0
    if y0 is None: y0 = ny / 2.0
    re_pix = re_arcsec / pixel_scale

    # Create an oversampled grid to render the profile at higher resolution.
    nx_os, ny_os = nx * oversample, ny * oversample
    x = (np.arange(nx_os) + 0.5) / oversample
    y = (np.arange(ny_os) + 0.5) / oversample
    xv, yv = np.meshgrid(x, y)
    
    # Rotate and scale coordinates to account for ellipticity and position angle.
    x_c, y_c = xv - x0, yv - y0
    theta_rad = np.deg2rad(90.0 - theta_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    x_rot = x_c * cos_t - y_c * sin_t
    y_rot = x_c * sin_t + y_c * cos_t
    
    # Calculate elliptical radius for each pixel.
    q = 1.0 - ellip
    r = np.sqrt(x_rot**2 + (y_rot / q)**2)

    # Calculate intensity and rebin back to the original resolution.
    I_os = sersic_intensity(r, Ie, re_pix, n)
    I = I_os.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))
    
    # Convolve the galaxy profile with the atmospheric PSF.
    if psf_kernel is not None:
        # Create a zero-padded array with the same dimensions as the galaxy stamp.
        padded_kernel = np.zeros_like(I)
        
        # Get shapes and centers for slicing.
        stamp_h, stamp_w = I.shape
        psf_h, psf_w = psf_kernel.shape
        
        # Calculate the slice to place the smaller PSF kernel in the center of the larger padded array.
        y_start = (stamp_h - psf_h) // 2
        x_start = (stamp_w - psf_w) // 2
        
        padded_kernel[y_start : y_start + psf_h, x_start : x_start + psf_w] = psf_kernel
        
        # Perform convolution using FFTs. The kernel must be shifted so its center is at (0,0) for the FFT.
        I_fft = fftpack.fft2(I)
        kernel_fft = fftpack.fft2(fftpack.ifftshift(padded_kernel))
        
        # Multiply in the frequency domain and transform back.
        convolved_fft = I_fft * kernel_fft
        I = np.real(fftpack.ifft2(convolved_fft))

    return I

# Renders all galaxies from the dataframe onto the image.
def add_galaxies_to_image(image, galaxies_df, pixel_scale, zero_point, exposure_s, seeing_fwhm_arcsec):
    fwhm_px = seeing_fwhm_arcsec / pixel_scale
    psf_kernel = gaussian_psf_kernel(fwhm_px)
    
    for _, gal in galaxies_df.iterrows():
        # Skip galaxies with missing magnitude or size data.
        if 'j_m_ext' not in gal or pd.isna(gal['j_m_ext']): continue
        re_arcsec = gal.get('j_r_eff', gal.get('r_3sig')) # Effective radius.
        if pd.isna(re_arcsec) or re_arcsec <= 0: continue
            
        # Get galaxy parameters from the dataframe.
        mag = gal['j_m_ext']
        re_pix = re_arcsec / pixel_scale
        x, y = gal['x'], gal['y']
        n = estimate_n_from_concentration(gal.get('j_con_indx'))
        axis_ratio = gal.get('sup_ba', 1.0)
        if pd.isna(axis_ratio) or axis_ratio <= 0: axis_ratio = 1.0
        ellip = 1.0 - axis_ratio
        pos_angle = gal.get('sup_phi', 0.0)
        if pd.isna(pos_angle): pos_angle = 0.0
        
        # Calculate total flux.
        total_flux_e = zero_point * 10**(-0.4 * mag) * exposure_s
        
        # Dynamically calculate the stamp size based on galaxy size and exposure time.
        # This prevents the "boxy" cutoff artifact for bright objects in long exposures.
        base_radius_pix = int(round(8 * re_pix))
        exposure_scale_factor = max(1.0, np.sqrt(exposure_s / 60.0))
        radius_pix = max(25, int(base_radius_pix * exposure_scale_factor))
        stamp_shape = (2 * radius_pix + 1, 2 * radius_pix + 1)
        
        # Generate a unit-flux Sersic stamp.
        unit_stamp = sample_sersic_to_image(
            shape=stamp_shape, pixel_scale=pixel_scale, Ie=1.0,
            re_arcsec=re_arcsec, n=n, ellip=ellip,
            theta_deg=pos_angle, x0=radius_pix, y0=radius_pix,
            oversample=7, psf_kernel=psf_kernel
        )
        
        # Normalize the stamp and scale it by the total flux.
        stamp_sum = unit_stamp.sum()
        if stamp_sum > 0:
            normalized_stamp = unit_stamp / stamp_sum
            add_stamp(image, normalized_stamp, x, y, total_flux_e)
            
    return image


