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

# Use relative imports for package structure
from .utils import add_stamp, gaussian_psf_kernel

def estimate_n_from_concentration(c_index):
    if pd.isna(c_index) or c_index <= 0:
        return 2.5
    if c_index < 2.5: n = 0.5 + (c_index - 2.0)
    elif c_index < 4.0: n = 1.0 + (c_index - 2.8) * 1.5
    else: n = 4.0 + (c_index - 4.5) * 2.0
    return np.clip(n, 0.5, 8.0)

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

def bn_of_n(n):
    def f(b):
        return gammainc(2.0 * n, b) - 0.5
    try:
        return brentq(f, 1e-6, 200.0)
    except ValueError:
        return 1.9992 * n - 0.3271

def sersic_intensity(r, Ie, re, n):
    if re <= 0 or n <= 0: return np.zeros_like(r)
    b = bn_of_n(n)
    with np.errstate(over='ignore'):
        x = (r / re)**(1.0 / n)
    return Ie * np.exp(-b * (x - 1.0))

def sample_sersic_to_image(shape, pixel_scale, Ie, re_arcsec, n, ellip, theta_deg, 
                           x0=None, y0=None, oversample=5, psf_kernel=None):
    ny, nx = shape
    if x0 is None: x0 = nx / 2.0
    if y0 is None: y0 = ny / 2.0
    re_pix = re_arcsec / pixel_scale

    nx_os, ny_os = nx * oversample, ny * oversample
    x = (np.arange(nx_os) + 0.5) / oversample
    y = (np.arange(ny_os) + 0.5) / oversample
    xv, yv = np.meshgrid(x, y)
    
    x_c, y_c = xv - x0, yv - y0
    theta_rad = np.deg2rad(90.0 - theta_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    x_rot = x_c * cos_t - y_c * sin_t
    y_rot = x_c * sin_t + y_c * cos_t
    
    q = 1.0 - ellip
    r = np.sqrt(x_rot**2 + (y_rot / q)**2)

    I_os = sersic_intensity(r, Ie, re_pix, n)
    I = I_os.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))
    
    if psf_kernel is not None:
        k_shape = I.shape
        k_ny, k_nx = psf_kernel.shape
        padded_kernel = np.zeros(k_shape)
        c_y, c_x = k_shape[0] // 2, k_shape[1] // 2
        k_c_y, k_c_x = k_ny // 2, k_nx // 2
        y_slice = slice(c_y - k_c_y, c_y - k_c_y + k_ny)
        x_slice = slice(c_x - k_c_x, c_x - k_c_x + k_nx)
        padded_kernel[y_slice, x_slice] = psf_kernel
        padded_kernel = fftpack.ifftshift(padded_kernel)
        fy = fftpack.fft2(I)
        fk = fftpack.fft2(padded_kernel)
        I = np.real(fftpack.ifft2(fy * fk))
    return I

def add_galaxies_to_image(image, galaxies_df, pixel_scale, zero_point, exposure_s, seeing_fwhm_arcsec):
    fwhm_px = seeing_fwhm_arcsec / pixel_scale
    psf_size = max(21, int(round(fwhm_px * 6)) + 1)
    if psf_size % 2 == 0: psf_size += 1
    psf_kernel = gaussian_psf_kernel(fwhm_px, size=psf_size)
    
    for _, gal in galaxies_df.iterrows():
        if 'j_m_ext' not in gal or pd.isna(gal['j_m_ext']): continue
        re_arcsec = gal.get('j_r_eff', gal.get('r_3sig'))
        if pd.isna(re_arcsec) or re_arcsec <= 0: continue
            
        mag = gal['j_m_ext']
        re_pix = re_arcsec / pixel_scale
        x, y = gal['x'], gal['y']
        
        n = estimate_n_from_concentration(gal.get('j_con_indx'))
        
        axis_ratio = gal.get('sup_ba', 1.0)
        if pd.isna(axis_ratio) or axis_ratio <= 0: axis_ratio = 1.0
        ellip = 1.0 - axis_ratio
        
        pos_angle = gal.get('sup_phi', 0.0)
        if pd.isna(pos_angle): pos_angle = 0.0
        
        total_flux_e = zero_point * 10**(-0.4 * mag) * exposure_s
        
        radius_pix = max(15, int(round(6 * re_pix)))
        stamp_shape = (2 * radius_pix + 1, 2 * radius_pix + 1)
        
        unit_stamp = sample_sersic_to_image(
            shape=stamp_shape, 
            pixel_scale=pixel_scale, 
            Ie=1.0,
            re_arcsec=re_arcsec, 
            n=n, 
            ellip=ellip,
            theta_deg=pos_angle,
            x0=radius_pix, 
            y0=radius_pix, 
            oversample=4, 
            psf_kernel=psf_kernel
        )
        
        stamp_sum = unit_stamp.sum()
        if stamp_sum > 0:
            normalized_stamp = unit_stamp / stamp_sum
            add_stamp(image, normalized_stamp, x, y, total_flux_e)
            
    return image


