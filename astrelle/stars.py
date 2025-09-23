# stars.py
# Handles querying, projecting, and rendering stars from Gaia.

import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning, message='.*update_default_config*')

import numpy as np
import pandas as pd
import time
from socket import gaierror
from requests.exceptions import RequestException
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.wcs import WCS
from astroquery.gaia import Gaia

# Use relative imports for package structure.
from .utils import rebin, add_stamp

# Configure Gaia query to return all results and have a long timeout.
Gaia.ROW_LIMIT = -1
Gaia.TIMEOUT = 600

# Queries Gaia DR3 for stars within a rectangular field of view.
def query_stars_gaia_rectangular(center_ra, center_dec, width_arcmin, height_arcmin, mag_limit=20):
    # Uses a precise polygon query to account for projection distortion on the sky.
    center = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg, frame='icrs')
    
    # Create a tangent plane projection centered on the field of view.
    offset_frame = SkyOffsetFrame(origin=center)
    
    half_w_deg, half_h_deg = (width_arcmin / 120.0), (height_arcmin / 120.0)

    # Define the corners of the flat sensor in the offset frame.
    corners_offset = [
        (+half_w_deg, +half_h_deg), (+half_w_deg, -half_h_deg),
        (-half_w_deg, -half_h_deg), (-half_w_deg, +half_h_deg)
    ]
    
    # Transform the flat corners back to the curved sky (ICRS frame).
    corner_icrs = [
        SkyCoord(lon=lon*u.deg, lat=lat*u.deg, frame=offset_frame).transform_to('icrs')
        for lon, lat in corners_offset
    ]
    
    # Create the polygon string for the ADQL query.
    poly_coords = ",".join(f"{c.ra.deg:.9f},{c.dec.deg:.9f}" for c in corner_icrs)
    
    # Construct the ADQL query string.
    adql = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS', ra, dec), POLYGON('ICRS', {poly_coords}))
          AND phot_g_mean_mag <= {mag_limit}
    """
    
    # Retry the query up to 3 times in case of network errors.
    max_retries = 3
    for attempt in range(max_retries):
        try:
            job = Gaia.launch_job_async(adql)
            res = job.get_results()
            log_msg = f"Gaia query returned {len(res)} stars."
            return res.to_pandas(), log_msg
        except (gaierror, RequestException) as e:
            log_msg = f"Network error querying Gaia (attempt {attempt + 1}/{max_retries}): {e}"
            print(log_msg)
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1)) # Exponential backoff.
            else:
                log_msg = f"CRITICAL: Gaia query failed after {max_retries} attempts. Error: {e}"
                return pd.DataFrame(), log_msg
    
    return pd.DataFrame(), "Gaia query failed after all retries."

# Creates an Astropy WCS object for the given image parameters.
def create_wcs(center_ra, center_dec, nx, ny, pixel_scale_arcsec):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2.0, ny / 2.0] # Set the reference pixel to the image center.
    wcs.wcs.cdelt = np.array([-pixel_scale_arcsec / 3600, pixel_scale_arcsec / 3600]) # Pixel scale in degrees.
    wcs.wcs.crval = [center_ra, center_dec] # RA/Dec of the reference pixel.
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"] # Set the projection type (Gnomonic).
    return wcs

# Projects RA/Dec of stars to pixel coordinates using a WCS object.
def project_stars_to_pixels(df, wcs):
    ra_vals, dec_vals = df['ra'].values, df['dec'].values
    pix_coords = wcs.all_world2pix(ra_vals, dec_vals, 0)
    df['x'], df['y'] = pix_coords[0], pix_coords[1]
    return df

# Calculates the total electron flux for each star based on its magnitude.
def add_star_fluxes(df, zero_point, exposure_s):
    df['total_electrons'] = zero_point * 10**(-0.4 * df['phot_g_mean_mag']) * exposure_s
    return df

# Generates a normalized 2D Gaussian profile stamp in polar coordinates.
def gaussian_stamp(radius_px, fwhm_px):
    final_dim = 2 * radius_px + 1
    oversample = 5
    oversampled_dim = final_dim * oversample
    center = (oversampled_dim - 1) / 2.0
    sigma_os = (fwhm_px * oversample) / 2.35482
    
    # Create a grid of radial distances from the center.
    y, x = np.indices((oversampled_dim, oversampled_dim))
    r_sq = (x - center)**2 + (y - center)**2
    
    stamp_oversampled = np.exp(-r_sq / (2 * sigma_os**2))
    stamp_rebinned = rebin(stamp_oversampled, oversample)
    
    # Normalize the stamp to have a total sum of 1.
    return stamp_rebinned / stamp_rebinned.sum() if stamp_rebinned.sum() > 0 else stamp_rebinned

# Renders all stars from the dataframe onto the image.
def render_stars(image, df, fwhm_px):
    ny, nx = image.shape
    stamp_radius = max(5, int(round(4 * fwhm_px)))
    stamp = gaussian_stamp(stamp_radius, fwhm_px)
    
    # Add a stamp for each star at its pixel location.
    for _, star in df.iterrows():
        if 0 <= star['x'] < nx and 0 <= star['y'] < ny:
            add_stamp(image, stamp, star['x'], star['y'], star['total_electrons'])
    return image


