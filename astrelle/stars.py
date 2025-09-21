# stars.py
# Handles querying, projecting, and rendering stars from Gaia.

import numpy as np
import pandas as pd
import time
from socket import gaierror
from requests.exceptions import RequestException

from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.wcs import WCS
from astroquery.gaia import Gaia

from .utils import rebin, add_stamp

Gaia.ROW_LIMIT = -1
Gaia.TIMEOUT = 600

def query_stars_gaia_rectangular(center_ra, center_dec, width_arcmin, height_arcmin, mag_limit=20):
    """
    Queries Gaia DR3 for stars within a rectangular field of view using a precise
    polygon query to account for projection distortion.
    """
    center = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg, frame='icrs')
    # Create a tangent plane projection centered on the FOV
    offset_frame = SkyOffsetFrame(origin=center)
    
    half_w_deg, half_h_deg = (width_arcmin / 120.0), (height_arcmin / 120.0)

    # Define the corners of the flat sensor in the offset frame
    corners_offset = [
        (+half_w_deg, +half_h_deg), (+half_w_deg, -half_h_deg),
        (-half_w_deg, -half_h_deg), (-half_w_deg, +half_h_deg)
    ]
    
    # Transform the flat corners back to the curved sky (ICRS frame)
    corner_icrs = [
        SkyCoord(lon=lon*u.deg, lat=lat*u.deg, frame=offset_frame).transform_to('icrs')
        for lon, lat in corners_offset
    ]
    
    # Create the polygon string for the ADQL query
    poly_coords = ",".join(f"{c.ra.deg:.9f},{c.dec.deg:.9f}" for c in corner_icrs)
    
    adql = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS', ra, dec), POLYGON('ICRS', {poly_coords}))
          AND phot_g_mean_mag <= {mag_limit}
    """
    
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
                time.sleep(2 * (attempt + 1))
            else:
                log_msg = f"CRITICAL: Gaia query failed after {max_retries} attempts. Continuing without stars. Error: {e}"
                return pd.DataFrame(), log_msg
    
    return pd.DataFrame(), "Gaia query failed after all retries."


def create_wcs(center_ra, center_dec, nx, ny, pixel_scale_arcsec):
    """Creates an Astropy WCS object for the given pointing."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2.0, ny / 2.0]
    wcs.wcs.cdelt = np.array([-pixel_scale_arcsec / 3600, pixel_scale_arcsec / 3600])
    wcs.wcs.crval = [center_ra, center_dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs

def project_stars_to_pixels(df, wcs):
    """Projects RA/Dec of stars to pixel coordinates using a WCS object."""
    ra_vals, dec_vals = df['ra'].values, df['dec'].values
    pix_coords = wcs.all_world2pix(ra_vals, dec_vals, 0)
    df['x'], df['y'] = pix_coords[0], pix_coords[1]
    return df

def add_star_fluxes(df, zero_point, exposure_s):
    """Calculates the total electron flux for each star."""
    df['total_electrons'] = zero_point * 10**(-0.4 * df['phot_g_mean_mag']) * exposure_s
    return df

def gaussian_stamp(radius_px, fwhm_px):
    """Generates a normalized 2D Gaussian profile stamp."""
    final_dim = 2 * radius_px + 1
    oversample = 5
    oversampled_dim = final_dim * oversample
    center = (oversampled_dim - 1) / 2.0
    sigma_os = (fwhm_px * oversample) / 2.35482
    y, x = np.indices((oversampled_dim, oversampled_dim))
    stamp_oversampled = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma_os**2))
    stamp_rebinned = rebin(stamp_oversampled, oversample)
    return stamp_rebinned / stamp_rebinned.sum() if stamp_rebinned.sum() > 0 else stamp_rebinned

def render_stars(image, df, fwhm_px):
    """Renders all stars from the dataframe onto the image."""
    ny, nx = image.shape
    stamp_radius = max(5, int(round(4 * fwhm_px)))
    stamp = gaussian_stamp(stamp_radius, fwhm_px)
    
    for _, star in df.iterrows():
        if 0 <= star['x'] < nx and 0 <= star['y'] < ny:
            add_stamp(image, stamp, star['x'], star['y'], star['total_electrons'])
    return image


