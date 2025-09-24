# satellite.py
# Handles querying and rendering of satellite trails.

import os
import time
import json
import math
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# SGP4 is now used for propagation
from sgp4.api import Satrec, jday

# Astropy is used for coordinate transformations
from astropy.coordinates import EarthLocation, GCRS, ITRS, AltAz, SkyCoord
from astropy.time import Time
from astropy import units as u

# Use relative imports for package structure.
from .utils import add_stamp, gaussian_psf_kernel

# Configuration constants.
CELESTRAK_BASE_URL = "https://celestrak.com/NORAD/elements/gp.php"
# Dictionary of available TLE sources from CelesTrak, mapping user-friendly names to API group names.
TLE_SOURCES = {
    "Active Satellites": "active",
    "All Space Stations": "stations",
    "100 Brightest": "visual",
    "GPS Satellites": "gps-ops",
    "GLONASS Satellites": "glo-ops",
    "Galileo Satellites": "galileo",
    "Starlink Constellation": "starlink",
    "Weather Satellites": "weather",
    "Communications Satellites (GEO)": "geo",
    "Last 30 Days' Launches": "last-30-days",
}
# REF_MAG and REF_ALT_KM are from a standard model consistent with the provided paper.
REF_MAG = 5.0 
REF_ALT_KM = 1000.0
MAG_LIMIT = 21.0 # Faintest satellite magnitude to simulate.
FRAME_DRIFT_ARCSEC_PER_S = 15.0 # Apparent sky drift for non-sidereal tracking.
WGS72_EARTH_RADIUS_KM = 6378.135
CACHE_DIR = Path.home() / ".cache" / "astrelle"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Downloads the latest satellite TLEs from CelesTrak for a specified group.
def download_tles(group="active", force=False, timeout=30):
    tle_cache_path = CACHE_DIR / f"celestrak_{group}.txt"
    # Use cached file if it exists, is not empty, and is less than 24 hours old, unless forced.
    if tle_cache_path.exists() and tle_cache_path.stat().st_size > 0 and not force:
        file_mod_time = datetime.fromtimestamp(tle_cache_path.stat().st_mtime, timezone.utc)
        if (datetime.now(timezone.utc) - file_mod_time) < timedelta(days=1):
            return tle_cache_path.read_text()
    try:
        url = f"{CELESTRAK_BASE_URL}?GROUP={group}&FORMAT=tle"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        txt = r.text
        tle_cache_path.write_text(txt)
        return txt
    except requests.RequestException:
        # If download fails, fall back to cached version if it exists
        if tle_cache_path.exists():
            return tle_cache_path.read_text()
        return ""

# Parses a raw TLE text file into a list of dictionaries.
def parse_tles(txt):
    lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
    out = []
    i = 0
    while i + 2 < len(lines):
        name = lines[i].strip()
        if lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            out.append({"name": name, "line1": lines[i+1], "line2": lines[i+2]})
            i += 3
        else:
            i += 1
    return out

# Calculates a satellite's topocentric RA, Dec, and geocentric altitude at a specific time.
def _get_sat_coords_at_time(sat, t_dt, observer_loc):
    # SGP4 propagation: This is the core propagation step.
    jd, fr = jday(t_dt.year, t_dt.month, t_dt.day, t_dt.hour, t_dt.minute, t_dt.second + t_dt.microsecond*1e-6)
    error, position_teme, _ = sat.sgp4(jd, fr)
    if error != 0: return None, None, None, None

    # Coordinate Transformation: Convert TEME to standard celestial and local frames.
    obstime = Time(t_dt, scale='utc')
    # GCRS is a standard, modern celestial frame (similar to J2000/ICRS).
    sat_gcrs = GCRS(x=position_teme[0]*u.km, y=position_teme[1]*u.km, z=position_teme[2]*u.km, 
                    representation_type='cartesian', obstime=obstime)

    # Transform to the observer's local horizon system (Alt/Az).
    altaz_frame = AltAz(obstime=obstime, location=observer_loc)
    sat_altaz = sat_gcrs.transform_to(altaz_frame)
    
    # Return local Az/Alt and celestial RA/Dec.
    return sat_altaz.az.deg, sat_altaz.alt.deg, sat_gcrs.ra.deg, sat_gcrs.dec.deg

# Cohen-Sutherland line clipping algorithm constants.
INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

def _compute_outcode(x, y, x_min, y_min, x_max, y_max):
    code = INSIDE
    if x < x_min: code |= LEFT
    elif x > x_max: code |= RIGHT
    if y < y_min: code |= BOTTOM
    elif y > y_max: code |= TOP
    return code

def _cohen_sutherland_clip(x0, y0, x1, y1, x_min, y_min, x_max, y_max):
    code0, code1 = _compute_outcode(x0, y0, x_min, y_min, x_max, y_max), _compute_outcode(x1, y1, x_min, y_min, x_max, y_max)
    while True:
        if not (code0 | code1): return True
        if code0 & code1: return False
        dy, dx = y1 - y0, x1 - x0
        code_out = code1 if code1 > code0 else code0
        x, y = 0.0, 0.0
        if code_out & TOP:
            x, y = x0 + dx * (y_max - y0) / dy if dy != 0 else x0, y_max
        elif code_out & BOTTOM:
            x, y = x0 + dx * (y_min - y0) / dy if dy != 0 else x0, y_min
        elif code_out & RIGHT:
            y, x = y0 + dy * (x_max - x0) / dx if dx != 0 else y0, x_max
        elif code_out & LEFT:
            y, x = y0 + dy * (x_min - x0) / dx if dx != 0 else y0, x_min
        if code_out == code0:
            x0, y0 = x, y
            code0 = _compute_outcode(x0, y0, x_min, y_min, x_max, y_max)
        else:
            x1, y1 = x, y
            code1 = _compute_outcode(x1, y1, x_min, y_min, x_max, y_max)

# Worker function for get_sats_in_fov
def _check_tle_visibility(tle, dt_utc, exposure_s, observer_loc, center_skycoord, search_radius_deg, ra_min, ra_max, dec_min, dec_max):
    try:
        sat = Satrec.twoline2rv(tle["line1"], tle["line2"])
        mid_time = dt_utc + timedelta(seconds=exposure_s / 2.0)
        
        # 1. First coarse check: Is the satellite above the horizon and within a large search radius at mid-exposure?
        _, alt_mid, ra_mid, dec_mid = _get_sat_coords_at_time(sat, mid_time, observer_loc)
        if alt_mid is None or alt_mid < 0: return None
        
        sat_mid_skycoord = SkyCoord(ra=ra_mid*u.deg, dec=dec_mid*u.deg, frame='gcrs')
        if center_skycoord.separation(sat_mid_skycoord).deg > search_radius_deg: return None

        # 2. Second, precise check: Does the satellite's path segment actually cross the FOV rectangle?
        _, _, ra0, dec0 = _get_sat_coords_at_time(sat, dt_utc, observer_loc)
        _, _, ra1, dec1 = _get_sat_coords_at_time(sat, dt_utc + timedelta(seconds=exposure_s), observer_loc)
        if ra0 is None or ra1 is None: return None
        
        # Handle RA wrap-around near 360 degrees.
        if abs(ra1 - ra0) > 180: ra1 += 360 if ra1 < ra0 else -360
        
        # Correct for sky drift if telescope is tracking sidereally.
        drift_deg = (FRAME_DRIFT_ARCSEC_PER_S * exposure_s) / 3600.0
        ra1_corrected = ra1 - drift_deg * np.cos(np.deg2rad(dec1))

        # Use Cohen-Sutherland algorithm to clip the path to the FOV rectangle.
        if _cohen_sutherland_clip(ra0, dec0, ra1_corrected, dec1, ra_min, ra_max, dec_min, dec_max):
            return tle
        return None
    except Exception:
        return None

def get_sats_in_fov(presets, tel_key, dt_utc, exposure_s, fov_arcmin, center_ra_dec, tle_group="active", num_threads=8, progress_callback=None, log_callback=None):
    def log(msg):
        if log_callback: log_callback(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}")
    def update_progress(val):
        if progress_callback: progress_callback(val)

    tel = presets["telescopes"][tel_key]
    tle_txt = download_tles(group=tle_group)
    if not tle_txt: return pd.DataFrame(), pd.DataFrame(), "ERROR: Could not download TLE data."
    all_tles = parse_tles(tle_txt); log(f"Parsed {len(all_tles)} TLEs.")
    all_tles_df = pd.DataFrame(all_tles)
    
    observer_loc = EarthLocation(lon=tel["longitude"]*u.deg, lat=tel["latitude"]*u.deg, height=tel["elevation_m"]*u.m)
    
    # Define FOV boundaries, correcting RA width for declination.
    fov_w_deg, fov_h_deg = fov_arcmin[0] / 60.0, fov_arcmin[1] / 60.0
    cos_dec = np.cos(np.deg2rad(center_ra_dec[1]))
    fov_w_deg_corrected = fov_w_deg / max(cos_dec, 0.1) # Avoid division by zero at poles.
    ra_min, ra_max = center_ra_dec[0] - fov_w_deg_corrected / 2, center_ra_dec[0] + fov_w_deg_corrected / 2
    dec_min, dec_max = center_ra_dec[1] - fov_h_deg / 2, center_ra_dec[1] + fov_h_deg / 2
    
    # Define a larger search radius for the coarse filter.
    center_skycoord = SkyCoord(ra=center_ra_dec[0]*u.deg, dec=center_ra_dec[1]*u.deg, frame='gcrs')
    max_angular_speed_deg_s = 1.5 # A reasonable upper limit for LEO satellite speed.
    max_travel_deg = max_angular_speed_deg_s * exposure_s
    fov_radius_deg = np.sqrt(fov_w_deg_corrected**2 + fov_h_deg**2) / 2.0
    search_radius_deg = fov_radius_deg + max_travel_deg / 2.0 + 1.0 # Add buffer

    sats_in_fov_list = []
    
    log(f"Checking {len(all_tles)} satellites with {num_threads} threads...")
    total_tles, completed_count = len(all_tles), 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(_check_tle_visibility, tle, dt_utc, exposure_s, observer_loc, center_skycoord, search_radius_deg, ra_min, ra_max, dec_min, dec_max): tle for tle in all_tles}
        for future in as_completed(futures):
            if result := future.result(): sats_in_fov_list.append(result)
            completed_count += 1
            if completed_count % 100 == 0 and total_tles > 0:
                update_progress(60 + int(25 * (completed_count / total_tles)))

    sats_df = pd.DataFrame(sats_in_fov_list)
    update_progress(85)
    return sats_df, all_tles_df, f"Found {len(sats_df)} satellites passing through FOV."

# Renders satellite trails onto the image.
def render_satellite_trails(image, sats_df, wcs, zero_point, exposure_s, tel_key, presets, dt_utc, seeing_fwhm_arcsec, pixel_scale):
    trail_details = []
    ny, nx = image.shape
    tel = presets["telescopes"][tel_key]
    observer_loc = EarthLocation(lon=tel["longitude"]*u.deg, lat=tel["latitude"]*u.deg, height=tel["elevation_m"]*u.m)

    for _, sat_info in sats_df.iterrows():
        sat = Satrec.twoline2rv(sat_info["line1"], sat_info["line2"])
        
        mid_time = dt_utc + timedelta(seconds=exposure_s / 2.0)
        jd_mid, fr_mid = jday(mid_time.year, mid_time.month, mid_time.day, mid_time.hour, mid_time.minute, mid_time.second + mid_time.microsecond*1e-6)
        _, position_teme, _ = sat.sgp4(jd_mid, fr_mid)
        alt_km = np.linalg.norm(position_teme) - WGS72_EARTH_RADIUS_KM
        
        apparent_mag = mag_from_altitude_km(alt_km)
        if apparent_mag > MAG_LIMIT: continue
            
        flux_e_per_sec = zero_point * 10**(-0.4 * apparent_mag)
        total_flux = flux_e_per_sec * exposure_s
        psf_stamp = gaussian_psf_kernel(fwhm_pix=seeing_fwhm_arcsec / pixel_scale)

        _, _, ra0, dec0 = _get_sat_coords_at_time(sat, dt_utc, observer_loc)
        _, _, ra1, dec1 = _get_sat_coords_at_time(sat, dt_utc + timedelta(seconds=exposure_s), observer_loc)
        if ra0 is None or ra1 is None: continue

        if abs(ra1 - ra0) > 180: ra1 += 360 if ra1 < ra0 else -360

        start_coords_pix = wcs.all_world2pix([[ra0, dec0]], 0)[0]
        end_coords_pix = wcs.all_world2pix([[ra1, dec1]], 0)[0]
        x_start, y_start = start_coords_pix[0], start_coords_pix[1]
        x_end, y_end = end_coords_pix[0], end_coords_pix[1]
        
        drift_pix_per_s = (FRAME_DRIFT_ARCSEC_PER_S / pixel_scale)
        x_end -= drift_pix_per_s * exposure_s

        dx, dy = x_end - x_start, y_end - y_start
        trail_length_pix = np.sqrt(dx**2 + dy**2)
        
        if trail_length_pix < 1:
            if 0 <= x_start < nx and 0 <= y_start < ny:
                add_stamp(image, psf_stamp, x_start, y_start, total_flux)
        else:
            num_stamps = int(np.ceil(trail_length_pix))
            flux_per_stamp = total_flux / max(1, num_stamps)
            for i in range(num_stamps):
                fraction = i / max(1, num_stamps - 1) if num_stamps > 1 else 0.0
                x_pix = x_start + dx * fraction
                y_pix = y_start + dy * fraction
                if 0 <= x_pix < nx and 0 <= y_pix < ny:
                     add_stamp(image, psf_stamp, x_pix, y_pix, flux_per_stamp)

        trail_details.append({
            "satname": sat_info.get('name', 'UNKNOWN'),
            "x_center": (x_start + x_end) / 2, 
            "y_center": (y_start + y_end) / 2
        })
    return image, trail_details

# Renders synthetic satellite trails based on user-defined parameters.
def generate_synthetic_trails(image, trail_params_list, zero_point, exposure_s):
    ny, nx = image.shape
    trail_details = []

    for i, p in enumerate(trail_params_list):
        flux_e_per_sec = zero_point * 10**(-0.4 * p['apparent_mag'])
        total_flux = flux_e_per_sec * exposure_s
        psf_stamp = gaussian_psf_kernel(fwhm_pix=p['trail_fwhm_arcsec'] / p['pixel_scale'])

        x_start, y_start = p['start_x_pix'], p['start_y_pix']
        angle_rad = np.deg2rad(p['angle_deg'])
        speed_pix_s = p['speed_pixels_s']
        
        x_end = x_start + speed_pix_s * np.cos(angle_rad) * exposure_s
        y_end = y_start + speed_pix_s * np.sin(angle_rad) * exposure_s
        
        dx, dy = x_end - x_start, y_end - y_start
        trail_length_pix = np.sqrt(dx**2 + dy**2)

        if trail_length_pix < 1:
            if 0 <= x_start < nx and 0 <= y_start < ny:
                add_stamp(image, psf_stamp, x_start, y_start, total_flux)
        else:
            num_stamps = int(np.ceil(trail_length_pix))
            flux_per_stamp = total_flux / max(1, num_stamps)
            for j in range(num_stamps):
                fraction = j / max(1, num_stamps - 1) if num_stamps > 1 else 0.0
                x_pix = x_start + dx * fraction
                y_pix = y_start + dy * fraction
                if 0 <= x_pix < nx and 0 <= y_pix < ny:
                    add_stamp(image, psf_stamp, x_pix, y_pix, flux_per_stamp)

        trail_details.append({
            "satname": f"SYNTHETIC-{i+1}",
            "x_center": (x_start + x_end) / 2, 
            "y_center": (y_start + y_end) / 2
        })
    return image, trail_details

# Calculates an approximate visual magnitude for a satellite based on its altitude.
def mag_from_altitude_km(alt_km):
    if alt_km is None: return 22.0
    mag = REF_MAG + 5.0 * math.log10(max(1.0, float(alt_km) / REF_ALT_KM))
    return float(mag)


