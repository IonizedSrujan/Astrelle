# satellite.py
# Handles querying and rendering of satellite trails.

import os
import time
import json
import math
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from skyfield.api import load, EarthSatellite, wgs84
from astropy.coordinates import SkyCoord
import astropy.units as u

# Use relative imports for package structure
from .utils import add_stamp, gaussian_psf_kernel

# --- Configuration ---
CELESTRAK_URL = "https://celestrak.com/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
MAX_WORKERS = 8
REF_ALT_KM = 400.0
REF_MAG = 1.0
MAG_LIMIT = 21.0
FRAME_DRIFT_ARCSEC_PER_S = 15.0

# --- TLE Handling ---
def download_tles(force=False, timeout=30, cache_dir=Path(".")):
    tle_cache_path = cache_dir / "celestrak_tles.txt"
    if tle_cache_path.exists() and not force:
        return tle_cache_path.read_text()
    try:
        r = requests.get(CELESTRAK_URL, timeout=timeout)
        r.raise_for_status()
        txt = r.text
        tle_cache_path.write_text(txt)
        return txt
    except requests.RequestException as e:
        return ""

def parse_tles(txt):
    lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
    out = []
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        if i + 2 < len(lines) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            out.append({"name": name, "line1": lines[i+1], "line2": lines[i+2]})
            i += 3
        else:
            i += 1
    return out

def _get_sat_coords_at_time(sat, topos, t_dt):
    ts = get_timescale()
    t = ts.from_datetime(t_dt)
    topo = (sat - topos).at(t)
    ra, dec, _ = topo.radec()
    alt, _, _ = topo.altaz()
    return ra.hours * 15.0, dec.degrees, alt.degrees > 0

# --- Constants for Cohen-Sutherland clipping ---
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
        if not (code0 | code1):
            return True
        if code0 & code1:
            return False
        
        dy, dx = y1 - y0, x1 - x0
        code_out = code1 if code1 > code0 else code0
        x, y = 0.0, 0.0

        if code_out & TOP:
            x = x0 + dx * (y_max - y0) / dy if dy != 0 else x0
            y = y_max
        elif code_out & BOTTOM:
            x = x0 + dx * (y_min - y0) / dy if dy != 0 else x0
            y = y_min
        elif code_out & RIGHT:
            y = y0 + dy * (x_max - x0) / dx if dx != 0 else y0
            x = x_max
        elif code_out & LEFT:
            y = y0 + dy * (x_min - x0) / dx if dx != 0 else y0
            x = x_min

        if code_out == code0:
            x0, y0 = x, y
            code0 = _compute_outcode(x0, y0, x_min, y_min, x_max, y_max)
        else:
            x1, y1 = x, y
            code1 = _compute_outcode(x1, y1, x_min, y_min, x_max, y_max)

def get_sats_in_fov(presets, tel_key, dt_utc, exposure_s, fov_arcmin, center_ra_dec, progress_callback=None, log_callback=None, cache_dir=Path(".")):
    def log(msg):
        if log_callback: log_callback(f"[{datetime.now():%H:%M:%S}] {msg}")
    
    def update_progress(val):
        if progress_callback: progress_callback(val)

    tel = presets["telescopes"][tel_key]
    
    tle_txt = download_tles(cache_dir=cache_dir)
    if not tle_txt:
        return pd.DataFrame(), "ERROR: Could not download TLE data from CelesTrak."
    
    all_tles = parse_tles(tle_txt)
    log(f"Parsed {len(all_tles)} TLEs.")
    
    ts = get_timescale()
    topos = wgs84.latlon(tel["latitude"], tel["longitude"], elevation_m=tel.get("elevation_m", 0))
    
    fov_w_deg, fov_h_deg = fov_arcmin[0] / 60.0, fov_arcmin[1] / 60.0
    ra_min, ra_max = center_ra_dec[0] - fov_w_deg / 2, center_ra_dec[0] + fov_w_deg / 2
    dec_min, dec_max = center_ra_dec[1] - fov_h_deg / 2, center_ra_dec[1] + fov_h_deg / 2
    
    center_skycoord = SkyCoord(ra=center_ra_dec[0]*u.deg, dec=center_ra_dec[1]*u.deg, frame='icrs')
    max_travel_deg = (3.0 / 60.0) * exposure_s 
    fov_radius_deg = np.sqrt(fov_w_deg**2 + fov_h_deg**2) / 2.0
    search_radius_deg = fov_radius_deg + max_travel_deg / 2.0 + 1.0

    sats_in_fov_list = []
    frame_drift_deg = (FRAME_DRIFT_ARCSEC_PER_S * exposure_s) / 3600.0

    def check_tle(tle):
        try:
            sat = EarthSatellite(tle["line1"], tle["line2"], name=tle.get("name", "SAT"), ts=ts)
            mid_time = dt_utc + timedelta(seconds=exposure_s / 2.0)
            ra_mid, dec_mid, above_horizon = _get_sat_coords_at_time(sat, topos, mid_time)
            if not above_horizon: return None

            sat_mid_skycoord = SkyCoord(ra=ra_mid*u.deg, dec=dec_mid*u.deg, frame='icrs')
            separation = center_skycoord.separation(sat_mid_skycoord).deg
            if separation > search_radius_deg: return None

            ra0, dec0, _ = _get_sat_coords_at_time(sat, topos, dt_utc)
            ra1, dec1, _ = _get_sat_coords_at_time(sat, topos, dt_utc + timedelta(seconds=exposure_s))
            
            # RA can wrap around 360 degrees, handle this properly
            if ra1 < ra0 - 180: ra1 += 360
            if ra1 > ra0 + 180: ra1 -= 360
                
            ra1_corrected = ra1 - frame_drift_deg
            
            if _cohen_sutherland_clip(ra0, dec0, ra1_corrected, dec1, ra_min, ra_max, dec_min, dec_max):
                return tle
            return None
        except Exception:
            return None

    log(f"Checking {len(all_tles)} satellites with optimized filter...")
    total_tles = len(all_tles); completed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_tle, tle): tle for tle in all_tles}
        for future in as_completed(futures):
            if result := future.result(): sats_in_fov_list.append(result)
            completed_count += 1
            if completed_count % 100 == 0 and total_tles > 0:
                update_progress(60 + int(25 * (completed_count / total_tles)))

    sats_df = pd.DataFrame(sats_in_fov_list)
    update_progress(85)
    return sats_df, f"Found {len(sats_df)} satellites passing through FOV."

def render_satellite_trails(image, sats_df, wcs, zero_point, exposure_s, tel_key, sen_key, presets, dt_utc, seeing_fwhm_arcsec, pixel_scale):
    trail_details = []
    ny, nx = image.shape
    tel = presets["telescopes"][tel_key]
    sen = presets["sensors"][sen_key]
    fps = sen.get("fps", 10.0)
    num_frames = int(exposure_s * fps)
    if num_frames == 0: num_frames = 1
    time_per_frame = 1.0 / fps
    
    ts = get_timescale()
    topos = wgs84.latlon(tel["latitude"], tel["longitude"], elevation_m=tel.get("elevation_m", 0))

    frame_drift_deg_per_s = FRAME_DRIFT_ARCSEC_PER_S / 3600.0

    for _, sat_info in sats_df.iterrows():
        sat = EarthSatellite(sat_info["line1"], sat_info["line2"], name=sat_info.get("name", "SAT"), ts=ts)
        
        mid_time = dt_utc + timedelta(seconds=exposure_s / 2.0)
        t_mid = ts.from_datetime(mid_time)
        geocentric = sat.at(t_mid)
        alt_km = wgs84.height_of(geocentric).km
        apparent_mag = mag_from_altitude_km(alt_km)

        if apparent_mag > MAG_LIMIT: continue
            
        flux_e_per_sec = zero_point * 10**(-0.4 * apparent_mag)
        flux_per_frame = flux_e_per_sec * time_per_frame

        psf_stamp = gaussian_psf_kernel(fwhm_pix=seeing_fwhm_arcsec / pixel_scale)

        ra0, dec0, _ = _get_sat_coords_at_time(sat, topos, dt_utc)
        ra1, dec1, _ = _get_sat_coords_at_time(sat, topos, dt_utc + timedelta(seconds=exposure_s))

        if ra1 < ra0 - 180: ra1 += 360
        if ra1 > ra0 + 180: ra1 -= 360

        start_coords_pix = wcs.all_world2pix([[ra0, dec0]], 0)[0]
        end_coords_pix = wcs.all_world2pix([[ra1, dec1]], 0)[0]

        x_start, y_start = start_coords_pix[0], start_coords_pix[1]
        x_end, y_end = end_coords_pix[0], end_coords_pix[1]
        
        # Correct for frame drift in pixel space
        drift_pix_per_s = (FRAME_DRIFT_ARCSEC_PER_S / pixel_scale)
        x_end -= drift_pix_per_s * exposure_s

        for i in range(num_frames):
            fraction = i / max(1, num_frames -1)
            x_pix = x_start + (x_end - x_start) * fraction
            y_pix = y_start + (y_end - y_start) * fraction

            if 0 <= x_pix < nx and 0 <= y_pix < ny:
                 add_stamp(image, psf_stamp, x_pix, y_pix, flux_per_frame)

        trail_details.append({
            "satname": sat_info.get('name', 'UNKNOWN'),
            "x_center": (x_start + x_end) / 2, 
            "y_center": (y_start + y_end) / 2
        })
        
    return image, trail_details

def generate_synthetic_trails(image, wcs, trail_params_list, pixel_scale, zero_point, exposure_s, fps):
    ny, nx = image.shape
    trail_details = []
    
    num_frames = int(exposure_s * fps)
    if num_frames == 0: num_frames = 1
    time_per_frame = 1.0 / fps

    for i, trail_params in enumerate(trail_params_list):
        apparent_mag = trail_params['apparent_mag']
        trail_fwhm_arcsec = trail_params['trail_fwhm_arcsec']
        start_ra_deg = trail_params['start_ra_deg']
        start_dec_deg = trail_params['start_dec_deg']
        angle_deg = trail_params['angle_deg']
        speed_arcsec_s = trail_params['speed_arcsec_s']

        flux_e_per_sec = zero_point * 10**(-0.4 * apparent_mag)
        flux_per_frame = flux_e_per_sec * time_per_frame

        psf_stamp = gaussian_psf_kernel(fwhm_pix=trail_fwhm_arcsec / pixel_scale)

        start_coords_pix = wcs.all_world2pix([[start_ra_deg, start_dec_deg]], 0)[0]
        x_start, y_start = start_coords_pix[0], start_coords_pix[1]

        angle_rad = np.deg2rad(90.0 - angle_deg)
        speed_pix_s = speed_arcsec_s / pixel_scale
        vx_pix_s = speed_pix_s * np.cos(angle_rad)
        vy_pix_s = speed_pix_s * np.sin(angle_rad)

        x_end = x_start + vx_pix_s * exposure_s
        y_end = y_start + vy_pix_s * exposure_s

        for j in range(num_frames):
            frame_time_s = j * time_per_frame
            x_pix = x_start + vx_pix_s * frame_time_s
            y_pix = y_start + vy_pix_s * frame_time_s

            if 0 <= x_pix < nx and 0 <= y_pix < ny:
                add_stamp(image, psf_stamp, x_pix, y_pix, flux_per_frame)

        trail_details.append({
            "satname": f"SYNTHETIC-{i+1}",
            "x_center": (x_start + x_end) / 2, 
            "y_center": (y_start + y_end) / 2
        })
        
    return image, trail_details

_timescale = None
def get_timescale():
    global _timescale
    if _timescale is None: _timescale = load.timescale()
    return _timescale

def mag_from_altitude_km(alt_km):
    if alt_km is None: return 22.0
    mag = REF_MAG + 5.0 * math.log10(max(1.0, float(alt_km) / REF_ALT_KM))
    return float(mag)


