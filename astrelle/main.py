# main.py
# Main simulation orchestrator for Astrelle.

import os
import time
from datetime import datetime, timezone
from pathlib import Path
import random
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation

# Use relative imports for package structure.
from . import background
from . import stars
from . import galaxy
from . import satellite

def run_simulation(tel_key, sen_key, dt_utc, exposure_s,
                   pointing_mode, sat_mode, synth_sat_params,
                   ra_deg=None, dec_deg=None, alt_deg=None, az_deg=None,
                   mag_limit=20.0, seeing_fwhm_arcsec=1.5,
                   sky_mag_per_arcsec2=21.0,
                   bias_level_e=100, bias_spread_e=2.0,
                   tle_group="active", num_threads=8,
                   log_callback=None, progress_callback=None):
    # Main simulation entry point.

    def log(msg):
        if log_callback:
            log_callback(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}")

    def update_progress(val):
        if progress_callback:
            progress_callback(val)

    presets = background.load_presets()
    tel = presets["telescopes"][tel_key]
    sen = presets["sensors"][sen_key]
    nx, ny = sen["resolution"]
    
    if pointing_mode == "Local (Alt/Az)":
        log("Calculating RA/Dec from Alt/Az pointing..."); update_progress(15)
        loc = EarthLocation(lat=tel["latitude"]*u.deg, lon=tel["longitude"]*u.deg, height=tel["elevation_m"]*u.m)
        obstime = Time(dt_utc)
        pointing_altaz = SkyCoord(alt=alt_deg*u.deg, az=az_deg*u.deg, frame=AltAz(obstime=obstime, location=loc))
        pointing_icrs = pointing_altaz.transform_to('icrs')
        center_ra, center_dec = float(pointing_icrs.ra.deg), float(pointing_icrs.dec.deg)
        log(f"Pointing (Alt/Az): Alt={alt_deg:.2f}°, Az={az_deg:.2f}° -> RA={center_ra:.4f}°, Dec={center_dec:.4f}° at {dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        log("Using direct RA/Dec pointing..."); update_progress(15)
        center_ra, center_dec = ra_deg, dec_deg
        log(f"Pointing (RA/Dec): RA={center_ra:.4f}°, Dec={center_dec:.4f}° at {dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Check for a preset plate scale, otherwise calculate it.
    if "plate_scale_arcsec_per_mm" in tel and tel["plate_scale_arcsec_per_mm"] > 0:
        pixel_scale = tel["plate_scale_arcsec_per_mm"] * (sen["pixel_size_um"] / 1000.0)
        log(f"Using telescope preset plate scale: {pixel_scale:.4f}\" / pixel")
    else:
        focal_length_mm = tel['diameter_mm'] * tel['f_number']
        pixel_scale = (sen["pixel_size_um"] / focal_length_mm) * 206.265
        log(f"Calculated plate scale: {pixel_scale:.4f}\" / pixel")

    width_arcmin = (nx * pixel_scale) / 60.0
    height_arcmin = (ny * pixel_scale) / 60.0
    wcs = stars.create_wcs(center_ra, center_dec, nx, ny, pixel_scale)

    log("Generating background and calculating zero point..."); update_progress(20)
    zero_point = background.calculate_zero_point(tel, sen)
    ideal_signal = background.generate_background_e(zero_point, sky_mag_per_arcsec2, pixel_scale, exposure_s, sen["dark_current_e_per_s"], (ny, nx))
    
    log("Querying Gaia for stars..."); update_progress(25)
    stars_df, log_msg = stars.query_stars_gaia_rectangular(center_ra, center_dec, width_arcmin, height_arcmin, mag_limit)
    log(log_msg)
    if not stars_df.empty:
        stars_df = stars.add_star_fluxes(stars_df, zero_point, exposure_s)
        stars_df = stars.project_stars_to_pixels(stars_df, wcs)
        log("Rendering stars..."); update_progress(35)
        ideal_signal = stars.render_stars(ideal_signal, stars_df, seeing_fwhm_arcsec / pixel_scale)

    log("Querying 2MASS for galaxies..."); update_progress(40)
    galaxies_df, log_msg = galaxy.query_galaxies_irsa_xsc(center_ra, center_dec, width_arcmin, height_arcmin)
    log(log_msg)
    if not galaxies_df.empty:
        galaxies_df = stars.project_stars_to_pixels(galaxies_df, wcs)
        log("Rendering galaxies..."); update_progress(45)
        ideal_signal = galaxy.add_galaxies_to_image(ideal_signal, galaxies_df, pixel_scale, zero_point, exposure_s, seeing_fwhm_arcsec)
    
    sats_out_df = pd.DataFrame()

    if sat_mode == "Catalog":
        log("Processing satellites from catalog..."); update_progress(50)
        satellites_df_tle, all_tles_df, log_msg = satellite.get_sats_in_fov(
            presets, tel_key, dt_utc, exposure_s, (width_arcmin, height_arcmin), (center_ra, center_dec),
            tle_group=tle_group, num_threads=num_threads, progress_callback=progress_callback, 
            log_callback=log_callback
        )
        log(log_msg)
        
        if not satellites_df_tle.empty:
            log("Rendering satellite trails from catalog..."); update_progress(85)
            ideal_signal, sat_trail_details = satellite.render_satellite_trails(
                ideal_signal, satellites_df_tle, wcs, zero_point, exposure_s, 
                tel_key, presets, dt_utc, 
                seeing_fwhm_arcsec, pixel_scale
            )
            sats_out_df = pd.DataFrame(sat_trail_details)
            if not sats_out_df.empty:
                sats_out_df = pd.merge(sats_out_df, satellites_df_tle, left_on='satname', right_on='name', how='left')
                sats_out_df.drop(columns=['name'], inplace=True, errors='ignore')

    elif sat_mode == "Synthetic":
        if synth_sat_params and synth_sat_params[0].get('gen_mode') == 'Random':
            log("Generating random parameters for synthetic trails...")
            for p in synth_sat_params:
                p['start_x_pix'] = random.uniform(0, nx)
                p['start_y_pix'] = random.uniform(0, ny)
                p['angle_deg'] = random.uniform(0, 360)

        log(f"Generating {len(synth_sat_params)} synthetic satellite trails..."); update_progress(50)
        for p in synth_sat_params: p['pixel_scale'] = pixel_scale
        ideal_signal, sat_trail_details = satellite.generate_synthetic_trails(
            image=ideal_signal, trail_params_list=synth_sat_params,
            zero_point=zero_point, exposure_s=exposure_s
        )
        sats_out_df = pd.DataFrame(sat_trail_details)
        update_progress(85)

    log("Applying noise model..."); update_progress(95)
    final_image = background.generate_noise(ideal_signal, bias_level_e, bias_spread_e, sen['read_noise_e'], sen['saturation_level_e'])

    return {
        "image": final_image, "stars_df": stars_df, "galaxies_df": galaxies_df, 
        "satellites_df": sats_out_df, "wcs": wcs
    }


