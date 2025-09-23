# background.py
# Handles atmospheric physics, zero-point calculation, and background/noise generation.

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pkg_resources

from astropy import units as u
from astropy.constants import h, c
from astropy.io import fits
from astropy.utils.iers import conf as iers_conf

# Disable auto-downloading of IERS data to prevent issues on offline systems.
iers_conf.auto_download = False
iers_conf.auto_max_age = None

# Gets the full path to a data file within the package data directory.
def get_data_path(filename):
    return pkg_resources.resource_filename('astrelle', f'data/{filename}')

# Define paths to data files.
PRESET_FILE = get_data_path("presets.json")
PASSBAND_FILE = get_data_path("passband.dat")
VEGA_SPECTRUM_FITS_FILE = get_data_path("alpha_lyr_stis_005.fits")

# Physical constants in convenient units.
h_planck_erg_s = h.to('erg s').value
c_angstrom_s = c.to('angstrom/s').value
MAG_VEGA = 0.03 # Apparent magnitude of Vega in the G band.
GAIA_G_EFF_LAM_NM = 639.02 # Effective wavelength of Gaia's G band.

# Caching for frequently accessed data to avoid repeated file I/O.
_presets_cache = None
_passband_cache = None
_vega_cache = None
_zero_point_cache = {}

# Loads telescope and sensor presets from the JSON file.
def load_presets():
    global _presets_cache
    if _presets_cache is None:
        with open(PRESET_FILE, 'r') as f: _presets_cache = json.load(f)
    return _presets_cache

# Saves user-modified presets to a config directory.
def save_presets(presets: dict):
    user_config_dir = Path.home() / ".config" / "astrelle"
    user_config_dir.mkdir(parents=True, exist_ok=True)
    user_preset_file = user_config_dir / "presets.json"
    with open(user_preset_file, 'w') as f:
        json.dump(presets, f, indent=2)
    global _presets_cache
    _presets_cache = presets

# Loads the Gaia G-band passband data.
def load_passband_data():
    global _passband_cache
    if _passband_cache is None:
        df = pd.read_csv(PASSBAND_FILE, delim_whitespace=True,
                         names=['lambda', 'GPb', 'e_GPb', 'BPPb', 'e_BPPb', 'RPPb', 'e_RPPb'], comment='#')
        _passband_cache = df[df['GPb'] < 99.0].copy() # Filter out invalid data points.
    return _passband_cache

# Loads the reference spectrum of Vega.
def load_vega_spectrum():
    global _vega_cache
    if _vega_cache is None:
        with fits.open(VEGA_SPECTRUM_FITS_FILE) as hdul:
            data = hdul[1].data
            _vega_cache = pd.DataFrame({'wavelength_ang': data['WAVELENGTH'], 'flux_flam': data['FLUX']})
    return _vega_cache

# Calculates atmospheric transmission based on a model by Mohan et al.
def calculate_atmospheric_transmission_mohan(lams_nm, altitude_km):
    lams_um = lams_nm / 1000.0
    h = altitude_km
    
    # Calculate Rayleigh scattering, aerosol, and ozone absorption components.
    lams_um_sq_inv = np.divide(1.0, lams_um**2, out=np.zeros_like(lams_um), where=lams_um != 0)
    term1_denom, term2_denom = 146 - lams_um_sq_inv, 41 - lams_um_sq_inv
    term1_denom[term1_denom == 0], term2_denom[term2_denom == 0] = np.inf, np.inf # Avoid division by zero.
    k_r = (0.0095 * np.exp(-h / 8.0) / lams_um**4) * (0.23465 + 107.6 / term1_denom + 0.93161 / term2_denom)
    k_a = (0.087 * np.exp(-h / 1.5) / lams_um**0.8)
    safe_lams_um = np.clip(lams_um, 0.3, 1.0) # Clip to valid range for ozone model.
    k_o = 839.4375 * np.exp(-131 * (safe_lams_um - 0.26)) + 0.0381562 * np.exp(-188 * (safe_lams_um - 0.59)**2)
    
    # Combine components and convert from optical depth to transmission.
    return 10**(-0.4 * (k_r + k_a + k_o))

# Calculates the instrumental zero point (e-/s for a 0-magnitude star).
def calculate_zero_point(tel, sen):
    cache_key = (tel['name'], sen['name'])
    if cache_key in _zero_point_cache: return _zero_point_cache[cache_key]
    
    # Load required data and interpolate Vega's spectrum to the passband wavelengths.
    passband, vega = load_passband_data(), load_vega_spectrum()
    lams_nm_grid = passband['lambda'].values
    lams_ang_grid = lams_nm_grid * 10.0
    vega_interp = interp1d(vega['wavelength_ang'], vega['flux_flam'], bounds_error=False, fill_value=0.0)
    resampled_vega_flux = vega_interp(lams_ang_grid)
    
    # Calculate atmospheric transmission and telescope collecting area.
    transmission = calculate_atmospheric_transmission_mohan(lams_nm_grid, tel.get('elevation_m', 2450) / 1000.0)
    area_cm2 = math.pi * (((tel['diameter_mm']/10.0)/2.0)**2 - ((tel.get('inner_diameter_mm', 0.0)/10.0)/2.0)**2)
    
    # Integrate the flux of Vega over the passband.
    energy_rate = resampled_vega_flux * transmission * tel['throughput'] * area_cm2 * passband['GPb'].values
    total_energy_per_s = np.trapz(energy_rate, lams_ang_grid)
    
    # Convert energy rate to electron rate and calculate the zero point.
    photon_energy_erg = h_planck_erg_s * c_angstrom_s / (GAIA_G_EFF_LAM_NM * 10.0)
    electron_rate_vega = (total_energy_per_s / photon_energy_erg) * sen['qe']
    zero_point = electron_rate_vega / (10**(-0.4 * MAG_VEGA))
    
    _zero_point_cache[cache_key] = zero_point
    return zero_point

# Generates the base background signal from skyglow and dark current.
def generate_background_e(zero_point, sky_mag_per_arcsec2, pixel_scale, exposure_s, dark_current_e_per_s, shape):
    sky_e_per_s_arcsec2 = zero_point * 10**(-0.4 * sky_mag_per_arcsec2)
    sky_e_per_s_pixel = sky_e_per_s_arcsec2 * (pixel_scale**2)
    sky_level_e = sky_e_per_s_pixel * exposure_s
    dark_level_e = dark_current_e_per_s * exposure_s
    return np.full(shape, sky_level_e + dark_level_e, dtype=np.float32)

# Applies Poisson, read, and bias noise to an ideal signal image.
def generate_noise(ideal_signal, bias_level_e, bias_spread_e, read_noise_e, saturation_e):
    rng = np.random.default_rng()
    # Apply Poisson (shot) noise.
    noisy_image = rng.poisson(np.maximum(0, ideal_signal)).astype(np.float32)
    # Add bias offset and noise.
    noisy_image += rng.normal(bias_level_e, bias_spread_e, size=ideal_signal.shape)
    # Add read noise.
    noisy_image += rng.normal(0.0, read_noise_e, size=ideal_signal.shape)
    # Clip at the sensor's saturation level.
    np.clip(noisy_image, 0, saturation_e, out=noisy_image)
    return noisy_image


