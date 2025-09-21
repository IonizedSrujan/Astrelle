# utils.py
# Shared utility functions

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def rebin(arr, factor):
    """Rebins a 2D array by summing blocks."""
    new_shape = (arr.shape[0] // factor, arr.shape[1] // factor)
    shape = (new_shape[0], factor, new_shape[1], factor)
    return arr.reshape(shape).sum(-1).sum(1)

def add_stamp(image, stamp, x, y, flux_e):
    """Adds a flux-scaled stamp to an image at a given location."""
    h, w = image.shape
    r = stamp.shape[0] // 2
    ix, iy = int(round(x)), int(round(y))
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    if x1 > x0 and y1 > y0:
        sx0, sy0 = x0 - (ix - r), y0 - (iy - r)
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
        image[y0:y1, x0:x1] += flux_e * stamp[sy0:sy1, sx0:sx1]

def gaussian_psf_kernel(fwhm_pix, size=None, oversample=5):
    """
    Generates a normalized 2D Gaussian PSF kernel from FWHM.
    """
    if size is None:
        size = max(11, int(round(fwhm_pix * 5)) | 1) # Ensure odd size
    
    radius = (size - 1) / 2.0
    center = radius * oversample + (oversample -1) / 2.0
    sigma_os = (fwhm_pix * oversample) / 2.35482
    
    oversampled_dim = size * oversample
    y, x = np.indices((oversampled_dim, oversampled_dim))
    
    stamp_oversampled = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma_os**2))
    stamp_rebinned = rebin(stamp_oversampled, oversample)
    
    kernel_sum = stamp_rebinned.sum()
    if kernel_sum > 0:
        return stamp_rebinned / kernel_sum
    return stamp_rebinned

def export_png(path, arr):
    """Saves a NumPy array as a PNG image."""
    vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99.8)
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_fits(path, arr, header_extra=None):
    """Saves a NumPy array as a FITS file."""
    hdu = fits.PrimaryHDU(arr.astype(np.float32))
    if header_extra:
        for k, v in header_extra.items():
            hdu.header[k] = v if isinstance(v, tuple) else (v, "")
    hdu.writeto(str(path), overwrite=True)

