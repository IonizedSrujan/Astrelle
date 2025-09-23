# utils.py
# Shared utility functions for the Astrelle package.

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Rebins a 2D array by an integer factor by summing blocks of pixels.
def rebin(arr, factor):
    new_shape = (arr.shape[0] // factor, arr.shape[1] // factor)
    shape = (new_shape[0], factor, new_shape[1], factor)
    return arr.reshape(shape).sum(-1).sum(1)

# Adds a flux-scaled stamp to an image at a sub-pixel location.
def add_stamp(image, stamp, x, y, flux_e):
    h, w = image.shape
    r = stamp.shape[0] // 2
    ix, iy = int(round(x)), int(round(y))
    
    # Determine the overlapping region between the stamp and the image.
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    
    # Add the stamp if there is any overlap.
    if x1 > x0 and y1 > y0:
        sx0, sy0 = x0 - (ix - r), y0 - (iy - r)
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
        image[y0:y1, x0:x1] += flux_e * stamp[sy0:sy1, sx0:sx1]

# Generates a normalized 2D Gaussian PSF kernel from FWHM.
def gaussian_psf_kernel(fwhm_pix, size=None, oversample=5):
    # Determine a suitable size for the kernel if not provided.
    if size is None:
        size = max(11, int(round(fwhm_pix * 5)) | 1) # Ensure size is odd.
    
    # Oversample to get a more accurate, smoother kernel.
    radius = (size - 1) / 2.0
    center = radius * oversample + (oversample -1) / 2.0
    sigma_os = (fwhm_pix * oversample) / 2.35482
    
    # Create the oversampled Gaussian profile.
    oversampled_dim = size * oversample
    y, x = np.indices((oversampled_dim, oversampled_dim))
    r_sq = (x - center)**2 + (y - center)**2 # Use polar coordinates for calculation.
    stamp_oversampled = np.exp(-r_sq / (2 * sigma_os**2))
    
    # Rebin the oversampled stamp back to the target resolution.
    stamp_rebinned = rebin(stamp_oversampled, oversample)
    
    # Normalize the kernel so that its sum is 1.
    kernel_sum = stamp_rebinned.sum()
    if kernel_sum > 0:
        return stamp_rebinned / kernel_sum
    return stamp_rebinned

# Saves a NumPy array as a PNG image with automatic contrast scaling.
def export_png(path, arr):
    vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99.8)
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

# Saves a NumPy array as a FITS file with an optional header.
def save_fits(path, arr, header_extra=None):
    hdu = fits.PrimaryHDU(arr.astype(np.float32))
    if header_extra:
        hdu.header.update(header_extra)
    hdu.writeto(str(path), overwrite=True)


