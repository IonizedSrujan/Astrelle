# Astrelle Sky Simulator

Astrelle is a high-fidelity astronomical image simulator designed to generate realistic synthetic sky images in FITS format. It models the photon-to-pixel pipeline: sourcing celestial objects from catalogs, applying atmospheric effects, and simulating detector noise. Its core purpose is to provide ground-truth datasets for developing and validating algorithms, especially for detecting and mitigating satellite trails in astronomical observations.

---

## Key Features
- **Catalog-Based Sky Simulation**: Generates star fields and galaxies using Gaia DR3 and 2MASS XSC catalogs.  
- **Physical Accuracy**: Models PSF (seeing), sky background, and detector noise (shot, read, dark current).  
- **Satellite Trail Simulation**: Supports live TLE-based real passes and configurable synthetic trails.  
- **Configurable Instruments**: JSON-based presets for telescope and sensor configurations.  
- **Cross-Platform GUI**: PyQt6-based interface available on Linux, Windows, and macOS.  
- **Standard Outputs**: Produces FITS with valid WCS headers, PNG images, and CSV ground-truth tables.  

---

## Installation & Usage

No external dependencies are required. Download the package for your operating system from the [Releases](https://github.com/YOURUSER/astrelle/releases) page and run directly.

### Linux
1. Download `astrelle_v1.0.1.AppImage`.  
2. Make it executable:  
   ```bash
   chmod +x astrelle_v1.0.1.AppImage
   ```
3. Run:  
   ```bash
   ./astrelle_v1.0.1.AppImage
   ```

### Windows
1. Download `astrelle_v1.0.1.exe`.  
2. Double-click to launch the installer and follow on-screen instructions.  
3. Open *Astrelle Sky Simulator* from the Start Menu after installation.  

### macOS
1. Download `astrelle_v1.0.1_macOS.zip`.  
2. Extract the archive.  
3. Open `Astrelle.app` and, if prompted, allow it through *System Preferences > Security & Privacy*.  

---

## Version Information
- **v1.0.1** (Latest): Multi-platform builds (Windows, macOS, Linux), stability fixes.  
- **v1.0.0**: Initial public release.  

---

## License
BSD 3-Clause License. See [LICENSE](LICENSE) file.

---

## Acknowledgments
Astrelle builds upon open-source astronomical software:
- [Astropy](https://www.astropy.org/)  
- [Skyfield](https://rhodesmill.org/skyfield/)  
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)  
