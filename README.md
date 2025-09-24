# Astrelle Sky Simulator

Astrelle is a high-fidelity astronomical image simulator designed to generate realistic synthetic sky images in FITS format. It models the photon-to-pixel pipeline: sourcing celestial objects from catalogs, applying atmospheric effects, and simulating detector noise. Its core purpose is to provide ground-truth datasets for developing and validating algorithms, especially for detecting and mitigating satellite trails in astronomical observations.

---

## Key Features
- **Catalog-Based Sky Simulation**: Generates star fields and galaxies using Gaia DR3 and 2MASS XSC catalogs.  
- **Physical Accuracy**: Models PSF (seeing), sky background, and detector noise (shot, read, dark current).  
- **Satellite Trail Simulation**:  
  - Live TLE-based real passes using the **SGP4 propagation model**.  
  - Integrated multiple CelesTrak sets (Starlink, OneWeb, GPS, Galileo, GLONASS, ISS, NOAA/Meteor).  
  - Configurable synthetic trails with pixel-based start points and velocities (pixels, pixels/s).  
  - Smooth continuous trail rendering for realistic appearance.  
- **Configurable Instruments**: JSON-based presets for telescope and sensor configurations.  
- **Cross-Platform GUI**: PyQt6-based interface available on Linux, Windows, and macOS.  
  - Export TLEs to `.txt`.  
  - Improved cross-platform widget consistency.  
  - Enhanced image preview with sharp contrast.  
  - Simulation can now be stopped mid-run.  
- **Standard Outputs**: Produces FITS with valid WCS headers, PNG images, and CSV ground-truth tables.  

---

## Installation & Usage

No external dependencies are required. Download the package for your operating system from the [Releases](https://github.com/IonizedSrujan/Astrelle/releases) page and run directly.

### Linux
1. Download `astrelle_v1.1.0.AppImage`.  
2. Make it executable:  
   ```bash
   chmod +x astrelle_v1.1.0.AppImage
   ```
3. Run:  
   ```bash
   ./astrelle_v1.1.0.AppImage
   ```

### Windows
1. Download `astrelle_v1.1.0.exe`.  
2. Double-click to launch the installer and follow on-screen instructions.  
3. Open *Astrelle Sky Simulator* from the Start Menu after installation.  

### macOS
1. Download `astrelle_v1.1.0_macOS.zip`.  
2. Extract the archive.  
3. Open `Astrelle.app` and, if prompted, allow it through *System Preferences > Security & Privacy*.  

---

## Version Information
- **v1.1.0** (Latest): Continuous satellite trails, SGP4 propagation, integrated CelesTrak sets, GUI improvements (TLE export, stop simulation, contrast fix).
- **v1.0.1**: Multi-platform builds (Windows, macOS, Linux), stability fixes.  
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
