# Astrelle Sky Simulator

Astrelle is a high-fidelity sky simulation software designed to generate realistic, synthetic astronomical images in the FITS format. It models the entire photon-to-pixel pipeline, from sourcing celestial objects from real-world catalogs to simulating atmospheric effects and detector noise. Its primary purpose is to create ground-truth datasets for the development and validation of algorithms that detect and mitigate satellite trails in astronomical data.

---

## Key Features
- **Realistic Sky Simulation**: Generates star fields and galaxies sourced from major astronomical catalogs (Gaia DR3, 2MASS XSC).
- **Physical Modeling**: Accurately models atmospheric seeing (PSF), sky background, and detector noise (shot, read, dark current).
- **Satellite Trail Generation**: Simulates both real satellite passes by fetching live TLE data, and allows creation of configurable synthetic trails.
- **Configurable Instruments**: JSON-based preset system defines telescope and sensor combinations.
- **Graphical User Interface**: Intuitive GUI built with PyQt6 provides full control over simulation parameters.
- **Standard Data Formats**: Outputs FITS files with valid WCS headers, along with optional PNG images and CSV ground-truth lists.

---

## Installation

Astrelle is designed for Debian-based Linux systems (e.g., Ubuntu).

### 1. System Dependencies
```bash
sudo apt-get update
sudo apt-get install python3-numpy python3-pandas python3-scipy python3-astropy python3-astroquery python3-pyvo python3-matplotlib python3-requests
```

### 2. Python-Specific Dependencies
```bash
pip3 install PyQt6 skyfield
```

### 3. Application Installation
Download the latest `.deb` package from the [Releases](https://github.com/YOURUSER/astrelle/releases) page and install:

```bash
sudo dpkg -i astrelle_*.deb
# Fix missing dependencies if needed
sudo apt-get install -f
```

---

## Usage

Launch in two ways:

- **From Application Menu**: Search for *Astrelle Sky Simulator*.  
- **From Terminal**:
  ```bash
  astrelle-gui
  ```

### Example: Synthetic Trail Simulation

**Scenario**: Generate a 300-second exposure of the Coma Galaxy Cluster with a bright artificial satellite trail.

- **Instrument**: DFOT preset  
- **Pointing**:  
  - RA: `12h 59m 49s`  
  - Dec: `+27d 58m 50s`  
- **Observation**:  
  - Exposure Time: `300 s`  
  - Seeing: `1.2 arcsec`  
  - Sky Brightness: `22.0 mag/arcsec²`  
- **Satellites**:  
  - Mode: `Synthetic`  
  - Trail Parameters:  
    - Magnitude: `7.0`  
    - Start Pixel (X,Y): `0, 1024`  
    - End Pixel (X,Y): `2048, 1024`

Run the simulation and monitor the log window for progress.

---

## License
This project is licensed under the [BSD 3-Clause License](LICENSE).

---

## Acknowledgments
Astrelle is built upon a foundation of open-source astronomical software, including:

- [Astropy](https://www.astropy.org/)  
- [Skyfield](https://rhodesmill.org/skyfield/)  
- [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/)  
- and many others.
