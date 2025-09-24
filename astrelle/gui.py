# gui.py
# PyQt6 GUI for Astrelle Sky Simulator.

import sys
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QFileDialog,
    QProgressBar, QToolBar, QMenu, QDialog, QFormLayout,
    QLineEdit, QMessageBox, QDateEdit, QTimeEdit, QSplitter, QToolButton,
    QCheckBox, QGroupBox, QSizePolicy, QSpinBox, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt6.QtGui import QPalette, QColor, QAction

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import EarthLocation
from astropy import units as u


# Use relative imports for package structure.
from .main import run_simulation
from .utils import export_png, save_fits
from .background import load_presets, save_presets
from .satellite import TLE_SOURCES, download_tles

# Dark theme stylesheet for the application.
DARK_STYLESHEET = """
QWidget {
    color: #e0e0e0;
    background-color: #2e2e2e;
    font-size: 10pt;
}
QMainWindow {
    background-color: #252525;
}
QToolBar {
    background-color: #383838;
    border: 1px solid #444;
}
QToolButton, QPushButton {
    background-color: #4a4a4a;
    border: 1px solid #555;
    padding: 6px;
    border-radius: 4px;
}
QToolButton:hover, QPushButton:hover {
    background-color: #5a5a5a;
}
QToolButton:checked {
    background-color: #007acc;
}
QToolButton:pressed, QPushButton:pressed {
    background-color: #6a6a6a;
}
QToolButton:menu-indicator {
    image: none;
}
QMenu {
    background-color: #383838;
    border: 1px solid #555;
}
QMenu::item:selected {
    background-color: #5a5a5a;
}
QLineEdit, QTextEdit, QComboBox, QDateEdit, QTimeEdit, QSpinBox {
    background-color: #383838;
    border: 1px solid #555;
    padding: 4px;
    border-radius: 4px;
}
QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button {
    border: none;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #444;
    border-radius: 6px;
    margin-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 10px;
}
QSplitter::handle {
    background: #444;
}
QProgressBar {
    border: 1px solid #555;
    border-radius: 4px;
    text-align: center;
    background-color: #383838;
}
QProgressBar::chunk {
    background-color: #007acc;
    border-radius: 3px;
}
QLabel {
    background-color: transparent;
}
"""

# Worker thread for running the main simulation.
class SimWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    log = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.params['log_callback'] = self.log.emit
            self.params['progress_callback'] = self.progress.emit
            out = run_simulation(**self.params)
            self.finished.emit(out)
        except Exception as e:
            import traceback
            self.log.emit(f"[{datetime.now(timezone.utc):%H:%M:%S}] FATAL ERROR: {e}\n{traceback.format_exc()}")
            self.finished.emit(None)

# Dialog window for adding new telescope or sensor presets.
class AddPresetDialog(QDialog):
    def __init__(self, mode="telescope", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Add New {mode.capitalize()} Preset")
        self.mode = mode
        form = QFormLayout(self)
        self.fields = {'key': QLineEdit(), 'name': QLineEdit()}
        form.addRow("Key (e.g., 'dfot_new'):", self.fields['key'])
        form.addRow("Name:", self.fields['name'])
        
        if mode == "telescope":
            self.fields.update({k: QLineEdit(v) for k, v in {
                'diameter_mm': "1300", 'inner_diameter_mm': "400", 'f_number': "4.0",
                'throughput': "0.65", 'latitude': "29.3609", 'longitude': "79.6843", 'elevation_m': "2540"
            }.items()})
            for k, v_str in [("Diameter (mm)", 'diameter_mm'), ("Inner Diameter (mm)", 'inner_diameter_mm'),
                             ("F-Number", 'f_number'), ("Throughput (0-1)", 'throughput'),
                             ("Latitude (deg)", 'latitude'), ("Longitude (deg)", 'longitude'), ("Elevation (m)", 'elevation_m')]:
                form.addRow(k, self.fields[v_str])
        else: # sensor
            self.fields.update({k: QLineEdit(v) for k, v in {
                'resolution': "2048x2048", 'pixel_size_um': "13.5", 'read_noise_e': "3.0",
                'dark_current_e_per_s': "0.001", 'qe': "0.95", 'saturation_level_e': "100000"
            }.items()})
            for k, v_str in [("Resolution (WxH)", 'resolution'), ("Pixel Size (µm)", 'pixel_size_um'),
                             ("Read Noise (e-)", 'read_noise_e'), ("Dark Current (e-/pixel/s)", 'dark_current_e_per_s'),
                             ("Quantum Efficiency (0-1)", 'qe'), ("Saturation (e-)", 'saturation_level_e')]:
                form.addRow(k, self.fields[v_str])
        
        save_btn = QPushButton("Save Preset"); save_btn.clicked.connect(self.save); form.addRow(save_btn)

    def save(self):
        try:
            key = self.fields['key'].text().strip()
            if not key: raise ValueError("Key cannot be empty.")
            presets, new = load_presets(), {"name": self.fields['name'].text().strip() or key}
            
            if self.mode == "telescope":
                for k in ['diameter_mm', 'inner_diameter_mm', 'f_number', 'throughput', 'latitude', 'longitude', 'elevation_m']:
                    new[k] = float(self.fields[k].text())
                presets["telescopes"][key] = new
            else: # sensor
                res = self.fields['resolution'].text().lower().split('x')
                if len(res) != 2: raise ValueError("Resolution must be WxH")
                new["resolution"] = [int(res[0]), int(res[1])]
                for k in ['pixel_size_um', 'read_noise_e', 'dark_current_e_per_s', 'qe']: new[k] = float(self.fields[k].text())
                new['saturation_level_e'] = int(self.fields['saturation_level_e'].text())
                presets["sensors"][key] = new
            
            save_presets(presets)
            QMessageBox.information(self, "Success", f"Preset '{key}' saved.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Could not save preset: {e}")

# Widget for editing parameters of a single synthetic satellite trail.
class SyntheticTrailEditor(QWidget):
    def __init__(self, trail_number):
        super().__init__()
        self.trail_number = trail_number
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edits = {k: QLineEdit(v) for k, v in {
            "sat_mag": "7.0", "trail_fwhm": "1.0",
            "start_x_pix": "1024", "start_y_pix": "1024",
            "sat_angle": f"{45.0 + (trail_number - 1) * 5.0:.1f}",
            "sat_speed": "100.0"
        }.items()}
        layout.addRow("Apparent Mag:", self.edits['sat_mag'])
        layout.addRow("Trail Width FWHM (\")", self.edits['trail_fwhm'])
        layout.addRow("Start X (pix):", self.edits['start_x_pix'])
        layout.addRow("Start Y (pix):", self.edits['start_y_pix'])
        layout.addRow("Angle (deg, 0=right):", self.edits['sat_angle'])
        layout.addRow("Speed (pixels/s):", self.edits['sat_speed'])

    def set_manual_mode(self, is_manual):
        self.layout().setRowVisible(2, is_manual)
        self.layout().setRowVisible(3, is_manual)

    def get_params(self):
        return {
            "apparent_mag": float(self.edits['sat_mag'].text()),
            "trail_fwhm_arcsec": float(self.edits['trail_fwhm'].text()),
            "start_x_pix": float(self.edits['start_x_pix'].text()),
            "start_y_pix": float(self.edits['start_y_pix'].text()),
            "angle_deg": float(self.edits['sat_angle'].text()),
            "speed_pixels_s": float(self.edits['sat_speed'].text()),
        }

# Main application window.
class SkySimGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astrelle"); self.resize(1400, 900)
        self.presets = load_presets(); self.current_out = None
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(DARK_STYLESHEET)
        
        tb = QToolBar("Main"); self.addToolBar(tb)
        self.export_btn = QToolButton(); self.export_btn.setText("Export")
        self.export_menu = QMenu()
        self.export_menu.addAction("Export PNG", lambda: self.export("png"))
        self.export_menu.addAction("Export FITS", lambda: self.export("fits"))
        self.export_menu.addSeparator()
        self.export_stars_action = self.export_menu.addAction("Export Star List (CSV)", lambda: self.export_csv("stars"))
        self.export_gals_action = self.export_menu.addAction("Export Galaxy List (CSV)", lambda: self.export_csv("galaxies"))
        self.export_sats_action = self.export_menu.addAction("Export Satellite List (CSV)", lambda: self.export_csv("satellites"))
        self.export_menu.addSeparator()
        self.export_full_tle_action = self.export_menu.addAction("Export Full TLE Catalog (.txt)", self.export_full_tle_catalog)
        self.export_btn.setMenu(self.export_menu); self.export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.export_btn.setEnabled(False) # Disabled until a sim is run
        self.export_stars_action.setEnabled(False)
        self.export_gals_action.setEnabled(False)
        self.export_sats_action.setEnabled(False)
        self.export_full_tle_action.setEnabled(True) # Can always export TLE data
        tb.addWidget(self.export_btn)
        
        add_preset_btn = QToolButton(); add_preset_btn.setText("Add Preset")
        add_menu = QMenu()
        add_menu.addAction("Add Telescope", lambda: self.add_preset("telescope"))
        add_menu.addAction("Add Sensor", lambda: self.add_preset("sensor"))
        add_preset_btn.setMenu(add_menu); add_preset_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        tb.addWidget(add_preset_btn)

        central = QWidget(); self.setCentralWidget(central)
        splitter = QSplitter(Qt.Orientation.Horizontal, self); QVBoxLayout(central).addWidget(splitter)
        
        left_panel = QWidget(); self.controls = QVBoxLayout(left_panel)
        left_panel.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        
        inst_group = QGroupBox("Instrument"); inst_layout = QVBoxLayout(); inst_group.setLayout(inst_layout)
        inst_form = QFormLayout()
        self.tel_cb = QComboBox(); self.tel_cb.addItems(self.presets["telescopes"].keys())
        self.tel_cb.currentTextChanged.connect(self.update_info_displays)
        inst_form.addRow("Telescope:", self.tel_cb)
        self.tel_info = QTextEdit(); self.tel_info.setReadOnly(True); self.tel_info.setMinimumHeight(100)
        inst_layout.addLayout(inst_form)
        inst_layout.addWidget(self.tel_info)
        
        sensor_form = QFormLayout()
        self.sen_cb = QComboBox(); self.sen_cb.addItems(self.presets["sensors"].keys())
        self.sen_cb.currentTextChanged.connect(self.update_info_displays)
        sensor_form.addRow("Sensor:", self.sen_cb)
        self.sen_info = QTextEdit(); self.sen_info.setReadOnly(True); self.sen_info.setMinimumHeight(100)
        inst_layout.addLayout(sensor_form)
        inst_layout.addWidget(self.sen_info)
        self.controls.addWidget(inst_group)

        point_group = QGroupBox("Pointing"); point_layout = QVBoxLayout(); point_group.setLayout(point_layout)
        self.point_mode_cb = QComboBox(); self.point_mode_cb.addItems(["Local (Alt/Az)", "Equatorial (RA/Dec)"])
        self.point_mode_cb.currentTextChanged.connect(self.toggle_pointing_mode)
        point_layout.addWidget(self.point_mode_cb)
        self.pointing_widgets = QWidget(); point_form = QFormLayout(self.pointing_widgets)
        self.date_edit = QDateEdit(calendarPopup=True); self.date_edit.setDateTime(QDateTime.currentDateTime())
        self.time_edit = QTimeEdit(displayFormat="HH:mm:ss"); self.time_edit.setDateTime(QDateTime.currentDateTime())
        self.tz_cb = QComboBox(); self.tz_cb.addItems(["UTC", "IST"]); self.tz_cb.setCurrentText("IST")
        self.edits = {k: QLineEdit(v) for k, v in { "alt": "90.0", "az": "0.0", "ra": "270.26", "dec": "29.35" }.items()}
        self.alt_az_rows = [
            ("Date:", self.date_edit), ("Time:", self.time_edit), ("Timezone:", self.tz_cb),
            ("Altitude (deg):", self.edits['alt']), ("Azimuth (deg):", self.edits['az'])
        ]
        self.ra_dec_rows = [ ("RA (deg):", self.edits['ra']), ("Dec (deg):", self.edits['dec']) ]
        for label, widget in self.alt_az_rows + self.ra_dec_rows: point_form.addRow(label, widget)
        point_layout.addWidget(self.pointing_widgets)
        self.controls.addWidget(point_group)

        obs_group = QGroupBox("Observation"); obs_form = QFormLayout(); obs_group.setLayout(obs_form)
        self.edits.update({k: QLineEdit(v) for k, v in {
            "exp": "120", "mag": "20.0", "see": "1.5", "sky": "21.0",
            "bias": "100", "spread": "2.0"
        }.items()})
        obs_form.addRow("Exposure (s):", self.edits['exp'])
        obs_form.addRow("Limiting Mag:", self.edits['mag'])
        obs_form.addRow("Seeing FWHM (\"):", self.edits['see'])
        obs_form.addRow("Sky Brightness (mag/\"):", self.edits['sky'])
        obs_form.addRow("Bias Level (e-):", self.edits['bias'])
        obs_form.addRow("Bias Spread (e-):", self.edits['spread'])
        self.controls.addWidget(obs_group)

        sat_group = QGroupBox("Satellite Trails"); self.sat_layout = QVBoxLayout(); sat_group.setLayout(self.sat_layout)
        tle_source_layout = QFormLayout()
        self.tle_source_cb = QComboBox()
        self.tle_source_cb.addItems(TLE_SOURCES.keys())
        tle_source_layout.addRow("TLE Catalog:", self.tle_source_cb)
        self.sat_layout.addLayout(tle_source_layout)
        
        self.sat_mode_cb = QComboBox(); self.sat_mode_cb.addItems(["Catalog", "Synthetic"])
        self.sat_mode_cb.currentTextChanged.connect(self.toggle_sat_mode)
        self.sat_layout.addWidget(self.sat_mode_cb)
        
        self.synth_sat_widgets = QWidget(); synth_layout = QVBoxLayout(self.synth_sat_widgets)
        synth_layout.setContentsMargins(0, 5, 0, 0)
        synth_mode_layout = QHBoxLayout()
        synth_mode_layout.addWidget(QLabel("Generation Mode:"))
        self.synth_gen_mode_cb = QComboBox(); self.synth_gen_mode_cb.addItems(["Manual", "Random"])
        self.synth_gen_mode_cb.currentTextChanged.connect(self.toggle_synth_gen_mode)
        synth_mode_layout.addWidget(self.synth_gen_mode_cb)
        synth_mode_layout.addStretch()
        synth_layout.addLayout(synth_mode_layout)
        num_trails_layout = QHBoxLayout()
        num_trails_layout.addWidget(QLabel("Number of Trails:"))
        self.num_trails_spinbox = QSpinBox(); self.num_trails_spinbox.setMinimum(1); self.num_trails_spinbox.setValue(1)
        self.num_trails_spinbox.valueChanged.connect(self.update_trail_editors)
        num_trails_layout.addWidget(self.num_trails_spinbox)
        self.trail_selector_cb = QComboBox(); self.trail_selector_cb.setVisible(False)
        self.trail_selector_cb.currentIndexChanged.connect(self.switch_trail_editor)
        num_trails_layout.addWidget(self.trail_selector_cb)
        num_trails_layout.addStretch()
        synth_layout.addLayout(num_trails_layout)
        self.trail_editors_stack = QStackedWidget()
        synth_layout.addWidget(self.trail_editors_stack)
        self.trail_editors = []
        self.sat_layout.addWidget(self.synth_sat_widgets)
        self.controls.addWidget(sat_group)
        
        anno_group = QGroupBox("Annotations"); anno_layout = QVBoxLayout(); anno_group.setLayout(anno_layout)
        self.annotate_stars_cb = QCheckBox("Annotate Bright Stars"); self.annotate_stars_cb.stateChanged.connect(self.redraw_plot)
        self.annotate_gals_cb = QCheckBox("Annotate Galaxies"); self.annotate_gals_cb.stateChanged.connect(self.redraw_plot)
        self.annotate_sats_cb = QCheckBox("Annotate Satellite Trails"); self.annotate_sats_cb.stateChanged.connect(self.redraw_plot)
        anno_layout.addWidget(self.annotate_stars_cb); anno_layout.addWidget(self.annotate_gals_cb); anno_layout.addWidget(self.annotate_sats_cb)
        self.controls.addWidget(anno_group)

        adv_group = QGroupBox("Advanced"); adv_form = QFormLayout(); adv_group.setLayout(adv_form)
        self.threads_spinbox = QSpinBox()
        self.threads_spinbox.setMinimum(1)
        self.threads_spinbox.setMaximum(os.cpu_count() or 16)
        self.threads_spinbox.setValue(8)
        adv_form.addRow("Satellite Threads:", self.threads_spinbox)
        self.controls.addWidget(adv_group)

        self.run_btn = QPushButton("Run Simulation"); self.run_btn.clicked.connect(self.toggle_simulation)
        self.controls.addWidget(self.run_btn)
        self.progress = QProgressBar()
        self.controls.addWidget(self.progress)
        self.controls.addStretch(1)
        splitter.addWidget(left_panel)

        right_panel = QWidget(); rv = QVBoxLayout(right_panel)
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.fig = Figure(figsize=(8, 6)); self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.fig.patch.set_facecolor('#2e2e2e')
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #383838;")
        rv.addWidget(self.toolbar)
        rv.addWidget(self.canvas, 1)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.log.setMaximumHeight(160)
        rv.addWidget(self.log)
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 1000])
        self._init_plot()
        self.update_info_displays()
        self.toggle_pointing_mode("Local (Alt/Az)")
        self.toggle_sat_mode("Catalog")
        self.update_trail_editors(1)
        self.toggle_synth_gen_mode("Manual")

    def _init_plot(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        self.ax.set_title("Run a simulation to see the output image", color='white')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def toggle_synth_gen_mode(self, mode):
        is_manual = mode == "Manual"
        for editor in self.trail_editors:
            editor.set_manual_mode(is_manual)

    def update_trail_editors(self, count):
        while len(self.trail_editors) > count:
            editor = self.trail_editors.pop()
            self.trail_editors_stack.removeWidget(editor)
            editor.deleteLater()
        
        while len(self.trail_editors) < count:
            trail_num = len(self.trail_editors) + 1
            editor = SyntheticTrailEditor(trail_num)
            self.trail_editors.append(editor)
            self.trail_editors_stack.addWidget(editor)

        self.toggle_synth_gen_mode(self.synth_gen_mode_cb.currentText())

        self.trail_selector_cb.clear()
        if count > 1:
            self.trail_selector_cb.addItems([f"Edit Trail #{i+1}" for i in range(count)])
            self.trail_selector_cb.setVisible(True)
        else:
            self.trail_selector_cb.setVisible(False)

        if self.trail_editors_stack.count() > 0:
            current_index = self.trail_editors_stack.currentIndex()
            if current_index >= self.trail_selector_cb.count():
                current_index = 0
            self.trail_selector_cb.setCurrentIndex(current_index)

    def switch_trail_editor(self, index):
        if index >= 0:
            self.trail_editors_stack.setCurrentIndex(index)

    def toggle_pointing_mode(self, mode):
        is_alt_az = mode == "Local (Alt/Az)"
        form_layout = self.pointing_widgets.layout()
        for i in range(form_layout.rowCount()):
            label_item = form_layout.itemAt(i, QFormLayout.ItemRole.LabelRole)
            if label_item:
                is_in_alt_az = any(label_item.widget().text() == label for label, _ in self.alt_az_rows)
                is_in_ra_dec = any(label_item.widget().text() == label for label, _ in self.ra_dec_rows)
                if is_in_alt_az:
                    form_layout.setRowVisible(i, is_alt_az)
                elif is_in_ra_dec:
                    form_layout.setRowVisible(i, not is_alt_az)

    def toggle_sat_mode(self, mode):
        self.synth_sat_widgets.setVisible(mode == "Synthetic")
        is_catalog = mode == "Catalog"
        tle_source_form = self.sat_layout.itemAt(0).layout()
        if tle_source_form:
            for i in range(tle_source_form.rowCount()):
                tle_source_form.setRowVisible(i, is_catalog)

    def add_preset(self, mode):
        dialog = AddPresetDialog(mode, self)
        if dialog.exec():
            self.presets = load_presets()
            cb = self.tel_cb if mode == 'telescope' else self.sen_cb
            p_key = f"{mode}s"
            current = cb.currentText()
            cb.clear()
            cb.addItems(self.presets[p_key].keys())
            if current in self.presets[p_key]:
                cb.setCurrentText(current)
            self.update_info_displays()

    def update_info_displays(self):
        tel_text = "\n".join(f"{k}: {v}" for k, v in self.presets["telescopes"].get(self.tel_cb.currentText(), {}).items())
        self.tel_info.setPlainText(tel_text)
        sen_text = "\n".join(f"{k}: {v}" for k, v in self.presets["sensors"].get(self.sen_cb.currentText(), {}).items())
        self.sen_info.setPlainText(sen_text)

    def _get_params(self):
        dt = self.date_edit.dateTime().toPyDateTime().replace(
            hour=self.time_edit.time().hour(), minute=self.time_edit.time().minute(), second=self.time_edit.time().second()
        )
        if self.tz_cb.currentText() == "IST":
            dt -= timedelta(hours=5, minutes=30)
        dt_utc = dt.replace(tzinfo=timezone.utc)
        
        selected_tle_source_name = self.tle_source_cb.currentText()

        synth_params = None
        if self.sat_mode_cb.currentText() == "Synthetic":
            synth_params = [editor.get_params() for editor in self.trail_editors]
            for p in synth_params:
                p['gen_mode'] = self.synth_gen_mode_cb.currentText()

        params = {
            "tel_key": self.tel_cb.currentText(), 
            "sen_key": self.sen_cb.currentText(),
            "dt_utc": dt_utc,
            "exposure_s": float(self.edits['exp'].text()),
            "mag_limit": float(self.edits['mag'].text()), 
            "seeing_fwhm_arcsec": float(self.edits['see'].text()),
            "sky_mag_per_arcsec2": float(self.edits['sky'].text()), 
            "bias_level_e": float(self.edits['bias'].text()),
            "bias_spread_e": float(self.edits['spread'].text()),
            "num_threads": self.threads_spinbox.value(),
            "pointing_mode": self.point_mode_cb.currentText(),
            "sat_mode": self.sat_mode_cb.currentText(),
            "synth_sat_params": synth_params,
            "tle_group": TLE_SOURCES.get(selected_tle_source_name, "active"),
        }

        if params["pointing_mode"] == "Local (Alt/Az)":
            params.update({"alt_deg": float(self.edits['alt'].text()), "az_deg": float(self.edits['az'].text())})
        else: # Equatorial
            params.update({"ra_deg": float(self.edits['ra'].text()), "dec_deg": float(self.edits['dec'].text())})
        
        return params

    def toggle_simulation(self):
        if self.worker and self.worker.isRunning():
            self.stop_sim()
        else:
            self.run_sim()

    def run_sim(self):
        try:
            params = self._get_params()
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {e}")
            return

        self.run_btn.setText("Stop Simulation")
        self.export_btn.setEnabled(False)
        
        self.worker = SimWorker(params)
        self.worker.log.connect(self.log.append)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.sim_done)
        self.worker.start()

    def stop_sim(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.log.append(f"[{datetime.now(timezone.utc):%H:%M:%S}] Simulation stopped by user.")
            self.sim_done(None) # Call with None to reset UI state

    def sim_done(self, out):
        self.run_btn.setText("Run Simulation")
        self.run_btn.setEnabled(True)
        self.current_out = out
        
        if out is not None:
            self.redraw_plot()
            self.export_btn.setEnabled(True)
            self.export_stars_action.setEnabled('stars_df' in out and not out['stars_df'].empty)
            self.export_gals_action.setEnabled('galaxies_df' in out and not out['galaxies_df'].empty)
            self.export_sats_action.setEnabled('satellites_df' in out and not out['satellites_df'].empty)
            self.log.append(f"[{datetime.now(timezone.utc):%H:%M:%S}] Simulation finished successfully.")
            self.progress.setValue(100)
        else:
            # Check if worker exists to avoid logging on initial startup
            if self.worker:
                self.log.append(f"[{datetime.now(timezone.utc):%H:%M:%S}] Simulation failed or was aborted.")
            self.progress.setValue(0)
        
        self.worker = None # Clear worker reference

    def redraw_plot(self):
        if not self.current_out or self.current_out.get("image") is None: return
        arr = self.current_out["image"]
        wcs = self.current_out.get("wcs")
        
        # Use a robust ZScale for automatic contrast stretching
        try:
            mean, median, std = sigma_clipped_stats(arr, sigma=3.0, maxiters=5)
            vmin = median - 2 * std
            vmax = median + 10 * std
        except Exception:
            # Fallback for very empty or unusual images
            vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99.8)
        
        self.fig.clear()

        if wcs:
            self.ax = self.fig.add_subplot(111, projection=wcs)
            self.ax.coords.grid(True, color='white', ls=':', alpha=0.5)
            self.ax.coords['ra'].set_axislabel('Right Ascension', color='white')
            self.ax.coords['dec'].set_axislabel('Declination', color='white')
            self.ax.coords['ra'].set_major_formatter('hh:mm:ss')
            self.ax.coords['dec'].set_major_formatter('dd:mm:ss')
            self.ax.coords['ra'].set_ticklabel(color='white')
            self.ax.coords['dec'].set_ticklabel(color='white')
            self.ax.tick_params(axis='both', colors='white')
            for spine in self.ax.spines.values():
                spine.set_edgecolor('white')
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.axis("off")

        self.ax.imshow(arr, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, interpolation="none")

        if self.annotate_stars_cb.isChecked() and self.current_out.get('stars_df') is not None and not self.current_out['stars_df'].empty:
            df = self.current_out['stars_df'][self.current_out['stars_df']['phot_g_mean_mag'] < 14]
            self.ax.scatter(df['x'], df['y'], s=40, facecolor='none', edgecolor='cyan', alpha=0.7)
        if self.annotate_gals_cb.isChecked() and self.current_out.get('galaxies_df') is not None and not self.current_out['galaxies_df'].empty:
            df = self.current_out['galaxies_df']
            self.ax.scatter(df['x'], df['y'], s=50, marker='s', facecolor='none', edgecolor='lime', alpha=0.7)
        if self.annotate_sats_cb.isChecked() and self.current_out.get('satellites_df') is not None and not self.current_out['satellites_df'].empty:
            for _, s in self.current_out['satellites_df'].iterrows():
                self.ax.text(s['x_center'], s['y_center'], s['satname'], color='yellow', fontsize=7, ha='center',
                             bbox={'facecolor':'black', 'alpha':0.5, 'edgecolor':'yellow', 'boxstyle':'round,pad=0.2'})

        self.ax.set_title("Simulated Image", color='white', pad=20)
        self.ax.set_facecolor('black')
        self.canvas.draw()

    def export(self, kind):
        if not self.current_out: return
        default_name = f"astrelle_output.{kind}"
        path, _ = QFileDialog.getSaveFileName(self, f"Save {kind.upper()}", default_name, filter=f"Files (*.{kind})")
        if not path: return
        try:
            if kind == "png":
                export_png(path, self.current_out["image"])
            elif kind == "fits":
                header = self.current_out["wcs"].to_header()
                save_fits(path, self.current_out["image"], header)
            self.log.append(f"Successfully saved {kind.upper()} to {os.path.basename(path)}")
        except Exception as e:
            self.log.append(f"ERROR: Could not save {kind.upper()}: {e}")

    def export_csv(self, kind):
        if not self.current_out:
            self.log.append("No simulation data available to export.")
            return

        df_key = f"{kind}s_df"
        df = self.current_out.get(df_key)
        
        if df is None or df.empty:
            self.log.append(f"The simulation found no {kind} to export.")
            return
            
        default_name = f"astrelle_{kind}_list.csv"
        path, _ = QFileDialog.getSaveFileName(self, f"Save {kind.capitalize()} List", default_name, filter="CSV Files (*.csv)")
        if not path: return
        try:
            df.to_csv(path, index=False)
            self.log.append(f"Successfully saved {kind} list to {os.path.basename(path)}")
        except Exception as e:
            self.log.append(f"ERROR: Could not save {kind} list: {e}")

    def export_full_tle_catalog(self):
        self.log.append("Fetching latest TLE catalog for export...")
        try:
            selected_group_name = self.tle_source_cb.currentText()
            group_key = TLE_SOURCES.get(selected_group_name, "active")
            
            tle_txt = download_tles(group=group_key)
            if not tle_txt:
                QMessageBox.warning(self, "TLE Export", "Could not download or find cached TLE data.")
                return

            default_name = f"celestrak_{group_key}_catalog.txt"
            path, _ = QFileDialog.getSaveFileName(self, "Save Full TLE Catalog", default_name, filter="Text Files (*.txt)")
            if not path: return

            with open(path, 'w') as f:
                f.write(tle_txt)
            self.log.append(f"Successfully saved full TLE catalog to {os.path.basename(path)}")
        except Exception as e:
            self.log.append(f"ERROR: Could not save TLE catalog: {e}")
            QMessageBox.critical(self, "Error", f"Could not save TLE catalog: {e}")
            
# Main function to launch the application.
def main_func():
    app = QApplication(sys.argv)
    w = SkySimGUI()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main_func()
