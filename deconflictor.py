# Deconflicter UI — PyQtGraph 3D + live collision (point-visibility + export)


from __future__ import annotations
import os
import random
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import numpy as np
import pyqtgraph.opengl as gl

from GaussianPoints import GaussianPoints, severity_to_rgb_hue  # type: ignore
from DronePath import DronePath  # type: ignore


# ---------------------- Helpers ----------------------

def qcolor_from_rgb01(rgb: Tuple[float, float, float]) -> QtGui.QColor:
    r = int(max(0, min(255, round(rgb[0] * 255))))
    g = int(max(0, min(255, round(rgb[1] * 255))))
    b = int(max(0, min(255, round(rgb[2] * 255))))
    return QtGui.QColor(r, g, b)


def rgb01_from_qcolor(c: QtGui.QColor) -> Tuple[float, float, float]:
    return (c.red() / 255.0, c.green() / 255.0, c.blue() / 255.0)


# ---------------------- Engine ----------------------
class EngineBundle(QtCore.QObject):
    dataLoaded = QtCore.pyqtSignal()
    collisionsComputed = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.gauss = GaussianPoints(cloud_strength=300)
        self.linear = DronePath()

        self.type_idx = 0
        self.mode = 'Gaussian'

        self.drones_path: Optional[Path] = None
        self.leader_path: Optional[Path] = None
        self.leader_id: Optional[str] = 'Leader'

        self.cloud_strength = 300
        self.spread = 0.1
        self.width = 2.0
        self.section_enabled = False
        self.distance_factor = 0.1
        self.confidence_factor = 0.8

        self.tmin = 0.0
        self.tmax = 1.0
        self.tcurrent = 0.0

        self.base_colors: Dict[str, Tuple[float, float, float]] = {}
        self.dot_enabled: Dict[str, bool] = {}
        self.eye_enabled: Dict[str, bool] = {}

        # leader visibility controls
        self.leader_visible = True
        self.leader_dots = True

        self._last_collision_rows: List[Tuple[str, float, float, float]] = []
        self.drone_count = 0

    def load_drones(self, file: Path):
        self.drones_path = file
        self.gauss.load_json_points(str(file), type=self.type_idx)
        self.linear.load_json_points(str(file))

        ids = sorted(self.linear.drone_data.keys())
        for did in ids:
            self.dot_enabled.setdefault(did, True)
            self.eye_enabled.setdefault(did, True)
            if did not in self.base_colors:
                self.base_colors[did] = (random.random()*0.8+0.1, random.random()*0.8+0.1, random.random()*0.8+0.1)

        self.drone_count = len(ids)
        if hasattr(self.linear, 'tmin') and self.linear.tmin is not None:
            self.tmin = float(self.linear.tmin)
        if hasattr(self.linear, 'tmax') and self.linear.tmax is not None:
            self.tmax = float(self.linear.tmax)
        self.tcurrent = self.tmin
        # ensure leader colors reserved
        if self.leader_path:
            self._ensure_leader_palette()
        self.dataLoaded.emit()

    def load_leader(self, file: Path):
        self.leader_path = file
        self.sp_linear = self.linear.create_drone(str(file))
        self.sp_gauss = self.gauss.create_drone(str(file), type=self.type_idx)
        self._ensure_leader_palette()
        self.tmin = min(self.tmin, float(self.sp_linear.get('tmin', self.tmin)))
        self.tmax = max(self.tmax, float(self.sp_linear.get('tmax', self.tmax)))
        self.tcurrent = self.tmin
        self.dataLoaded.emit()

    def _ensure_leader_palette(self):
        lid = self.leader_id or 'Leader'
        self.base_colors.setdefault(lid, (1.0, 1.0, 0.0))
        self.eye_enabled.setdefault(lid, True)
        self.dot_enabled.setdefault(lid, True)

    def compute_collisions(self) -> List[Tuple[str, float, float, float]]:
        rows: List[Tuple[str, float, float, float]] = []
        if self.mode == 'Linear':
            if not hasattr(self, 'sp_linear'):
                return rows
            inter = self.linear.check_spline_with_drones(self.sp_linear, dist=float(self.distance_factor), detailed=True, refine=True)
            if inter and isinstance(inter, dict):
                for did, recs in inter.items():
                    for r in recs:
                        sev = float(max(0.0, min(1.0, r.get('severity', 0.0))))
                        rows.append((did, float(r.get('t_rep', 0.0)), sev, float(r.get('d_min', 0.0))))
        else:
            if not hasattr(self, 'sp_gauss'):
                return rows
            inter = self.gauss.check_spline_with_drones(self.sp_gauss, confidence_level=float(self.confidence_factor),
                                                        plot_tmin=self.tmin, plot_tmax=self.tmax, width=(self.width if self.section_enabled else 0.0),
                                                        detailed=True, refine=True)
            if inter and isinstance(inter, dict):
                for did, recs in inter.items():
                    for r in recs:
                        sev = float(max(0.0, min(1.0, r.get('severity', 0.0))))
                        rows.append((did, float(r.get('t_rep', 0.0)), sev, float(r.get('m2_min', 0.0))))
        self._last_collision_rows = sorted(rows, key=lambda x: x[1])
        self.collisionsComputed.emit(self._last_collision_rows)
        return self._last_collision_rows


# ---------------------- UI Widgets ----------------------
class LoadingScreen(QtWidgets.QWidget):
    proceed = QtCore.pyqtSignal()

    def __init__(self, engine: EngineBundle):
        super().__init__(); self.engine = engine
        self.setAutoFillBackground(True)
        self.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0b0011, stop:1 #220022);")

        main = QtWidgets.QVBoxLayout(self); main.setContentsMargins(60,60,60,60); main.setSpacing(18); main.setAlignment(Qt.AlignCenter)

        title = QtWidgets.QLabel('Deconflicter'); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('color:#f8f6ff; font-size:56px; font-weight:900; letter-spacing:2px;')
        main.addWidget(title, alignment=Qt.AlignHCenter)

        card = QtWidgets.QFrame(); card.setFrameShape(QtWidgets.QFrame.StyledPanel)
        card.setStyleSheet('background: #141417; border-radius:12px; padding:22px;')
        card_layout = QtWidgets.QGridLayout(card); card_layout.setColumnStretch(0,0); card_layout.setColumnStretch(1,1); card_layout.setHorizontalSpacing(18); card_layout.setVerticalSpacing(14)

        drones_label = QtWidgets.QLabel('Upload your drones dataset here'); drones_label.setStyleSheet('color:#cfcfcf; font-size:13px;')
        drones_btn = QtWidgets.QPushButton('Load drones'); drones_btn.setCursor(Qt.PointingHandCursor); drones_btn.clicked.connect(self._pick_drones)
        self.drones_status = QtWidgets.QLabel('0 Drones loaded'); self.drones_status.setStyleSheet('color:#6cff9d; font-weight:600')

        leader_label = QtWidgets.QLabel('Upload your leader dataset here'); leader_label.setStyleSheet('color:#cfcfcf; font-size:13px;')
        leader_btn = QtWidgets.QPushButton('Load leader'); leader_btn.setCursor(Qt.PointingHandCursor); leader_btn.clicked.connect(self._pick_leader)
        self.leader_status = QtWidgets.QLabel('Leader file not loaded'); self.leader_status.setStyleSheet('color:#6cff9d; font-weight:600')

        type_label = QtWidgets.QLabel('Select the type'); type_label.setStyleSheet('color:#cfcfcf; font-size:13px;')
        self.type_cb = QtWidgets.QComboBox(); self.type_cb.addItems(['Type 0','Type 1']); self.type_cb.currentIndexChanged.connect(self._type_changed)
        self.type_desc = QtWidgets.QLabel('Type 0 - Only position dataset, Type 1 - Position and velocity dataset'); self.type_desc.setStyleSheet('color:#9f9f9f;'); self.type_desc.setAlignment(Qt.AlignLeft)

        card_layout.addWidget(drones_label, 0, 0); card_layout.addWidget(drones_btn, 0, 1); card_layout.addWidget(self.drones_status, 0, 2)
        card_layout.addWidget(leader_label, 1, 0); card_layout.addWidget(leader_btn, 1, 1); card_layout.addWidget(self.leader_status, 1, 2)
        card_layout.addWidget(type_label, 2, 0); card_layout.addWidget(self.type_cb, 2, 1); card_layout.addWidget(self.type_desc, 3, 0, 1, 3)

        main.addWidget(card, alignment=Qt.AlignCenter)

        self.go_btn = QtWidgets.QPushButton('Continue')
        self.go_btn.setCursor(Qt.PointingHandCursor)
        self.go_btn.setMinimumHeight(72)
        self.go_btn.setMinimumWidth(320)
        self.go_btn.setStyleSheet("QPushButton{background:#3a2a6a;color:#fff;border-radius:12px;font-weight:900;font-size:20px;padding:14px 28px;} QPushButton:pressed{background:#52398a;} QPushButton:disabled{background:#0f0f10;color:#777;}")
        self.go_btn.setEnabled(False)
        self.go_btn.clicked.connect(self.proceed.emit)
        main.addWidget(self.go_btn, alignment=Qt.AlignHCenter)
        main.addSpacing(8)

        self.engine.dataLoaded.connect(self._on_engine_loaded)

    def _type_changed(self, idx: int):
        self.engine.type_idx = idx

    def _pick_drones(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Pick drones JSON…', '', 'JSON (*.json)')
        if path:
            try: self.engine.load_drones(Path(path))
            except Exception as e: QtWidgets.QMessageBox.critical(self, 'Load error', str(e))

    def _pick_leader(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Pick leader JSON…', '', 'JSON (*.json)')
        if path:
            try: self.engine.load_leader(Path(path))
            except Exception as e: QtWidgets.QMessageBox.critical(self, 'Load error', str(e))

    def _on_engine_loaded(self):
        n = self.engine.drone_count
        self.drones_status.setText(f'{n} Drones loaded')
        if self.engine.leader_path: self.leader_status.setText('Leader file loaded')
        self.go_btn.setEnabled(self.engine.drone_count > 0 and self.engine.leader_path is not None)


class OutlinerItem(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, drone_id: str, base_rgb: Tuple[float, float, float], eye=True, dots=True, name_override: Optional[str]=None):
        super().__init__(); self.drone_id = drone_id
        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(6,6,6,6); lay.setSpacing(8)
        self.eye_btn = QtWidgets.QToolButton(); self.eye_btn.setCheckable(True); self.eye_btn.setChecked(eye); self.eye_btn.setText('👁')
        # make pressed/checked look different
        self.eye_btn.setStyleSheet("QToolButton{color:#bdbdbd;background:transparent;border:none;} QToolButton:checked{color:#ffffff;background:#2b2b2b;border-radius:6px;padding:2px;}")
        display_name = name_override if name_override else drone_id
        self.name = QtWidgets.QLabel(display_name); self.name.setStyleSheet('color:#e6e6e6')
        self.dot_btn = QtWidgets.QToolButton(); self.dot_btn.setCheckable(True); self.dot_btn.setChecked(dots); self.dot_btn.setText('•')
        self.color_btn = QtWidgets.QPushButton(); self.color_btn.setFixedSize(28,18); self._qcolor = qcolor_from_rgb01(base_rgb); self._apply_btn_color(); self.color_btn.clicked.connect(self._pick_color)
        lay.addWidget(self.eye_btn); lay.addWidget(self.name, 1); lay.addWidget(self.dot_btn); lay.addWidget(self.color_btn)
        self.eye_btn.toggled.connect(self.changed.emit); self.dot_btn.toggled.connect(self.changed.emit)

    def _apply_btn_color(self): self.color_btn.setStyleSheet(f'background: {self._qcolor.name()}; border: 1px solid #222;')

    def _pick_color(self):
        c = QtWidgets.QColorDialog.getColor(self._qcolor, self, 'Pick color (HSV)')
        if c.isValid():
            self._qcolor = c; self._apply_btn_color(); self.changed.emit()

    def get_state(self): return (self.eye_btn.isChecked(), self.dot_btn.isChecked(), rgb01_from_qcolor(self._qcolor))


class CollisionsTable(QtWidgets.QTableWidget):
    def __init__(self):
        super().__init__(0, 4)
        self.setHorizontalHeaderLabels(['Drone','Time','Severity','Distance/Metric'])
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

    def populate(self, rows: List[Tuple[str, float, float, float]]):
        self.setRowCount(0)
        for (did, t_rep, sev, metric) in rows:
            r = self.rowCount(); self.insertRow(r)
            self.setItem(r,0,QtWidgets.QTableWidgetItem(did)); self.setItem(r,1,QtWidgets.QTableWidgetItem(f'{t_rep:.3f}'))
            self.setItem(r,2,QtWidgets.QTableWidgetItem(f'{sev:.3f}' if sev is not None else 'None'))
            self.setItem(r,3,QtWidgets.QTableWidgetItem(f'{metric:.4f}'))


# ---------------------- Main Window ----------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: EngineBundle):
        super().__init__(); self.engine = engine; self.setWindowTitle('Deconflicter'); self.resize(1400,900)

        self.stack = QtWidgets.QStackedWidget(); self.loading = LoadingScreen(engine); self.app = self._build_app_ui(); self.stack.addWidget(self.loading); self.stack.addWidget(self.app)
        self.setCentralWidget(self.stack)
        self.loading.proceed.connect(self._animate_to_app)
        self.engine.dataLoaded.connect(self._refresh_outliner)
        self.engine.collisionsComputed.connect(self._refresh_collisions_table)

    def _build_app_ui(self):
        w = QtWidgets.QWidget(); root_l = QtWidgets.QVBoxLayout(w); root_l.setContentsMargins(10,10,10,10); root_l.setSpacing(8)

        toolbar = QtWidgets.QHBoxLayout()
        self.back_btn = QtWidgets.QPushButton('Back')
        self.back_btn.setMinimumWidth(110); self.back_btn.setMinimumHeight(36)
        self.back_btn.setStyleSheet("QPushButton{background:#2b2b38;color:#fff;border-radius:8px;font-weight:700;padding:6px 10px;} QPushButton:pressed{background:#3b3b46;}")
        self.back_btn.clicked.connect(self._go_back_home)
        self.mode_cb = QtWidgets.QComboBox(); self.mode_cb.addItems(['Gaussian','Linear']); self.mode_cb.setCurrentText(self.engine.mode); self.mode_cb.currentTextChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self.back_btn); toolbar.addWidget(self.mode_cb); toolbar.addStretch(1)
        self.check_btn = QtWidgets.QPushButton('Check collisions'); self.check_btn.setMinimumHeight(36); self.check_btn.clicked.connect(self._check_collisions_clicked)
        toolbar.addWidget(self.check_btn)
        root_l.addLayout(toolbar)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.view = gl.GLViewWidget(); self.view.opts['distance']=40; self.view.setBackgroundColor((10,10,12)); grid = gl.GLGridItem(); grid.setSize(40,40); grid.setSpacing(1,1); self.view.addItem(grid)
        left_container = QtWidgets.QWidget(); left_l = QtWidgets.QVBoxLayout(left_container); left_l.setContentsMargins(0,0,0,0); left_l.addWidget(self.view)

        right = QtWidgets.QWidget(); right_l = QtWidgets.QVBoxLayout(right); right_l.setContentsMargins(6,6,6,6); right_l.setSpacing(8)
        out_frame = QtWidgets.QFrame(); out_frame.setFrameShape(QtWidgets.QFrame.StyledPanel); out_layout = QtWidgets.QVBoxLayout(out_frame); out_layout.setContentsMargins(6,6,6,6); out_title = QtWidgets.QLabel('Outliner'); out_title.setStyleSheet('color:#e6e6e6; font-weight:700'); out_layout.addWidget(out_title)
        self.outliner_scroll = QtWidgets.QScrollArea(); self.outliner_scroll.setWidgetResizable(True); self.outliner_body = QtWidgets.QWidget(); self.outliner_v = QtWidgets.QVBoxLayout(self.outliner_body); self.outliner_v.addStretch(1); self.outliner_scroll.setWidget(self.outliner_body); out_layout.addWidget(self.outliner_scroll)

        par_frame = QtWidgets.QFrame(); par_frame.setFrameShape(QtWidgets.QFrame.StyledPanel); par_layout = QtWidgets.QFormLayout(par_frame); par_layout.setLabelAlignment(Qt.AlignLeft)
        self.cloud_strength_sb = QtWidgets.QSpinBox(); self.cloud_strength_sb.setRange(0,20000); self.cloud_strength_sb.setValue(self.engine.cloud_strength)
        self.spread_dsb = QtWidgets.QDoubleSpinBox(); self.spread_dsb.setRange(0.0,5.0); self.spread_dsb.setDecimals(3); self.spread_dsb.setValue(self.engine.spread)
        self.section_chk = QtWidgets.QCheckBox('Section view'); self.section_chk.toggled.connect(self._on_params_changed)
        self.width_dsb = QtWidgets.QDoubleSpinBox(); self.width_dsb.setRange(0.0,60.0); self.width_dsb.setDecimals(3); self.width_dsb.setValue(self.engine.width)
        self.cloud_strength_sb.valueChanged.connect(self._on_params_changed); self.spread_dsb.valueChanged.connect(self._on_params_changed); self.width_dsb.valueChanged.connect(self._on_params_changed)
        self.dist_dsb = QtWidgets.QDoubleSpinBox(); self.dist_dsb.setRange(0.0,100.0); self.dist_dsb.setDecimals(4); self.dist_dsb.setValue(self.engine.distance_factor)
        self.conf_dsb = QtWidgets.QDoubleSpinBox(); self.conf_dsb.setRange(0.0,0.999); self.conf_dsb.setDecimals(3); self.conf_dsb.setValue(self.engine.confidence_factor)
        self.dist_dsb.valueChanged.connect(self._on_params_changed); self.conf_dsb.valueChanged.connect(self._on_params_changed)

        par_layout.addRow('Cloud strength', self.cloud_strength_sb); par_layout.addRow('Spread', self.spread_dsb); par_layout.addRow(self.section_chk, self.width_dsb)
        par_layout.addRow('Distance factor (Linear)', self.dist_dsb); par_layout.addRow('Confidence factor (Gaussian)', self.conf_dsb)

        coll_label = QtWidgets.QLabel('Collisions'); coll_label.setStyleSheet('color:#e6e6e6; font-weight:700')
        self.col_table = CollisionsTable()

        right_l.addWidget(out_frame, 3); right_l.addWidget(par_frame, 1); right_l.addWidget(coll_label); right_l.addWidget(self.col_table, 2)

        splitter.addWidget(left_container); splitter.addWidget(right)
        splitter.setStretchFactor(0,3); splitter.setStretchFactor(1,2)
        root_l.addWidget(splitter, 1)

        tl = QtWidgets.QHBoxLayout(); tl.setContentsMargins(0,0,0,0)
        self.tmin_sb = QtWidgets.QDoubleSpinBox(); self.tmin_sb.setDecimals(3); self.tmin_sb.setRange(-1e9,1e9)
        self.tmax_sb = QtWidgets.QDoubleSpinBox(); self.tmax_sb.setDecimals(3); self.tmax_sb.setRange(-1e9,1e9)
        self.tcurrent_sb = QtWidgets.QDoubleSpinBox(); self.tcurrent_sb.setDecimals(3); self.tcurrent_sb.setRange(-1e9,1e9)
        self.tmin_sb.valueChanged.connect(self._time_bounds_changed); self.tmax_sb.valueChanged.connect(self._time_bounds_changed); self.tcurrent_sb.valueChanged.connect(self._tcurrent_edited)
        self.timeline = QtWidgets.QSlider(Qt.Horizontal); self.timeline.setMinimum(0); self.timeline.setMaximum(1000); self.timeline.valueChanged.connect(self._timeline_changed)
        tl.addWidget(QtWidgets.QLabel('tmin')); tl.addWidget(self.tmin_sb); tl.addWidget(self.timeline, 1); tl.addWidget(QtWidgets.QLabel('t')); tl.addWidget(self.tcurrent_sb); tl.addWidget(QtWidgets.QLabel('tmax')); tl.addWidget(self.tmax_sb)
        root_l.addLayout(tl)

        # export button at bottom-right
        br = QtWidgets.QHBoxLayout(); br.addStretch(1)
        self.export_btn = QtWidgets.QPushButton('Export collisions')
        self.export_btn.setMinimumWidth(160); self.export_btn.setMinimumHeight(36)
        self.export_btn.setStyleSheet("QPushButton{background:#1f1f2a;color:#fff;border-radius:8px;padding:6px 10px;} QPushButton:pressed{background:#2b2b38;}" )
        self.export_btn.clicked.connect(self._export_collisions_clicked)
        br.addWidget(self.export_btn)
        root_l.addLayout(br)

        # plot containers
        self.splines: Dict[str, gl.GLLinePlotItem] = {}; self.waypoints: Dict[str, gl.GLScatterPlotItem] = {}
        self.clouds: Dict[str, gl.GLScatterPlotItem] = {}; self.moving_dots: Dict[str, gl.GLScatterPlotItem] = {}
        self.leader_cloud: Optional[gl.GLScatterPlotItem] = None; self.leader_spline: Optional[gl.GLLinePlotItem] = None
        self.leader_points: Optional[gl.GLScatterPlotItem] = None; self.leader_marker: Optional[gl.GLScatterPlotItem] = None

        self._splitter = splitter
        QtCore.QTimer.singleShot(150, self._apply_default_splitter_sizes)

        return w

    def _apply_default_splitter_sizes(self):
        total = self._splitter.size().width()
        left_w = int(total * 0.60)
        right_w = max(200, total - left_w)
        self._splitter.setSizes([left_w, right_w])

    def _animate_to_app(self):
        self.stack.setCurrentIndex(1)
        QtCore.QTimer.singleShot(120, self._refresh_outliner)

    def _go_back_home(self):
        self.stack.setCurrentIndex(0)

    def _refresh_outliner(self):
        for i in reversed(range(self.outliner_v.count()-1)):
            item = self.outliner_v.itemAt(i).widget()
            if item: item.setParent(None)
        # Add leader first if present
        if self.engine.leader_path:
            lid = self.engine.leader_id or 'Leader'
            base = self.engine.base_colors.get(lid, (1.0,1.0,0.0))
            oi = OutlinerItem(lid, base, eye=self.engine.leader_visible, dots=self.engine.leader_dots, name_override='Leader')
            oi.changed.connect(self._outliner_item_changed); self.outliner_v.insertWidget(self.outliner_v.count()-1, oi)
        # then normal drones
        ids = sorted(self.engine.linear.drone_data.keys())
        for did in ids:
            base = self.engine.base_colors.get(did, (random.random(),random.random(),random.random()))
            oi = OutlinerItem(did, base, eye=self.engine.eye_enabled.get(did, True), dots=self.engine.dot_enabled.get(did, True))
            oi.changed.connect(self._outliner_item_changed); self.outliner_v.insertWidget(self.outliner_v.count()-1, oi)
        self._set_time_bounds(self.engine.tmin, self.engine.tmax)
        self._run_sim()

    def _outliner_item_changed(self):
        idx = 0
        if self.engine.leader_path:
            w = self.outliner_v.itemAt(idx).widget()
            if isinstance(w, OutlinerItem):
                eye, dot, rgb = w.get_state(); self.engine.leader_visible = bool(eye); self.engine.leader_dots = bool(dot); self.engine.base_colors[w.drone_id] = rgb
            idx += 1
        for i in range(idx, self.outliner_v.count()-1):
            w = self.outliner_v.itemAt(i).widget()
            if isinstance(w, OutlinerItem):
                eye, dot, rgb = w.get_state(); self.engine.eye_enabled[w.drone_id] = eye; self.engine.dot_enabled[w.drone_id] = dot; self.engine.base_colors[w.drone_id] = rgb
        # apply changes to existing view immediately
        self._apply_visibility_changes()
        self._run_sim()

    def _apply_visibility_changes(self):
        # drones
        for did, line in list(self.splines.items()):
            visible = bool(self.engine.eye_enabled.get(did, True))
            try:
                line.setVisible(visible)
            except Exception:
                if not visible:
                    try: self.view.removeItem(line)
                    except: pass
        for did, wp in list(self.waypoints.items()):
            # waypoints visible only if eye is on AND dot is on
            visible = bool(self.engine.eye_enabled.get(did, True)) and bool(self.engine.dot_enabled.get(did, True))
            try:
                wp.setVisible(visible)
                if not visible:
                    # clear data to avoid lingering points
                    wp.setData(pos=np.zeros((1,3)))
            except Exception:
                if not visible:
                    try: self.view.removeItem(wp)
                    except: pass
        for did, cloud in list(self.clouds.items()):
            visible = bool(self.engine.eye_enabled.get(did, True))
            try:
                cloud.setVisible(visible)
                if not visible:
                    cloud.setData(pos=np.zeros((1,3)))
            except Exception:
                if not visible:
                    try: self.view.removeItem(cloud)
                    except: pass
        for did, md in list(self.moving_dots.items()):
            visible = bool(self.engine.eye_enabled.get(did, True))
            try:
                md.setVisible(visible)
            except Exception:
                if not visible:
                    try: self.view.removeItem(md)
                    except: pass
        # leader
        if self.leader_spline:
            try: self.leader_spline.setVisible(bool(self.engine.leader_visible))
            except Exception:
                if not self.engine.leader_visible:
                    try: self.view.removeItem(self.leader_spline)
                    except: pass
        if self.leader_points:
            try:
                self.leader_points.setVisible(bool(self.engine.leader_visible and self.engine.leader_dots))
                if not (self.engine.leader_visible and self.engine.leader_dots):
                    self.leader_points.setData(pos=np.zeros((1,3)))
            except Exception:
                if not (self.engine.leader_visible and self.engine.leader_dots):
                    try: self.view.removeItem(self.leader_points)
                    except: pass
        if self.leader_cloud:
            try: self.leader_cloud.setVisible(bool(self.engine.leader_visible))
            except Exception:
                if not self.engine.leader_visible:
                    try: self.view.removeItem(self.leader_cloud)
                    except: pass
        if self.leader_marker:
            try: self.leader_marker.setVisible(bool(self.engine.leader_visible))
            except Exception:
                if not self.engine.leader_visible:
                    try: self.view.removeItem(self.leader_marker)
                    except: pass

    def _check_collisions_clicked(self):
        rows = self.engine.compute_collisions()
        if not rows:
            QtWidgets.QMessageBox.information(self, 'Clear', 'The drone is clear to move.')
        else:
            QtWidgets.QMessageBox.warning(self, 'Collision', 'Collision detected! See table for details.')
        self._refresh_collisions_table(rows)

    def _on_mode_changed(self, text: str):
        self.engine.mode = text
        is_linear = (text == 'Linear')
        self.cloud_strength_sb.setEnabled(not is_linear)
        self.spread_dsb.setEnabled(not is_linear)
        self.conf_dsb.setEnabled(not is_linear)
        self.dist_dsb.setEnabled(is_linear)
        self._run_sim()

    def _on_params_changed(self):
        self.engine.cloud_strength = int(self.cloud_strength_sb.value())
        self.engine.spread = float(self.spread_dsb.value())
        self.engine.section_enabled = bool(self.section_chk.isChecked())
        self.engine.width = float(self.width_dsb.value())
        self.engine.distance_factor = float(self.dist_dsb.value())
        self.engine.confidence_factor = float(self.conf_dsb.value())
        self._run_sim()

    def _clear_view(self):
        for item in list(self.splines.values()): self.view.removeItem(item)
        for item in list(self.waypoints.values()): self.view.removeItem(item)
        for item in list(self.clouds.values()): self.view.removeItem(item)
        for item in list(self.moving_dots.values()): self.view.removeItem(item)
        if self.leader_spline: self.view.removeItem(self.leader_spline); self.leader_spline=None
        if self.leader_points: self.view.removeItem(self.leader_points); self.leader_points=None
        if self.leader_cloud: self.view.removeItem(self.leader_cloud); self.leader_cloud=None
        if self.leader_marker: self.view.removeItem(self.leader_marker); self.leader_marker=None
        self.splines.clear(); self.waypoints.clear(); self.clouds.clear(); self.moving_dots.clear()

    def _run_sim(self):
        self._clear_view()
        if not self.engine.linear.drone_data and not self.engine.leader_path:
            return
        ids = sorted(self.engine.linear.drone_data.keys())
        tmin = self.engine.tmin; tmax = self.engine.tmax; samples = 400

        # draw other drones
        for did in ids:
            if not self.engine.eye_enabled.get(did, True):
                # ensure any previous visuals are removed/hidden
                if did in self.splines:
                    try: self.view.removeItem(self.splines[did])
                    except: pass
                if did in self.waypoints:
                    try: self.view.removeItem(self.waypoints[did])
                    except: pass
                if did in self.clouds:
                    try: self.view.removeItem(self.clouds[did])
                    except: pass
                if did in self.moving_dots:
                    try: self.view.removeItem(self.moving_dots[did])
                    except: pass
                continue
            entry = self.engine.linear.drone_data[did]
            fx, fy, fz = entry['fx'], entry['fy'], entry['fz']
            if not self.engine.section_enabled or self.engine.width==0.0:
                t_vals = np.linspace(tmin, tmax, samples)
            else:
                center = self.engine.tcurrent; t_vals = np.linspace(max(tmin, center-0.5*self.engine.width), min(tmax, center+0.5*self.engine.width), samples)
            x = np.atleast_1d(fx(t_vals)); y = np.atleast_1d(fy(t_vals)); z = np.atleast_1d(fz(t_vals))
            color = self.engine.base_colors.get(did, (0.6,0.6,0.6))
            line = gl.GLLinePlotItem(pos=np.vstack([x,y,z]).T, color=(*color,0.9), width=2.0, antialias=True)
            self.view.addItem(line); self.splines[did]=line

            if self.engine.dot_enabled.get(did, True) and self.engine.eye_enabled.get(did, True):
                wps = entry['waypoints']; px=np.array([p[1] for p in wps]); py=np.array([p[2] for p in wps]); pz=np.array([p[3] for p in wps])
                sc = gl.GLScatterPlotItem(pos=np.vstack([px,py,pz]).T, size=6.0, color=(*tuple(np.array(color)*0.6),1.0))
                self.view.addItem(sc); self.waypoints[did]=sc

            sc_empty = gl.GLScatterPlotItem(pos=np.zeros((1,3)), size=4.0, color=(*color,0.35))
            self.view.addItem(sc_empty); self.clouds[did]=sc_empty
            self._add_or_update_moving_dot(did)

        # draw leader spline/points/cloud only if visible
        if self.engine.leader_path and self.engine.leader_visible and hasattr(self.engine, 'sp_linear') and hasattr(self.engine, 'sp_gauss'):
            lid = self.engine.leader_id or 'Leader'
            sp = self.engine.sp_gauss if self.engine.mode=='Gaussian' else self.engine.sp_linear
            if not self.engine.section_enabled or self.engine.width == 0.0:
                t_vals = np.linspace(tmin, tmax, samples)
            else:
                center = self.engine.tcurrent; t_vals = np.linspace(max(tmin, center-0.5*self.engine.width), min(tmax, center+0.5*self.engine.width), samples)
            fx,fy,fz = sp['fx'], sp['fy'], sp['fz']
            x = np.atleast_1d(fx(t_vals)); y = np.atleast_1d(fy(t_vals)); z = np.atleast_1d(fz(t_vals))
            leader_color = self.engine.base_colors.get(lid, (1.0,1.0,0.0))
            self.leader_spline = gl.GLLinePlotItem(pos=np.vstack([x,y,z]).T, color=(*leader_color,0.95), width=3.0, antialias=True)
            self.view.addItem(self.leader_spline)
            if 'waypoints' in sp and self.engine.leader_dots and self.engine.leader_visible:
                wps = sp['waypoints']
                px = np.array([p[0] for p in wps]); py = np.array([p[1] for p in wps]); pz = np.array([p[2] for p in wps])
                self.leader_points = gl.GLScatterPlotItem(pos=np.vstack([px,py,pz]).T, size=7.0, color=(0.9,0.85,0.0,0.95))
                self.view.addItem(self.leader_points)
            self.leader_cloud = gl.GLScatterPlotItem(pos=np.zeros((1,3)), size=5.0, color=(1,1,0,0.8)); self.view.addItem(self.leader_cloud)
            self._add_or_update_leader_marker(sp)

        self._set_time_bounds(self.engine.tmin, self.engine.tmax)
        for did in ids: self._update_cloud_for_drone(did)
        self._update_leader_collision_cloud()
        self.engine.compute_collisions()

    def _add_or_update_moving_dot(self, did: str):
        if did not in self.moving_dots:
            sc = gl.GLScatterPlotItem(pos=np.zeros((1,3)), size=8.0, color=(1,1,1,1)); self.view.addItem(sc); self.moving_dots[did]=sc
        entry=self.engine.linear.drone_data[did]; fx,fy,fz=entry['fx'],entry['fy'],entry['fz']; t=self.engine.tcurrent
        try: p=np.array([[float(fx(t)), float(fy(t)), float(fz(t))]]); self.moving_dots[did].setData(pos=p)
        except Exception: pass

    def _add_or_update_leader_marker(self, sp_data: Dict[str,Any]):
        if self.leader_marker is None:
            sc = gl.GLScatterPlotItem(pos=np.zeros((1,3)), size=10.0, color=(1,1,1,1)); self.view.addItem(sc); self.leader_marker=sc
        t=self.engine.tcurrent; fx,fy,fz = sp_data['fx'], sp_data['fy'], sp_data['fz']
        try: p=np.array([[float(fx(t)), float(fy(t)), float(fz(t))]]); self.leader_marker.setData(pos=p)
        except Exception: pass

    def _update_cloud_for_drone(self, did: str):
        # hide clouds if drone visibility is off
        if not self.engine.eye_enabled.get(did, True):
            if did in self.clouds:
                try: self.clouds[did].setData(pos=np.zeros((1,3))); self.clouds[did].setVisible(False)
                except Exception:
                    try: self.view.removeItem(self.clouds[did])
                    except: pass
            return

        if self.engine.mode != 'Gaussian':
            if did in self.clouds:
                try: self.clouds[did].setData(pos=np.zeros((1,3))); self.clouds[did].setVisible(False)
                except: pass
            return
        if did not in self.engine.linear.drone_data: return
        cloud = self.engine.gauss.get_cloud_samples(did, self.engine.tcurrent, self.engine.width if self.engine.section_enabled else 0.0,
                                                   spread=self.engine.spread, cloud_strength=self.engine.cloud_strength,
                                                   plot_tmin=self.engine.tmin, plot_tmax=self.engine.tmax)
        if cloud is None or len(cloud)==0:
            if did in self.clouds:
                try: self.clouds[did].setData(pos=np.zeros((1,3)))
                except: pass
            return
        color = self.engine.base_colors.get(did, (0.6,0.6,0.6))
        dot_rgb = tuple(min(1.0, c*1.3) for c in color)
        n = len(cloud)
        colors = np.tile(np.array([dot_rgb[0],dot_rgb[1],dot_rgb[2],0.55], dtype=float), (n,1))
        if did in self.clouds and self.clouds[did] is not None:
            try: self.clouds[did].setData(pos=cloud[:, :3], color=colors); self.clouds[did].setVisible(True)
            except Exception:
                try: self.clouds[did].setData(pos=cloud[:, :3])
                except: pass
        else:
            sc = gl.GLScatterPlotItem(pos=cloud[:, :3], size=4.0, color=colors); self.view.addItem(sc); self.clouds[did]=sc

    def _update_leader_collision_cloud(self):
        # leader cloud follows leader and uses gradient according to nearby collisions
        if not self.engine.leader_visible:
            if self.leader_cloud:
                try: self.leader_cloud.setData(pos=np.zeros((1,3))); self.leader_cloud.setVisible(False)
                except: pass
            return
        if self.engine.mode != 'Gaussian' or not hasattr(self.engine,'sp_gauss'):
            if self.leader_cloud:
                try: self.leader_cloud.setData(pos=np.zeros((1,3))); self.leader_cloud.setVisible(False)
                except: pass
            return
        sp = self.engine.sp_gauss
        t0 = float(self.engine.tcurrent)
        width = self.engine.width if self.engine.section_enabled else (self.engine.tmax - self.engine.tmin)
        if width <= 0:
            t_start = float(self.engine.tmin); t_end = float(self.engine.tmax)
        else:
            t_start = max(self.engine.tmin, t0 - 0.5*width); t_end = min(self.engine.tmax, t0 + 0.5*width)
        if t_end <= t_start:
            if self.leader_cloud:
                try: self.leader_cloud.setData(pos=np.zeros((1,3)))
                except: pass
            return

        total_slices = max(12, int(min(60, self.engine.cloud_strength // 10)))
        times = np.linspace(t_start, t_end, total_slices)
        slice_width = max( (t_end - t_start) / float(total_slices), 1e-6 )

        rows = self.engine._last_collision_rows
        all_pos = []
        all_cols = []
        for ti in times:
            try:
                driver_cloud = self.engine.gauss.get_driver_cloud_samples(sp, float(ti), slice_width,
                                                                          spread=self.engine.spread,
                                                                          cloud_strength=max(6, int(self.engine.cloud_strength // total_slices)),
                                                                          plot_tmin=self.engine.tmin, plot_tmax=self.engine.tmax)
            except Exception:
                driver_cloud = None
            if driver_cloud is None or len(driver_cloud)==0:
                continue
            sev = 0.0
            if rows:
                times_rows = np.array([r[1] for r in rows])
                idx = int(np.argmin(np.abs(times_rows - ti)))
                sev = float(rows[idx][2]) if len(rows)>0 else 0.0
            col = np.array(severity_to_rgb_hue(sev))
            alpha = 0.35 + 0.6 * float(sev)
            cols = np.tile(np.array([col[0],col[1],col[2],alpha]), (driver_cloud.shape[0],1))
            all_pos.append(driver_cloud[:, :3]); all_cols.append(cols)

        if all_pos:
            pos = np.vstack(all_pos); colors = np.vstack(all_cols)
            if self.leader_cloud:
                try: self.leader_cloud.setData(pos=pos, color=colors); self.leader_cloud.setVisible(True)
                except Exception: pass
            else:
                self.leader_cloud = gl.GLScatterPlotItem(pos=pos, size=5.0, color=colors); self.view.addItem(self.leader_cloud)
        else:
            sp_samples = self.engine.gauss.get_driver_cloud_samples(sp, t0, slice_width,
                                                                    spread=self.engine.spread, cloud_strength=max(20, int(self.engine.cloud_strength//4)),
                                                                    plot_tmin=self.engine.tmin, plot_tmax=self.engine.tmax)
            if sp_samples is None or len(sp_samples)==0:
                if self.leader_cloud:
                    try: self.leader_cloud.setData(pos=np.zeros((1,3)))
                    except: pass
                return
            cols = np.tile(np.array([0.9,0.8,0.0,0.6]), (sp_samples.shape[0],1))
            if self.leader_cloud:
                try: self.leader_cloud.setData(pos=sp_samples[:, :3], color=cols); self.leader_cloud.setVisible(True)
                except: pass
            else:
                self.leader_cloud = gl.GLScatterPlotItem(pos=sp_samples[:, :3], size=5.0, color=cols); self.view.addItem(self.leader_cloud)

    def _set_time_bounds(self, tmin: float, tmax: float):
        self.tmin_sb.blockSignals(True); self.tmax_sb.blockSignals(True); self.tcurrent_sb.blockSignals(True)
        self.tmin_sb.setValue(tmin); self.tmax_sb.setValue(tmax); self.tcurrent_sb.setValue(self.engine.tcurrent)
        self.tmin_sb.blockSignals(False); self.tmax_sb.blockSignals(False); self.tcurrent_sb.blockSignals(False)
        def to_slider(t):
            if tmax==tmin: return 0
            return int(((t - tmin) / (tmax - tmin)) * self.timeline.maximum())
        self.timeline.blockSignals(True); self.timeline.setValue(to_slider(self.engine.tcurrent)); self.timeline.blockSignals(False)

    def _time_bounds_changed(self):
        tmin = float(self.tmin_sb.value()); tmax=float(self.tmax_sb.value());
        if tmax<=tmin: return
        self.engine.tmin=tmin; self.engine.tmax=tmax; self._run_sim()

    def _tcurrent_edited(self, v: float):
        self.engine.tcurrent = float(v)
        self.timeline.blockSignals(True); self.timeline.setValue(self._slider_value_from_time(self.engine.tcurrent)); self.timeline.blockSignals(False)
        for did in list(self.moving_dots.keys()): self._add_or_update_moving_dot(did)
        for did in list(self.clouds.keys()): self._update_cloud_for_drone(did)
        if self.engine.section_enabled and self.engine.width > 0.0:
            tmin = self.engine.tmin; tmax = self.engine.tmax; samples = 400
            center = self.engine.tcurrent
            t_vals = np.linspace(max(tmin, center - 0.5 * self.engine.width), min(tmax, center + 0.5 * self.engine.width), samples)
            for did, line in list(self.splines.items()):
                entry = self.engine.linear.drone_data.get(did)
                if not entry: continue
                fx, fy, fz = entry['fx'], entry['fy'], entry['fz']
                try:
                    x = np.atleast_1d(fx(t_vals)); y = np.atleast_1d(fy(t_vals)); z = np.atleast_1d(fz(t_vals))
                    line.setData(pos=np.vstack([x, y, z]).T)
                except Exception: pass
            if self.engine.leader_path and self.leader_spline is not None and self.engine.leader_visible:
                sp = self.engine.sp_gauss if self.engine.mode=='Gaussian' else self.engine.sp_linear
                try:
                    x = np.atleast_1d(sp['fx'](t_vals)); y = np.atleast_1d(sp['fy'](t_vals)); z = np.atleast_1d(sp['fz'](t_vals))
                    self.leader_spline.setData(pos=np.vstack([x, y, z]).T)
                except Exception: pass
        if hasattr(self.engine, 'sp_gauss') or hasattr(self.engine, 'sp_linear'):
            sp = self.engine.sp_gauss if self.engine.mode=='Gaussian' and hasattr(self.engine,'sp_gauss') else (self.engine.sp_linear if hasattr(self.engine,'sp_linear') else None)
            if sp: self._add_or_update_leader_marker(sp)
        self._update_leader_collision_cloud(); self.engine.compute_collisions()

    def _slider_value_from_time(self, t: float) -> int:
        tmin = self.engine.tmin; tmax = self.engine.tmax
        if tmax==tmin: return 0
        return int(((t - tmin) / (tmax - tmin)) * self.timeline.maximum())

    def _timeline_changed(self, v: int):
        self.engine.tcurrent = self._slider_to_time(v)
        self.tcurrent_sb.blockSignals(True); self.tcurrent_sb.setValue(self.engine.tcurrent); self.tcurrent_sb.blockSignals(False)
        for did in list(self.moving_dots.keys()): self._add_or_update_moving_dot(did)
        for did in list(self.clouds.keys()): self._update_cloud_for_drone(did)
        if self.engine.section_enabled and self.engine.width > 0.0:
            tmin = self.engine.tmin; tmax = self.engine.tmax; samples = 400
            center = self.engine.tcurrent
            t_vals = np.linspace(max(tmin, center - 0.5 * self.engine.width), min(tmax, center + 0.5 * self.engine.width), samples)
            for did, line in list(self.splines.items()):
                entry = self.engine.linear.drone_data.get(did)
                if not entry: continue
                fx, fy, fz = entry['fx'], entry['fy'], entry['fz']
                try:
                    x = np.atleast_1d(fx(t_vals)); y = np.atleast_1d(fy(t_vals)); z = np.atleast_1d(fz(t_vals))
                    line.setData(pos=np.vstack([x, y, z]).T)
                except Exception: pass
            if self.engine.leader_path and self.leader_spline is not None and self.engine.leader_visible:
                sp = self.engine.sp_gauss if self.engine.mode=='Gaussian' else self.engine.sp_linear
                try:
                    x = np.atleast_1d(sp['fx'](t_vals)); y = np.atleast_1d(sp['fy'](t_vals)); z = np.atleast_1d(sp['fz'](t_vals))
                    self.leader_spline.setData(pos=np.vstack([x, y, z]).T)
                except Exception: pass
        if hasattr(self.engine, 'sp_gauss') or hasattr(self.engine, 'sp_linear'):
            sp = self.engine.sp_gauss if self.engine.mode=='Gaussian' and hasattr(self.engine,'sp_gauss') else (self.engine.sp_linear if hasattr(self.engine,'sp_linear') else None)
            if sp: self._add_or_update_leader_marker(sp)
        self._update_leader_collision_cloud(); self.engine.compute_collisions()

    def _slider_to_time(self, v: int) -> float:
        tmin = self.engine.tmin; tmax = self.engine.tmax
        return tmin + (tmax - tmin) * (v / float(self.timeline.maximum()))

    def _refresh_collisions_table(self, rows): self.col_table.populate(rows)

    # ------------------ Export collisions ------------------
    def _export_collisions_clicked(self):
        rows = self.engine._last_collision_rows
        if not rows:
            QtWidgets.QMessageBox.information(self, 'Export', 'No collision data to export.')
            return
        fn, fmt = QtWidgets.QFileDialog.getSaveFileName(self, 'Export collisions', '', 'CSV (*.csv);;JSON (*.json)')
        if not fn:
            return
        try:
            if fn.lower().endswith('.csv') or 'CSV' in fmt:
                with open(fn, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Drone','Time','Severity','Metric'])
                    for r in rows:
                        writer.writerow([r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])
            else:
                data = [{'drone':r[0], 'time':float(r[1]), 'severity':float(r[2]), 'metric':float(r[3])} for r in rows]
                with open(fn, 'w') as f:
                    json.dump(data, f, indent=2)
            QtWidgets.QMessageBox.information(self, 'Export', f'Collisions exported to {fn}')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Export error', str(e))


# ---------------------- App Entry ----------------------
class DeconflicterApp(QtWidgets.QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setStyle('Fusion')
        dark = QtGui.QPalette(); dark.setColor(QtGui.QPalette.Window, QtGui.QColor(18,18,22)); dark.setColor(QtGui.QPalette.WindowText, Qt.white)
        dark.setColor(QtGui.QPalette.Base, QtGui.QColor(28,28,34)); dark.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(40,40,48))
        dark.setColor(QtGui.QPalette.Text, Qt.white); dark.setColor(QtGui.QPalette.Button, QtGui.QColor(28,28,34)); dark.setColor(QtGui.QPalette.ButtonText, Qt.white)
        self.setPalette(dark)
        self.setStyleSheet("QWidget:disabled{color:#7a7a7a;} QDoubleSpinBox:disabled,QSpinBox:disabled{background:#2b2b2b;color:#7a7a7a;} QPushButton:disabled{background:#151515;color:#444;}")
        self.engine = EngineBundle(); self.win = MainWindow(self.engine); self.win.show()


def main():
    import sys
    app = DeconflicterApp(sys.argv); sys.exit(app.exec_())


if __name__ == '__main__':
    main()
