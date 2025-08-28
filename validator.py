
"""
validator.py
Creates randomized drone datasets and ground-truth collision times.
Generates: drone_points.json, leader.json, results.json
"""

import sys, os, json, random
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QSpinBox, QFileDialog, QDoubleSpinBox,
    QMessageBox, QFrame, QGridLayout, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt

# -------------------------
# Utilities
# -------------------------
def lerp(a, b, t):
    return a + (b - a) * t

def ensure_sorted_by_t(path):
    return sorted(path, key=lambda p: float(p["t"]))

def compute_velocities_for_path(path):
    if len(path) < 2:
        for p in path:
            p.update({"vx": 0.0, "vy": 0.0, "vz": 0.0})
        return path

    times = np.array([float(p["t"]) for p in path], dtype=float)
    xs = np.array([float(p["x"]) for p in path], dtype=float)
    ys = np.array([float(p["y"]) for p in path], dtype=float)
    zs = np.array([float(p["z"]) for p in path], dtype=float)
    vx = np.zeros_like(xs)
    vy = np.zeros_like(ys)
    vz = np.zeros_like(zs)
    n = len(times)
    for i in range(n):
        if i == 0:
            dt = times[1] - times[0] if times[1] != times[0] else 1.0
            vx[i] = (xs[1] - xs[0]) / dt
            vy[i] = (ys[1] - ys[0]) / dt
            vz[i] = (zs[1] - zs[0]) / dt
        elif i == n - 1:
            dt = times[-1] - times[-2] if times[-1] != times[-2] else 1.0
            vx[i] = (xs[-1] - xs[-2]) / dt
            vy[i] = (ys[-1] - ys[-2]) / dt
            vz[i] = (zs[-1] - zs[-2]) / dt
        else:
            dt = times[i+1] - times[i-1]
            if dt == 0:
                vx[i] = vy[i] = vz[i] = 0.0
            else:
                vx[i] = (xs[i+1] - xs[i-1]) / dt
                vy[i] = (ys[i+1] - ys[i-1]) / dt
                vz[i] = (zs[i+1] - zs[i-1]) / dt
    for i, p in enumerate(path):
        p.update({"vx": float(vx[i]), "vy": float(vy[i]), "vz": float(vz[i])})
    return path

def interpolate_position(path, t):
    pts = ensure_sorted_by_t(path)
    if not pts:
        return (0.0, 0.0, 0.0)
    times = [float(p["t"]) for p in pts]
    xs = [float(p["x"]) for p in pts]
    ys = [float(p["y"]) for p in pts]
    zs = [float(p["z"]) for p in pts]
    if t <= times[0]:
        return (xs[0], ys[0], zs[0])
    if t >= times[-1]:
        return (xs[-1], ys[-1], zs[-1])
    for i in range(1, len(times)):
        if times[i] >= t:
            u = (t - times[i-1]) / (times[i] - times[i-1]) if (times[i] - times[i-1]) != 0 else 0.0
            x = lerp(xs[i-1], xs[i], u)
            y = lerp(ys[i-1], ys[i], u)
            z = lerp(zs[i-1], zs[i], u)
            return (x, y, z)
    return (xs[-1], ys[-1], zs[-1])

# -------------------------
# Path generation
# -------------------------
def random_path(tmin, tmax, n_points=5, type_mode=0, spread_scale=1.0):
    times = np.linspace(float(tmin), float(tmax), n_points)
    ox, oy, oz = random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(0, 4)
    step_xy = 2.0 * spread_scale
    step_z = 0.8 * spread_scale
    xs = np.cumsum(np.random.uniform(-step_xy, step_xy, size=n_points)) + ox
    ys = np.cumsum(np.random.uniform(-step_xy, step_xy, size=n_points)) + oy
    zs = np.cumsum(np.random.uniform(-step_z, step_z, size=n_points)) + oz
    path = []
    for i, t in enumerate(times):
        p = {"x": float(xs[i]), "y": float(ys[i]), "z": float(zs[i]), "t": float(times[i])}
        path.append(p)
    path = ensure_sorted_by_t(path)
    if type_mode == 1:
        path = compute_velocities_for_path(path)
    return path

# -------------------------
# Collision injection (robust)
# -------------------------
def inject_collisions_strict(leader_path, drones_dict, mode, tmin, tmax):
    results = {}
    drone_ids = list(drones_dict.keys())
    if mode == "No collision":
        return results

    def add_collision_to_drone(did, t_c):
        pos = interpolate_position(leader_path, t_c)
        eps = max(0.02 * (float(tmax) - float(tmin)) / 10.0, 0.05)
        before_t = max(float(tmin), t_c - eps)
        after_t = min(float(tmax), t_c + eps)
        jitter = 1e-6
        seg = [
            {"x": float(pos[0] - 0.01 + random.uniform(-jitter, jitter)),
             "y": float(pos[1] - 0.01 + random.uniform(-jitter, jitter)),
             "z": float(pos[2] + random.uniform(-jitter, jitter)),
             "t": float(before_t)},
            {"x": float(pos[0] + random.uniform(-jitter, jitter)),
             "y": float(pos[1] + random.uniform(-jitter, jitter)),
             "z": float(pos[2] + random.uniform(-jitter, jitter)),
             "t": float(t_c)},
            {"x": float(pos[0] + 0.01 + random.uniform(-jitter, jitter)),
             "y": float(pos[1] + 0.01 + random.uniform(-jitter, jitter)),
             "z": float(pos[2] + random.uniform(-jitter, jitter)),
             "t": float(after_t)}
        ]
        drones_dict[did].extend(seg)
        drones_dict[did] = ensure_sorted_by_t(drones_dict[did])
        return round(float(t_c), 3)

    if mode == "Single point collision":
        did = random.choice(drone_ids)
        t_c = random.uniform(float(tmin), float(tmax))
        results[did] = [add_collision_to_drone(did, t_c)]
        return results

    if mode == "Multi point collision":
        n_collisions = random.randint(2, max(2, min(5, len(drone_ids) + 1)))
        results = {}
        for _ in range(n_collisions):
            did = random.choice(drone_ids)
            t_c = random.uniform(float(tmin), float(tmax))
            val = add_collision_to_drone(did, t_c)
            results.setdefault(did, []).append(val)
        for k in list(results.keys()):
            results[k] = sorted(list(set(results[k])))
        return results

    return results

# -------------------------
# Finalize output
# -------------------------
def finalize_paths_for_output(drones_dict, leader_path, type_mode):
    drones_out = {}
    for did, pts in drones_dict.items():
        pts_sorted = ensure_sorted_by_t(pts)
        if type_mode == 1:
            pts_sorted = compute_velocities_for_path(pts_sorted)
        drones_out[did] = [{k: float(v) for k, v in p.items()} for p in pts_sorted]
    leader_out = [ {k: float(v) for k, v in p.items()} for p in ensure_sorted_by_t(leader_path) ]
    if type_mode == 1:
        leader_out = compute_velocities_for_path(leader_out)
    return drones_out, leader_out

# -------------------------
# GUI
# -------------------------
class ValidatorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Validator for Deconflictor")
        self.setMinimumSize(640, 360)
        self._apply_dark_theme()
        self._build_ui()

    def _apply_dark_theme(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(12, 12, 12))
        pal.setColor(QPalette.WindowText, QColor(230, 230, 230))
        pal.setColor(QPalette.Base, QColor(20, 20, 20))
        pal.setColor(QPalette.Button, QColor(28, 28, 28))
        pal.setColor(QPalette.ButtonText, QColor(230, 230, 230))
        self.setPalette(pal)
        self.setStyleSheet("""
            QLabel { color: #EEE; }
            QComboBox, QSpinBox, QDoubleSpinBox { background: #1f1f1f; color: #eee; border: 1px solid #333; padding: 3px; }
            QPushButton#generate { background: #27ae60; color: white; font-weight: bold; padding: 10px; border-radius: 6px; }
            QFrame#panel { background: #141414; border-radius: 8px; padding: 12px; }
        """)

    def _build_ui(self):
        main = QVBoxLayout(self)
        title = QLabel("Validator for Deconflictor")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main.addWidget(title)

        frame = QFrame()
        frame.setObjectName("panel")
        main.addWidget(frame)
        grid = QGridLayout(frame)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(12)

        grid.addWidget(QLabel("Collision type"), 0, 0)
        self.combo_mode = QComboBox(); self.combo_mode.addItems(["No collision", "Single point collision", "Multi point collision"])
        grid.addWidget(self.combo_mode, 0, 1)

        grid.addWidget(QLabel("Number of drones"), 1, 0)
        self.spin_n = QSpinBox(); self.spin_n.setRange(1, 30); self.spin_n.setValue(5)
        grid.addWidget(self.spin_n, 1, 1)

        grid.addWidget(QLabel("Type"), 2, 0)
        self.combo_type = QComboBox(); self.combo_type.addItems(["0", "1"]); self.combo_type.setCurrentIndex(0)
        grid.addWidget(self.combo_type, 2, 1)

        grid.addWidget(QLabel("tmin"), 3, 0)
        self.spin_tmin = QDoubleSpinBox(); self.spin_tmin.setRange(-1e6, 1e6); self.spin_tmin.setDecimals(3); self.spin_tmin.setValue(0.0)
        grid.addWidget(self.spin_tmin, 3, 1)
        grid.addWidget(QLabel("tmax"), 4, 0)
        self.spin_tmax = QDoubleSpinBox(); self.spin_tmax.setRange(-1e6, 1e6); self.spin_tmax.setDecimals(3); self.spin_tmax.setValue(20.0)
        grid.addWidget(self.spin_tmax, 4, 1)

        spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        grid.addItem(spacer, 5, 0, 1, 2)

        self.btn_generate = QPushButton("Generate")
        self.btn_generate.setObjectName("generate")
        self.btn_generate.clicked.connect(self.on_generate)
        main.addWidget(self.btn_generate, alignment=Qt.AlignHCenter)

        note = QLabel("Files saved: drone_points.json  leader.json  results.json")
        note.setStyleSheet("color:#bbb; font-size:11px;")
        note.setAlignment(Qt.AlignCenter)
        main.addWidget(note)

    def on_generate(self):
        try:
            tmin = float(self.spin_tmin.value())
            tmax = float(self.spin_tmax.value())
        except Exception:
            QMessageBox.warning(self, "Invalid", "tmin/tmax invalid")
            return
        if tmax <= tmin:
            QMessageBox.warning(self, "Invalid", "tmax must be > tmin")
            return
        n = int(self.spin_n.value())
        type_mode = int(self.combo_type.currentText())
        mode = self.combo_mode.currentText()

        n_leader_pts = random.randint(4, 6)
        leader = random_path(tmin, tmax, n_points=n_leader_pts, type_mode=0, spread_scale=1.2)

        drones = {}
        for i in range(1, n+1):
            drones[f"drone_{i}"] = random_path(tmin, tmax, n_points=random.randint(4,6), type_mode=0, spread_scale=1.0)

        results = inject_collisions_strict(leader, drones, mode, tmin, tmax)

        drones_out, leader_out = finalize_paths_for_output(drones, leader, type_mode)

        folder = QFileDialog.getExistingDirectory(self, "Select folder to save JSON files")
        if not folder:
            return
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(folder, "drone_points.json"), "w") as f:
                json.dump(drones_out, f, indent=2)
            with open(os.path.join(folder, "leader.json"), "w") as f:
                json.dump(leader_out, f, indent=2)
            with open(os.path.join(folder, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Could not save files: {e}")
            return

        QMessageBox.information(self, "Saved", f"Saved dataset and results to:\n{folder}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = ValidatorUI()
    ui.show()
    sys.exit(app.exec_())
