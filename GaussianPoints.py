import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Union, Set
from scipy.interpolate import CubicSpline
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D plotting)
from matplotlib import colors as mcolors

# --------------------------
# Utility helpers
# --------------------------
def _ensure_array(x):
    return np.atleast_1d(x)

def _darken_rgb(rgb, factor=0.6):
    """Return darker version of rgb triple (each 0..1)."""
    arr = np.clip(np.array(rgb, dtype=float), 0.0, 1.0)
    return tuple((arr * factor).tolist())

def _rgb_from_cmap(cmap, i):
    c = cmap(i)
    if len(c) >= 3:
        return (c[0], c[1], c[2])
    return (c, c, c)

def _lerp(a, b, t):
    return a + (b - a) * t

def severity_to_rgb_hue(sev: float) -> Tuple[float, float, float]:
    """
    Map severity in [0,1] to a colour shifting hue from green -> yellow -> red smoothly,
    using HSV interpolation for smooth hue rotation.
    """
    sev = float(np.clip(sev, 0.0, 1.0))
    green_rgb = np.array([0.0, 1.0, 0.0])
    red_rgb = np.array([1.0, 0.0, 0.0])
    # Convert to HSV to interpolate hue correctly
    g_hsv = mcolors.rgb_to_hsv(green_rgb)
    r_hsv = mcolors.rgb_to_hsv(red_rgb)
    hsv = _lerp(g_hsv, r_hsv, sev)
    rgb = mcolors.hsv_to_rgb(hsv)
    return tuple(rgb.tolist())

# --------------------------
# Updated GaussianPoints class
# --------------------------
class GaussianPoints:
    """Build cubic spline parametric trajectories with Gaussian uncertainty clouds for multiple drones.

    Extended with:
      - colour manager (random per-drone palette; reserved colours for driver)
      - improved collision detection with sampling/refinement and detailed record output
      - pyqtgraph-friendly data access wrappers
    """

    # reserved colours (RGB tuples)
    _RESERVED_DRIVER_SPLINE_RGB = (1.0, 1.0, 0.0)  # yellow
    _RESERVED_DRIVER_CLOUD_BASE_RGB = (0.0, 1.0, 0.0)  # green

    def __init__(self, cloud_strength: int = 300):
        """
        Args:
            cloud_strength: default total number of points used to represent the probability cloud
                            for the *visible* time window (higher => denser clouds).
        """
        self.drone_data: Dict[str, Dict[str, Any]] = {}
        self.tmin = None
        self.tmax = None
        self.uncertainty_type = 0  # 0: position-only, 1: position+velocity
        self.cloud_strength = int(cloud_strength)

        # Colour manager: mapping drone_id -> (base_rgb, dot_rgb)
        self._palette: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {}
        self._palette_seed = None  # not used (user said no seed)
        self._cmap = plt.get_cmap('tab10')

    # ---------------------
    # Loading / building (unchanged)
    # ---------------------
    def load_json_points(self, file: Union[str, Path], type: int = 0) -> None:
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        self.uncertainty_type = type

        try:
            raw = json.loads(file_path.read_text())
            rows = self._parse_json_data(raw, type)
            self._build_from_rows(rows, type)
            # regenerate palette for new drones (lazy)
            self.generate_palette_for_current_drones()
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {e}")

    def _parse_json_data(self, raw, type: int) -> List[Tuple]:
        rows = []
        expected_len = 5 if type == 0 else 8
        if isinstance(raw, dict):
            for drone_id, wps in raw.items():
                for wp in wps:
                    if isinstance(wp, dict):
                        if type == 0:
                            x, y, z, t = float(wp['x']), float(wp['y']), float(wp['z']), float(wp['t'])
                            rows.append((str(drone_id), x, y, z, t))
                        else:
                            x, y, z = float(wp['x']), float(wp['y']), float(wp['z'])
                            vx, vy, vz = float(wp['vx']), float(wp['vy']), float(wp['vz'])
                            t = float(wp['t'])
                            rows.append((str(drone_id), x, y, z, vx, vy, vz, t))
                    elif isinstance(wp, (list, tuple)) and len(wp) >= expected_len:
                        if type == 0:
                            rows.append((str(drone_id), float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3])))
                        else:
                            rows.append((str(drone_id), float(wp[0]), float(wp[1]), float(wp[2]),
                                       float(wp[3]), float(wp[4]), float(wp[5]), float(wp[6])))
        elif isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                for item in raw:
                    if type == 0 and all(k in item for k in ['id', 'x', 'y', 'z', 't']):
                        rows.append((str(item['id']), float(item['x']),
                                     float(item['y']), float(item['z']), float(item['t'])))
                    elif type == 1 and all(k in item for k in ['id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 't']):
                        rows.append((str(item['id']), float(item['x']), float(item['y']), float(item['z']),
                                     float(item['vx']), float(item['vy']), float(item['vz']), float(item['t'])))
            elif raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) >= expected_len:
                for item in raw:
                    if type == 0:
                        rows.append((str(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4])))
                    else:
                        rows.append((str(item[0]), float(item[1]), float(item[2]), float(item[3]),
                                     float(item[4]), float(item[5]), float(item[6]), float(item[7])))
        return rows

    def _build_from_rows(self, rows: List[Tuple], type: int):
        grouped = defaultdict(list)
        for r in rows:
            drone_id = r[0]
            if type == 0:
                _, x, y, z, t = r
                grouped[drone_id].append((t, x, y, z, None, None, None))
            else:
                _, x, y, z, vx, vy, vz, t = r
                grouped[drone_id].append((t, x, y, z, vx, vy, vz))

        if self.tmin is None or self.tmax is None:
            all_times = []
            for pts in grouped.values():
                all_times.extend([p[0] for p in pts])
            if self.tmin is None:
                self.tmin = min(all_times) if all_times else 0.0
            if self.tmax is None:
                self.tmax = max(all_times) if all_times else 0.0

        for drone_id, pts in grouped.items():
            pts_sorted = sorted(pts, key=lambda p: p[0])
            t_arr = np.array([p[0] for p in pts_sorted], dtype=float)
            x_arr = np.array([p[1] for p in pts_sorted], dtype=float)
            y_arr = np.array([p[2] for p in pts_sorted], dtype=float)
            z_arr = np.array([p[3] for p in pts_sorted], dtype=float)

            if type == 1:
                vx_arr = np.array([p[4] for p in pts_sorted], dtype=float)
                vy_arr = np.array([p[5] for p in pts_sorted], dtype=float)
                vz_arr = np.array([p[6] for p in pts_sorted], dtype=float)

            if len(t_arr) == 1:
                cx, cy, cz = x_arr[0], y_arr[0], z_arr[0]

                def make_const(v):
                    def f(t):
                        t_a = np.atleast_1d(t)
                        return np.full(t_a.shape, v, dtype=float)
                    return f

                fx = make_const(cx)
                fy = make_const(cy)
                fz = make_const(cz)

                if type == 1:
                    cvx, cvy, cvz = vx_arr[0], vy_arr[0], vz_arr[0]
                    fvx = make_const(cvx)
                    fvy = make_const(cvy)
                    fvz = make_const(cvz)
            else:
                diffs = np.diff(t_arr)
                if np.any(diffs <= 0):
                    eps = 1e-9
                    for i in range(1, len(t_arr)):
                        if t_arr[i] <= t_arr[i - 1]:
                            t_arr[i] = t_arr[i - 1] + eps
                            eps *= 1.000001

                fx = CubicSpline(t_arr, x_arr, extrapolate=True)
                fy = CubicSpline(t_arr, y_arr, extrapolate=True)
                fz = CubicSpline(t_arr, z_arr, extrapolate=True)

                if type == 1:
                    fvx = CubicSpline(t_arr, vx_arr, extrapolate=True)
                    fvy = CubicSpline(t_arr, vy_arr, extrapolate=True)
                    fvz = CubicSpline(t_arr, vz_arr, extrapolate=True)

            drone_entry = {
                'fx': fx, 'fy': fy, 'fz': fz,
                'tmin': self.tmin, 'tmax': self.tmax,
                'waypoints': pts_sorted,
                'type': type
            }

            if type == 1:
                drone_entry.update({
                    'fvx': fvx, 'fvy': fvy, 'fvz': fvz
                })

            self.drone_data[drone_id] = drone_entry

        # regenerate palette to include new drones
        self.generate_palette_for_current_drones()

    # ---------------------
    # Colour manager
    # ---------------------
    def generate_palette_for_current_drones(self, reserved_rgb: Optional[Set[Tuple[float, float, float]]] = None):
        """
        Generate base+dot colours for all current drones (only assigns to missing ones).
        Does not overwrite existing assigned colours unless force=True in future enhancements.
        """
        if reserved_rgb is None:
            reserved_rgb = {self._RESERVED_DRIVER_SPLINE_RGB, self._RESERVED_DRIVER_CLOUD_BASE_RGB}

        ids = sorted(self.drone_data.keys())
        for i, drone_id in enumerate(ids):
            if drone_id in self._palette:
                continue
            # choose a colour from cmap (tab10) ensuring it's not reserved
            attempt = 0
            base_rgb = _rgb_from_cmap(self._cmap, i)
            while tuple(np.round(base_rgb, 6)) in reserved_rgb and attempt < 20:
                # pick next colour
                attempt += 1
                base_rgb = _rgb_from_cmap(self._cmap, (i + attempt) % 10)
            dot_rgb = _darken_rgb(base_rgb, 0.55)
            self._palette[drone_id] = (tuple(base_rgb), tuple(dot_rgb))

    def set_palette_for_drone(self, drone_id: str, base_rgb: Tuple[float, float, float], dot_rgb: Optional[Tuple[float, float, float]] = None):
        """Explicitly set palette for a single drone."""
        if dot_rgb is None:
            dot_rgb = _darken_rgb(base_rgb, 0.55)
        self._palette[drone_id] = (tuple(base_rgb), tuple(dot_rgb))

    def get_palette_for_drone(self, drone_id: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return (base_rgb, dot_rgb) for given drone; generate palette if missing."""
        if drone_id not in self._palette:
            self.generate_palette_for_current_drones()
        return self._palette.get(drone_id, ((0.5, 0.5, 0.5), (0.3, 0.3, 0.3)))

    # ---------------------
    # Utilities for uncertainty (unchanged)
    # ---------------------
    def _get_uncertainty_function(self, drone_id: str):
        entry = self.drone_data[drone_id]

        if entry['type'] == 0:
            def uncertainty_func(t, base_sigma=0.1):
                return np.eye(3) * (base_sigma ** 2)
        else:
            def uncertainty_func(t, base_sigma=0.1, velocity_factor=0.05):
                try:
                    vx = float(np.atleast_1d(entry['fvx'](t))[0])
                    vy = float(np.atleast_1d(entry['fvy'](t))[0])
                    vz = float(np.atleast_1d(entry['fvz'](t))[0])

                    v_mag = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
                    if v_mag < 1e-6:
                        return np.eye(3) * (base_sigma ** 2)

                    v_unit = np.array([vx, vy, vz]) / v_mag
                    cov = np.eye(3) * (base_sigma ** 2)
                    velocity_cov = velocity_factor * v_mag * np.outer(v_unit, v_unit)
                    cov += velocity_cov
                    return cov
                except Exception:
                    return np.eye(3) * (base_sigma ** 2)
        return uncertainty_func

    def _cov_from_sp_data(self, sp_data: Dict[str, Any], t: float, base_sigma: float = 0.1, velocity_factor: float = 0.05):
        try:
            if sp_data['type'] == 0:
                return np.eye(3) * (base_sigma ** 2)
            else:
                try:
                    vx = float(np.atleast_1d(sp_data['fvx'](t))[0])
                    vy = float(np.atleast_1d(sp_data['fvy'](t))[0])
                    vz = float(np.atleast_1d(sp_data['fvz'](t))[0])
                    v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)
                    cov = np.eye(3) * (base_sigma ** 2)
                    if v_mag > 1e-9:
                        v_unit = np.array([vx, vy, vz]) / v_mag
                        cov += velocity_factor * v_mag * np.outer(v_unit, v_unit)
                    return cov
                except Exception:
                    return np.eye(3) * (base_sigma ** 2)
        except Exception:
            return np.eye(3) * (base_sigma ** 2)

    # ---------------------
    # Spline accessor (unchanged)
    # ---------------------
    def get_gaussian_spline(self, drone_id: str) -> Dict[str, Any]:
        if drone_id not in self.drone_data:
            raise KeyError(f"Drone id '{drone_id}' not found.")

        entry = self.drone_data[drone_id]
        result = {
            'fx': entry['fx'],
            'fy': entry['fy'],
            'fz': entry['fz'],
            'uncertainty_func': self._get_uncertainty_function(drone_id)
        }

        if entry['type'] == 1:
            result.update({
                'fvx': entry['fvx'],
                'fvy': entry['fvy'],
                'fvz': entry['fvz']
            })

        return result

    # ---------------------
    # Cloud generation across a visible time window (unchanged)
    # ---------------------
    def _generate_time_window_cloud(self, drone_id: str, t_center: float, width: float,
                                    spread: float = 0.1, cloud_strength: Optional[int] = None,
                                    plot_tmin: Optional[float] = None, plot_tmax: Optional[float] = None):
        entry = self.drone_data.get(drone_id)
        if entry is None:
            return None

        if cloud_strength is None:
            cloud_strength = int(self.cloud_strength)

        global_tmin = plot_tmin if plot_tmin is not None else entry.get('tmin', self.tmin)
        global_tmax = plot_tmax if plot_tmax is not None else entry.get('tmax', self.tmax)
        if global_tmin is None or global_tmax is None:
            return None

        if width == 0:
            t_start, t_end = float(global_tmin), float(global_tmax)
            n_slices = max(2, int(min(50, cloud_strength // 4)))
            times = np.linspace(t_start, t_end, n_slices)
            weights = np.ones_like(times, dtype=float)
        else:
            t_start = max(global_tmin, t_center - width / 2.0)
            t_end = min(global_tmax, t_center + width / 2.0)
            if t_end <= t_start:
                t_start, t_end = float(global_tmin), float(global_tmax)
            n_slices = int(min(80, max(3, round((t_end - t_start) * 20))))
            times = np.linspace(t_start, t_end, n_slices)
            sigma_time = max((t_end - t_start) / 4.0, 1e-6)
            weights = np.exp(-0.5 * ((times - t_center) / sigma_time) ** 2)

        weights_sum = float(np.sum(weights))
        if weights_sum <= 0:
            weights = np.ones_like(weights)
            weights_sum = float(np.sum(weights))
        samples_per_slice = (weights / weights_sum) * float(cloud_strength)
        samples_per_slice = np.maximum(1, np.round(samples_per_slice).astype(int))

        all_samples = []
        uncertainty_func = self._get_uncertainty_function(drone_id)

        for ti, npt in zip(times, samples_per_slice):
            try:
                mx = float(np.atleast_1d(entry['fx'](ti))[0])
                my = float(np.atleast_1d(entry['fy'](ti))[0])
                mz = float(np.atleast_1d(entry['fz'](ti))[0])
                mean = np.array([mx, my, mz])

                cov = uncertainty_func(ti, base_sigma=float(spread))
                cov = np.asarray(cov, dtype=float)
                cov += 1e-12 * np.eye(3)

                if npt > 0:
                    tries = 0
                    while tries < 3:
                        try:
                            s = np.random.multivariate_normal(mean, cov, size=int(npt))
                            break
                        except Exception:
                            cov += (1e-8 * (tries + 1)) * np.eye(3)
                            tries += 1
                    else:
                        continue
                    all_samples.append(s)
            except Exception:
                continue

        if not all_samples:
            return None
        return np.vstack(all_samples)

    # ---------------------
    # Plotting (updated to use palette)
    # ---------------------
    def plot_all(self, spread: float = 0.1, width: float = 2.0, show_width: bool = False,
                 tmin: float = None, tmax: float = None, samples: int = 300,
                 cloud_strength: Optional[int] = None,
                 figsize: Tuple[float, float] = (12, 8)):
        if not self.drone_data:
            raise RuntimeError("No drone data available to plot.")
        plot_tmin = tmin if tmin is not None else self.tmin
        plot_tmax = tmax if tmax is not None else self.tmax

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ids = sorted(self.drone_data.keys())
        # ensure palette
        self.generate_palette_for_current_drones()

        for i, drone_id in enumerate(ids):
            base_color_rgb, dot_color_rgb = self.get_palette_for_drone(drone_id)
            if width == 0:
                t_vals = np.linspace(plot_tmin, plot_tmax, samples)
            else:
                t_center = (plot_tmin + plot_tmax) / 2
                t_vals = np.linspace(max(plot_tmin, t_center - width / 2),
                                     min(plot_tmax, t_center + width / 2), samples)

            entry = self.drone_data[drone_id]
            fx, fy, fz = entry['fx'], entry['fy'], entry['fz']
            x_vals = np.atleast_1d(fx(t_vals))
            y_vals = np.atleast_1d(fy(t_vals))
            z_vals = np.atleast_1d(fz(t_vals))

            # Plot spline path in base colour (slightly faded)
            ax.plot(x_vals, y_vals, z_vals, color=base_color_rgb,
                    linewidth=2.0, alpha=0.6, label=f'Drone {drone_id}')

            # waypoints as darker scatter points
            waypoints = entry['waypoints']
            wp_x = [wp[1] for wp in waypoints]
            wp_y = [wp[2] for wp in waypoints]
            wp_z = [wp[3] for wp in waypoints]
            ax.scatter(wp_x, wp_y, wp_z, color=dot_color_rgb, s=50,
                       edgecolor='black', alpha=0.9, zorder=10)

            # Add Gaussian point cloud for visible window
            t_center_vis = float(np.mean([t_vals[0], t_vals[-1]])) if len(t_vals) > 0 else float((plot_tmin + plot_tmax) / 2.0)
            cloud = self._generate_time_window_cloud(drone_id, t_center_vis, width,
                                                     spread=spread, cloud_strength=cloud_strength,
                                                     plot_tmin=plot_tmin, plot_tmax=plot_tmax)
            if cloud is not None:
                ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                           color=dot_color_rgb, s=8, alpha=0.25)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Drone Trajectories with Gaussian Clouds (t={plot_tmin:.1f} to {plot_tmax:.1f})')
        ax.legend()
        plt.tight_layout()
        plt.show()
        return ax

    def plot_all_slider(self, spread: float = 0.1, width: float = 2.0, show_width: bool = False,
                       tmin: float = None, tmax: float = None, samples: int = 300,
                       cloud_strength: Optional[int] = None,
                       figsize: Tuple[float, float] = (12, 8)):
        if not self.drone_data:
            raise RuntimeError("No drone data available to plot.")
        plot_tmin = tmin if tmin is not None else self.tmin
        plot_tmax = tmax if tmax is not None else self.tmax
        t_init = (plot_tmin + plot_tmax) / 2

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ids = sorted(self.drone_data.keys())
        self.generate_palette_for_current_drones()

        spline_handles = {}
        cloud_handles = {}

        for i, drone_id in enumerate(ids):
            base_color_rgb, dot_color_rgb = self.get_palette_for_drone(drone_id)
            t_vals = np.linspace(plot_tmin, plot_tmax, samples)
            entry = self.drone_data[drone_id]
            fx, fy, fz = entry['fx'], entry['fy'], entry['fz']
            x_vals = np.atleast_1d(fx(t_vals))
            y_vals = np.atleast_1d(fy(t_vals))
            z_vals = np.atleast_1d(fz(t_vals))

            spline_handles[drone_id], = ax.plot(x_vals, y_vals, z_vals,
                                               color=base_color_rgb, linewidth=1.0,
                                               alpha=0.3, label=f'Drone {drone_id}')

            cloud_handles[drone_id] = ax.scatter([np.nan], [np.nan], [np.nan],
                                                 color=dot_color_rgb, s=8, alpha=0.25)

        def update_plot(t_val):
            for drone_id in ids:
                cloud = self._generate_time_window_cloud(drone_id, t_val, width,
                                                        spread=spread, cloud_strength=cloud_strength,
                                                        plot_tmin=plot_tmin, plot_tmax=plot_tmax)
                if cloud is not None:
                    cloud_handles[drone_id]._offsets3d = (cloud[:, 0], cloud[:, 1], cloud[:, 2])
                else:
                    cloud_handles[drone_id]._offsets3d = ([], [], [])

        update_plot(t_init)

        time_text = ax.text2D(0.02, 0.95, f"t = {t_init:.2f}",
                              transform=ax.transAxes, fontsize=12)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Trajectories with Gaussian Clouds (Time Slider)')
        ax.legend()

        ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Time', plot_tmin, plot_tmax,
                        valinit=t_init, valfmt='%0.2f')

        def slider_update(val):
            t_val = slider.val
            update_plot(t_val)
            time_text.set_text(f"t = {t_val:.2f}")
            fig.canvas.draw_idle()

        slider.on_changed(slider_update)
        plt.tight_layout()
        plt.show()
        return ax

    # ---------------------
    # Position / velocity getters (unchanged)
    # ---------------------
    def get_point(self, time: float) -> List[Tuple]:
        results = []
        for drone_id in self.drone_data.keys():
            entry = self.drone_data[drone_id]
            pos = self._get_position(drone_id, time)
            if pos is not None:
                if entry['type'] == 0:
                    results.append((drone_id, pos[0], pos[1], pos[2], time))
                else:
                    vel = self._get_velocity(drone_id, time)
                    if vel is not None:
                        results.append((drone_id, pos[0], pos[1], pos[2],
                                        vel[0], vel[1], vel[2], time))
                    else:
                        results.append((drone_id, pos[0], pos[1], pos[2],
                                        0.0, 0.0, 0.0, time))
        return results

    def _get_position(self, drone_id: str, time: float) -> Optional[Tuple[float, float, float]]:
        if drone_id not in self.drone_data:
            return None
        entry = self.drone_data[drone_id]
        fx, fy, fz = entry['fx'], entry['fy'], entry['fz']
        try:
            x = float(np.atleast_1d(fx(time))[0])
            y = float(np.atleast_1d(fy(time))[0])
            z = float(np.atleast_1d(fz(time))[0])
            return (x, y, z)
        except Exception:
            return None

    def _get_velocity(self, drone_id: str, time: float) -> Optional[Tuple[float, float, float]]:
        if drone_id not in self.drone_data:
            return None
        entry = self.drone_data[drone_id]
        if entry['type'] != 1:
            return None
        try:
            vx = float(np.atleast_1d(entry['fvx'](time))[0])
            vy = float(np.atleast_1d(entry['fvy'](time))[0])
            vz = float(np.atleast_1d(entry['fvz'](time))[0])
            return (vx, vy, vz)
        except Exception:
            return None

    # ---------------------
    # Driver creation (unchanged)
    # ---------------------
    def create_drone(self, file: Union[str, Path], type: int = 0) -> Dict[str, Any]:
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        try:
            raw = json.loads(file_path.read_text())
            if isinstance(raw, list):
                if raw and isinstance(raw[0], dict):
                    if type == 0 and all(k in raw[0] for k in ['x', 'y', 'z', 't']):
                        points = [(float(item['x']), float(item['y']),
                                   float(item['z']), float(item['t'])) for item in raw]
                    elif type == 1 and all(k in raw[0] for k in ['x', 'y', 'z', 'vx', 'vy', 'vz', 't']):
                        points = [(float(item['x']), float(item['y']), float(item['z']),
                                   float(item['vx']), float(item['vy']), float(item['vz']),
                                   float(item['t'])) for item in raw]
                elif raw and isinstance(raw[0], (list, tuple)):
                    expected_len = 4 if type == 0 else 7
                    if len(raw[0]) >= expected_len:
                        if type == 0:
                            points = [(float(item[0]), float(item[1]),
                                       float(item[2]), float(item[3])) for item in raw]
                        else:
                            points = [(float(item[0]), float(item[1]), float(item[2]), 
                                       float(item[3]), float(item[4]), float(item[5]), 
                                       float(item[6])) for item in raw]
                    else:
                        raise ValueError("Insufficient data in driver file")
                else:
                    raise ValueError("Unsupported driver data format")
            else:
                raise ValueError("Driver file must contain a list of points")

            if type == 0:
                points_sorted = sorted(points, key=lambda p: p[3])
                t_arr = np.array([p[3] for p in points_sorted], dtype=float)
                x_arr = np.array([p[0] for p in points_sorted], dtype=float)
                y_arr = np.array([p[1] for p in points_sorted], dtype=float)
                z_arr = np.array([p[2] for p in points_sorted], dtype=float)
            else:
                points_sorted = sorted(points, key=lambda p: p[6])
                t_arr = np.array([p[6] for p in points_sorted], dtype=float)
                x_arr = np.array([p[0] for p in points_sorted], dtype=float)
                y_arr = np.array([p[1] for p in points_sorted], dtype=float)
                z_arr = np.array([p[2] for p in points_sorted], dtype=float)
                vx_arr = np.array([p[3] for p in points_sorted], dtype=float)
                vy_arr = np.array([p[4] for p in points_sorted], dtype=float)
                vz_arr = np.array([p[5] for p in points_sorted], dtype=float)

            if len(t_arr) > 1:
                diffs = np.diff(t_arr)
                if np.any(diffs <= 0):
                    eps = 1e-9
                    for i in range(1, len(t_arr)):
                        if t_arr[i] <= t_arr[i - 1]:
                            t_arr[i] = t_arr[i - 1] + eps
                            eps *= 1.000001

            if len(t_arr) == 1:
                def make_const(v):
                    def f(t): return np.full(np.atleast_1d(t).shape, v, dtype=float)
                    return f
                fx = make_const(x_arr[0])
                fy = make_const(y_arr[0])
                fz = make_const(z_arr[0])
                if type == 1:
                    fvx = make_const(vx_arr[0])
                    fvy = make_const(vy_arr[0])
                    fvz = make_const(vz_arr[0])
            else:
                fx = CubicSpline(t_arr, x_arr, extrapolate=True)
                fy = CubicSpline(t_arr, y_arr, extrapolate=True)
                fz = CubicSpline(t_arr, z_arr, extrapolate=True)
                if type == 1:
                    fvx = CubicSpline(t_arr, vx_arr, extrapolate=True)
                    fvy = CubicSpline(t_arr, vy_arr, extrapolate=True)
                    fvz = CubicSpline(t_arr, vz_arr, extrapolate=True)

            sp_data = {
                'fx': fx, 'fy': fy, 'fz': fz,
                'tmin': float(np.min(t_arr)),
                'tmax': float(np.max(t_arr)),
                'waypoints': points_sorted,
                'type': type
            }

            if type == 1:
                sp_data.update({
                    'fvx': fvx, 'fvy': fvy, 'fvz': fvz
                })

            return sp_data

        except Exception as e:
            raise ValueError(f"Error creating driver drone: {e}")

    # ---------------------
    # Collision checking (improved)
    # ---------------------
    def check_spline_with_drones(self, sp_data: Dict[str, Any], confidence_level: float = 0.8,
                                 plot_tmin: Optional[float] = None, plot_tmax: Optional[float] = None,
                                 width: float = 0.0, method: str = 'analytic',
                                 detailed: bool = False, debug: bool = False,
                                 n_samples: Optional[int] = None, refine: bool = True) -> Union[bool, Dict[str, List[float]]]:
        """
        Check if driver drone spline intersects with existing drone Gaussian clouds.

        New options:
            detailed: if True return per-drone list of collision records:
                {
                  'droneA': [
                    {'t_start':..., 't_end':..., 't_rep':..., 'm2_min':..., 'severity':...},
                    ...
                  ]
                }
            n_samples: override computed number of sample times (int). If provided and > 0 used as absolute sample count.
            refine: if True perform local refinement around coarse collision detections.

        Backwards-compatible:
            if detailed=False, returns False or {"drone_id": [t1, t2, ...]} similar to prior behaviour.
        """
        if not self.drone_data:
            return False

        driver_fx = sp_data['fx']
        driver_fy = sp_data['fy']
        driver_fz = sp_data['fz']
        driver_type = sp_data['type']

        overall_tmin = self.tmin if self.tmin is not None else sp_data.get('tmin', None)
        overall_tmax = self.tmax if self.tmax is not None else sp_data.get('tmax', None)
        check_tmin = max(overall_tmin if overall_tmin is not None else sp_data.get('tmin', -np.inf),
                         sp_data.get('tmin', -np.inf))
        check_tmax = min(overall_tmax if overall_tmax is not None else sp_data.get('tmax', np.inf),
                         sp_data.get('tmax', np.inf))

        if plot_tmin is not None:
            check_tmin = max(check_tmin, plot_tmin)
        if plot_tmax is not None:
            check_tmax = min(check_tmax, plot_tmax)

        if check_tmax <= check_tmin:
            return False

        if width > 0:
            t_center = 0.5 * (check_tmin + check_tmax)
            check_tmin = max(check_tmin, t_center - width / 2.0)
            check_tmax = min(check_tmax, t_center + width / 2.0)
            if check_tmax <= check_tmin:
                return False

        intersections: Dict[str, List[Any]] = {}

        try:
            r_threshold = float(np.sqrt(chi2.ppf(float(confidence_level), df=3)))
        except Exception:
            r_threshold = 1.28

        # compute sample count
        total_duration = float(check_tmax - check_tmin)
        if n_samples is not None and int(n_samples) > 0:
            n_samples_use = int(n_samples)
        else:
            # default denser sampling: ~200 samples per unit time (min 400)
            n_samples_use = max(400, int(max(300, round(total_duration * 200.0))))
        t_samples = np.linspace(check_tmin, check_tmax, n_samples_use)

        # iterate over other drones
        for drone_id, drone_info in self.drone_data.items():
            drone_fx = drone_info['fx']
            drone_fy = drone_info['fy']
            drone_fz = drone_info['fz']
            uncertainty_func = self._get_uncertainty_function(drone_id)

            collision_times = []
            m2_values = []

            for t in t_samples:
                try:
                    dx = float(np.atleast_1d(driver_fx(t))[0])
                    dy = float(np.atleast_1d(driver_fy(t))[0])
                    dz = float(np.atleast_1d(driver_fz(t))[0])
                    driver_pos = np.array([dx, dy, dz])

                    ex = float(np.atleast_1d(drone_fx(t))[0])
                    ey = float(np.atleast_1d(drone_fy(t))[0])
                    ez = float(np.atleast_1d(drone_fz(t))[0])
                    drone_pos = np.array([ex, ey, ez])

                    cov_other = uncertainty_func(t, base_sigma=0.1)
                    cov_driver = self._cov_from_sp_data(sp_data, t, base_sigma=0.1)

                    cov_combined = np.asarray(cov_driver, dtype=float) + np.asarray(cov_other, dtype=float)
                    cov_combined += 1e-12 * np.eye(3)

                    diff = driver_pos - drone_pos
                    cov_inv = np.linalg.pinv(cov_combined)
                    m2 = float(np.dot(diff, np.dot(cov_inv, diff)))

                    if np.sqrt(m2) <= r_threshold:
                        collision_times.append(float(t))
                        m2_values.append(m2)
                except Exception:
                    continue

            if not collision_times:
                if debug:
                    print(f"[check] Drone {drone_id}: no collisions (coarse sampling)")
                continue

            # group consecutive times into intervals
            grouped_intervals = []
            current_group = [collision_times[0]]
            current_m2s = [m2_values[0]]
            # sampling interval approximate
            if len(t_samples) > 1:
                approx_dt = float(t_samples[1] - t_samples[0])
            else:
                approx_dt = max(1e-6, total_duration / max(1, n_samples_use))
            max_gap = approx_dt * 2.5

            for i in range(1, len(collision_times)):
                if collision_times[i] - collision_times[i - 1] <= max_gap:
                    current_group.append(collision_times[i])
                    current_m2s.append(m2_values[i])
                else:
                    grouped_intervals.append((current_group, current_m2s))
                    current_group = [collision_times[i]]
                    current_m2s = [m2_values[i]]
            if current_group:
                grouped_intervals.append((current_group, current_m2s))

            records = []
            for group_times, group_m2s in grouped_intervals:
                coarse_t_start = float(group_times[0])
                coarse_t_end = float(group_times[-1])

                # choose coarse representative as the time in the coarse group with minimal m2
                try:
                    coarse_idx_min = int(np.argmin(group_m2s))
                    coarse_t_rep = float(group_times[coarse_idx_min])
                    coarse_m2_min = float(group_m2s[coarse_idx_min])
                except Exception:
                    coarse_t_rep = float(group_times[len(group_times) // 2])
                    coarse_m2_min = float(min(group_m2s) if group_m2s else np.inf)

                refined_t_rep = coarse_t_rep
                refined_m2_min = coarse_m2_min
                t_start_ref = coarse_t_start
                t_end_ref = coarse_t_end

                # refinement: dense sampling around [coarse_t_start - pad, coarse_t_end + pad]
                if refine:
                    pad = max((coarse_t_end - coarse_t_start) * 0.5, approx_dt * 5.0)
                    r_start = max(check_tmin, coarse_t_start - pad)
                    r_end = min(check_tmax, coarse_t_end + pad)
                    if r_end > r_start:
                        dense_n = max(100, int((r_end - r_start) / (total_duration + 1e-12) * n_samples_use * 2))
                        dense_t = np.linspace(r_start, r_end, dense_n)
                        local_m2 = []
                        for tt in dense_t:
                            try:
                                dx = float(np.atleast_1d(driver_fx(tt))[0])
                                dy = float(np.atleast_1d(driver_fy(tt))[0])
                                dz = float(np.atleast_1d(driver_fz(tt))[0])
                                driver_pos = np.array([dx, dy, dz])

                                ex = float(np.atleast_1d(drone_fx(tt))[0])
                                ey = float(np.atleast_1d(drone_fy(tt))[0])
                                ez = float(np.atleast_1d(drone_fz(tt))[0])
                                drone_pos = np.array([ex, ey, ez])

                                cov_other = uncertainty_func(tt, base_sigma=0.1)
                                cov_driver = self._cov_from_sp_data(sp_data, tt, base_sigma=0.1)
                                cov_combined = np.asarray(cov_driver, dtype=float) + np.asarray(cov_other, dtype=float)
                                cov_combined += 1e-12 * np.eye(3)
                                diff = driver_pos - drone_pos
                                cov_inv = np.linalg.pinv(cov_combined)
                                m2 = float(np.dot(diff, np.dot(cov_inv, diff)))
                                local_m2.append((tt, m2))
                            except Exception:
                                continue
                        if local_m2:
                            # find min m2 in refined window
                            tt_vals, m2_vals = zip(*local_m2)
                            idx_min = int(np.argmin(m2_vals))
                            refined_t_rep = float(tt_vals[idx_min])
                            refined_m2_min = float(m2_vals[idx_min])

                            # compute refined interval by threshold crossing around refined_t_rep
                            below = [ (t_, m2_) for t_, m2_ in local_m2 if np.sqrt(m2_) <= r_threshold ]
                            if below:
                                t_below = [t for t, _ in below]
                                t_start_ref = float(min(t_below))
                                t_end_ref = float(max(t_below))
                            else:
                                # if no points below threshold in refined window, fall back to coarse
                                t_start_ref = coarse_t_start
                                t_end_ref = coarse_t_end
                        else:
                            # dense sampling produced no values (unlikely). fall back to coarse
                            t_start_ref = coarse_t_start
                            t_end_ref = coarse_t_end
                            refined_t_rep = coarse_t_rep
                            refined_m2_min = coarse_m2_min
                    else:
                        # pad produced invalid window - fallback
                        t_start_ref = coarse_t_start
                        t_end_ref = coarse_t_end
                        refined_t_rep = coarse_t_rep
                        refined_m2_min = coarse_m2_min
                else:
                    # no refinement performed, keep coarse values
                    t_start_ref = coarse_t_start
                    t_end_ref = coarse_t_end
                    refined_t_rep = coarse_t_rep
                    refined_m2_min = coarse_m2_min

                # severity mapping: based on refined_m2_min distance compared to threshold
                d_min = float(np.sqrt(max(0.0, refined_m2_min)))
                severity = float(np.clip(1.0 - (d_min / (r_threshold + 1e-12)), 0.0, 1.0))

                record = {
                    't_start': float(t_start_ref),
                    't_end': float(t_end_ref),
                    't_rep': float(refined_t_rep),
                    'm2_min': float(refined_m2_min),
                    'severity': float(severity)
                }
                records.append(record)

            if records:
                if detailed:
                    intersections[drone_id] = records
                else:
                    intersections[drone_id] = [r['t_rep'] for r in records]

            if debug:
                if records:
                    print(f"[check] Drone {drone_id}: {len(records)} collision group(s).")
                else:
                    print(f"[check] Drone {drone_id}: none after grouping.")

        return intersections if intersections else False

    # ---------------------
    # Plotting driver with collision visualization (updated colours + severity)
    # ---------------------
    def plot_all_driver(self, sp_data: Dict[str, Any], spread: float = 0.1,
                       width: float = 2.0, show_width: bool = False,
                       tmin: float = None, tmax: float = None, samples: int = 300,
                       cloud_strength: Optional[int] = None,
                       figsize: Tuple[float, float] = (12, 8)):
        if not self.drone_data:
            raise RuntimeError("No drone data available to plot.")
        plot_tmin = tmin if tmin is not None else self.tmin
        plot_tmax = tmax if tmax is not None else self.tmax

        if sp_data:
            plot_tmin = min(plot_tmin, sp_data['tmin']) if plot_tmin is not None else sp_data['tmin']
            plot_tmax = max(plot_tmax, sp_data['tmax']) if plot_tmax is not None else sp_data['tmax']

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ids = sorted(self.drone_data.keys())
        self.generate_palette_for_current_drones()

        # Plot existing drones with their palette
        for i, drone_id in enumerate(ids):
            base_color_rgb, dot_color_rgb = self.get_palette_for_drone(drone_id)
            if width == 0:
                t_vals = np.linspace(plot_tmin, plot_tmax, samples)
            else:
                t_center = (plot_tmin + plot_tmax) / 2
                t_vals = np.linspace(max(plot_tmin, t_center - width / 2),
                                     min(plot_tmax, t_center + width / 2), samples)

            entry = self.drone_data[drone_id]
            fx, fy, fz = entry['fx'], entry['fy'], entry['fz']
            x_vals = np.atleast_1d(fx(t_vals))
            y_vals = np.atleast_1d(fy(t_vals))
            z_vals = np.atleast_1d(fz(t_vals))

            ax.plot(x_vals, y_vals, z_vals, color=base_color_rgb,
                    linewidth=2.0, alpha=0.6, label=f'Drone {drone_id}')

            waypoints = entry['waypoints']
            wp_x = [wp[1] for wp in waypoints]
            wp_y = [wp[2] for wp in waypoints]
            wp_z = [wp[3] for wp in waypoints]
            ax.scatter(wp_x, wp_y, wp_z, color=dot_color_rgb, s=50,
                       edgecolor='black', alpha=0.9, zorder=10)

            t_center_vis = float(np.mean([t_vals[0], t_vals[-1]])) if len(t_vals) > 0 else (plot_tmin + plot_tmax) / 2
            cloud = self._generate_time_window_cloud(drone_id, t_center_vis, width,
                                                     spread=spread, cloud_strength=cloud_strength,
                                                     plot_tmin=plot_tmin, plot_tmax=plot_tmax)
            if cloud is not None:
                ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                           color=dot_color_rgb, s=10, alpha=0.25, zorder=5)

        # Plot driver drone spline and clouds
        if sp_data:
            driver_fx = sp_data['fx']
            driver_fy = sp_data['fy']
            driver_fz = sp_data['fz']

            if width == 0:
                t_vals_driver = np.linspace(plot_tmin, plot_tmax, samples)
            else:
                t_center = (plot_tmin + plot_tmax) / 2
                t_vals_driver = np.linspace(max(plot_tmin, t_center - width / 2),
                                            min(plot_tmax, t_center + width / 2), samples)

            x_vals_driver = np.atleast_1d(driver_fx(t_vals_driver))
            y_vals_driver = np.atleast_1d(driver_fy(t_vals_driver))
            z_vals_driver = np.atleast_1d(driver_fz(t_vals_driver))

            # Driver spline = reserved yellow
            ax.plot(x_vals_driver, y_vals_driver, z_vals_driver, color=self._RESERVED_DRIVER_SPLINE_RGB,
                    linewidth=3.0, alpha=0.8, label='Driver Drone')

            if 'waypoints' in sp_data:
                driver_waypoints = sp_data['waypoints']
                if sp_data['type'] == 0:
                    dwp_x = [wp[0] for wp in driver_waypoints]
                    dwp_y = [wp[1] for wp in driver_waypoints]
                    dwp_z = [wp[2] for wp in driver_waypoints]
                else:
                    dwp_x = [wp[0] for wp in driver_waypoints]
                    dwp_y = [wp[1] for wp in driver_waypoints]
                    dwp_z = [wp[2] for wp in driver_waypoints]
                ax.scatter(dwp_x, dwp_y, dwp_z, color=self._RESERVED_DRIVER_SPLINE_RGB, s=60,
                           edgecolor='black', alpha=0.9, zorder=15)

            intersections = self.check_spline_with_drones(sp_data, confidence_level=0.8,
                                                         plot_tmin=plot_tmin, plot_tmax=plot_tmax, width=width,
                                                         detailed=True, debug=False, n_samples=None, refine=True)

            if intersections:
                # intersections is a dict drone_id -> list of records
                for drone_id, records in intersections.items():
                    for rec in records:
                        t = rec['t_rep']
                        severity = rec['severity']
                        driver_cloud = self._generate_time_window_cloud_for_spdata(sp_data, t, width,
                                                                                   spread=spread,
                                                                                   cloud_strength=cloud_strength,
                                                                                   plot_tmin=plot_tmin, plot_tmax=plot_tmax)
                        if driver_cloud is not None:
                            col = severity_to_rgb_hue(severity)
                            alpha = 0.35 + 0.6 * float(severity)
                            ax.scatter(driver_cloud[:, 0], driver_cloud[:, 1], driver_cloud[:, 2],
                                       color=col, s=15, alpha=alpha, zorder=20)
            else:
                n_driver_clouds = min(10, len(t_vals_driver))
                driver_cloud_times = t_vals_driver[::max(1, len(t_vals_driver) // n_driver_clouds)]
                for t in driver_cloud_times:
                    driver_cloud = self._generate_time_window_cloud_for_spdata(sp_data, t, width,
                                                                               spread=spread,
                                                                               cloud_strength=cloud_strength,
                                                                               plot_tmin=plot_tmin, plot_tmax=plot_tmax)
                    if driver_cloud is not None:
                        ax.scatter(driver_cloud[:, 0], driver_cloud[:, 1], driver_cloud[:, 2],
                                   color=self._RESERVED_DRIVER_CLOUD_BASE_RGB, s=12, alpha=0.45, zorder=12)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Drone Trajectories with Driver Drone (t={plot_tmin:.1f} to {plot_tmax:.1f})')
        ax.legend()
        plt.tight_layout()
        plt.show()
        return ax

    def _generate_time_window_cloud_for_spdata(self, sp_data: Dict[str, Any], t_center: float, width: float,
                                               spread: float = 0.1, cloud_strength: Optional[int] = None,
                                               plot_tmin: Optional[float] = None, plot_tmax: Optional[float] = None):
        if cloud_strength is None:
            cloud_strength = int(self.cloud_strength)

        global_tmin = plot_tmin if plot_tmin is not None else sp_data.get('tmin', None)
        global_tmax = plot_tmax if plot_tmax is not None else sp_data.get('tmax', None)
        if global_tmin is None or global_tmax is None:
            return None

        if width == 0:
            t_start, t_end = float(global_tmin), float(global_tmax)
            n_slices = max(2, int(min(50, cloud_strength // 4)))
            times = np.linspace(t_start, t_end, n_slices)
            weights = np.ones_like(times)
        else:
            t_start = max(global_tmin, t_center - width / 2.0)
            t_end = min(global_tmax, t_center + width / 2.0)
            if t_end <= t_start:
                t_start, t_end = float(global_tmin), float(global_tmax)

            n_slices = int(min(80, max(3, round((t_end - t_start) * 20))))
            times = np.linspace(t_start, t_end, n_slices)
            sigma_time = max((t_end - t_start) / 4.0, 1e-6)
            weights = np.exp(-0.5 * ((times - t_center) / sigma_time) ** 2)

        weights_sum = float(np.sum(weights))
        if weights_sum <= 0:
            weights = np.ones_like(weights)
            weights_sum = float(np.sum(weights))
        samples_per_slice = (weights / weights_sum) * float(cloud_strength)
        samples_per_slice = np.maximum(1, np.round(samples_per_slice).astype(int))

        all_samples = []
        for ti, npt in zip(times, samples_per_slice):
            try:
                mx = float(np.atleast_1d(sp_data['fx'](ti))[0])
                my = float(np.atleast_1d(sp_data['fy'](ti))[0])
                mz = float(np.atleast_1d(sp_data['fz'](ti))[0])
                mean = np.array([mx, my, mz])

                cov = self._cov_from_sp_data(sp_data, ti, base_sigma=float(spread))
                cov = np.asarray(cov, dtype=float)
                cov += 1e-12 * np.eye(3)

                if npt > 0:
                    tries = 0
                    while tries < 3:
                        try:
                            s = np.random.multivariate_normal(mean, cov, size=int(npt))
                            break
                        except Exception:
                            cov += (1e-8 * (tries + 1)) * np.eye(3)
                            tries += 1
                    else:
                        continue
                    all_samples.append(s)
            except Exception:
                continue

        if not all_samples:
            return None
        return np.vstack(all_samples)

    # ---------------------
    # Pyqtgraph-friendly helper wrappers
    # ---------------------
    def get_spline_points(self, drone_id: str, tmin: float, tmax: float, samples: int = 200) -> Optional[np.ndarray]:
        """Return (N,3) array of positions sampled uniformly along spline between tmin..tmax."""
        if drone_id not in self.drone_data:
            return None
        fx = self.drone_data[drone_id]['fx']
        fy = self.drone_data[drone_id]['fy']
        fz = self.drone_data[drone_id]['fz']
        t_vals = np.linspace(tmin, tmax, samples)
        xs = np.atleast_1d(fx(t_vals))
        ys = np.atleast_1d(fy(t_vals))
        zs = np.atleast_1d(fz(t_vals))
        return np.vstack([xs, ys, zs]).T

    def get_cloud_samples(self, drone_id: str, t_center: float, width: float,
                          spread: float = 0.1, cloud_strength: Optional[int] = None,
                          plot_tmin: Optional[float] = None, plot_tmax: Optional[float] = None) -> Optional[np.ndarray]:
        """Wrapper for _generate_time_window_cloud -> returns (N,3) sample positions."""
        return self._generate_time_window_cloud(drone_id, t_center, width, spread, cloud_strength, plot_tmin, plot_tmax)

    def get_driver_cloud_samples(self, sp_data: Dict[str, Any], t_center: float, width: float,
                                 spread: float = 0.1, cloud_strength: Optional[int] = None,
                                 plot_tmin: Optional[float] = None, plot_tmax: Optional[float] = None) -> Optional[np.ndarray]:
        return self._generate_time_window_cloud_for_spdata(sp_data, t_center, width, spread, cloud_strength, plot_tmin, plot_tmax)

    def generate_palette_for_drones(self, drone_ids: Iterable[str], reserved_ids: Optional[Set[str]] = None):
        """Generate palette entries for a custom list of drone IDs (useful before plotting)."""
        if reserved_ids is None:
            reserved_ids = set()
        for i, drone_id in enumerate(drone_ids):
            if drone_id in reserved_ids:
                continue
            if drone_id not in self._palette:
                base_rgb = _rgb_from_cmap(self._cmap, i % 10)
                # ensure not equal to reserved driver colours
                if tuple(np.round(base_rgb, 6)) == tuple(np.round(self._RESERVED_DRIVER_SPLINE_RGB, 6)):
                    base_rgb = (0.2, 0.4, 0.8)
                dot_rgb = _darken_rgb(base_rgb, 0.55)
                self._palette[drone_id] = (tuple(base_rgb), tuple(dot_rgb))

    def compute_collision_details(self, sp_data: Dict[str, Any], confidence_level: float = 0.8,
                                  plot_tmin: Optional[float] = None, plot_tmax: Optional[float] = None,
                                  width: float = 0.0, detailed: bool = True, debug: bool = False,
                                  n_samples: Optional[int] = None, refine: bool = True,
                                  include_driver_samples: bool = False, driver_cloud_kwargs: Optional[Dict[str, Any]] = None) -> Union[bool, Dict[str, Any]]:
        """
        Friendly wrapper which returns rich collision details usable by a pyqtgraph GUI.

        If include_driver_samples=True, each record will include a 'driver_cloud' key with (N,3) points
        generated by get_driver_cloud_samples(...) using driver_cloud_kwargs.
        """
        res = self.check_spline_with_drones(sp_data, confidence_level=confidence_level,
                                            plot_tmin=plot_tmin, plot_tmax=plot_tmax,
                                            width=width, method='analytic',
                                            detailed=detailed, debug=debug, n_samples=n_samples, refine=refine)
        if not res:
            return False
        if not detailed:
            return res

        if include_driver_samples:
            if driver_cloud_kwargs is None:
                driver_cloud_kwargs = {}
            for drone_id, records in res.items():
                for rec in records:
                    t = rec['t_rep']
                    driver_cloud = self.get_driver_cloud_samples(sp_data, t, width,
                                                                spread=driver_cloud_kwargs.get('spread', 0.1),
                                                                cloud_strength=driver_cloud_kwargs.get('cloud_strength', None),
                                                                plot_tmin=driver_cloud_kwargs.get('plot_tmin', None),
                                                                plot_tmax=driver_cloud_kwargs.get('plot_tmax', None))
                    rec['driver_cloud'] = driver_cloud
        return res

    # ---------------------
    # Simple helpers (unchanged)
    # ---------------------
    def get_drone_ids(self) -> List[str]:
        return list(self.drone_data.keys())

    def has_drone(self, drone_id: str) -> bool:
        return drone_id in self.drone_data

    def get_time_bounds(self) -> Tuple[float, float]:
        return (self.tmin, self.tmax)

    def set_time_bounds(self, tmin: float, tmax: float):
        self.tmin = tmin
        self.tmax = tmax
        for drone_info in self.drone_data.values():
            drone_info['tmin'] = tmin
            drone_info['tmax'] = tmax
