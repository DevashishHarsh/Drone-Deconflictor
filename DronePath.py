"""
DronePath class for plotting drone trajectories with cubic spline interpolation.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Union, Set
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D


def _darken_rgb(rgb, factor=0.6):
    """Return darker version of rgb triple (each 0..1)."""
    arr = np.clip(np.array(rgb, dtype=float), 0.0, 1.0)
    return tuple((arr * factor).tolist())


def _rgb_from_cmap(cmap, i):
    c = cmap(i)
    if len(c) >= 3:
        return (c[0], c[1], c[2])
    return (c, c, c)


def severity_to_rgb_hue(sev: float) -> Tuple[float, float, float]:
    """
    Map severity in [0,1] to a colour shifting hue from green -> red smoothly,
    using HSV interpolation for smooth hue rotation.
    """
    sev = float(np.clip(sev, 0.0, 1.0))
    green_rgb = np.array([0.0, 1.0, 0.0])
    red_rgb = np.array([1.0, 0.0, 0.0])
    g_hsv = mcolors.rgb_to_hsv(green_rgb)
    r_hsv = mcolors.rgb_to_hsv(red_rgb)
    hsv = g_hsv + (r_hsv - g_hsv) * sev
    rgb = mcolors.hsv_to_rgb(hsv)
    return tuple(rgb.tolist())


class DronePath:
    """Build cubic spline parametric trajectories x(t), y(t), z(t) for multiple drones."""

    # reserved colours (RGB tuples)
    _RESERVED_DRIVER_SPLINE_RGB = (1.0, 1.0, 0.0)  # yellow
    _RESERVED_DRIVER_POINT_RGB = (0.0, 1.0, 0.0)  # green

    def __init__(self):
        self.drone_data: Dict[str, Dict[str, Any]] = {}
        self.tmin = None
        self.tmax = None

        # simple palette manager: drone_id -> (base_rgb, waypoint_rgb)
        self._palette: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {}
        self._cmap = plt.get_cmap('tab10')

    # -------------------------
    # Palette helpers
    # -------------------------
    def generate_palette_for_current_drones(self, reserved_rgb: Optional[Set[Tuple[float, float, float]]] = None):
        """Generate base+waypoint colours for all current drones (only assigns to missing ones)."""
        if reserved_rgb is None:
            reserved_rgb = {self._RESERVED_DRIVER_SPLINE_RGB, self._RESERVED_DRIVER_POINT_RGB}
        ids = sorted(self.drone_data.keys())
        for i, drone_id in enumerate(ids):
            if drone_id in self._palette:
                continue
            base_rgb = _rgb_from_cmap(self._cmap, i % 10)
            attempt = 0
            while tuple(np.round(base_rgb, 6)) in reserved_rgb and attempt < 20:
                attempt += 1
                base_rgb = _rgb_from_cmap(self._cmap, (i + attempt) % 10)
            waypoint_rgb = _darken_rgb(base_rgb, 0.55)
            self._palette[drone_id] = (tuple(base_rgb), tuple(waypoint_rgb))

    def set_palette_for_drone(self, drone_id: str, base_rgb: Tuple[float, float, float], waypoint_rgb: Optional[Tuple[float, float, float]] = None):
        """Explicitly set palette for a single drone."""
        if waypoint_rgb is None:
            waypoint_rgb = _darken_rgb(base_rgb, 0.55)
        self._palette[drone_id] = (tuple(base_rgb), tuple(waypoint_rgb))

    def get_palette_for_drone(self, drone_id: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return (base_rgb, waypoint_rgb) for given drone; generate palette if missing."""
        if drone_id not in self._palette:
            self.generate_palette_for_current_drones()
        return self._palette.get(drone_id, ((0.5, 0.5, 0.5), (0.3, 0.3, 0.3)))

    # -------------------------
    # Loading / building
    # -------------------------
    def load_json_points(self, file: Union[str, Path]) -> None:
        """
        Load drone points from JSON file.
        Expected format: ["drone_id", x, y, z, t] for each point
        """
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        try:
            raw = json.loads(file_path.read_text())
            rows = self._parse_json_data(raw)
            self._build_from_rows(rows)
            # ensure palette updated
            self.generate_palette_for_current_drones()
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {e}")

    def _parse_json_data(self, raw) -> List[Tuple[str, float, float, float, float]]:
        """Parse different JSON formats to standardized rows format."""
        rows = []

        if isinstance(raw, dict):
            for drone_id, wps in raw.items():
                for wp in wps:
                    if isinstance(wp, dict):
                        x = float(wp['x'])
                        y = float(wp['y'])
                        z = float(wp['z'])
                        t = float(wp['t'])
                    elif isinstance(wp, (list, tuple)) and len(wp) >= 4:
                        x = float(wp[0])
                        y = float(wp[1])
                        z = float(wp[2])
                        t = float(wp[3])
                    else:
                        raise ValueError(f"Unsupported waypoint format for drone {drone_id}")
                    rows.append((str(drone_id), x, y, z, t))

        elif isinstance(raw, list):
            if raw and isinstance(raw[0], dict) and all(k in raw[0] for k in ['id', 'x', 'y', 'z', 't']):
                for item in raw:
                    rows.append((str(item['id']), float(item['x']),
                                float(item['y']), float(item['z']), float(item['t'])))
            elif raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 5:
                for item in raw:
                    rows.append((str(item[0]), float(item[1]),
                                float(item[2]), float(item[3]), float(item[4])))
            else:
                raise ValueError("Unsupported JSON list format")
        else:
            raise ValueError("Unsupported JSON format")

        return rows

    def _build_from_rows(self, rows: List[Tuple[str, float, float, float, float]]):
        """Build spline data from parsed rows."""
        grouped = defaultdict(list)
        for r in rows:
            drone_id, x, y, z, t = r
            grouped[drone_id].append((t, x, y, z))

        # Calculate global time bounds if not set
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

            if len(t_arr) == 1:
                # Single point - create constant functions
                cx, cy, cz = x_arr[0], y_arr[0], z_arr[0]

                def make_const(v):
                    def f(t):
                        t_a = np.atleast_1d(t)
                        return np.full(t_a.shape, v, dtype=float)
                    return f

                fx = make_const(cx)
                fy = make_const(cy)
                fz = make_const(cz)
            else:
                # Multiple points - create splines
                # Fix duplicate time values
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

            self.drone_data[drone_id] = {
                'fx': fx, 'fy': fy, 'fz': fz,
                'tmin': self.tmin, 'tmax': self.tmax,
                'waypoints': pts_sorted
            }

    def get_spline(self, drone_id: str) -> Dict[str, Any]:
        """
        Get spline functions for a specific drone.
        Returns: {'fx': function, 'fy': function, 'fz': function}
        """
        if drone_id not in self.drone_data:
            raise KeyError(f"Drone id '{drone_id}' not found.")

        entry = self.drone_data[drone_id]
        return {
            'fx': entry['fx'],
            'fy': entry['fy'],
            'fz': entry['fz']
        }

    # -------------------------
    # Plotting (uses palette)
    # -------------------------
    def plot_all(self, width: float = 2.0, show_width: bool = False,
                 tmin: float = None, tmax: float = None, samples: int = 300,
                 figsize: Tuple[float, float] = (12, 8)):
        """
        Plot all drone splines.

        Args:
            width: Time width around current time (0 = show complete path)
            show_width: Whether to show the width visualization
            tmin: Minimum time (uses self.tmin if None)
            tmax: Maximum time (uses self.tmax if None)
            samples: Number of points to sample for spline visualization
        """
        if not self.drone_data:
            raise RuntimeError("No drone data available to plot.")

        plot_tmin = tmin if tmin is not None else self.tmin
        plot_tmax = tmax if tmax is not None else self.tmax

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('tab10')
        ids = sorted(self.drone_data.keys())

        # ensure palette assigned
        self.generate_palette_for_current_drones()

        for i, drone_id in enumerate(ids):
            base_color, waypoint_color = self.get_palette_for_drone(drone_id)

            if width == 0:
                # Show complete spline path
                t_vals = np.linspace(plot_tmin, plot_tmax, samples)
            else:
                # Show width around center time
                t_center = (plot_tmin + plot_tmax) / 2
                t_vals = np.linspace(max(plot_tmin, t_center - width / 2),
                                     min(plot_tmax, t_center + width / 2), samples)

            entry = self.drone_data[drone_id]
            fx, fy, fz = entry['fx'], entry['fy'], entry['fz']

            x_vals = np.atleast_1d(fx(t_vals))
            y_vals = np.atleast_1d(fy(t_vals))
            z_vals = np.atleast_1d(fz(t_vals))

            ax.plot(x_vals, y_vals, z_vals, color=base_color,
                    linewidth=2.0, alpha=0.8, label=f'Drone {drone_id}')

            # Show waypoints as scatter points
            waypoints = entry['waypoints']
            wp_x = [wp[1] for wp in waypoints]
            wp_y = [wp[2] for wp in waypoints]
            wp_z = [wp[3] for wp in waypoints]
            ax.scatter(wp_x, wp_y, wp_z, color=waypoint_color, s=50,
                       edgecolor='black', alpha=0.9, zorder=10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Drone Trajectories (t={plot_tmin:.1f} to {plot_tmax:.1f})')
        ax.legend()

        plt.tight_layout()
        plt.show()
        return ax

    def plot_all_slider(self, width: float = 2.0, show_width: bool = False,
                       tmin: float = None, tmax: float = None, samples: int = 300,
                       figsize: Tuple[float, float] = (12, 8)):
        """
        Plot all drone splines with interactive time slider.

        Args:
            width: Time width around slider time (0 = show complete path)
            show_width: Whether to show the width visualization
            tmin: Minimum time (uses self.tmin if None)
            tmax: Maximum time (uses self.tmax if None)
            samples: Number of points to sample for spline visualization
        """
        if not self.drone_data:
            raise RuntimeError("No drone data available to plot.")

        plot_tmin = tmin if tmin is not None else self.tmin
        plot_tmax = tmax if tmax is not None else self.tmax
        t_init = (plot_tmin + plot_tmax) / 2

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('tab10')
        ids = sorted(self.drone_data.keys())

        # Store handles for dynamic updates
        spline_handles = {}
        highlight_handles = {}

        # Calculate global bounds for consistent axis limits
        all_x, all_y, all_z = [], [], []

        for drone_id in ids:
            t_vals = np.linspace(plot_tmin, plot_tmax, samples)
            entry = self.drone_data[drone_id]
            fx, fy, fz = entry['fx'], entry['fy'], entry['fz']

            x_vals = np.atleast_1d(fx(t_vals))
            y_vals = np.atleast_1d(fy(t_vals))
            z_vals = np.atleast_1d(fz(t_vals))

            all_x.extend(x_vals[~np.isnan(x_vals)])
            all_y.extend(y_vals[~np.isnan(y_vals)])
            all_z.extend(z_vals[~np.isnan(z_vals)])

        # Set axis limits with padding
        if all_x and all_y and all_z:
            x_range = max(all_x) - min(all_x) if max(all_x) != min(all_x) else 1.0
            y_range = max(all_y) - min(all_y) if max(all_y) != min(all_y) else 1.0
            z_range = max(all_z) - min(all_z) if max(all_z) != min(all_z) else 1.0

            padding = 0.1
            ax.set_xlim(min(all_x) - padding * x_range, max(all_x) + padding * x_range)
            ax.set_ylim(min(all_y) - padding * y_range, max(all_y) + padding * y_range)
            ax.set_zlim(min(all_z) - padding * z_range, max(all_z) + padding * z_range)

        # ensure palette
        self.generate_palette_for_current_drones()

        # Initialize plots for each drone
        for i, drone_id in enumerate(ids):
            base_color, waypoint_color = self.get_palette_for_drone(drone_id)

            if width == 0 or not show_width:
                # Show complete spline (faint)
                t_vals = np.linspace(plot_tmin, plot_tmax, samples)
                entry = self.drone_data[drone_id]
                fx, fy, fz = entry['fx'], entry['fy'], entry['fz']

                x_vals = np.atleast_1d(fx(t_vals))
                y_vals = np.atleast_1d(fy(t_vals))
                z_vals = np.atleast_1d(fz(t_vals))

                spline_handles[drone_id], = ax.plot(x_vals, y_vals, z_vals,
                                                   color=base_color, linewidth=1.0,
                                                   alpha=0.3, label=f'Drone {drone_id}')
            else:
                spline_handles[drone_id], = ax.plot([np.nan], [np.nan], [np.nan],
                                                   color=base_color, linewidth=1.0,
                                                   alpha=0.3, label=f'Drone {drone_id}')

            # Create highlighted segment line (updated by slider)
            highlight_handles[drone_id], = ax.plot([np.nan], [np.nan], [np.nan],
                                                   color=base_color, linewidth=3.0, alpha=1.0)

        # Current positions scatter
        pos_x, pos_y, pos_z = [], [], []
        for drone_id in ids:
            pos = self._get_position(drone_id, t_init)
            if pos is not None:
                pos_x.append(pos[0])
                pos_y.append(pos[1])
                pos_z.append(pos[2])
            else:
                pos_x.append(np.nan)
                pos_y.append(np.nan)
                pos_z.append(np.nan)

        current_scatter = ax.scatter(pos_x, pos_y, pos_z, color='green',
                                     s=120, depthshade=True, zorder=20)

        def update_plot(t_val):
            # Update highlighted segments
            for drone_id in ids:
                if width > 0:
                    t_start = max(plot_tmin, t_val - width / 2)
                    t_end = min(plot_tmax, t_val + width / 2)
                    # avoid zero division
                    seg_samples = max(2, int(samples * (t_end - t_start) / max(1e-9, (plot_tmax - plot_tmin))))
                    t_seg = np.linspace(t_start, t_end, seg_samples)

                    entry = self.drone_data[drone_id]
                    fx, fy, fz = entry['fx'], entry['fy'], entry['fz']

                    x_seg = np.atleast_1d(fx(t_seg))
                    y_seg = np.atleast_1d(fy(t_seg))
                    z_seg = np.atleast_1d(fz(t_seg))

                    highlight_handles[drone_id].set_data(x_seg, y_seg)
                    highlight_handles[drone_id].set_3d_properties(z_seg)

            # Update current positions
            new_pos_x, new_pos_y, new_pos_z = [], [], []
            for drone_id in ids:
                pos = self._get_position(drone_id, t_val)
                if pos is not None:
                    new_pos_x.append(pos[0])
                    new_pos_y.append(pos[1])
                    new_pos_z.append(pos[2])
                else:
                    new_pos_x.append(np.nan)
                    new_pos_y.append(np.nan)
                    new_pos_z.append(np.nan)

            try:
                current_scatter._offsets3d = (np.array(new_pos_x),
                                              np.array(new_pos_y),
                                              np.array(new_pos_z))
            except:
                pass  # Fallback if update fails

        # Initial update
        update_plot(t_init)

        # Time display
        time_text = ax.text2D(0.02, 0.95, f"t = {t_init:.2f}",
                              transform=ax.transAxes, fontsize=12)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Trajectories with Time Slider')
        ax.legend()

        # Create slider
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

    # -------------------------
    # Getters
    # -------------------------
    def get_point(self, time: float) -> List[Tuple[str, float, float, float, float]]:
        """
        Get positions of all drones at a specific time using spline interpolation.
        Returns: List of [drone_id, x, y, z, t] for all drones
        """
        results = []
        for drone_id in self.drone_data.keys():
            pos = self._get_position(drone_id, time)
            if pos is not None:
                results.append((drone_id, pos[0], pos[1], pos[2], time))
        return results

    def _get_position(self, drone_id: str, time: float) -> Optional[Tuple[float, float, float]]:
        """Get position of specific drone at given time."""
        if drone_id not in self.drone_data:
            return None

        entry = self.drone_data[drone_id]
        fx, fy, fz = entry['fx'], entry['fy'], entry['fz']

        try:
            x = float(np.atleast_1d(fx(time))[0])
            y = float(np.atleast_1d(fy(time))[0])
            z = float(np.atleast_1d(fz(time))[0])
            return (x, y, z)
        except:
            return None

    def create_drone(self, file: Union[str, Path]) -> Dict[str, Any]:
        """
        Create spline data for driver drone from file.
        File should contain [x, y, z, t] data without drone_id.
        Returns: sp_data dictionary for the driver drone
        """
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        try:
            raw = json.loads(file_path.read_text())

            # Parse driver drone data (no drone_id in file)
            if isinstance(raw, list):
                if raw and isinstance(raw[0], dict) and all(k in raw[0] for k in ['x', 'y', 'z', 't']):
                    points = [(float(item['x']), float(item['y']),
                               float(item['z']), float(item['t'])) for item in raw]
                elif raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 4:
                    points = [(float(item[0]), float(item[1]),
                               float(item[2]), float(item[3])) for item in raw]
                else:
                    raise ValueError("Unsupported driver data format")
            else:
                raise ValueError("Driver file must contain a list of points")

            # Sort by time and create splines
            points_sorted = sorted(points, key=lambda p: p[3])
            t_arr = np.array([p[3] for p in points_sorted], dtype=float)
            x_arr = np.array([p[0] for p in points_sorted], dtype=float)
            y_arr = np.array([p[1] for p in points_sorted], dtype=float)
            z_arr = np.array([p[2] for p in points_sorted], dtype=float)

            # Handle duplicate times
            if len(t_arr) > 1:
                diffs = np.diff(t_arr)
                if np.any(diffs <= 0):
                    eps = 1e-9
                    for i in range(1, len(t_arr)):
                        if t_arr[i] <= t_arr[i - 1]:
                            t_arr[i] = t_arr[i - 1] + eps
                            eps *= 1.000001

            # Create splines
            if len(t_arr) == 1:
                def make_const(v):
                    def f(t): return np.full(np.atleast_1d(t).shape, v, dtype=float)
                    return f
                fx = make_const(x_arr[0])
                fy = make_const(y_arr[0])
                fz = make_const(z_arr[0])
            else:
                fx = CubicSpline(t_arr, x_arr, extrapolate=True)
                fy = CubicSpline(t_arr, y_arr, extrapolate=True)
                fz = CubicSpline(t_arr, z_arr, extrapolate=True)

            sp_data = {
                'fx': fx,
                'fy': fy,
                'fz': fz,
                'tmin': float(np.min(t_arr)),
                'tmax': float(np.max(t_arr)),
                'waypoints': points_sorted
            }

            return sp_data

        except Exception as e:
            raise ValueError(f"Error creating driver drone: {e}")

    # -------------------------
    # Simple geometric collision checking (distance-based) - enhanced
    # -------------------------
    def check_spline_with_drones(self, sp_data: Dict[str, Any], dist: float = 0.1,
                                 detailed: bool = False, debug: bool = False,
                                 n_samples: Optional[int] = None, refine: bool = True) -> Union[bool, Dict[str, List[float]]]:
        """
        Check if driver drone spline intersects with any existing drone splines using Euclidean distance.

        Args:
            sp_data: Spline data for driver drone (from create_drone)
            dist: Distance threshold for intersection detection
            detailed: If True return detailed records per collision group:
                      {'drone_id': [{'t_start', 't_end', 't_rep', 'd_min', 'severity'}, ...], ...}
                      If False returns {'drone_id': [t_rep, ...], ...} (backwards-compatible)
            debug: Print per-drone debug info
            n_samples: override number of time samples for coarse scanning (int). If None a default density is used.
            refine: If True do local dense search around coarse hits to find true minimum distance

        Returns:
            False if no intersections, or intersections dict as described above.
        """
        if not self.drone_data:
            return False

        driver_fx = sp_data['fx']
        driver_fy = sp_data['fy']
        driver_fz = sp_data['fz']

        # Use intersection of time ranges
        check_tmin = max(self.tmin if self.tmin is not None else sp_data['tmin'], sp_data['tmin'])
        check_tmax = min(self.tmax if self.tmax is not None else sp_data['tmax'], sp_data['tmax'])

        if check_tmax <= check_tmin:
            return False

        intersections: Dict[str, Any] = {}

        # decide number of coarse samples
        total_dur = float(check_tmax - check_tmin)
        if n_samples is not None and int(n_samples) > 0:
            n_samples_use = int(n_samples)
        else:
            # default coarse density: 200 samples per unit time, min 300
            n_samples_use = max(300, int(max(300, round(total_dur * 200.0))))

        t_samples = np.linspace(check_tmin, check_tmax, n_samples_use)

        for drone_id, drone_info in self.drone_data.items():
            drone_fx = drone_info['fx']
            drone_fy = drone_info['fy']
            drone_fz = drone_info['fz']

            intersection_times = []
            d_values = []

            for t in t_samples:
                try:
                    dx = float(np.atleast_1d(driver_fx(t))[0])
                    dy = float(np.atleast_1d(driver_fy(t))[0])
                    dz = float(np.atleast_1d(driver_fz(t))[0])
                    ex = float(np.atleast_1d(drone_fx(t))[0])
                    ey = float(np.atleast_1d(drone_fy(t))[0])
                    ez = float(np.atleast_1d(drone_fz(t))[0])
                    distance = np.sqrt((dx - ex) ** 2 + (dy - ey) ** 2 + (dz - ez) ** 2)
                    if distance <= dist:
                        intersection_times.append(float(t))
                        d_values.append(float(distance))
                except Exception:
                    continue

            if not intersection_times:
                if debug:
                    print(f"[distance-check] {drone_id}: no coarse hits")
                continue

            # group consecutive times into intervals
            grouped_intervals = []
            current_group = [intersection_times[0]]
            current_ds = [d_values[0]]
            max_gap = (check_tmax - check_tmin) / float(n_samples_use) * 2.5

            for i in range(1, len(intersection_times)):
                if intersection_times[i] - intersection_times[i - 1] <= max_gap:
                    current_group.append(intersection_times[i])
                    current_ds.append(d_values[i])
                else:
                    grouped_intervals.append((current_group, current_ds))
                    current_group = [intersection_times[i]]
                    current_ds = [d_values[i]]
            if current_group:
                grouped_intervals.append((current_group, current_ds))

            records = []
            for group_times, group_ds in grouped_intervals:
                coarse_t_start = float(group_times[0])
                coarse_t_end = float(group_times[-1])
                coarse_t_rep = float(group_times[len(group_times) // 2])
                coarse_d_min = float(min(group_ds)) if group_ds else float(np.min(group_ds))

                refined_t_rep = coarse_t_rep
                refined_d_min = coarse_d_min
                refined_t_start = coarse_t_start
                refined_t_end = coarse_t_end

                if refine:
                    # define pad and refined search window
                    pad = max((coarse_t_end - coarse_t_start) * 0.5, (check_tmax - check_tmin) / float(max(1, n_samples_use)) * 5.0)
                    r_start = max(check_tmin, coarse_t_start - pad)
                    r_end = min(check_tmax, coarse_t_end + pad)
                    if r_end > r_start:
                        # dense local sampling
                        dense_n = max(150, int((r_end - r_start) / (max(1e-9, (check_tmax - check_tmin))) * n_samples_use * 2))
                        dense_t = np.linspace(r_start, r_end, dense_n)
                        local_pairs = []
                        for tt in dense_t:
                            try:
                                dx = float(np.atleast_1d(driver_fx(tt))[0])
                                dy = float(np.atleast_1d(driver_fy(tt))[0])
                                dz = float(np.atleast_1d(driver_fz(tt))[0])
                                ex = float(np.atleast_1d(drone_fx(tt))[0])
                                ey = float(np.atleast_1d(drone_fy(tt))[0])
                                ez = float(np.atleast_1d(drone_fz(tt))[0])
                                dval = float(np.sqrt((dx - ex) ** 2 + (dy - ey) ** 2 + (dz - ez) ** 2))
                                local_pairs.append((tt, dval))
                            except Exception:
                                continue
                        if local_pairs:
                            t_vals_local, d_vals_local = zip(*local_pairs)
                            idx_min = int(np.argmin(d_vals_local))
                            refined_t_rep = float(t_vals_local[idx_min])
                            refined_d_min = float(d_vals_local[idx_min])
                            # compute refined interval where distance <= dist
                            below = [(t_, d_) for t_, d_ in local_pairs if d_ <= dist]
                            if below:
                                t_below = [t for t, _ in below]
                                refined_t_start = float(min(t_below))
                                refined_t_end = float(max(t_below))
                            else:
                                # fallback to coarse interval
                                refined_t_start = coarse_t_start
                                refined_t_end = coarse_t_end

                # severity mapping for simple distance: deeper inside threshold -> higher severity
                severity = float(np.clip(1.0 - (refined_d_min / (dist + 1e-12)), 0.0, 1.0))

                rec = {
                    't_start': float(refined_t_start),
                    't_end': float(refined_t_end),
                    't_rep': float(refined_t_rep),
                    'd_min': float(refined_d_min),
                    'severity': float(severity)
                }
                records.append(rec)

            if records:
                if detailed:
                    intersections[drone_id] = records
                else:
                    intersections[drone_id] = [r['t_rep'] for r in records]

            if debug:
                if records:
                    print(f"[distance-check] {drone_id}: {len(records)} group(s); reps = {[r['t_rep'] for r in records]}")
                else:
                    print(f"[distance-check] {drone_id}: groups found but empty records")

        return intersections if intersections else False

    # -------------------------
    # Plotting driver view (distance-based collision highlighting)
    # -------------------------
    def plot_all_driver(self, sp_data: Dict[str, Any], dist: float = 0.1,
                        width: float = 2.0, tmin: float = None, tmax: float = None,
                        samples: int = 300, cloud_samples: int = 100, cloud_span: float = 0.5,
                        show_nominal_driver_points: bool = True, detailed: bool = True,
                        figsize: Tuple[float, float] = (12, 8), debug: bool = False) -> Any:
        """
        Plot all existing drone trajectories and the driver spline with distance-based collision visualization.

        Args:
            sp_data: driver spline (from create_drone)
            dist: distance threshold used for collision detection (used for severity mapping)
            width: visible time window width (0 => full spline)
            tmin/tmax: explicit overall bounds (default to self.tmin/self.tmax and driver's bounds)
            samples: number of points for plotting splines
            cloud_samples: number of samples to draw around each collision representative time for driver markers
            cloud_span: seconds total span around t_rep to sample driver points for colored markers
            show_nominal_driver_points: if True and no collisions, plot green markers along driver spline
            detailed: pass-through to check_spline_with_drones() to get intervals+severity
            debug: verbose prints
        """
        if not self.drone_data:
            raise RuntimeError("No drone data available to plot.")

        plot_tmin = tmin if tmin is not None else self.tmin
        plot_tmax = tmax if tmax is not None else self.tmax

        # extend plot bounds to include driver
        if sp_data:
            plot_tmin = min(plot_tmin, sp_data['tmin']) if plot_tmin is not None else sp_data['tmin']
            plot_tmax = max(plot_tmax, sp_data['tmax']) if plot_tmax is not None else sp_data['tmax']

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        ids = sorted(self.drone_data.keys())
        self.generate_palette_for_current_drones()

        # Plot existing drones
        for drone_id in ids:
            base_color, waypoint_color = self.get_palette_for_drone(drone_id)
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

            ax.plot(x_vals, y_vals, z_vals, color=base_color,
                    linewidth=2.0, alpha=0.6, label=f'Drone {drone_id}')

            # waypoints
            waypoints = entry['waypoints']
            wp_x = [wp[1] for wp in waypoints]
            wp_y = [wp[2] for wp in waypoints]
            wp_z = [wp[3] for wp in waypoints]
            ax.scatter(wp_x, wp_y, wp_z, color=waypoint_color, s=50,
                       edgecolor='black', alpha=0.9, zorder=10)

        # Plot driver spline
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

            # Plot driver spline in yellow
            ax.plot(x_vals_driver, y_vals_driver, z_vals_driver, color=self._RESERVED_DRIVER_SPLINE_RGB,
                    linewidth=3.0, alpha=0.9, label='Driver Drone')

            # driver waypoints
            if 'waypoints' in sp_data:
                driver_waypoints = sp_data['waypoints']
                dwp_x = [wp[0] for wp in driver_waypoints]
                dwp_y = [wp[1] for wp in driver_waypoints]
                dwp_z = [wp[2] for wp in driver_waypoints]
                ax.scatter(dwp_x, dwp_y, dwp_z, color=self._RESERVED_DRIVER_SPLINE_RGB, s=60,
                           edgecolor='black', alpha=0.95, zorder=15)

            # check collisions
            intersections = self.check_spline_with_drones(sp_data, dist=dist, detailed=detailed, debug=debug, n_samples=None, refine=True)

            if intersections:
                # intersections is dict drone_id -> list of records (if detailed=True)
                for drone_id, records in intersections.items():
                    for rec in records:
                        t_rep = rec.get('t_rep', None)
                        severity = float(rec.get('severity', 0.0))
                        # sample driver around t_rep
                        if t_rep is None:
                            continue
                        t0 = max(plot_tmin, t_rep - cloud_span / 2.0)
                        t1 = min(plot_tmax, t_rep + cloud_span / 2.0)
                        if t1 <= t0:
                            continue
                        t_cloud = np.linspace(t0, t1, max(2, int(cloud_samples)))
                        try:
                            xs = np.atleast_1d(driver_fx(t_cloud))
                            ys = np.atleast_1d(driver_fy(t_cloud))
                            zs = np.atleast_1d(driver_fz(t_cloud))
                            col = severity_to_rgb_hue(severity)
                            alpha = 0.35 + 0.6 * float(severity)
                            ax.scatter(xs, ys, zs, color=col, s=18, alpha=alpha, zorder=20)
                        except Exception:
                            continue
            else:
                # no intersections -> show green markers along driver spline if requested
                if show_nominal_driver_points:
                    n_points = min(12, len(t_vals_driver))
                    idxs = np.linspace(0, len(t_vals_driver) - 1, n_points).astype(int)
                    xs = x_vals_driver[idxs]
                    ys = y_vals_driver[idxs]
                    zs = z_vals_driver[idxs]
                    ax.scatter(xs, ys, zs, color=self._RESERVED_DRIVER_POINT_RGB, s=10, alpha=0.6, zorder=12)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Drone Trajectories with Driver (t={plot_tmin:.1f} to {plot_tmax:.1f})')
        ax.legend()
        plt.tight_layout()
        plt.show()
        return ax

    # -------------------------
    # Pyqtgraph/OpenGL helpers
    # -------------------------
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

    def compute_collision_details(self, sp_data: Dict[str, Any], dist: float = 0.1,
                                  detailed: bool = True, debug: bool = False,
                                  n_samples: Optional[int] = None, refine: bool = True,
                                  include_driver_sample: bool = False, driver_sample_kwargs: Optional[Dict[str, Any]] = None) -> Union[bool, Dict[str, Any]]:
        """
        Wrapper that returns detailed collision records (distance-based) ready for GUI consumption.

        If include_driver_sample=True and driver_sample_kwargs is provided (e.g. {'samples':200}),
        each record will include a 'driver_sample' key with an (N,3) array sampled around t_rep.
        """
        res = self.check_spline_with_drones(sp_data, dist=dist, detailed=detailed, debug=debug, n_samples=n_samples, refine=refine)
        if not res:
            return False
        if not detailed or not include_driver_sample:
            return res

        if driver_sample_kwargs is None:
            driver_sample_kwargs = {}

        # If caller wants driver samples, they should sample sp_data themselves; keep placeholder None for now.
        for drone_id, records in res.items():
            for rec in records:
                rec['driver_sample'] = None
        return res

    # -------------------------
    # Simple helpers
    # -------------------------
    def get_drone_ids(self) -> List[str]:
        """Get list of all drone IDs."""
        return list(self.drone_data.keys())

    def has_drone(self, drone_id: str) -> bool:
        """Check if drone exists."""
        return drone_id in self.drone_data

    def get_time_bounds(self) -> Tuple[float, float]:
        """Get global time bounds."""
        return (self.tmin, self.tmax)

    def set_time_bounds(self, tmin: float, tmax: float):
        """Set global time bounds."""
        self.tmin = tmin
        self.tmax = tmax
        # Update all drone data time bounds
        for drone_info in self.drone_data.values():
            drone_info['tmin'] = tmin
            drone_info['tmax'] = tmax
