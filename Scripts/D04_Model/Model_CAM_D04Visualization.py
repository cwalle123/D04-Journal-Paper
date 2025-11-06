"""
Pure rendering:
- Renders ALL bins (num_bins shown == num_bins set).
- Uses TRUE bin widths [x_min, x_max]; no min-thickness inflation, no shifting.
- No center line in the curtain (optional faint side edges to perceive thickness).
- No axis padding/locking; Matplotlib autoscale is used.
- Data-faithful: no sigma exaggeration, no smoothing.
"""

##############################################################################################################

# External imports
import math, inspect
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Internal imports
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path, generate_starting_error

##############################################################################################################
"""Functions"""

# ----------------------------
# Visual style
# ----------------------------
COLORS = {
    "curtain_base": "#B3BAC5",   # muted gray-blue
    "curtain_hi":   "#3B82F6",   # highlight
    "edge_line":    "#8A949F",   # faint side edges (left/right)
    "mean_base":    "#5F6368",
    "mean_hi":      "#111827",
    "regression":   "#222222",
    "scatter":      "#9AA0A6",
    "slab":         "#ECEFF4",
    "background":   "#F8FAFC",
}

STYLE = {
    "scatter_alpha": 0.16,
    "reg_line_width": 2.4,

    "curtain_alpha_min": 0.20,
    "curtain_alpha_hi":  0.62,

    "edge_line_width":   1.0,   # side edges only (no center line)
    "mean_marker_size":    24,
    "mean_marker_size_hi": 34,

    "bin_slab_alpha": 0.06,

    "hud_fontsize": 11,
    "axis_label_size": 11,
    "title_size": 13,
    "trail_width": 2.2,
}

ANIM = {
    "step_stride": 3,
    "trail_len": 300,
    "interval_ms": 25,
    "camera_orbit": True,
    "orbit_deg_per_frame": 0.12,
}

DENSITY = {
    "pdf_points": 160,
    "max_cloud_points": 25000,  # for scatter only
}

# ----------------------------
# Utilities
# ----------------------------
def _edge_values(x_sorted: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    return np.array([x_sorted[idx] for idx in bin_edges])

def _locate_bin(x_val: float, x_sorted: np.ndarray, bin_edges: np.ndarray) -> int:
    edges = _edge_values(x_sorted, bin_edges)
    if x_val <= edges[0]: return 0
    if x_val >= edges[-1]: return len(edges) - 2
    i = np.searchsorted(edges, x_val, side="right") - 1
    return max(0, min(i, len(edges) - 2))

def _call_generate_error_path(start_error, n_steps, slope, intercept,
                              x_sorted, bin_edges, deviations_per_bin, bin_stats_df):
    params = list(inspect.signature(generate_error_path).parameters.keys())
    if "bin_stats_df" in params:
        return generate_error_path(start_error, n_steps, bin_stats_df,
                                   slope, intercept, x_sorted, bin_edges, deviations_per_bin)
    return generate_error_path(start_error, n_steps,
                               slope, intercept, x_sorted, bin_edges, deviations_per_bin)

# ----------------------------
# Axes & ground
# ----------------------------
def _style_axes(ax, title, sensor):
    ax.set_facecolor(COLORS["background"])
    ax.set_title(title, fontsize=STYLE["title_size"], pad=10)
    ax.set_xlabel(f"{sensor} Error (n)", fontsize=STYLE["axis_label_size"])
    ax.set_ylabel(f"{sensor} Error (n+1)", fontsize=STYLE["axis_label_size"])
    ax.set_zlabel("Residual PDF", fontsize=STYLE["axis_label_size"])
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.03)
    ax.view_init(elev=25, azim=-45)

def _draw_ground_plane(ax, x_min, x_max, y_min, y_max, alpha=0.06):
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 2),
                       np.linspace(y_min, y_max, 2))
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, color=COLORS["slab"], alpha=alpha, zorder=0)

# ----------------------------
# Curtain builder
# ----------------------------
def _build_bin_curtain(ax,
                       x_sorted: np.ndarray, bin_edges: np.ndarray, i: int,
                       slope: float, intercept: float,
                       mu_i: float, sd_i: float,
                       base_alpha: float) -> Dict[str, object]:
    """Ribbon surface for the bin using TRUE bounds (no padding/min-thickness)."""
    x_min = x_sorted[bin_edges[i]]
    x_max = x_sorted[bin_edges[i + 1]]
    x_center = 0.5 * (x_min + x_max)

    y_center = slope * x_center + intercept + mu_i
    y_grid   = np.linspace(y_center - 4 * sd_i, y_center + 4 * sd_i, DENSITY["pdf_points"])
    z_pdf    = (1.0 / (sd_i * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y_grid - y_center) / sd_i) ** 2)

    X = np.vstack([np.full_like(y_grid, x_min), np.full_like(y_grid, x_max)])
    Y = np.vstack([y_grid, y_grid])
    Z = np.vstack([z_pdf,  z_pdf])

    surf = ax.plot_surface(
        X, Y, Z, linewidth=0, antialiased=False,
        alpha=base_alpha, color=COLORS["curtain_base"], shade=True
    )

    # faint side edges only
    ax.plot(np.full_like(y_grid, x_min), y_grid, z_pdf,
            linewidth=STYLE["edge_line_width"], alpha=0.5, color=COLORS["edge_line"])
    ax.plot(np.full_like(y_grid, x_max), y_grid, z_pdf,
            linewidth=STYLE["edge_line_width"], alpha=0.5, color=COLORS["edge_line"])

    mean = ax.scatter([x_center], [y_center], [0.0],
                      s=STYLE["mean_marker_size"], color=COLORS["mean_base"], marker="s")
    return {"surface": surf, "mean": mean}

# ----------------------------
# Build figure (ALL bins)
# ----------------------------
def build_all_bins_figure(sensor, xt, yt, slope, intercept, x_sorted, bin_edges, bin_stats_df):
    nb = len(bin_edges) - 1
    edges = _edge_values(x_sorted, bin_edges)

    if len(xt) > DENSITY["max_cloud_points"]:
        idx = np.linspace(0, len(xt) - 1, DENSITY["max_cloud_points"], dtype=int)
        xt_plot, yt_plot = xt[idx], yt[idx]
    else:
        xt_plot, yt_plot = xt, yt

    fig = plt.figure(figsize=(11.5, 7))
    ax  = fig.add_subplot(111, projection="3d")
    _style_axes(ax, f"D04 Bin Neighborhoods – {sensor}", sensor)

    x_gp0, x_gp1 = xt_plot.min(), xt_plot.max()
    y_gp0, y_gp1 = min(yt_plot.min(), (slope*xt_plot+intercept).min()), max(yt_plot.max(), (slope*xt_plot+intercept).max())
    _draw_ground_plane(ax, x_gp0, x_gp1, y_gp0, y_gp1, alpha=0.06)

    ax.scatter(xt_plot, yt_plot, np.zeros_like(xt_plot), s=1,
               alpha=STYLE["scatter_alpha"], color=COLORS["scatter"])
    x_line = np.linspace(edges[0], edges[-1], 250)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, np.zeros_like(x_line),
            color=COLORS["regression"], linewidth=STYLE["reg_line_width"])

    artists_by_bin: Dict[int, Dict[str, object]] = {}
    for i in range(nb):
        mu_i  = float(bin_stats_df.iloc[i]["deviation_mean"])
        var_i = float(bin_stats_df.iloc[i]["deviation_variance"])
        sd_i  = math.sqrt(max(var_i, 1e-12))
        artists_by_bin[i] = _build_bin_curtain(
            ax, x_sorted, bin_edges, i, slope, intercept, mu_i, sd_i,
            STYLE["curtain_alpha_min"]
        )

    fig.tight_layout()
    return fig, ax, artists_by_bin, edges

# ----------------------------
# Animation
# ----------------------------
def animate(fig, ax, xt, yt, x_sorted, bin_edges, artists_by_bin, edges):
    bins = np.array([_locate_bin(x, x_sorted, bin_edges) for x in xt])

    point, = ax.plot([xt[0]], [yt[0]], [0.0], marker="o", markersize=6,
                     color=COLORS["curtain_hi"], linestyle="None")
    trail, = ax.plot([], [], [], color=COLORS["curtain_hi"], linewidth=STYLE["trail_width"])
    hud = ax.text2D(0.02, 0.96, "", transform=ax.transAxes,
                    fontsize=STYLE["hud_fontsize"], color=COLORS["curtain_hi"])
    current = {"i": None}

    def _dim(i):
        if i not in artists_by_bin: return
        a = artists_by_bin[i]
        a["surface"].set_facecolor(COLORS["curtain_base"])
        a["surface"].set_alpha(STYLE["curtain_alpha_min"])
        a["mean"].set_color(COLORS["mean_base"])
        a["mean"].set_sizes([STYLE["mean_marker_size"]])

    def _hi(i):
        if i not in artists_by_bin: return
        a = artists_by_bin[i]
        a["surface"].set_facecolor(COLORS["curtain_hi"])
        a["surface"].set_alpha(STYLE["curtain_alpha_hi"])
        a["mean"].set_color(COLORS["mean_hi"])
        a["mean"].set_sizes([STYLE["mean_marker_size_hi"]])

    for i in artists_by_bin.keys(): _dim(i)

    def init():
        trail.set_data_3d([], [], [])
        return point, trail

    def update(frame):
        t = frame * ANIM["step_stride"]
        if t >= len(xt): t = len(xt) - 1

        bi = bins[t]
        if bi != current["i"]:
            if current["i"] is not None: _dim(current["i"])
            _hi(bi)
            current["i"] = bi

        point.set_data_3d([xt[t]], [yt[t]], [0.0])
        t0 = max(0, t - ANIM["trail_len"])
        trail.set_data_3d(xt[t0:t+1], yt[t0:t+1], np.zeros(t - t0 + 1))

        x0, x1 = edges[bi], edges[bi+1]
        hud.set_text(f"t={t}   bin={bi}   range=[{x0:.3f}, {x1:.3f}]")

        if ANIM["camera_orbit"]:
            ax.view_init(elev=ax.elev, azim=ax.azim + ANIM["orbit_deg_per_frame"])
        return point, trail, hud

    frames = int(np.ceil(len(xt) / ANIM["step_stride"]))
    return FuncAnimation(fig, update, init_func=init, frames=frames,
                         interval=ANIM["interval_ms"], blit=False)

# ----------------------------
# Runner (identical seeding logic as Model_ALL_Simulation)
# ----------------------------
def run(sensor: str = "CAM",
        used_tows: Optional[list] = None,
        num_bins: int = 10,
        n_steps: int = 373,
        test_ratio: float = 0.5,
        seed: Optional[int] = None):
    """Visualizes the same process that Model_ALL_Simulation.py simulates."""
    if seed is not None:
        import random
        np.random.seed(seed)
        random.seed(seed)

    if used_tows is None:
        used_tows = list(range(2, 10))

    # Fit model
    (bin_stats_df, slope, intercept, _r, _p, _se,
     x_sorted, bin_edges, deviations_per_bin) = consecutive_error(
        sensor, used_tows=used_tows, test_ratio=test_ratio,
        num_bins=num_bins, bins_show=False, plot_fit=False
    )

    # --- identical to Model_ALL_Simulation.py start logic ---
    start_value = generate_starting_error(sensor)

    # Generate path
    path = _call_generate_error_path(
        start_value, n_steps, slope, intercept,
        x_sorted, bin_edges, deviations_per_bin, bin_stats_df
    )
    path = np.asarray(path, float)
    xt, yt = path[:-1], path[1:]

    fig, ax, artists_by_bin, edges = build_all_bins_figure(
        sensor, xt, yt, slope, intercept, x_sorted, bin_edges, bin_stats_df
    )

    # Create and save animation
    anim = animate(fig, ax, xt, yt, x_sorted, bin_edges, artists_by_bin, edges)

    # --- SAVE TO GIF ---
    from matplotlib.animation import PillowWriter
    from pathlib import Path
    downloads = Path.home() / "Downloads"
    gif_path = downloads / "D04_bins_3D.gif"

    anim.save(gif_path, writer=PillowWriter(fps=5))
    print(f"\n✅ Saved animation to:\n{gif_path}\n")

    plt.show()

##############################################################################################################
"""Run this file"""

def main():
    # Exactly 10 curtains; exactly 373 steps
    run(sensor="CAM", used_tows=list(range(2,10)), num_bins=10, n_steps=373, seed=42)

if __name__ == "__main__":
    main()
