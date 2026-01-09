#!/usr/bin/env python3
"""
plot_spatial_treated_signals.py

Spatial-domain treated signal plots (centerline only), like-for-like with your FFT preprocessing math.

Creates 3 figures (PDFs):
  3) Spatial-domain treated signal: Experimental only
  4) Spatial-domain treated signal: Random Walk only
  5) Spatial-domain treated signal: Random Sampling only

Each figure has 3 subplots:
  (1) Raw tow (resampled):                 yg
  (2) Raw tow with detrending (D1P0W0):    linear_detrend(yg, x_grid_mm*1e-3)
  (3) Raw tow with Tukey window (D0P0W1):  yg * tukey(N, alpha=0.05)

Defaults (as requested):
  python plot_spatial_treated_signals.py --tow 7 --rw-seed 42 --rs-seed 1234 --rs-method Sidd
These are already the internal defaults, so running with no args is equivalent.

NOTE on windowing choice:
- For the spatial-domain "physical effect" plot, we show yg*w (NOT divided by coherent gain CG).
  CG is an FFT amplitude normalization; dividing by CG in spatial domain would re-scale the signal
  and reduce the intuitive taper effect you want to visualize.

Output:
- Saves 3 PDFs to Outputs/ by default.
- X-axis fixed to 0..1000 mm.
- Y-axis autoscaled but consistent across the 3 subplots within each figure.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

##############################################################################################################
# Path setup (match your pattern)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

##############################################################################################################
# Internal imports (same sources as your FFT script)

from Model_ALL_RandomWalk import generate_RW_multitow
from Data_ALL_traverse import traverse_tow_constructor
from Model_ALL_RandomSampling import generate_RS_multitow

# Optional styling constants (fallback if not available)
try:
    from constants import (
        font_TNR, font_label, font_axis_ticks, font_legend,
        color_exp, color_RS, color_RW,
    )
except Exception:
    font_TNR = "Times New Roman"
    font_label = 12
    font_axis_ticks = 10
    font_legend = 10
    color_exp = "black"
    color_RW = "tab:orange"
    color_RS = "tab:blue"

##############################################################################################################
# Processing settings (fixed per your requirements)

TUKEY_ALPHA = 0.05   # same alpha used in FFT code when window enabled
X_START_MM  = 0.0
X_END_MM    = 1000.0
DX_MM       = 1.0    # x_grid_mm like-for-like

##############################################################################################################
# Helpers (copied like-for-like from your FFT script)

def linear_detrend(y, x=None):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    A = np.vstack([x, np.ones(n)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return y - (A @ coef)

def interp_to_grid(x_src, y_src, x_grid):
    return np.interp(x_grid, x_src, y_src)

##############################################################################################################
# Data extraction (single tow; same logic as your FFT script)

def extract_rw_centerline(seed=42):
    """
    Generate ONE RW tow and return arrays: x_mm, centerline
    """
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)

    # generate exactly one tow
    _, _, _, _, _, rw_list = generate_RW_multitow(num_tows=1, proposal_type="RWM")
    tow_df = rw_list[0]

    return tow_df["x_mm"].to_numpy(), tow_df["centerline"].to_numpy()

def extract_rs_centerline(
    n_steps=1001,
    tow_width_mm=6.35,
    tow_length_mm=1000.0,
    method="Sidd",
    seed=1234
):
    """
    Generate ONE RS tow and return arrays: x_mm, centerline
    """
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)

    _, RS_all_tows_data, _, _ = generate_RS_multitow(
        num_tows=1,
        n_steps=n_steps,
        tow_spacing_mm=tow_width_mm,
        tow_width_mm=tow_width_mm,
        tow_length_mm=tow_length_mm,
        method=method,
        print_statement=False
    )
    tow_df = RS_all_tows_data[0]

    return tow_df["x_mm"].to_numpy(), tow_df["centerline"].to_numpy()

def extract_experimental_centerline(tow, normalize=True):
    """
    Extract experimental centerline for one tow.
    Returns (x_mm, y_mm) or None.
    """
    df = traverse_tow_constructor(tow, normalize=normalize)
    if df is None:
        return None
    return df["x_centerline"].to_numpy(), df["y_centerline"].to_numpy()

##############################################################################################################
# Plotting + preprocessing

def setup_matplotlib():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [font_TNR]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.labelsize"] = font_label
    plt.rcParams["xtick.labelsize"] = font_axis_ticks
    plt.rcParams["ytick.labelsize"] = font_axis_ticks
    plt.rcParams["legend.fontsize"] = font_legend

def treated_signals_on_grid(x_src_mm, y_src_mm, x_grid_mm):
    """
    Like-for-like with FFT preprocessing:
      - always resample to x_grid_mm
      - detrend uses x in meters: x_grid_mm*1e-3
      - window uses Tukey(alpha=TUKEY_ALPHA) applied to (non-detrended) yg
    """
    yg = interp_to_grid(x_src_mm, y_src_mm, x_grid_mm)   # raw resampled
    yd = linear_detrend(yg, x_grid_mm * 1e-3)            # exact same detrend (x in meters)
    w  = tukey(len(yg), alpha=TUKEY_ALPHA)
    yw = yg * w                                          # physical window effect
    return yg, yd, yw

def plot_spatial_triptych(x_grid_mm, yg, yd, yw, title, color, outpath_pdf):
    """
    Plot 3 stacked subplots with consistent y-limits within the figure.
    """
    y_all = np.concatenate([yg, yd, yw])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y_lim = (y_min - pad, y_max + pad)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, y=0.98)

    axes[0].plot(x_grid_mm, yg, color=color, linewidth=1.2)
    axes[0].set_ylabel("y (mm)")
    axes[0].set_title("1) Raw (resampled) centerline: yg")
    axes[0].set_ylim(*y_lim)

    axes[1].plot(x_grid_mm, yd, color=color, linewidth=1.2)
    axes[1].set_ylabel("y (mm)")
    axes[1].set_title("2) Detrended (D1P0W0): linear_detrend(yg, x[m])")
    axes[1].set_ylim(*y_lim)

    axes[2].plot(x_grid_mm, yw, color=color, linewidth=1.2)
    axes[2].set_ylabel("y (mm)")
    axes[2].set_title(f"3) Tukey window applied (D0P0W1): yg * tukey(alpha={TUKEY_ALPHA})")
    axes[2].set_ylim(*y_lim)
    axes[2].set_xlabel("x (mm)")
    axes[2].set_xlim(X_START_MM, X_END_MM)

    for ax in axes:
        ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath_pdf, format="pdf")
    return fig

##############################################################################################################
# CLI + main

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot spatial-domain treated centerline signals (Exp / RW / RS), like-for-like with FFT preprocessing."
    )
    # Defaults exactly as you requested:
    p.add_argument("--tow", type=int, default=7, help="Tow index to plot (2..30 recommended).")
    p.add_argument("--rw-seed", type=int, default=42, help="RW RNG seed.")
    p.add_argument("--rs-seed", type=int, default=1234, help="RS RNG seed.")
    p.add_argument("--rs-method", choices=["Sidd", "Random"], default="Sidd", help="RS method.")
    p.add_argument("--outdir", type=str, default="Outputs", help="Directory to save PDFs.")
    p.add_argument("--no-show", action="store_true", help="Do not display figures (still saves PDFs).")
    return p.parse_args()

def main():
    args = parse_args()
    setup_matplotlib()
    os.makedirs(args.outdir, exist_ok=True)

    # Like-for-like common grid
    x_grid_mm = np.arange(X_START_MM, X_END_MM + DX_MM, DX_MM)

    # --- Experimental
    exp = extract_experimental_centerline(args.tow, normalize=True)
    if exp is None:
        raise ValueError(
            "traverse_tow_constructor returned None; choose tow in [2..30] for experimental data."
        )
    x_exp_mm, y_exp = exp
    yg_exp, yd_exp, yw_exp = treated_signals_on_grid(x_exp_mm, y_exp, x_grid_mm)

    # --- Random Walk
    x_rw_mm, y_rw = extract_rw_centerline(seed=args.rw_seed)
    yg_rw, yd_rw, yw_rw = treated_signals_on_grid(x_rw_mm, y_rw, x_grid_mm)

    # --- Random Sampling
    x_rs_mm, y_rs = extract_rs_centerline(
        n_steps=len(x_grid_mm),
        tow_width_mm=6.35,
        tow_length_mm=float(x_grid_mm[-1]),
        method=args.rs_method,
        seed=args.rs_seed
    )
    yg_rs, yd_rs, yw_rs = treated_signals_on_grid(x_rs_mm, y_rs, x_grid_mm)

    # Save PDFs
    out_exp = os.path.join(args.outdir, f"Spatial_Treated_EXP_tow{args.tow}.pdf")
    out_rw  = os.path.join(args.outdir, f"Spatial_Treated_RW_tow{args.tow}_seed{args.rw_seed}.pdf")
    out_rs  = os.path.join(args.outdir, f"Spatial_Treated_RS_tow{args.tow}_seed{args.rs_seed}_{args.rs_method}.pdf")

    fig1 = plot_spatial_triptych(
        x_grid_mm, yg_exp, yd_exp, yw_exp,
        title=f"Spatial-domain treated signal — EXP only (Tow {args.tow})",
        color=color_exp,
        outpath_pdf=out_exp
    )
    fig2 = plot_spatial_triptych(
        x_grid_mm, yg_rw, yd_rw, yw_rw,
        title=f"Spatial-domain treated signal — RW only (Tow {args.tow}, Seed {args.rw_seed})",
        color=color_RW,
        outpath_pdf=out_rw
    )
    fig3 = plot_spatial_triptych(
        x_grid_mm, yg_rs, yd_rs, yw_rs,
        title=f"Spatial-domain treated signal — RS only (Tow {args.tow}, Seed {args.rs_seed}, Method {args.rs_method})",
        color=color_RS,
        outpath_pdf=out_rs
    )

    print("Saved PDFs:")
    print(" ", out_exp)
    print(" ", out_rw)
    print(" ", out_rs)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

if __name__ == "__main__":
    main()
