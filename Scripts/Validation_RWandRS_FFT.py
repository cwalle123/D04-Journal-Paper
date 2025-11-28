#!/usr/bin/env python3
"""
Written by: Giovanni Zattoni

---------------------------------------
FFT amplitude spectrum comparison between:
- Random-Walk model (RW) tow (top/bottom edges + centerline)
- Experimental traverse tow (top/bottom edges + centerline)
- Random Sampling model (RS) tow (top/bottom edges + centerline)

Features:
- Choose tow via CLI (--tow N)
- Common grid (0–1000 mm, Δx = 1 mm)
- Linear detrend, Hann window, zero-padding ×4
- One-sided amplitude spectrum (cycles/m)
- Compare TOP / BOTTOM / CENTERLINE
- Compute total spectral Mean Squared Error (MSE) vs Experiment (for RW and RS)
- Optional: --loglog for extra log–log spectra
- Reproducible RW/RS (fixed seeds)
- Publication formatting (Times New Roman; exp=blue, RW=green, RS=yellow; no titles)
"""

##############################################################################################################

# External imports
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Ensure imports work -----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Internal imports
from Model_ALL_RandomWalk import generate_RW_multitow
from Data_ALL_traverse import traverse_tow_constructor
from Model_ALL_RandomSampling import generate_RS_multitow

##############################################################################################################
"""Styling & helpers"""

# ----------------- Global plot formatting -----------------
plt.rcParams.update({
    # Fonts
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L"],
    "mathtext.fontset": "stix",

    # Axes & layout
    "axes.grid": False,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "figure.figsize": (9.5, 4.8),
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,

    # Export-friendly
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

# ---------- Palette (Okabe–Ito colorblind-safe) ----------
COLORS = {
    "exp":   "#0072B2",  # Deep blue (Experiment)
    "rw":    "#009E73",  # Emerald green (Random Walk)
    "rs":    "#F1B047",  # Orange (Random Sampling)
}
LINEWIDTH = 1.6

# ----------------- FFT helpers -----------------
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

def one_sided_amplitude_spectrum(y, dx_m, pad_factor=4, window='hann'):
    y = np.asarray(y, dtype=float)
    N = len(y)
    w = np.hanning(N) if window == 'hann' else np.ones(N)
    y_win = y * w

    Nfft = int(pad_factor * N)
    Y = np.fft.rfft(y_win, n=Nfft)
    f_cperm = np.fft.rfftfreq(Nfft, d=dx_m)
    # Coherent gain (sum window)/N normalizes amplitude properly
    CG = w.sum() / N
    A = np.abs(Y) / (N * CG)

    # One-sided amplitude correction (preserve DC and Nyquist if even)
    if Nfft % 2 == 0:
        A[1:-1] *= 2
    else:
        A[1:] *= 2
    return f_cperm, A

def resample_and_fft(x_src_mm, y_src, x_grid_mm, detrend=True, window='hann', pad_factor=4):
    yg = interp_to_grid(x_src_mm, y_src, x_grid_mm)
    if detrend:
        # Detrend against meters to keep slopes in physical units
        yg = linear_detrend(yg, x_grid_mm * 1e-3)
    dx_m = (x_grid_mm[1] - x_grid_mm[0]) * 1e-3
    return one_sided_amplitude_spectrum(yg, dx_m, pad_factor=pad_factor, window=window)

##############################################################################################################
"""Data extraction"""

def extract_rw_edges_centerline(num_tows=1, seed=42):
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    _, _, _, _, _, rw_list = generate_RW_multitow(num_tows=num_tows, proposal_type="RWM")
    tow_df = rw_list[0]
    return (tow_df["x_mm"].to_numpy(),
            tow_df["top_edge"].to_numpy(),
            tow_df["bottom_edge"].to_numpy(),
            tow_df["centerline"].to_numpy())

def extract_rs_edges_centerline(
    num_tows=1,
    n_steps=1001,              # to match 0..1000 mm @ 1 mm
    tow_width_mm=6.35,
    tow_length_mm=1000.0,
    method="Sidd",
    seed=1234
):
    # RS randomness: standardize RNG for reproducibility
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)

    # Generate one RS tow and pull its geometry
    # RS returns: gap_overlap_df, RS_all_tows_data (list of DataFrames), gap%, overlap%
    _, RS_all_tows_data, _, _ = generate_RS_multitow(
        num_tows=num_tows,
        n_steps=n_steps,
        tow_spacing_mm=tow_width_mm,   # spacing is irrelevant for single tow, but set sensibly
        tow_width_mm=tow_width_mm,
        tow_length_mm=tow_length_mm,
        method=method,
        print_statement=False
    )
    tow_df = RS_all_tows_data[0]
    return (tow_df["x_mm"].to_numpy(),
            tow_df["top_edge"].to_numpy(),
            tow_df["bottom_edge"].to_numpy(),
            tow_df["centerline"].to_numpy())

def extract_experimental_edges_centerline(tow, normalize=True):
    df = traverse_tow_constructor(tow, normalize=normalize)
    if df is None:
        raise ValueError("traverse_tow_constructor returned None; choose tow in [2..30].")
    return ((df["x_left"].to_numpy(),       df["y_left"].to_numpy()),
            (df["x_right"].to_numpy(),      df["y_right"].to_numpy()),
            (df["x_centerline"].to_numpy(), df["y_centerline"].to_numpy()))

##############################################################################################################
"""Main comparison routine"""

def run_fft_compare(
    tow=5,
    rw_seed=42,
    rs_seed=1234,
    rs_method="Sidd",
    show_plots=True,
    show_loglog=False
):
    """Build FFT comparisons for TOP, BOTTOM, and CENTERLINE (Experiment vs RW vs RS)."""
    # Common 0..1000 mm grid, Δx = 1 mm  → 1001 points
    x_grid_mm = np.arange(0.0, 1000.0 + 1.0, 1.0)

    # --- Load data ---
    (x_top_exp, y_top_exp), (x_bot_exp, y_bot_exp), (x_ctr_exp, y_ctr_exp) = \
        extract_experimental_edges_centerline(tow, normalize=True)

    x_rw, y_top_rw, y_bot_rw, y_ctr_rw = extract_rw_edges_centerline(num_tows=1, seed=rw_seed)

    # n_steps must match the grid length; tow_length matches max x of grid
    x_rs, y_top_rs, y_bot_rs, y_ctr_rs = extract_rs_edges_centerline(
        num_tows=1,
        n_steps=len(x_grid_mm),
        tow_width_mm=6.35,
        tow_length_mm=float(x_grid_mm[-1]),
        method=rs_method,
        seed=rs_seed
    )

    detrend_flag, window_name, pad_factor = True, 'hann', 4

    # --- Compute spectra (Experiment / RW / RS) ---
    # TOP
    f_top_exp, A_top_exp = resample_and_fft(x_top_exp, y_top_exp, x_grid_mm, detrend_flag, window_name, pad_factor)
    f_top_rw,  A_top_rw  = resample_and_fft(x_rw,      y_top_rw,  x_grid_mm, detrend_flag, window_name, pad_factor)
    f_top_rs,  A_top_rs  = resample_and_fft(x_rs,      y_top_rs,  x_grid_mm, detrend_flag, window_name, pad_factor)

    # BOTTOM
    f_bot_exp, A_bot_exp = resample_and_fft(x_bot_exp, y_bot_exp, x_grid_mm, detrend_flag, window_name, pad_factor)
    f_bot_rw,  A_bot_rw  = resample_and_fft(x_rw,      y_bot_rw,  x_grid_mm, detrend_flag, window_name, pad_factor)
    f_bot_rs,  A_bot_rs  = resample_and_fft(x_rs,      y_bot_rs,  x_grid_mm, detrend_flag, window_name, pad_factor)

    # CENTERLINE
    f_ctr_exp, A_ctr_exp = resample_and_fft(x_ctr_exp, y_ctr_exp, x_grid_mm, detrend_flag, window_name, pad_factor)
    f_ctr_rw,  A_ctr_rw  = resample_and_fft(x_rw,      y_ctr_rw,  x_grid_mm, detrend_flag, window_name, pad_factor)
    f_ctr_rs,  A_ctr_rs  = resample_and_fft(x_rs,      y_ctr_rs,  x_grid_mm, detrend_flag, window_name, pad_factor)

    # --- Plotters ---
    def plot_linear(f_exp, A_exp, f_rw, A_rw, f_rs, A_rs, title_suffix=""):
        plt.figure()
        plt.plot(f_exp, A_exp, label="Experimental Data", color=COLORS["exp"], linewidth=LINEWIDTH, linestyle="-")
        plt.plot(f_rw,  A_rw,  label="Random Walk", color=COLORS["rw"],  linewidth=LINEWIDTH, linestyle="-")
        plt.plot(f_rs,  A_rs,  label="Random Sampling", color=COLORS["rs"], linewidth=LINEWIDTH, linestyle="-")
        plt.xlabel("Spatial frequency (cycles/m)")
        plt.ylabel("Amplitude (mm)")
        plt.grid(False)
        plt.legend(frameon=False)
        plt.xlim(0, 300)
        plt.tight_layout()

    def plot_loglog(f_exp, A_exp, f_rw, A_rw, f_rs, A_rs, title_suffix=""):
        m_exp = f_exp > 0
        m_rw  = f_rw  > 0
        m_rs  = f_rs  > 0
        plt.figure()
        plt.loglog(f_exp[m_exp], A_exp[m_exp], label="Experimental Data",      color=COLORS["exp"], linewidth=LINEWIDTH, linestyle="-")
        plt.loglog(f_rw[m_rw],   A_rw[m_rw],   label="Random Walk",      color=COLORS["rw"],  linewidth=LINEWIDTH, linestyle="--")
        plt.loglog(f_rs[m_rs],   A_rs[m_rs],   label="Random Sampling", color=COLORS["rs"],  linewidth=LINEWIDTH, linestyle="-.")
        plt.xlabel("Spatial frequency (cycles/m)")
        plt.ylabel("Amplitude (mm)")
        plt.grid(False, which="both")
        plt.legend(frameon=False)
        plt.tight_layout()

    if show_plots:
        # Linear amplitude spectra
        plot_linear(f_top_exp, A_top_exp, f_top_rw, A_top_rw, f_top_rs, A_top_rs, "Top")
        plot_linear(f_bot_exp, A_bot_exp, f_bot_rw, A_bot_rw, f_bot_rs, A_bot_rs, "Bottom")
        plot_linear(f_ctr_exp, A_ctr_exp, f_ctr_rw, A_ctr_rw, f_ctr_rs, A_ctr_rs, "Centerline")

    if show_loglog:
        # Log–log spectra
        plot_loglog(f_top_exp, A_top_exp, f_top_rw, A_top_rw, f_top_rs, A_top_rs, "Top")
        plot_loglog(f_bot_exp, A_bot_exp, f_bot_rw, A_bot_rw, f_bot_rs, A_bot_rs, "Bottom")
        plot_loglog(f_ctr_exp, A_ctr_exp, f_ctr_rw, A_ctr_rw, f_ctr_rs, A_ctr_rs, "Centerline")

    # --- Compute Mean Squared Errors vs Experiment ---
    def compute_mse(f_exp, A_exp, f_mod, A_mod):
        A_mod_i = np.interp(f_exp, f_mod, A_mod)
        return np.mean((A_mod_i - A_exp) ** 2)

    # RW MSE
    mse_top_rw = compute_mse(f_top_exp, A_top_exp, f_top_rw, A_top_rw)
    mse_bot_rw = compute_mse(f_bot_exp, A_bot_exp, f_bot_rw, A_bot_rw)
    mse_ctr_rw = compute_mse(f_ctr_exp, A_ctr_exp, f_ctr_rw, A_ctr_rw)

    # RS MSE
    mse_top_rs = compute_mse(f_top_exp, A_top_exp, f_top_rs, A_top_rs)
    mse_bot_rs = compute_mse(f_bot_exp, A_bot_exp, f_bot_rs, A_bot_rs)
    mse_ctr_rs = compute_mse(f_ctr_exp, A_ctr_exp, f_ctr_rs, A_ctr_rs)

    print(f"\nTow {tow} FFT comparison — Mean Squared Error vs Experiment:")
    print("  RW model:")
    print(f"    TOP edge:      {mse_top_rw:.6f}")
    print(f"    BOTTOM edge:   {mse_bot_rw:.6f}")
    print(f"    CENTERLINE:    {mse_ctr_rw:.6f}")
    print("  RS model:")
    print(f"    TOP edge:      {mse_top_rs:.6f}")
    print(f"    BOTTOM edge:   {mse_bot_rs:.6f}")
    print(f"    CENTERLINE:    {mse_ctr_rs:.6f}")

    if show_plots or show_loglog:
        plt.show()

##############################################################################################################
"""CLI"""

def parse_args():
    p = argparse.ArgumentParser(
        description="FFT amplitude spectrum comparison (Experiment vs RW vs RS) for Top/Bottom edges and Centerline."
    )
    p.add_argument("--tow", type=int, default=7, help="Tow index to compare (2..30 recommended).")
    p.add_argument("--rw-seed", type=int, default=42, help="Random-walk RNG seed for reproducibility.")
    p.add_argument("--rs-seed", type=int, default=1234, help="Random-sampling RNG seed for reproducibility.")
    p.add_argument("--rs-method", choices=["Sidd", "Random"], default="Sidd",
                   help="RS width generation method ('Sidd' enforces LLS_B >= LLS_A).")
    p.add_argument("--loglog", action="store_true", help="Also show log–log spectra (in addition to linear plots).")
    return p.parse_args()

##############################################################################################################
"""Run this file"""

def main():
    args = parse_args()
    run_fft_compare(
        tow=args.tow,
        rw_seed=args.rw_seed,
        rs_seed=args.rs_seed,
        rs_method=args.rs_method,
        show_plots=True,
        show_loglog=args.loglog
    )

if __name__ == "__main__":
    main()

