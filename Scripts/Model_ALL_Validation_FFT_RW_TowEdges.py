#!/usr/bin/env python3
"""

Written by: Giovanni Zattoni

Model_ALL_Validation_FFT_RW_TowEdges.py
---------------------------------------
FFT amplitude spectrum comparison between:
- Random-Walk model (RW) tow (top/bottom edges + centerline)
- Experimental traverse tow (top/bottom edges + centerline)

Features:
- Choose tow via CLI (--tow N)
- Common grid (0–1000 mm, Δx = 1 mm)
- Linear detrend, Hann window, zero-padding ×4
- One-sided amplitude spectrum (cycles/m)
- Compare TOP / BOTTOM / CENTERLINE
- Compute total spectral Mean Squared Error (MSE)
- Optional: --loglog for extra log–log spectra
- Reproducible RW (fixed seed)
- Publication formatting (Times New Roman, colorblind-safe blue/green, no titles)
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

# ---------- Palette (colorblind-safe: Okabe–Ito) ----------
COLORS = {
    "exp":   "#0072B2",  # Deep blue (Experiment)
    "model": "#009E73",  # Emerald green (Model / RW)
}
LINEWIDTH = 1.6

# ----------------- Ensure imports work -----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------- Project imports -----------------
from Model_ALL_RandomWalk import generate_RW_multitow
from Data_ALL_traverse import traverse_tow_constructor


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


# ----------------- Data extraction -----------------
def extract_model_edges_centerline(num_tows=1, seed=42):
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    _, _, _, _, _, rw_list = generate_RW_multitow(num_tows=num_tows, proposal_type="RWM")
    tow_df = rw_list[0]
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


# ----------------- Main comparison routine -----------------
def run_fft_compare(tow=5, seed=42, show_plots=True, show_loglog=False):
    """Build FFT comparisons for TOP, BOTTOM, and CENTERLINE (Experiment vs Model)."""
    # Common 0..1000 mm grid, Δx = 1 mm
    x_grid_mm = np.arange(0.0, 1000.0 + 1.0, 1.0)

    # --- Load data ---
    (x_top_exp, y_top_exp), (x_bot_exp, y_bot_exp), (x_ctr_exp, y_ctr_exp) = \
        extract_experimental_edges_centerline(tow, normalize=True)
    x_mod, y_top_mod, y_bot_mod, y_ctr_mod = extract_model_edges_centerline(num_tows=1, seed=seed)

    detrend_flag, window_name, pad_factor = True, 'hann', 4

    # --- Compute spectra ---
    f_top_exp, A_top_exp = resample_and_fft(x_top_exp, y_top_exp, x_grid_mm, detrend_flag, window_name, pad_factor)
    f_top_mod, A_top_mod = resample_and_fft(x_mod,     y_top_mod, x_grid_mm, detrend_flag, window_name, pad_factor)

    f_bot_exp, A_bot_exp = resample_and_fft(x_bot_exp, y_bot_exp, x_grid_mm, detrend_flag, window_name, pad_factor)
    f_bot_mod, A_bot_mod = resample_and_fft(x_mod,     y_bot_mod, x_grid_mm, detrend_flag, window_name, pad_factor)

    f_ctr_exp, A_ctr_exp = resample_and_fft(x_ctr_exp, y_ctr_exp, x_grid_mm, detrend_flag, window_name, pad_factor)
    f_ctr_mod, A_ctr_mod = resample_and_fft(x_mod,     y_ctr_mod, x_grid_mm, detrend_flag, window_name, pad_factor)

    # --- Plotters (Okabe–Ito colors) ---
    def plot_linear(f_exp, A_exp, f_mod, A_mod, title_suffix=""):
        plt.figure()
        plt.plot(f_exp, A_exp,  label="Experiment", color=COLORS["exp"],   linewidth=LINEWIDTH, linestyle="-")
        plt.plot(f_mod, A_mod,  label="Model (RW)", color=COLORS["model"], linewidth=LINEWIDTH, linestyle="-")
        plt.xlabel("Spatial frequency (cycles/m)")
        plt.ylabel("Amplitude (mm)")
        plt.grid(False)
        plt.legend(frameon=False)
        plt.tight_layout()

    def plot_loglog(f_exp, A_exp, f_mod, A_mod, title_suffix=""):
        m1, m2 = f_exp > 0, f_mod > 0
        plt.figure()
        plt.loglog(f_exp[m1], A_exp[m1], label="Experiment", color=COLORS["exp"],   linewidth=LINEWIDTH, linestyle="-")
        plt.loglog(f_mod[m2], A_mod[m2], label="Model (RW)", color=COLORS["model"], linewidth=LINEWIDTH, linestyle="--")
        plt.xlabel("Spatial frequency (cycles/m)")
        plt.ylabel("Amplitude (mm)")
        plt.grid(False, which="both")
        plt.legend(frameon=False)
        plt.tight_layout()

    if show_plots:
        # Linear amplitude spectra
        plot_linear(f_top_exp, A_top_exp, f_top_mod, A_top_mod, "Top")
        plot_linear(f_bot_exp, A_bot_exp, f_bot_mod, A_bot_mod, "Bottom")
        plot_linear(f_ctr_exp, A_ctr_exp, f_ctr_mod, A_ctr_mod, "Centerline")

    if show_loglog:
        # Log–log spectra
        plot_loglog(f_top_exp, A_top_exp, f_top_mod, A_top_mod, "Top")
        plot_loglog(f_bot_exp, A_bot_exp, f_bot_mod, A_bot_mod, "Bottom")
        plot_loglog(f_ctr_exp, A_ctr_exp, f_ctr_mod, A_ctr_mod, "Centerline")

    # --- Compute Mean Squared Errors ---
    def compute_mse(f_exp, A_exp, f_mod, A_mod):
        A_mod_i = np.interp(f_exp, f_mod, A_mod)
        return np.mean((A_mod_i - A_exp) ** 2)

    mse_top = compute_mse(f_top_exp, A_top_exp, f_top_mod, A_top_mod)
    mse_bot = compute_mse(f_bot_exp, A_bot_exp, f_bot_mod, A_bot_mod)
    mse_ctr = compute_mse(f_ctr_exp, A_ctr_exp, f_ctr_mod, A_ctr_mod)

    print(f"\nTow {tow} FFT comparison — Mean Squared Error:")
    print(f"TOP edge:      {mse_top:.6f}")
    print(f"BOTTOM edge:   {mse_bot:.6f}")
    print(f"CENTERLINE:    {mse_ctr:.6f}")

    if show_plots or show_loglog:
        plt.show()


# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="FFT amplitude spectrum comparison (RW vs Traverse) for Top/Bottom edges and Centerline."
    )
    p.add_argument("--tow", type=int, default=16, help="Tow index to compare (2..30 recommended).")
    p.add_argument("--seed", type=int, default=42, help="Random-walk RNG seed for reproducibility.")
    p.add_argument("--loglog", action="store_true", help="Also show log–log spectra (in addition to linear plots).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fft_compare(tow=args.tow, seed=args.seed, show_plots=True, show_loglog=args.loglog)
