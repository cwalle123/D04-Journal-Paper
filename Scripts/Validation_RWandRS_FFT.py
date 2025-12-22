#!/usr/bin/env python3
"""
Written by: Giovanni Zattoni

---------------------------------------
#!/usr/bin/env python3

Validation_RWandRS_FFT.py

FFT amplitude spectrum comparison between:
- Random-Walk model (RW) tow (MCMC / Random-Walk)
- Experimental traverse tow
- Random Sampling model (RS) tow (MC / Random Sampling)

This version:
- Tukey window (alpha=0.05) when USE_WINDOW=True
- DETREND / WINDOW / PADDING toggles at the top
- Writes FFT metrics CSV for Tow 2..30 + AVG row
- NEW: Exports BOTH the RW list and RS list as TWO SEPARATE CSV FILES
  (each list is a list of pandas DataFrames; we concatenate them with a sample_index column)
"""

# ======================================================================
# External imports
# ======================================================================
import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal.windows import tukey
from constants import (figure_width, min_figure_height, font_TNR, font_label, font_axis_ticks, font_legend, graph_line_thickness, 
                       legend_line_thickness, annotation_thickness, annotation_stripe_height, break_marker_thickness,
                       tick_width, tick_length, graph_box_thickness, legend_box_thickness, color_exp, color_RW, color_RS,
                       color_annotations, color_PDF_fits, color_borders, color_ideal_gap, transparency, color_gap, color_overlap, legend_space)

# ======================================================================
# FFT PROCESSING TOGGLES (EDIT THESE)
# ======================================================================
USE_DETREND  = True     # True / False
USE_WINDOW   = True     # True / False  (Tukey alpha=0.05 if True)
USE_PADDING  = True     # True / False
PAD_FACTOR   = 4        # Only used if USE_PADDING = True
TUKEY_ALPHA  = 0.05     # Tukey alpha if USE_WINDOW = True

# Metrics settings
FMAX_METRICS = 300.0    # cycles/m range used for MSE, rho, dominant freq
FMIN_DOM     = 1.0      # ignore DC/very-low freq when finding dominant freq

# Batch output (metrics table)
WRITE_BATCH_CSV = True
TOW_MIN_BATCH   = 2
TOW_MAX_BATCH   = 30
OUTDIR          = "Outputs"
CSV_NAME        = "FFT_metrics_alltows_centerline.csv"  # saved in Outputs/

# NEW: export the RW/RS lists as CSV files
EXPORT_LISTS_AS_CSV = True
LIST_OUTDIR         = "Outputs"
RW_LIST_CSV_NAME    = "RW_list_MCMC.csv"
RS_LIST_CSV_NAME    = "RS_list_MC.csv"

# How many simulated tows to generate for the exported lists (and for single-tow FFT, we still use index 0)
N_LIST_TOWS_EXPORT = 1   # set to e.g. 50 if you want a bigger chain/ensemble saved to CSV

# ======================================================================
# PATH SETUP
# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ======================================================================
# IMPORT PROJECT MODULES
# ======================================================================
from Model_ALL_RandomWalk import generate_RW_multitow
from Data_ALL_traverse import traverse_tow_constructor
from Model_ALL_RandomSampling import generate_RS_multitow

# ======================================================================
# Local plotting constants (so you don't depend on constants.py)
# ======================================================================
figure_width = 9.5
min_figure_height = 2.4

color_exp = "#0072B2"   # blue
color_RW  = "#009E73"   # green
color_RS  = "#F1B047"   # orange

font_TNR = "Times New Roman"

graph_box_thickness = 1.0
tick_length = 5
tick_width = 1.0
graph_line_thickness = 1.6

color_borders = "black"

legend_line_thickness = 1.6
legend_box_thickness = 1.0
font_legend = 11

# ----------------- Global plot formatting -----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L"],
    "mathtext.fontset": "stix",
    "axes.grid": False,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

# ======================================================================
# Helpers
# ======================================================================
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

def make_window(N: int) -> np.ndarray:
    """Rectangular if USE_WINDOW=False, otherwise Tukey(alpha=TUKEY_ALPHA)."""
    if not USE_WINDOW:
        return np.ones(N, dtype=float)
    return tukey(N, alpha=TUKEY_ALPHA)

def one_sided_amplitude_spectrum(y, dx_m):
    """
    One-sided amplitude spectrum with coherent-gain correction.
    Uses global toggles: USE_WINDOW (Tukey), USE_PADDING (PAD_FACTOR)
    """
    y = np.asarray(y, dtype=float)
    N = len(y)

    w = make_window(N)
    y_win = y * w

    Nfft = int(PAD_FACTOR * N) if USE_PADDING else int(N)

    Y = np.fft.rfft(y_win, n=Nfft)
    f_cperm = np.fft.rfftfreq(Nfft, d=dx_m)

    # Coherent gain normalization
    CG = w.sum() / N
    A = np.abs(Y) / (N * CG)

    # One-sided amplitude correction (preserve DC and Nyquist if even)
    if Nfft % 2 == 0:
        A[1:-1] *= 2
    else:
        A[1:] *= 2

    return f_cperm, A

def resample_and_fft(x_src_mm, y_src, x_grid_mm):
    """Resample to common grid, optional detrend, then FFT."""
    yg = interp_to_grid(x_src_mm, y_src, x_grid_mm)
    if USE_DETREND:
        yg = linear_detrend(yg, x_grid_mm * 1e-3)  # detrend vs meters
    dx_m = (x_grid_mm[1] - x_grid_mm[0]) * 1e-3
    return one_sided_amplitude_spectrum(yg, dx_m)

def compute_mse_and_rho(f_exp, A_exp, f_mod, A_mod, fmax=300.0):
    """
    Compare model spectrum to experimental spectrum over [0, fmax].
    Returns (mse, rho).
    """
    m = (f_exp >= 0.0) & (f_exp <= fmax)
    f = f_exp[m]
    Ae = A_exp[m]
    Am = np.interp(f, f_mod, A_mod)

    mse = float(np.mean((Am - Ae) ** 2))

    # Pearson correlation; handle degenerate cases
    if np.std(Ae) == 0.0 or np.std(Am) == 0.0:
        rho = np.nan
    else:
        rho = float(np.corrcoef(Ae, Am)[0, 1])
    return mse, rho

def dominant_frequency(f_exp, A_exp, fmin=1.0, fmax=300.0):
    """
    Dominant frequency of EXPERIMENT spectrum in [fmin, fmax].
    """
    m = (f_exp >= fmin) & (f_exp <= fmax)
    if not np.any(m):
        return np.nan
    f = f_exp[m]
    A = A_exp[m]
    if len(A) == 0:
        return np.nan
    return float(f[int(np.argmax(A))])

def save_df_list_to_csv(df_list, outpath, index_col_name="sample_index"):
    """
    Save a list of pandas DataFrames to ONE CSV file by concatenation.
    Adds a column (sample_index) so you know which list element each row came from.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required to export the tow lists to CSV. Install it or disable EXPORT_LISTS_AS_CSV.") from e

    if df_list is None or len(df_list) == 0:
        raise ValueError(f"Cannot save empty list to CSV: {outpath}")

    frames = []
    for i, df in enumerate(df_list):
        dfi = df.copy()
        dfi.insert(0, index_col_name, i)
        frames.append(dfi)

    big = pd.concat(frames, axis=0, ignore_index=True)
    big.to_csv(outpath, index=False)

# ======================================================================
# Data extraction
# ======================================================================
def extract_rw_edges_centerline(num_tows=1, seed=42):
    """
    Returns dict:
      - "list": full RW output list (MCMC / Random-Walk)
      - arrays for FIRST tow in list (index 0): x_mm, top, bottom, centerline
    """
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)

    _, _, _, _, _, rw_list = generate_RW_multitow(num_tows=num_tows, proposal_type="RWM")
    tow_df = rw_list[0]

    return {
        "list": rw_list,
        "x_mm": tow_df["x_mm"].to_numpy(),
        "top": tow_df["top_edge"].to_numpy(),
        "bottom": tow_df["bottom_edge"].to_numpy(),
        "centerline": tow_df["centerline"].to_numpy(),
    }

def extract_rs_edges_centerline(
    num_tows=1,
    n_steps=1001,
    tow_width_mm=6.35,
    tow_length_mm=1000.0,
    method="Sidd",
    seed=1234
):
    """
    Returns dict:
      - "list": full RS output list (MC / Random Sampling)
      - arrays for FIRST tow in list (index 0): x_mm, top, bottom, centerline
    """
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)

    _, RS_all_tows_data, _, _ = generate_RS_multitow(
        num_tows=num_tows,
        n_steps=n_steps,
        tow_spacing_mm=tow_width_mm,
        tow_width_mm=tow_width_mm,
        tow_length_mm=tow_length_mm,
        method=method,
        print_statement=False
    )
    tow_df = RS_all_tows_data[0]

    return {
        "list": RS_all_tows_data,
        "x_mm": tow_df["x_mm"].to_numpy(),
        "top": tow_df["top_edge"].to_numpy(),
        "bottom": tow_df["bottom_edge"].to_numpy(),
        "centerline": tow_df["centerline"].to_numpy(),
    }

def extract_experimental_edges_centerline(tow, normalize=True):
    df = traverse_tow_constructor(tow, normalize=normalize)
    if df is None:
        return None
    return ((df["x_left"].to_numpy(),       df["y_left"].to_numpy()),
            (df["x_right"].to_numpy(),      df["y_right"].to_numpy()),
            (df["x_centerline"].to_numpy(), df["y_centerline"].to_numpy()))

# ======================================================================
# Single-tow comparison routine (plots + prints)
# ======================================================================
def run_fft_compare(
    tow=7,
    rw_seed=42,
    rs_seed=1234,
    rs_method="Sidd",
    show_plots=True,
    show_loglog=False,
    save_PDF=True
):
    x_grid_mm = np.arange(0.0, 1000.0 + 1.0, 1.0)

    exp_data = extract_experimental_edges_centerline(tow, normalize=True)
    if exp_data is None:
        raise ValueError("traverse_tow_constructor returned None; choose tow in [2..30] for experimental data.")

    (_, _), (_, _), (x_ctr_exp, y_ctr_exp) = exp_data

    # Generate lists (exportable), but FFT uses element 0
    rw = extract_rw_edges_centerline(num_tows=N_LIST_TOWS_EXPORT, seed=rw_seed)
    rs = extract_rs_edges_centerline(
        num_tows=N_LIST_TOWS_EXPORT,
        n_steps=len(x_grid_mm),
        tow_width_mm=6.35,
        tow_length_mm=float(x_grid_mm[-1]),
        method=rs_method,
        seed=rs_seed
    )

    # Arrays for FFT (first element)
    x_rw, y_ctr_rw = rw["x_mm"], rw["centerline"]
    x_rs, y_ctr_rs = rs["x_mm"], rs["centerline"]

    print(f"\nModel list outputs:")
    print(f"  RW (MCMC) list length: {len(rw['list'])}")
    print(f"  RS (MC)   list length: {len(rs['list'])}")

    f_ctr_exp, A_ctr_exp = resample_and_fft(x_ctr_exp, y_ctr_exp, x_grid_mm)
    f_ctr_rw,  A_ctr_rw  = resample_and_fft(x_rw,      y_ctr_rw,  x_grid_mm)
    f_ctr_rs,  A_ctr_rs  = resample_and_fft(x_rs,      y_ctr_rs,  x_grid_mm)

    mse_rw, rho_rw = compute_mse_and_rho(f_ctr_exp, A_ctr_exp, f_ctr_rw, A_ctr_rw, fmax=FMAX_METRICS)
    mse_rs, rho_rs = compute_mse_and_rho(f_ctr_exp, A_ctr_exp, f_ctr_rs, A_ctr_rs, fmax=FMAX_METRICS)
    dom_f = dominant_frequency(f_ctr_exp, A_ctr_exp, fmin=FMIN_DOM, fmax=FMAX_METRICS)

    print(f"\nTow {tow} CENTERLINE FFT validation (vs Experiment)")
    print(f"  Settings: detrend={USE_DETREND}, window(Tukey a={TUKEY_ALPHA})={USE_WINDOW}, padding={USE_PADDING} (PAD_FACTOR={PAD_FACTOR})")
    print(f"  Dominant freq (EXP) [cycles/m]: {dom_f:.6f}")
    print("  RW model:  MSE={:.6e}, rho={}".format(mse_rw, "nan" if np.isnan(rho_rw) else f"{rho_rw:.6f}"))
    print("  RS model:  MSE={:.6e}, rho={}".format(mse_rs, "nan" if np.isnan(rho_rs) else f"{rho_rs:.6f}"))

    def plot_linear(f_exp, A_exp, f_rw, A_rw, f_rs, A_rs):
        fig, ax = plt.subplots(1, 1, figsize=(figure_width, 2*min_figure_height))
        ax.plot(f_exp, A_exp, label="Experimental",    color=color_exp, linewidth=graph_line_thickness, linestyle="-")
        ax.plot(f_rw,  A_rw,  label="MCMC simulation", color=color_RW,  linewidth=graph_line_thickness, linestyle="-")
        ax.plot(f_rs,  A_rs,  label="MC simulation",   color=color_RS,  linewidth=graph_line_thickness, linestyle="-")
        ax.set_xlabel("Spatial frequency (cycles/m)")
        ax.set_ylabel("Amplitude (mm)")
        ax.grid(False)
        ax.set_xlim(0, 50)

        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = [font_TNR]
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['xtick.labelsize'] = font_axis_ticks
        mpl.rcParams['ytick.labelsize'] = font_axis_ticks

        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(top=True, bottom=True, left=True, right=True, direction='in',
                       length=tick_length, width=tick_width)

        for spine in ax.spines.values():
            spine.set_linewidth(graph_box_thickness)
            spine.set_edgecolor(color_borders)

        handles, labels = ax.get_legend_handles_labels()
        fig.subplots_adjust(bottom=0.30)
        legend = fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=font_legend, fancybox=False)
        for legobj in legend.legend_handles:
            legobj.set_linewidth(legend_line_thickness)
        frame = legend.get_frame()
        frame.set_edgecolor(color_borders)
        frame.set_linewidth(legend_box_thickness)
        frame.set_facecolor('white')

    def plot_loglog(f_exp, A_exp, f_rw, A_rw, f_rs, A_rs):
        m_exp = f_exp > 0
        m_rw  = f_rw  > 0
        m_rs  = f_rs  > 0
        plt.figure()
        plt.loglog(f_exp[m_exp], A_exp[m_exp], label="Experimental",    color=color_exp, linewidth=graph_line_thickness, linestyle="-")
        plt.loglog(f_rw[m_rw],   A_rw[m_rw],   label="MCMC simulation", color=color_RW,  linewidth=graph_line_thickness, linestyle="--")
        plt.loglog(f_rs[m_rs],   A_rs[m_rs],   label="MC simulation",   color=color_RS,  linewidth=graph_line_thickness, linestyle="-.")
        plt.xlabel("Spatial frequency (cycles/m)")
        plt.ylabel("Amplitude (mm)")
        plt.grid(False, which="both")
        plt.legend(frameon=False)
        plt.tight_layout()

    if show_plots:
        plot_linear(f_ctr_exp, A_ctr_exp, f_ctr_rw, A_ctr_rw, f_ctr_rs, A_ctr_rs)

    if show_loglog:
        plot_loglog(f_ctr_exp, A_ctr_exp, f_ctr_rw, A_ctr_rw, f_ctr_rs, A_ctr_rs)

    if save_PDF:
        plt.savefig("FFT_RS_RW_2.pdf", format="pdf", bbox_inches=None)

    if show_plots or show_loglog:
        plt.show()

    # Return BOTH lists so main() can export them to CSV
    return rw["list"], rs["list"]

# ======================================================================
# Batch CSV output for Tow 2..30 (centerline only)
# ======================================================================
def write_batch_csv(
    tow_min=2,
    tow_max=30,
    rw_seed=42,
    rs_seed=1234,
    rs_method="Sidd",
    outdir="Outputs",
    csv_name="FFT_metrics_alltows_centerline.csv"
):
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, csv_name)

    x_grid_mm = np.arange(0.0, 1000.0 + 1.0, 1.0)

    rows = []
    for tow in range(tow_min, tow_max + 1):
        exp_data = extract_experimental_edges_centerline(tow, normalize=True)
        if exp_data is None:
            rows.append([tow, np.nan, np.nan, np.nan])
            print(f"Tow {tow}: missing/failed (traverse_tow_constructor returned None) -> filling NaNs")
            continue

        (_, _), (_, _), (x_ctr_exp, y_ctr_exp) = exp_data

        rw = extract_rw_edges_centerline(num_tows=1, seed=rw_seed)
        # (RS computed for plotting elsewhere; metrics table stays RW vs EXP like your current logic)
        f_exp, A_exp = resample_and_fft(x_ctr_exp, y_ctr_exp, x_grid_mm)
        f_rw,  A_rw  = resample_and_fft(rw["x_mm"], rw["centerline"], x_grid_mm)

        mse, rho = compute_mse_and_rho(f_exp, A_exp, f_rw, A_rw, fmax=FMAX_METRICS)
        dom_f = dominant_frequency(f_exp, A_exp, fmin=FMIN_DOM, fmax=FMAX_METRICS)

        rows.append([tow, mse, rho, dom_f])

    mse_vals = np.array([r[1] for r in rows], dtype=float)
    rho_vals = np.array([r[2] for r in rows], dtype=float)
    dom_vals = np.array([r[3] for r in rows], dtype=float)

    avg_mse = float(np.nanmean(mse_vals)) if np.any(~np.isnan(mse_vals)) else np.nan
    avg_rho = float(np.nanmean(rho_vals)) if np.any(~np.isnan(rho_vals)) else np.nan
    avg_dom = float(np.nanmean(dom_vals)) if np.any(~np.isnan(dom_vals)) else np.nan

    rows.append(["AVG", avg_mse, avg_rho, avg_dom])

    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Tow #", "MSE", "rho", "dominant freq"])
        for r in rows:
            w.writerow(r)

    print(f"Saved: {outpath}")

# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="FFT amplitude spectrum comparison (Experiment vs RW vs RS) for centerline + batch CSV + export RW/RS lists to CSV."
    )
    p.add_argument("--tow", type=int, default=7, help="Tow index to compare (2..30 recommended).")
    p.add_argument("--rw-seed", type=int, default=42, help="Random-walk RNG seed for reproducibility.")
    p.add_argument("--rs-seed", type=int, default=1234, help="Random-sampling RNG seed for reproducibility.")
    p.add_argument("--rs-method", choices=["Sidd", "Random"], default="Sidd",
                   help="RS width generation method ('Sidd' enforces LLS_B >= LLS_A).")
    p.add_argument("--loglog", action="store_true", help="Also show log–log spectra.")
    return p.parse_args()

def main():
    args = parse_args()

    # Single tow plot + print + returns both lists
    rw_list, rs_list = run_fft_compare(
        tow=args.tow,
        rw_seed=args.rw_seed,
        rs_seed=args.rs_seed,
        rs_method=args.rs_method,
        show_plots=True,
        show_loglog=args.loglog,
        save_PDF=False
    )

    # NEW: export BOTH lists as TWO SEPARATE CSV FILES
    if EXPORT_LISTS_AS_CSV:
        os.makedirs(LIST_OUTDIR, exist_ok=True)
        rw_csv_path = os.path.join(LIST_OUTDIR, RW_LIST_CSV_NAME)
        rs_csv_path = os.path.join(LIST_OUTDIR, RS_LIST_CSV_NAME)

        save_df_list_to_csv(rw_list, rw_csv_path, index_col_name="mcmc_index")
        save_df_list_to_csv(rs_list, rs_csv_path, index_col_name="sample_index")

        print(f"Saved RW (MCMC) list CSV: {rw_csv_path}")
        print(f"Saved RS (MC)   list CSV: {rs_csv_path}")

    # Batch metrics CSV (Tow 2..30) + AVG row
    if WRITE_BATCH_CSV:
        write_batch_csv(
            tow_min=TOW_MIN_BATCH,
            tow_max=TOW_MAX_BATCH,
            rw_seed=args.rw_seed,
            rs_seed=args.rs_seed,
            rs_method=args.rs_method,
            outdir=OUTDIR,
            csv_name=CSV_NAME
        )

if __name__ == "__main__":
    main()


