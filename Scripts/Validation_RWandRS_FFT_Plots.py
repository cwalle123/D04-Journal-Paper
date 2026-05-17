#!/usr/bin/env python3
"This script compares the FFT amplitude spectra of experimental tow centerlines against Random-Walk and Random-Sampling model prediction using plots"

##############################################################################################################
# External imports
import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal.windows import tukey

##############################################################################################################
# Path setup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

##############################################################################################################
# Internal imports
from Model_ALL_RandomWalk import generate_RW_multitow
from Data_ALL_traverse import traverse_tow_constructor
from Model_ALL_RandomSampling import generate_RS_multitow
from constants import (tow_width_specified, font_label, font_axis_ticks, figure_width, 
                        color_exp, color_RS, color_RW, font_TNR, tick_length, tick_width, graph_box_thickness, font_legend,
                        legend_box_thickness, color_borders, color_annotations, color_PDF_fits, transparency,
                        legend_space, annotation_thickness, annotation_stripe_height, legend_line_thickness, graph_line_thickness,
                        color_ideal_gap, transparency, left_margin, right_margin, top_margin, bottom_margin, 
                        legend_drop, legend_margin, inter_axes_gap, unit_box_height)

##############################################################################################################
# FFT processing toggles (EDIT THESE)

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

##############################################################################################################
"""Functions"""


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


# ======================================================================
# Data extraction (SINGLE TOW ONLY; no lists)
# ======================================================================
def extract_rw_edges_centerline(seed=42):
    """
    Generate ONE RW tow and return arrays: x_mm, top, bottom, centerline
    """
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)

    # generate exactly one tow
    _, _, _, _, _, rw_list = generate_RW_multitow(num_tows=1, proposal_type="RWM")
    tow_df = rw_list[0]

    return (
        tow_df["x_mm"].to_numpy(),
        tow_df["top_edge"].to_numpy(),
        tow_df["bottom_edge"].to_numpy(),
        tow_df["centerline"].to_numpy(),
    )


def extract_rs_edges_centerline(
    n_steps=1001,
    tow_width_mm=6.35,
    tow_length_mm=1000.0,
    method="Sidd",
    seed=1234
):
    """
    Generate ONE RS tow and return arrays: x_mm, top, bottom, centerline
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

    return (
        tow_df["x_mm"].to_numpy(),
        tow_df["top_edge"].to_numpy(),
        tow_df["bottom_edge"].to_numpy(),
        tow_df["centerline"].to_numpy(),
    )


def extract_experimental_edges_centerline(tow, normalize=True):
    df = traverse_tow_constructor(tow, normalize=normalize)
    if df is None:
        return None
    return (
        (df["x_left"].to_numpy(),       df["y_left"].to_numpy()),
        (df["x_right"].to_numpy(),      df["y_right"].to_numpy()),
        (df["x_centerline"].to_numpy(), df["y_centerline"].to_numpy()),
    )


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
    save_PDF=True,
    save_SVG=False
):
    x_grid_mm = np.arange(0.0, 1000.0 + 1.0, 1.0)

    exp_data = extract_experimental_edges_centerline(tow, normalize=True)
    if exp_data is None:
        raise ValueError("traverse_tow_constructor returned None; choose tow in [2..30] for experimental data.")

    (_, _), (_, _), (x_ctr_exp, y_ctr_exp) = exp_data

    # Generate ONE RW tow and ONE RS tow (no lists exported/returned)
    x_rw, _, _, y_ctr_rw = extract_rw_edges_centerline(seed=rw_seed)
    x_rs, _, _, y_ctr_rs = extract_rs_edges_centerline(
        n_steps=len(x_grid_mm),
        tow_width_mm=6.35,
        tow_length_mm=float(x_grid_mm[-1]),
        method=rs_method,
        seed=rs_seed
    )

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
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = [font_TNR]
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['xtick.labelsize'] = font_axis_ticks
        mpl.rcParams['ytick.labelsize'] = font_axis_ticks
        mpl.rcParams['axes.labelsize'] = font_label
        mpl.rcParams['legend.fontsize'] = font_legend
        mpl.rcParams['xtick.major.width'] = tick_width
        mpl.rcParams['ytick.major.width'] = tick_width
        mpl.rcParams['xtick.major.size'] = tick_length
        mpl.rcParams['ytick.major.size'] = tick_length
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'

        # Establish correct geometry
        axes_units_per_box = 2
        n_boxes = 1
        total_axes_units = n_boxes * axes_units_per_box
        figure_height = (total_axes_units * unit_box_height + (n_boxes - 1) * inter_axes_gap 
                        + top_margin + bottom_margin + legend_margin)
        fig = plt.figure(figsize=(figure_width, figure_height))
        axes_left = left_margin / figure_width
        axes_width = 1 - (left_margin + right_margin) / figure_width
        axes_bottom = (bottom_margin + legend_margin) / figure_height
        axes_height = (axes_units_per_box * unit_box_height) / figure_height
        ax = fig.add_axes([axes_left, axes_bottom, axes_width, axes_height])

        ax.plot(f_exp, A_exp, label="Experiment",    color=color_exp, linewidth=graph_line_thickness, linestyle="-")
        ax.plot(f_rs,  A_rs,  label="MC simulation",   color=color_RS,  linewidth=graph_line_thickness, linestyle="-")
        ax.plot(f_rw,  A_rw,  label="MCMC simulation", color=color_RW,  linewidth=graph_line_thickness, linestyle="-")
        ax.set_xlabel("Spatial frequency (cycles/m)")
        ax.set_ylabel("Amplitude (mm)")
        ax.grid(False)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 0.1)

        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(top=True, bottom=True, left=True, right=True)                    # tick locations
        for spine in ax.spines.values():
            spine.set_linewidth(graph_box_thickness)                                    # black box around figure
            spine.set_edgecolor(color_borders)

        legend_ax = fig.add_axes([axes_left, (bottom_margin - legend_drop) / figure_height, 
                                axes_width, legend_margin/figure_height], frameon=False)
        legend_ax.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        legend = legend_ax.legend(handles, labels, loc='center', ncol=1, fancybox=False) #create legend with black box
        fig.canvas.draw()
        legend_bbox = legend.get_window_extent()
        legend_height_fig = legend_bbox.height / fig.bbox.height
        desired_gap = legend_drop / figure_height
        legend_ax.set_position([axes_left, axes_bottom - desired_gap - legend_height_fig, axes_width, legend_margin / figure_height])
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
    
    if save_SVG:
        plt.savefig("FFT_RS_RW_2.svg", format="svg", bbox_inches=None)

    if show_plots or show_loglog:
        plt.show()


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

        x_rw, _, _, y_ctr_rw = extract_rw_edges_centerline(seed=rw_seed)

        # (RS computed for plotting elsewhere; metrics table stays RW vs EXP like your current logic)
        f_exp, A_exp = resample_and_fft(x_ctr_exp, y_ctr_exp, x_grid_mm)
        f_rw,  A_rw  = resample_and_fft(x_rw,      y_ctr_rw,  x_grid_mm)

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


##############################################################################################################
"""Run this file"""


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="FFT amplitude spectrum comparison (Experiment vs RW vs RS) for centerline + batch CSV."
    )
    p.add_argument("--tow", type=int, default=7, help="Tow index to compare (2..30 recommended).")
    p.add_argument("--rw-seed", type=int, default=42, help="Random-walk RNG seed for reproducibility.")
    p.add_argument("--rs-seed", type=int, default=1234, help="Random-sampling RNG seed for reproducibility.")
    p.add_argument(
        "--rs-method",
        choices=["Sidd", "Random"],
        default="Sidd",
        help="RS width generation method ('Sidd' enforces LLS_B >= LLS_A)."
    )
    p.add_argument("--loglog", action="store_true", help="Also show log–log spectra.")
    return p.parse_args()


def main():
    args = parse_args()

    # Single tow plot + print
    run_fft_compare(
        tow=args.tow,
        rw_seed=args.rw_seed,
        rs_seed=args.rs_seed,
        rs_method=args.rs_method,
        show_plots=True,
        show_loglog=args.loglog,
        save_PDF=True,
        save_SVG=False
    )

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

