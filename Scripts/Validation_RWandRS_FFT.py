#!/usr/bin/env python3
"""
Written by: Giovanni Zattoni

---------------------------------------
#!/usr/bin/env python3

Validation_RWandRS_FFT.py

Explanation:
- Each tow row uses a truly-random seed (new each run)
- RW and RS share the SAME seed for that tow row
- If EXP tow is missing -> fill NaNs and do NOT generate a seed
- CSV format unchanged
"""

# ======================================================================
# Imports
# ======================================================================
import argparse
import csv
import os
import sys
import secrets
import random  # <<< NEW (needed to save/restore Python RNG state)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

# ======================================================================
# PROCESSING TOGGLES
# ======================================================================
USE_DETREND      = False
USE_TUKEY_WINDOW = False
USE_PADDING      = False

TUKEY_ALPHA = 0.05
PAD_FACTOR  = 4

FMAX_METRICS = 300.0
FMIN_DOM     = 1.0

# Tow range
TOW_MIN = 1
TOW_MAX = 31

# RS settings
RS_METHOD = "Sidd"      # "Sidd" or "Random"

# Output
OUTDIR   = "Outputs"
CSV_EXP  = "FFT_metrics_EXP_centerline.csv"
CSV_RW   = "FFT_metrics_RW_centerline.csv"
CSV_RS   = "FFT_metrics_RS_centerline.csv"

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
from Model_ALL_RandomSampling import generate_RS_multitow
from Data_ALL_traverse import traverse_tow_constructor

# ======================================================================
# RNG ISOLATION HELPERS (FIX)
# ======================================================================
def _rng_snapshot():
    """Capture BOTH Python and NumPy RNG global states."""
    return random.getstate(), np.random.get_state()

def _rng_restore(py_state, np_state):
    """Restore BOTH Python and NumPy RNG global states."""
    random.setstate(py_state)
    np.random.set_state(np_state)

def _run_with_seed(seed: int, fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) with global RNGs seeded to `seed`,
    then restore original RNG states so nothing leaks to EXP.
    """
    py_state, np_state = _rng_snapshot()
    try:
        random.seed(seed)
        np.random.seed(seed)
        return fn(*args, **kwargs)
    finally:
        _rng_restore(py_state, np_state)

# ======================================================================
# FFT helpers
# ======================================================================
def linear_detrend(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return y - (A @ coef)

def make_window(N: int) -> np.ndarray:
    if USE_TUKEY_WINDOW:
        return tukey(N, alpha=TUKEY_ALPHA)
    return np.ones(N, dtype=float)

def one_sided_amplitude_spectrum(y, dx_m):
    y = np.asarray(y, dtype=float)
    N = len(y)

    w = make_window(N)
    y_win = y * w

    Nfft = int(PAD_FACTOR * N) if USE_PADDING else int(N)
    Y = np.fft.rfft(y_win, n=Nfft)
    f = np.fft.rfftfreq(Nfft, d=dx_m)

    # Coherent gain correction
    CG = w.sum() / N
    A = np.abs(Y) / (N * CG)

    # One-sided scaling
    if Nfft % 2 == 0:
        if len(A) > 2:
            A[1:-1] *= 2
    else:
        if len(A) > 1:
            A[1:] *= 2

    return f, A

def resample_and_fft(x_src_mm, y_src, x_grid_mm):
    x_src_mm = np.asarray(x_src_mm, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    xg = np.asarray(x_grid_mm, dtype=float)

    yg = np.interp(xg, x_src_mm, y_src)

    if USE_DETREND:
        yg = linear_detrend(yg, xg * 1e-3)

    dx_m = (xg[1] - xg[0]) * 1e-3
    return one_sided_amplitude_spectrum(yg, dx_m)

def compute_mse_and_rho(f_exp, A_exp, f_mod, A_mod, fmax=300.0):
    f_exp = np.asarray(f_exp)
    A_exp = np.asarray(A_exp)
    f_mod = np.asarray(f_mod)
    A_mod = np.asarray(A_mod)

    m = (f_exp >= 0.0) & (f_exp <= fmax)
    if not np.any(m):
        return np.nan, np.nan

    f = f_exp[m]
    Ae = A_exp[m]
    Am = np.interp(f, f_mod, A_mod)

    mse = float(np.mean((Am - Ae) ** 2))

    if np.std(Ae) == 0.0 or np.std(Am) == 0.0:
        rho = np.nan
    else:
        rho = float(np.corrcoef(Ae, Am)[0, 1])

    return mse, rho

def dominant_frequency(f, A, fmin=1.0, fmax=300.0):
    f = np.asarray(f)
    A = np.asarray(A)

    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return np.nan

    fm = f[m]
    Am = A[m]
    if len(Am) == 0:
        return np.nan

    return float(fm[int(np.argmax(Am))])

# ======================================================================
# Data extraction
# ======================================================================
def extract_experimental_centerline(tow, normalize=True):
    df = traverse_tow_constructor(tow, normalize=normalize)
    if df is None:
        return None
    return df["x_centerline"].to_numpy(), df["y_centerline"].to_numpy()

def extract_rw_centerline(seed: int):
    def _gen():
        _, _, _, _, _, rw_list = generate_RW_multitow(num_tows=1, proposal_type="RWM")
        tow_df = rw_list[0]
        return tow_df["x_mm"].to_numpy(), tow_df["centerline"].to_numpy()
    return _run_with_seed(seed, _gen)

def extract_rs_centerline(n_steps, tow_width_mm=6.35, tow_length_mm=1000.0, method="Sidd", seed: int = 1234):
    def _gen():
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
    return _run_with_seed(seed, _gen)

# ======================================================================
# Truly random per-tow row seed (RW + RS share it)
# ======================================================================
def random_row_seed_32bit() -> int:
    return secrets.randbits(32)

# ======================================================================
# Batch metrics tables: EXP + RW + RS
# ======================================================================
def write_batch_metrics_exp_rw_rs(
    tow_min=1,
    tow_max=31,
    rs_method="Sidd",
    outdir="Outputs",
    csv_exp="FFT_metrics_EXP_centerline.csv",
    csv_rw="FFT_metrics_RW_centerline.csv",
    csv_rs="FFT_metrics_RS_centerline.csv",
):
    os.makedirs(outdir, exist_ok=True)
    out_exp = os.path.join(outdir, csv_exp)
    out_rw  = os.path.join(outdir, csv_rw)
    out_rs  = os.path.join(outdir, csv_rs)

    x_grid_mm = np.arange(0.0, 1000.0 + 1.0, 1.0)

    exp_rows = []
    rw_rows  = []
    rs_rows  = []

    for tow in range(tow_min, tow_max + 1):
        exp_data = extract_experimental_centerline(tow, normalize=True)

        if exp_data is None:
            exp_rows.append([tow, np.nan, np.nan, np.nan])
            rw_rows.append([tow, np.nan, np.nan, np.nan])
            rs_rows.append([tow, np.nan, np.nan, np.nan])
            print(f"Tow {tow}: missing/failed (traverse_tow_constructor returned None) -> filling NaNs")
            continue

        x_exp, y_exp = exp_data

        # truly random per-tow seed; shared by RW and RS within the row
        row_seed = random_row_seed_32bit()

        x_rw, y_rw = extract_rw_centerline(seed=row_seed)
        x_rs, y_rs = extract_rs_centerline(
            n_steps=len(x_grid_mm),
            tow_width_mm=6.35,
            tow_length_mm=float(x_grid_mm[-1]),
            method=rs_method,
            seed=row_seed
        )

        f_exp, A_exp = resample_and_fft(x_exp, y_exp, x_grid_mm)
        f_rw,  A_rw  = resample_and_fft(x_rw,  y_rw,  x_grid_mm)
        f_rs,  A_rs  = resample_and_fft(x_rs,  y_rs,  x_grid_mm)

        dom_exp = dominant_frequency(f_exp, A_exp, fmin=FMIN_DOM, fmax=FMAX_METRICS)
        dom_rw  = dominant_frequency(f_rw,  A_rw,  fmin=FMIN_DOM, fmax=FMAX_METRICS)
        dom_rs  = dominant_frequency(f_rs,  A_rs,  fmin=FMIN_DOM, fmax=FMAX_METRICS)

        mse_rw, rho_rw = compute_mse_and_rho(f_exp, A_exp, f_rw, A_rw, fmax=FMAX_METRICS)
        mse_rs, rho_rs = compute_mse_and_rho(f_exp, A_exp, f_rs, A_rs, fmax=FMAX_METRICS)

        exp_rows.append([tow, 0.0, 1.0, dom_exp])
        rw_rows.append([tow, mse_rw, rho_rw, dom_rw])
        rs_rows.append([tow, mse_rs, rho_rs, dom_rs])

    def append_avg(rows):
        mse_vals = np.array([r[1] for r in rows], dtype=float)
        rho_vals = np.array([r[2] for r in rows], dtype=float)
        dom_vals = np.array([r[3] for r in rows], dtype=float)

        avg_mse = float(np.nanmean(mse_vals)) if np.any(~np.isnan(mse_vals)) else np.nan
        avg_rho = float(np.nanmean(rho_vals)) if np.any(~np.isnan(rho_vals)) else np.nan
        avg_dom = float(np.nanmean(dom_vals)) if np.any(~np.isnan(dom_vals)) else np.nan

        rows.append(["AVG", avg_mse, avg_rho, avg_dom])

    append_avg(exp_rows)
    append_avg(rw_rows)
    append_avg(rs_rows)

    header = ["Tow #", "MSE", "rho", "dominant freq"]

    for path, rows in [(out_exp, exp_rows), (out_rw, rw_rows), (out_rs, rs_rows)]:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    print(f"Saved: {out_exp}")
    print(f"Saved: {out_rw}")
    print(f"Saved: {out_rs}")

# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Write EXP/RW/RS FFT metric tables (Tow 1..31 + AVG)."
    )
    p.add_argument("--tow-min", type=int, default=TOW_MIN, help="Minimum tow index (default: 1).")
    p.add_argument("--tow-max", type=int, default=TOW_MAX, help="Maximum tow index (default: 31).")
    p.add_argument("--rs-method", choices=["Sidd", "Random"], default=RS_METHOD,
                   help="RS method (default: Sidd).")
    return p.parse_args()

def main():
    args = parse_args()
    write_batch_metrics_exp_rw_rs(
        tow_min=args.tow_min,
        tow_max=args.tow_max,
        rs_method=args.rs_method,
        outdir=OUTDIR,
        csv_exp=CSV_EXP,
        csv_rw=CSV_RW,
        csv_rs=CSV_RS
    )

if __name__ == "__main__":
    main()



