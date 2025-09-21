"""
FFT comparison where BOTH sides follow Model_ALL_Validation-Tow_Visualiser conventions:

REAL (Traverse):
  - Built from traverse_tow_constructor(tow)
  - Edges from traverse (x_right/y_right, x_left/y_left)
  - centerline = (y_left + y_right)/2
  - width = y_left - y_right
  - normalize by subtracting centerline[0]
  - resample to a uniform x-grid for FFT (recommended)

SIM (Model_ALL_Validation-Tow_Visualiser style):
  - CAM + LT consecutive-error paths -> centerline
  - LLS_B consecutive-error path -> width; width = 6.35 + width_error
  - edges = centerline ± 0.5*width
  - normalize by subtracting centerline[0]
"""

##############################################################################################################

# External imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Internal imports
from Data_ALL_traverse import traverse_tow_constructor
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path
import constants
# -*- coding: utf-8 -*-

##############################################################################################################
"""Run parameters"""
# ---------------- RUN PARAMS ----------------
tow_number = 8
length_tow_mm = 1000.0             # physical length used to define sim sampling
n_steps_sim = 360                   # sim samples (like Model_ALL_Validation-Tow_Visualiser's number_of_steps)
num_bins = 180                      # like Model_ALL_Validation-Tow_Visualiser's Consecutive_Error_Bins
zero_padding_factor = 2
use_seed = False
random_seed = 0
NOMINAL_WIDTH_MM = 6.35

##############################################################################################################
"""Functions"""

# ---------- REAL (TRAVERSE) ----------
def traverse_like_A(tow: int, resample_uniform: bool = True, target_steps: int | None = None):
    """
    Reconstruct a REAL tow exactly like Model_ALL_Validation-Tow_Visualiser:
      - edges from traverse_tow_constructor
      - centerline, width
      - normalization by subtracting centerline[0]
      - optional resampling to a uniform x-grid for FFT
    Returns dict with x, centerline, top_edge, bottom_edge, width.
    """
    df = traverse_tow_constructor(tow)

    # edges from traverse (Model_ALL_Validation-Tow_Visualiser source of truth)
    x_r = df["x_right"].to_numpy()
    y_r = df["y_right"].to_numpy()
    x_l = df["x_left"].to_numpy()
    y_l = df["y_left"].to_numpy()

    # choose a single x array (assume x_r ~ x_l)
    x = x_r
    centerline = 0.5 * (y_l + y_r)
    width = (y_l - y_r)

    # normalize like Model_ALL_Validation-Tow_Visualiser (subtract start value)
    offset0 = centerline[0]
    centerline = centerline - offset0
    y_l_n = y_l - offset0
    y_r_n = y_r - offset0
    top_edge = y_l_n
    bottom_edge = y_r_n

    # sort & NaN-guard
    m = np.isfinite(x) & np.isfinite(centerline)
    x, centerline, top_edge, bottom_edge, width = x[m], centerline[m], top_edge[m], bottom_edge[m], width[m]
    order = np.argsort(x)
    x, centerline, top_edge, bottom_edge, width = x[order], centerline[order], top_edge[order], bottom_edge[order], width[order]

    if resample_uniform:
        # build a uniform x-grid spanning the measured traverse length
        if target_steps is None:
            target_steps = len(x)
        x_uni = np.linspace(x[0], x[-1], target_steps)
        centerline = np.interp(x_uni, x, centerline)
        top_edge = np.interp(x_uni, x, top_edge)
        bottom_edge = np.interp(x_uni, x, bottom_edge)
        width = np.interp(x_uni, x, width)
        x = x_uni

    return {
        "x": x,
        "centerline": centerline,
        "top_edge": top_edge,
        "bottom_edge": bottom_edge,
        "width": width,
    }

# ---------- SIM ----------
def simulate_like_A(n_steps: int, num_bins: int, length_tow_mm: float, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # consecutive-error models for CAM, LT, LLS_B (same as Model_ALL_Validation-Tow_Visualiser)
    _, sc, ic, _, _, _, xsc, bec, dvc = consecutive_error("CAM",   test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False, random_state=np.random.randint(1e9))
    _, sl, il, _, _, _, xsl, bel, dvl = consecutive_error("LT",    test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False, random_state=np.random.randint(1e9))
    _, sw, iw, _, _, _, xsw, bew, dvw = consecutive_error("LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False, random_state=np.random.randint(1e9))

    # starting ranges copied from Model_ALL_Validation-Tow_Visualiser
    start_cam  = np.random.uniform(-0.75,  0.75)
    start_lt   = np.random.uniform(-0.90, -0.70)
    start_llsb = np.random.uniform(-0.21, -0.02)

    cam_path = generate_error_path(start_cam,  n_steps, sc, ic, xsc, bec, dvc)
    lt_path  = generate_error_path(start_lt,   n_steps, sl, il, xsl, bel, dvl)
    w_err    = generate_error_path(start_llsb, n_steps, sw, iw, xsw, bew, dvw)

    centerline = cam_path + lt_path
    width = NOMINAL_WIDTH_MM + w_err

    top_edge    = centerline + 0.5 * width
    bottom_edge = centerline - 0.5 * width

    # normalize like Model_ALL_Validation-Tow_Visualiser (subtract starting value)
    offset0 = centerline[0]
    centerline = centerline - offset0
    top_edge    = top_edge - offset0
    bottom_edge = bottom_edge - offset0

    x = np.linspace(0.0, length_tow_mm, n_steps)

    return {
        "x": x,
        "centerline": centerline,
        "top_edge": top_edge,
        "bottom_edge": bottom_edge,
        "width": width,
    }

# ---------- Helper: single-sided FFT ----------
def single_sided_fft(signal, fs, pad_factor=1):
    N = len(signal)
    Np = int(pad_factor * N)
    padded = np.pad(signal, (0, Np - N), mode="constant")
    fft_vals = np.fft.fft(padded)
    freqs = np.fft.fftfreq(Np, d=1.0/fs)
    amp = 2.0 * np.abs(fft_vals) / Np
    phase = np.angle(fft_vals)
    mask = freqs > 0
    return freqs[mask], amp[mask], phase[mask]

# ---------- Build both tows ----------
real = traverse_like_A(tow_number, resample_uniform=True, target_steps=None)  # uniform resample to len(x)
sim  = simulate_like_A(n_steps=n_steps_sim, num_bins=num_bins, length_tow_mm=length_tow_mm,
                       seed=(random_seed if use_seed else None))

# Sampling rates (samples per mm)
dx_real = (real["x"][-1] - real["x"][0]) / (len(real["x"]) - 1)
fs_real = 1.0 / dx_real
dx_sim  = length_tow_mm / n_steps_sim
fs_sim  = 1.0 / dx_sim

# ---------- FFTs on CENTERLINES ----------
f_real, A_real, P_real = single_sided_fft(real["centerline"], fs_real, pad_factor=zero_padding_factor)
f_sim,  A_sim,  P_sim  = single_sided_fft(sim["centerline"],  fs_sim,  pad_factor=zero_padding_factor)

# ---------- Compare on common frequency range ----------
fmax = min(f_real.max(), f_sim.max())
mask_c = f_real <= fmax
f_c = f_real[mask_c]
A_sim_i = np.interp(f_c, f_sim, A_sim)
P_sim_i = np.interp(f_c, f_sim, P_sim)
A_real_c = A_real[mask_c]
P_real_c = P_real[mask_c]

# ---------- Metrics ----------
mse_amp = mean_squared_error(A_real_c, A_sim_i)
phase_diff = np.angle(np.exp(1j * (P_real_c - P_sim_i)))  # wrapped difference
mse_phase = float(np.mean(phase_diff**2))

print(f"[Tow {tow_number}] MSE Amplitude Spectrum: {mse_amp:.6f}")
print(f"[Tow {tow_number}] MSE Phase Spectrum (wrapped): {mse_phase:.6f}")

# ---------- Plots ----------
plt.figure(figsize=(10,5))
plt.plot(f_c, A_real_c, label="Traverse Real FFT (centerline, A-style)", linewidth=1.5)
plt.plot(f_c, A_sim_i, "--", label="Sim FFT (A-style)", linewidth=1.5)
plt.xlabel("Spatial frequency (mm⁻¹)", fontsize=constants.font_large)
plt.ylabel("Amplitude (mm)", fontsize=constants.font_large)
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,5))
plt.plot(f_c, P_real_c, label="Traverse Real Phase (A-style)", linewidth=1.2)
plt.plot(f_c, P_sim_i,  "--", label="Sim Phase (A-style)", linewidth=1.2)
plt.xlabel("Spatial frequency (mm⁻¹)", fontsize=constants.font_large)
plt.ylabel("Phase (radians)", fontsize=constants.font_large)
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
