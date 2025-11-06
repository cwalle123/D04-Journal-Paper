"""
FFT comparison for tow EDGES (left/right) + CENTERLINE

Real (traverse) edges via traverse_tow_constructor(...)
Simulated edges built EXACTLY like the visualizer:
    centerline = CAM + LT
    width_nom  = constants.NOMINAL_WIDTH_MM (fallback 6.35)
    width      = width_nom + LLS_B_error
    sim_top    = centerline + 0.5 * width
    sim_bottom = centerline - 0.5 * width

Fixes:
  • Per-series zeroing (subtract first sample) for real & sim (edges & centerline)
  • Normalize FFT magnitudes by ORIGINAL N (not padded N)
  • Pairing SWAPPED: Top ↔ Right, Bottom ↔ Left

Written by: Giovanni Zattoni
"""

##############################################################################################################

# External imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Internal imports
from D04_Model.Model_ALL_Simulation import generate_error_path, consecutive_error
from Data_ALL_traverse import traverse_tow_constructor
import constants

##############################################################################################################
""""Functions and constants"""

# ---------------- PARAMETERS ----------------
tow_number = 3
length_tow = 1000.0      # mm
n_steps = 360
num_bins = 180
zero_padding_factor = 2
use_seed = False
random_seed = 0

# -------------- DATA LOAD (TRAVERSE EDGES) ---------------
trav = traverse_tow_constructor(tow_number, normalize=False)
x_left  = trav["x_left"].to_numpy()
y_left  = trav["y_left"].to_numpy()
x_right = trav["x_right"].to_numpy()
y_right = trav["y_right"].to_numpy()

# Sampling rates
x_pos = x_left
length_between_points = (x_pos[-1] - x_pos[0]) / (len(x_pos) - 1)
sampling_rate_real = 1.0 / length_between_points    # samples per mm
sampling_rate_sim  = n_steps / float(length_tow)    # samples per mm

# ---------------- MODEL FITTING (CAM, LT, LLS_B) ----------------
bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = consecutive_error(
    "CAM", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False)
bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = consecutive_error(
    "LT", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False)
bin_stats_llsb, slope_llsb, intercept_llsb, _, _, _, x_sorted_llsb, bin_edges_llsb, devs_llsb = consecutive_error(
    "LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False)

# ------------ SIMULATED PATHS (EDGES like visualizer) ------------
if use_seed:
    np.random.seed(random_seed)

start_cam  = np.random.uniform(-0.4,  0.6)
start_lt   = np.random.uniform(-1.0, -0.8)
start_llsb = np.random.uniform(-0.2,  0.2)

cam_path   = generate_error_path(start_cam,  n_steps, slope_cam,  intercept_cam,  x_sorted_cam,  bin_edges_cam,  devs_cam)
lt_path    = generate_error_path(start_lt,   n_steps, slope_lt,   intercept_lt,   x_sorted_lt,   bin_edges_lt,   devs_lt)
llsb_path  = generate_error_path(start_llsb, n_steps, slope_llsb, intercept_llsb, x_sorted_llsb, bin_edges_llsb, devs_llsb)

centerline = cam_path + lt_path
width_nom  = getattr(constants, "NOMINAL_WIDTH_MM", 6.35)
width      = width_nom + llsb_path

# Simulated edges
sim_top    = centerline + 0.5 * width      # ↔ Right (swapped pairing)
sim_bottom = centerline - 0.5 * width      # ↔ Left  (swapped pairing)
sim_centerline = centerline

# --------- Per-series zeroing (match visualizer style) ----------
def z0(a: np.ndarray) -> np.ndarray:
    return a - a[0]

# Real
real_left       = z0(y_left.copy())
real_right      = z0(y_right.copy())
real_centerline = z0(0.5 * (y_left + y_right))

# Sim
sim_top        = z0(sim_top.copy())
sim_bottom     = z0(sim_bottom.copy())
sim_centerline = z0(sim_centerline.copy())

# ---------------- REAL FFT (EDGES) ----------------
# Left edge
N_left = len(real_left)
N_left_padded = zero_padding_factor * N_left
left_padded = np.pad(real_left, (0, N_left_padded - N_left), mode='constant')

fft_left = np.fft.fft(left_padded)
freq_left = np.fft.fftfreq(N_left_padded, d=1.0 / sampling_rate_real)
amp_left = 2.0 * np.abs(fft_left) / N_left       # normalize by ORIGINAL N
phase_left = np.angle(fft_left)
mask_left = freq_left > 0
freq_left_pos = freq_left[mask_left]
amp_left_pos = amp_left[mask_left]
phase_left_pos = phase_left[mask_left]

# Right edge
N_right = len(real_right)
N_right_padded = zero_padding_factor * N_right
right_padded = np.pad(real_right, (0, N_right_padded - N_right), mode='constant')

fft_right = np.fft.fft(right_padded)
freq_right = np.fft.fftfreq(N_right_padded, d=1.0 / sampling_rate_real)
amp_right = 2.0 * np.abs(fft_right) / N_right    # normalize by ORIGINAL N
phase_right = np.angle(fft_right)
mask_right = freq_right > 0
freq_right_pos = freq_right[mask_right]
amp_right_pos = amp_right[mask_right]
phase_right_pos = phase_right[mask_right]

# -------------- SIMULATED FFT (EDGES) ---------------
# Top (↔ RIGHT)
N_sim_top = len(sim_top)
N_sim_top_padded = zero_padding_factor * N_sim_top
sim_top_padded = np.pad(sim_top, (0, N_sim_top_padded - N_sim_top), mode='constant')

fft_sim_top = np.fft.fft(sim_top_padded)
freq_sim_top = np.fft.fftfreq(N_sim_top_padded, d=1.0 / sampling_rate_sim)
amp_sim_top = 2.0 * np.abs(fft_sim_top) / N_sim_top   # normalize by ORIGINAL N
phase_sim_top = np.angle(fft_sim_top)
mask_sim_top = freq_sim_top > 0
freq_sim_top_pos = freq_sim_top[mask_sim_top]
amp_sim_top_pos = amp_sim_top[mask_sim_top]
phase_sim_top_pos = phase_sim_top[mask_sim_top]

# Bottom (↔ LEFT)
N_sim_bot = len(sim_bottom)
N_sim_bot_padded = zero_padding_factor * N_sim_bot
sim_bot_padded = np.pad(sim_bottom, (0, N_sim_bot_padded - N_sim_bot), mode='constant')

fft_sim_bot = np.fft.fft(sim_bot_padded)
freq_sim_bot = np.fft.fftfreq(N_sim_bot_padded, d=1.0 / sampling_rate_sim)
amp_sim_bot = 2.0 * np.abs(fft_sim_bot) / N_sim_bot  # normalize by ORIGINAL N
phase_sim_bot = np.angle(fft_sim_bot)
mask_sim_bot = freq_sim_bot > 0
freq_sim_bot_pos = freq_sim_bot[mask_sim_bot]
amp_sim_bot_pos = amp_sim_bot[mask_sim_bot]
phase_sim_bot_pos = phase_sim_bot[mask_sim_bot]

# ---------- INTERPOLATE SIM → REAL (EDGES, SWAPPED) -------------
# NEW pairing: sim_top ↔ real_right, sim_bottom ↔ real_left
amp_sim_top_interp_for_right   = np.interp(freq_right_pos, freq_sim_top_pos, amp_sim_top_pos)
phase_sim_top_interp_for_right = np.interp(freq_right_pos, freq_sim_top_pos, phase_sim_top_pos)

amp_sim_bot_interp_for_left    = np.interp(freq_left_pos,  freq_sim_bot_pos, amp_sim_bot_pos)
phase_sim_bot_interp_for_left  = np.interp(freq_left_pos,  freq_sim_bot_pos, phase_sim_bot_pos)

# ------------ ERROR METRICS (EDGES, SWAPPED) ----------
mse_amp_left  = mean_squared_error(amp_left_pos,  amp_sim_bot_interp_for_left)
mse_amp_right = mean_squared_error(amp_right_pos, amp_sim_top_interp_for_right)

phase_diff_left  = np.angle(np.exp(1j * (phase_left_pos  - phase_sim_bot_interp_for_left)))
phase_diff_right = np.angle(np.exp(1j * (phase_right_pos - phase_sim_top_interp_for_right)))
mse_phase_left  = np.mean(phase_diff_left**2)
mse_phase_right = np.mean(phase_diff_right**2)

mse_amp_avg   = 0.5 * (mse_amp_left + mse_amp_right)
mse_phase_avg = 0.5 * (mse_phase_left + mse_phase_right)

print(f"MSE Amplitude (L,R,avg): {mse_amp_left:.6f}, {mse_amp_right:.6f}, avg={mse_amp_avg:.6f}")
print(f"MSE Phase     (L,R,avg): {mse_phase_left:.6f}, {mse_phase_right:.6f}, avg={mse_phase_avg:.6f}")

# ---------------- REAL & SIM FFT (CENTERLINE) ----------------
# Real centerline
N_c_real = len(real_centerline)
N_c_real_padded = zero_padding_factor * N_c_real
real_c_padded = np.pad(real_centerline, (0, N_c_real_padded - N_c_real), mode='constant')

fft_c_real = np.fft.fft(real_c_padded)
freq_c_real = np.fft.fftfreq(N_c_real_padded, d=1.0 / sampling_rate_real)
amp_c_real = 2.0 * np.abs(fft_c_real) / N_c_real      # normalize by ORIGINAL N
phase_c_real = np.angle(fft_c_real)
mask_c_real = freq_c_real > 0
freq_c_pos = freq_c_real[mask_c_real]
amp_c_pos = amp_c_real[mask_c_real]
phase_c_pos = phase_c_real[mask_c_real]

# Sim centerline
N_c_sim = len(sim_centerline)
N_c_sim_padded = zero_padding_factor * N_c_sim
sim_c_padded = np.pad(sim_centerline, (0, N_c_sim_padded - N_c_sim), mode='constant')

fft_c_sim = np.fft.fft(sim_c_padded)
freq_c_sim = np.fft.fftfreq(N_c_sim_padded, d=1.0 / sampling_rate_sim)
amp_c_sim = 2.0 * np.abs(fft_c_sim) / N_c_sim         # normalize by ORIGINAL N
phase_c_sim = np.angle(fft_c_sim)
mask_c_sim = freq_c_sim > 0
freq_c_sim_pos = freq_c_sim[mask_c_sim]
amp_c_sim_pos = amp_c_sim[mask_c_sim]
phase_c_sim_pos = phase_c_sim[mask_c_sim]

# ---- INTERPOLATE & METRICS (CENTERLINE) ----
amp_c_sim_interp   = np.interp(freq_c_pos,  freq_c_sim_pos,  amp_c_sim_pos)
phase_c_sim_interp = np.interp(freq_c_pos,  freq_c_sim_pos,  phase_c_sim_pos)

mse_amp_C = mean_squared_error(amp_c_pos, amp_c_sim_interp)
phase_diff_C = np.angle(np.exp(1j * (phase_c_pos - phase_c_sim_interp)))
mse_phase_C = np.mean(phase_diff_C**2)

print(f"MSE Amplitude (Centerline): {mse_amp_C:.6f}")
print(f"MSE Phase     (Centerline): {mse_phase_C:.6f}")

# ----------------- PLOTS --------------------
FONT = getattr(constants, "font_large", 12)

# Amplitude — edges (swapped pairing)
plt.figure(figsize=(10, 5))
plt.plot(freq_left_pos,  amp_left_pos,                   label="Real Left FFT",  alpha=0.9)
plt.plot(freq_left_pos,  amp_sim_bot_interp_for_left,   "--", label="Sim Bottom FFT (↔ Left)")
plt.plot(freq_right_pos, amp_right_pos,                  label="Real Right FFT", alpha=0.9)
plt.plot(freq_right_pos, amp_sim_top_interp_for_right,  "--", label="Sim Top FFT (↔ Right)")
plt.xlabel("Frequency (mm⁻¹)", fontsize=FONT)
plt.ylabel("Amplitude (mm)",    fontsize=FONT)
plt.xlim(0, 0.2)
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# Phase — edges (swapped pairing)
plt.figure(figsize=(10, 5))
plt.plot(freq_left_pos,  phase_left_pos,                  label="Real Left Phase",  alpha=0.9)
plt.plot(freq_left_pos,  phase_sim_bot_interp_for_left,  "--", label="Sim Bottom Phase (↔ Left)")
plt.plot(freq_right_pos, phase_right_pos,                 label="Real Right Phase", alpha=0.9)
plt.plot(freq_right_pos, phase_sim_top_interp_for_right, "--", label="Sim Top Phase (↔ Right)")
plt.xlabel("Frequency (mm⁻¹)", fontsize=FONT)
plt.ylabel("Phase (radians)",   fontsize=FONT)
plt.xlim(0, 0.2)
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# Amplitude — centerline
plt.figure(figsize=(10, 5))
plt.plot(freq_c_pos, amp_c_pos,           label="Real Centerline FFT", alpha=0.9)
plt.plot(freq_c_pos, amp_c_sim_interp,   "--", label="Sim Centerline FFT")
plt.xlabel("Frequency (mm⁻¹)", fontsize=FONT)
plt.ylabel("Amplitude (mm)",    fontsize=FONT)
plt.xlim(0, 0.2)
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# Phase — centerline
plt.figure(figsize=(10, 5))
plt.plot(freq_c_pos, phase_c_pos,         label="Real Centerline Phase", alpha=0.9)
plt.plot(freq_c_pos, phase_c_sim_interp, "--", label="Sim Centerline Phase")
plt.xlabel("Frequency (mm⁻¹)", fontsize=FONT)
plt.ylabel("Phase (radians)",   fontsize=FONT)
plt.xlim(0, 0.2)
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
