import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Model_ALL_Simulation import generate_error_path, consecutive_error
from Handling_ALL_Functions import get_synced_data
import constants

# ---------------- PARAMETERS ----------------
tow_number = 3
spacesynced = True
length_tow = 1000  # in mm
n_steps = 360
num_bins = 180
zero_padding_factor = 2
use_seed = False  # <-- Toggle reproducibility ON/OFF
random_seed = 0

# -------------- DATA LOAD -------------------
df = get_synced_data(tow=tow_number, spacesynced=spacesynced)
cam = df["center_CAM"].dropna().values
lt = df["error_LT"].dropna().values
x_pos = df["x"].dropna().values
offset_real = cam + lt

# ---------- SAMPLING RATE FIX ---------------
length_between_points = (x_pos[-1] - x_pos[0]) / (len(x_pos) - 1)
sampling_rate_real = 1 / length_between_points
sampling_rate_sim = n_steps / length_tow

# ---------------- REAL FFT ------------------
real_centerline = offset_real
N_real = len(real_centerline)
N_real_padded = zero_padding_factor * N_real
real_centerline_padded = np.pad(real_centerline, (0, N_real_padded - N_real), mode='constant')

fft_real = np.fft.fft(real_centerline_padded)
freq_real = np.fft.fftfreq(N_real_padded, d=1 / sampling_rate_real)
amp_real = 2 * np.abs(fft_real) / N_real_padded
phase_real = np.angle(fft_real)
mask_real = freq_real > 0
freq_real_pos = freq_real[mask_real]
amp_real_pos = amp_real[mask_real]
phase_real_pos = phase_real[mask_real]

# ------------- MODEL FITTING ----------------
bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = consecutive_error(
    "CAM", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False
)
bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = consecutive_error(
    "LT", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False
)

# ------------ SIMULATED PATH ----------------
if use_seed:
    np.random.seed(random_seed)

start_cam = np.random.uniform(-0.4, 0.6)
start_lt = np.random.uniform(-1.0, -0.8)

cam_path = generate_error_path(start_cam, n_steps, slope_cam, intercept_cam, x_sorted_cam, bin_edges_cam, devs_cam)
lt_path = generate_error_path(start_lt, n_steps, slope_lt, intercept_lt, x_sorted_lt, bin_edges_lt, devs_lt)
simulated_centerline = cam_path + lt_path

# -------------- SIMULATED FFT ---------------
N_sim = len(simulated_centerline)
N_sim_padded = zero_padding_factor * N_sim
sim_centerline_padded = np.pad(simulated_centerline, (0, N_sim_padded - N_sim), mode='constant')

fft_sim = np.fft.fft(sim_centerline_padded)
freq_sim = np.fft.fftfreq(N_sim_padded, d=1 / sampling_rate_sim)
amp_sim = 2 * np.abs(fft_sim) / N_sim_padded
phase_sim = np.angle(fft_sim)
mask_sim = freq_sim > 0
freq_sim_pos = freq_sim[mask_sim]
amp_sim_pos = amp_sim[mask_sim]
phase_sim_pos = phase_sim[mask_sim]

# ---------- INTERPOLATE SIM FFT -------------
amp_sim_interp = np.interp(freq_real_pos, freq_sim_pos, amp_sim_pos)
phase_sim_interp = np.interp(freq_real_pos, freq_sim_pos, phase_sim_pos)

# ------------ ERROR METRICS -----------------
mse_amp = mean_squared_error(amp_real_pos, amp_sim_interp)

# Phase error using circular distance
phase_diff = np.angle(np.exp(1j * (phase_real_pos - phase_sim_interp)))
mse_phase = np.mean(phase_diff**2)

print(f"MSE Amplitude Spectrum: {mse_amp:.6f}")
print(f"MSE Phase Spectrum (wrapped): {mse_phase:.6f}")

# ----------------- PLOTS --------------------
# Amplitude
plt.figure(figsize=(10, 5))
plt.plot(freq_real_pos, amp_real_pos, label="Experimental Tow FFT", color='blue')
plt.plot(freq_real_pos, amp_sim_interp, linestyle="--", label="Simulated Tow FFT", color='orange')
plt.xlabel("Frequency (mm⁻¹)", fontsize=constants.font_large)
plt.ylabel("Amplitude (mm)", fontsize=constants.font_large)
plt.xlim(0, 0.2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Phase
plt.figure(figsize=(10, 5))
plt.plot(freq_real_pos, phase_real_pos, label="Experimental Tow Phase", color='blue')
plt.plot(freq_real_pos, phase_sim_interp, linestyle="--", label="Simulated Tow Phase", color='orange')
plt.xlabel("Frequency (mm⁻¹)", fontsize=constants.font_large)
plt.ylabel("Phase (radians)", fontsize=constants.font_large)
plt.xlim(0, 0.2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
