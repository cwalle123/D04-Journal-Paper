"""This file is used to find the optimum number of steps and bins."""

##############################################################################################################

# External imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

#Internal imports
from Data_ALL_traverse import traverse_tow_constructor
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path

##############################################################################################################
"""Functions"""

# ----------------- Helpers -----------------

def build_real_traverse_edges_like_A(
    tow: int,
    resample_uniform: bool = True,
    target_steps: int | None = None,
    per_edge_normalize: bool = True,
):
    """
    REAL (traverse) — EDGES version (aligned with your Data_ALL_traverse.traverse_tow_constructor):
      - uses x_left/y_left and x_right/y_right from traverse_tow_constructor
      - optional per-edge normalization: subtract first sample of each edge (like the centerline path did)
      - optional uniform resampling for FFT stability
    Returns (x_mm, left_edge_mm, right_edge_mm)
    """
    df = traverse_tow_constructor(tow)  # columns: x_right,y_right,x_left,y_left, ...  :contentReference[oaicite:0]{index=0}
    if df is None:
        raise ValueError(f"Tow {tow} not available from traverse.")

    x_r = df["x_right"].to_numpy()
    y_r = df["y_right"].to_numpy()
    x_l = df["x_left"].to_numpy()
    y_l = df["y_left"].to_numpy()

    # They’re already truncated to the same length in traverse_tow_constructor. :contentReference[oaicite:1]{index=1}
    # Choose a single x-axis; we’ll use right-edge x (matches your earlier pattern).
    x = x_r.copy()
    left = y_l.copy()
    right = y_r.copy()

    # Per-edge normalization (analogous to centerline normalization in your old helper)
    if per_edge_normalize:
        left = left - left[0]
        right = right - right[0]

    # Clean + sort
    m = np.isfinite(x) & np.isfinite(left) & np.isfinite(right)
    x = x[m]; left = left[m]; right = right[m]
    order = np.argsort(x)
    x = x[order]; left = left[order]; right = right[order]

    # Optional uniform resampling for FFT stability
    if resample_uniform:
        if target_steps is None:
            target_steps = len(x)
        x_uni = np.linspace(x[0], x[-1], target_steps)
        left = np.interp(x_uni, x, left)
        right = np.interp(x_uni, x, right)
        x = x_uni

    return x, left, right

def simulate_edges_like_visualizer(
    n_steps: int,
    num_bins: int,
    tow_length_mm: float = 1000.0,
    nominal_width_mm: float = 6.35,
):
    """
    SIM (exactly like Model_ALL_Validation_Tow_Visualiser):
      - Fit consecutive_error for CAM, LT, LLS_B (with random_state like visualizer)
      - Sample start states in the same ranges
      - Generate paths and construct edges:
            centerline = cam_path + lt_path
            widths     = nominal_width_mm + width_error
            top_edge    = centerline + 0.5 * widths
            bottom_edge = centerline - 0.5 * widths
      - x grid is linspace(0, tow_length_mm, n_steps)

    Returns:
        x_mm, sim_top, sim_bottom, centerline, widths
    """
    import random

    # --- Load error model fits (same calls/params as the visualizer) ---
    bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = consecutive_error(
        "CAM", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))  # matches visualizer behavior  :contentReference[oaicite:0]{index=0}
    bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = consecutive_error(
        "LT", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))  # :contentReference[oaicite:1]{index=1}
    bin_stats_llsb, slope_llsb, intercept_llsb, _, _, _, x_sorted_llsb, bin_edges_llsb, devs_llsb = consecutive_error(
        "LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))  # :contentReference[oaicite:2]{index=2}

    # --- Start values (exact ranges from the visualizer) ---
    start_cam   = random.uniform(-0.75,  0.75)  # :contentReference[oaicite:3]{index=3}
    start_lt    = random.uniform(-0.90, -0.70)  # :contentReference[oaicite:4]{index=4}
    start_llsb  = random.uniform(-0.21, -0.02)  # :contentReference[oaicite:5]{index=5}

    # --- Generate error paths (same function + arguments pattern) ---
    cam_path = generate_error_path(start_cam, n_steps, slope_cam, intercept_cam,
                                   x_sorted_cam, bin_edges_cam, devs_cam)  # :contentReference[oaicite:6]{index=6}
    lt_path  = generate_error_path(start_lt,  n_steps, slope_lt,  intercept_lt,
                                   x_sorted_lt, bin_edges_lt, devs_lt)      # :contentReference[oaicite:7]{index=7}
    width_err = generate_error_path(start_llsb, n_steps, slope_llsb, intercept_llsb,
                                    x_sorted_llsb, bin_edges_llsb, devs_llsb)  # :contentReference[oaicite:8]{index=8}

    centerline = cam_path + lt_path
    widths = nominal_width_mm + width_err  # 6.35 nominal + error (unchanged)  :contentReference[oaicite:9]{index=9}

    sim_top = centerline + 0.5 * widths
    sim_bottom = centerline - 0.5 * widths
    x_mm = np.linspace(0, tow_length_mm, n_steps)  # visualizer uses linspace over tow length  :contentReference[oaicite:10]{index=10}

    return x_mm, sim_top, sim_bottom, centerline, widths


def single_sided_fft(signal: np.ndarray, sampling_rate: float, pad_factor: int = 1):
    """
    Returns positive-frequency single-sided amplitude spectrum.
    sampling_rate: samples per mm (i.e., 1/Δx)
    """
    n = len(signal)
    n_pad = int(pad_factor * n)
    padded = np.pad(signal, (0, n_pad - n), mode="constant") if n_pad > n else signal
    # d = 1/sampling_rate = Δx (mm/sample)
    freq = np.fft.fftfreq(len(padded), d=1.0 / sampling_rate)
    fft_vals = np.fft.fft(padded)
    amp = 2.0 * np.abs(fft_vals) / len(padded)
    pos = freq > 0
    return freq[pos], amp[pos]


def fft_edges(left: np.ndarray, right: np.ndarray, fs: float, pad_factor: int):
    fL, aL = single_sided_fft(left,  sampling_rate=fs, pad_factor=pad_factor)
    fR, aR = single_sided_fft(right, sampling_rate=fs, pad_factor=pad_factor)
    return (fL, aL), (fR, aR)


def mse_over_common_freq_band(freq_a, amp_a, freq_b, amp_b):
    """
    Interpolates amp_b onto freq_a within the common max frequency range, then MSE.
    Avoids misleading tails from extrapolation.
    """
    fmax = min(freq_a.max(), freq_b.max())
    mask = freq_a <= fmax
    fa = freq_a[mask]
    Aa = amp_a[mask]
    Ab_i = np.interp(fa, freq_b, amp_b)
    return mean_squared_error(Aa, Ab_i)

# ----------------- Main optimizer -----------------

def find_best_nsteps_and_bins_edges(
    tow_range=range(2, 8),
    nsteps_candidates=None,
    bin_candidates=None,
    n_repeats=10,
    zero_padding_factor: int = 2,
    resample_uniform_real: bool = True,
    sim_length_mm: float = 1000.0,
    nominal_width_mm: float = 6.35,
):
    """
    Grid-search for (n_steps, num_bins) that minimize the average FFT-magnitude MSE
    **over BOTH EDGES**:
        0.5 * [ MSE(|FFT(sim_top)|, |FFT(real_left)|)
              + MSE(|FFT(sim_bottom)|, |FFT(real_right)|) ]

    Real edges: from traverse_tow_constructor (via build_real_traverse_edges_like_A).
    Sim edges:  EXACTLY like the visualizer (via simulate_edges_like_visualizer).
    """
    if nsteps_candidates is None:
        nsteps_candidates = list(range(100, 600, 10))
    if bin_candidates is None:
        bin_candidates = list(range(30, 300, 5))

    mse_surface = np.zeros((len(bin_candidates), len(nsteps_candidates)), dtype=float)

    for tow in tow_range:
        print(f"[INFO] Processing Tow {tow} ...")

        # -------- REAL EDGES (Traverse) --------
        x_real, left_real, right_real = build_real_traverse_edges_like_A(
            tow=tow,
            resample_uniform=resample_uniform_real,
            target_steps=None,  # keep native resolution unless you want to force it
            per_edge_normalize=True,  # mirrors old centerline normalization per edge
        )

        # Shared sampling rate for both edges
        dx_real = (x_real[-1] - x_real[0]) / (len(x_real) - 1)
        fs_real = 1.0 / dx_real

        # Real FFTs (left & right)
        (fL_real, aL_real), (fR_real, aR_real) = fft_edges(
            left_real, right_real, fs=fs_real, pad_factor=zero_padding_factor
        )

        # -------- GRID SEARCH --------
        for b_idx, num_bins in enumerate(bin_candidates):
            for s_idx, n_steps in enumerate(nsteps_candidates):
                total_mse = 0.0

                for _ in range(n_repeats):
                    # ---- SIMULATED EDGES (exact visualizer construction) ----
                    x_sim, sim_top, sim_bottom, _sim_centerline, _sim_widths = simulate_edges_like_visualizer(
                        n_steps=n_steps,
                        num_bins=num_bins,
                        tow_length_mm=sim_length_mm,
                        nominal_width_mm=nominal_width_mm,
                    )

                    # Sim sampling rate (uniform linspace over length)
                    fs_sim = n_steps / sim_length_mm

                    # Sim FFTs (top & bottom)
                    (f_top, a_top), (f_bot, a_bot) = fft_edges(
                        sim_top, sim_bottom, fs=fs_sim, pad_factor=zero_padding_factor
                    )

                    # --- MSE over common frequency band, edge-wise ---
                    mse_left  = mse_over_common_freq_band(fL_real, aL_real, f_top, a_top)
                    mse_right = mse_over_common_freq_band(fR_real, aR_real, f_bot, a_bot)

                    # Average the two edges
                    total_mse += 0.5 * (mse_left + mse_right)

                # accumulate over tows
                mse_surface[b_idx, s_idx] += total_mse

    # ---- Plot the surface (same as before) ----
    X, Y = np.meshgrid(nsteps_candidates, bin_candidates)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, mse_surface, cmap='viridis')
    ax.set_xlabel("number of steps")
    ax.set_ylabel("number of bins")
    ax.set_zlabel("Total MSE (sum over tows & repeats)")
    plt.tight_layout()
    plt.show()

    # ---- Best configuration ----
    min_idx = np.unravel_index(np.argmin(mse_surface), mse_surface.shape)
    optimal_bins  = bin_candidates[min_idx[0]]
    optimal_steps = nsteps_candidates[min_idx[1]]
    best_val      = mse_surface[min_idx]
    print(f"Optimal → n_steps: {optimal_steps}, num_bins: {optimal_bins}, Total MSE: {best_val:.6f}")

    return mse_surface, optimal_steps, optimal_bins

##############################################################################################################
"""Run this file"""

def main():
    find_best_nsteps_and_bins_edges()

if __name__ == "__main__":
    main()

