# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Project imports: match Code A sources ---
from Data_ALL_traverse import traverse_tow_constructor
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path

# ----------------- Helpers -----------------

def build_real_traverse_centerline_like_A(tow: int, resample_uniform: bool = True, target_steps: int | None = None):
    """
    REAL (traverse) like Code A:
      - edges from traverse_tow_constructor
      - centerline = (y_left + y_right)/2
      - normalize by subtracting first value
      - optional uniform resampling for FFT stability
    Returns (x_mm, centerline_mm)
    """
    df = traverse_tow_constructor(tow)

    x_r = df["x_right"].to_numpy()
    y_r = df["y_right"].to_numpy()
    x_l = df["x_left"].to_numpy()
    y_l = df["y_left"].to_numpy()

    # Use one x axis (assume x_r ~ x_l)
    x = x_r
    centerline = 0.5 * (y_l + y_r)

    # Normalize to start at 0 (Code A convention)
    centerline = centerline - centerline[0]

    # Clean + sort
    m = np.isfinite(x) & np.isfinite(centerline)
    x = x[m]; centerline = centerline[m]
    order = np.argsort(x)
    x = x[order]; centerline = centerline[order]

    if resample_uniform:
        if target_steps is None:
            target_steps = len(x)
        x_uni = np.linspace(x[0], x[-1], target_steps)
        centerline = np.interp(x_uni, x, centerline)
        x = x_uni

    return x, centerline


def simulate_centerline_like_A(n_steps: int, num_bins: int, length_tow_mm: float, nominal_width_mm: float = 6.35):
    """
    SIM (Code A):
      - consecutive_error for CAM, LT, LLS_B
      - centerline = CAM + LT
      - width = nominal + LLS_B error (not used in FFT but kept for parity)
      - normalize centerline by subtracting first value
    Returns centerline array of length n_steps.
    """
    # Fit models
    _, sc, ic, _, _, _, xsc, bec, dvc = consecutive_error("CAM",   test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False)
    _, sl, il, _, _, _, xsl, bel, dvl = consecutive_error("LT",    test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False)
    _, sw, iw, _, _, _, xsw, bew, dvw = consecutive_error("LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False)

    # Start ranges copied from Code A
    start_cam  = np.random.uniform(-0.75,  0.75)
    start_lt   = np.random.uniform(-0.90, -0.70)
    start_llsb = np.random.uniform(-0.21, -0.02)

    cam_path = generate_error_path(start_cam,  n_steps, sc,  ic,  xsc,  bec, dvc)
    lt_path  = generate_error_path(start_lt,   n_steps, sl,  il,  xsl,  bel, dvl)
    w_err    = generate_error_path(start_llsb, n_steps, sw,  iw,  xsw,  bew, dvw)

    centerline = cam_path + lt_path
    width = nominal_width_mm + w_err  # kept for parity with Code A (not used below)

    # Normalize like Code A
    centerline = centerline - centerline[0]

    # Sim x-grid is uniform [0, length_tow_mm], but FFT uses sampling rate only
    return centerline


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

def find_best_nsteps_and_bins(
    tow_range=range(2, 8),
    nsteps_candidates=None,
    bin_candidates=None,
    n_repeats=10,
    zero_padding_factor: int = 2,
    resample_uniform_real: bool = True,
    sim_length_mm: float = 1000.0
):
    """
    Grid-search for (n_steps, num_bins) that minimize summed amplitude-spectrum MSE
    across tows, using:
      - REAL: traverse centerline like Code A
      - SIM: Code A (CAM + LT + LLS_B), centerline-only FFT
    """
    if nsteps_candidates is None:
        nsteps_candidates = list(range(100, 600, 10))
    if bin_candidates is None:
        bin_candidates = list(range(30, 300, 5))
    mse_surface = np.zeros((len(bin_candidates), len(nsteps_candidates)), dtype=float)

    for tow in tow_range:
        print(f"[INFO] Processing Tow {tow} ...")

        # --- REAL (Traverse) like Code A ---
        x_real, cl_real = build_real_traverse_centerline_like_A(
            tow=tow,
            resample_uniform=resample_uniform_real,
            target_steps=None
        )
        # sampling rate (samples per mm)
        dx_real = (x_real[-1] - x_real[0]) / (len(x_real) - 1)
        fs_real = 1.0 / dx_real

        # REAL FFT
        freq_real, amp_real = single_sided_fft(cl_real, sampling_rate=fs_real, pad_factor=zero_padding_factor)

        for b_idx, num_bins in enumerate(bin_candidates):
            # Fit models ONCE per (tow, num_bins) for efficiency
            # (We still generate new random realizations per repeat.)
            for s_idx, n_steps in enumerate(nsteps_candidates):
                total_mse = 0.0

                for _ in range(n_repeats):
                    # --- SIM like Code A ---
                    cl_sim = simulate_centerline_like_A(
                        n_steps=n_steps,
                        num_bins=num_bins,
                        length_tow_mm=sim_length_mm,
                        nominal_width_mm=6.35
                    )
                    # sim sampling rate (samples per mm)
                    fs_sim = n_steps / sim_length_mm

                    # SIM FFT
                    freq_sim, amp_sim = single_sided_fft(cl_sim, sampling_rate=fs_sim, pad_factor=zero_padding_factor)

                    # MSE over common freq band
                    mse = mse_over_common_freq_band(freq_real, amp_real, freq_sim, amp_sim)
                    total_mse += mse

                mse_surface[b_idx, s_idx] += total_mse  # accumulate over tows

    # ---- Plot the surface ----
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
    optimal_bins = bin_candidates[min_idx[0]]
    optimal_steps = nsteps_candidates[min_idx[1]]
    best_val = mse_surface[min_idx]
    print(f"Optimal -> n_steps: {optimal_steps}, num_bins: {optimal_bins}, Total MSE: {best_val:.6f}")

    return mse_surface, optimal_steps, optimal_bins


# Run
if __name__ == "__main__":
    find_best_nsteps_and_bins()
