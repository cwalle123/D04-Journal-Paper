"""This file is used to find the optimum number of steps and bins.
   Written by: """

##############################################################################################################

# External imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import sys

#Internal imports
from Data_ALL_traverse import traverse_tow_constructor, traverse_tow_gaps_and_overlaps_lengths
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path
from Model_ALL_Validation_Tow_Visualiser import plot_simulated_vs_real_tow
from Model_ALL_Simulation import generate_multitow_layout_lengths

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
      - optional per-edge normalization: subtract first sample of each edge
      - optional uniform resampling for FFT stability (auto-matches native density)
    Returns (x_mm, left_edge_mm, right_edge_mm)
    """
    df = traverse_tow_constructor(tow)  # columns: x_right,y_right,x_left,y_left
    if df is None:
        raise ValueError(f"Tow {tow} not available from traverse.")

    x_r = df["x_right"].to_numpy(dtype=float)
    y_r = df["y_right"].to_numpy(dtype=float)
    x_l = df["x_left"].to_numpy(dtype=float)
    y_l = df["y_left"].to_numpy(dtype=float)

    # Choose a single x-axis; we’ll use right-edge x
    x = x_r.copy()
    left = y_l.copy()
    right = y_r.copy()

    # Per-edge normalization
    if per_edge_normalize and len(left) and len(right):
        left  = left  - left[0]
        right = right - right[0]

    # Clean + sort + make x strictly increasing
    m = np.isfinite(x) & np.isfinite(left) & np.isfinite(right)
    x, left, right = x[m], left[m], right[m]
    order = np.argsort(x, kind="mergesort")
    x, left, right = x[order], left[order], right[order]
    if len(x) > 1:
        uniq = np.ones_like(x, dtype=bool)
        uniq[1:] = x[1:] > x[:-1]
        x, left, right = x[uniq], left[uniq], right[uniq]

    if not resample_uniform or len(x) < 2:
        return x, left, right

    # ---------- AUTO-MATCH NATIVE DENSITY ----------
    if target_steps is None:
        dx_med = float(np.median(np.diff(x)))          # robust native spacing
        length = float(x[-1] - x[0])
        # number of points so Δx_uniform ≈ dx_med and both ends included
        target_steps = max(int(round(length / max(dx_med, 1e-12))) + 1, 2)

    x_uni = np.linspace(x[0], x[-1], target_steps)
    left  = np.interp(x_uni, x, left)
    right = np.interp(x_uni, x, right)
    x     = x_uni

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
    """
    import random

    bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = consecutive_error(
        "CAM", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = consecutive_error(
        "LT", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_llsb, slope_llsb, intercept_llsb, _, _, _, x_sorted_llsb, bin_edges_llsb, devs_llsb = consecutive_error(
        "LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))

    # Start values (same ranges as visualizer)
    start_cam   = random.uniform(-0.75,  0.75)
    start_lt    = random.uniform(-0.90, -0.70)
    start_llsb  = random.uniform(-0.21, -0.02)

    cam_path   = generate_error_path(start_cam,  n_steps, slope_cam,  intercept_cam,  x_sorted_cam,  bin_edges_cam,  devs_cam)
    lt_path    = generate_error_path(start_lt,   n_steps, slope_lt,   intercept_lt,   x_sorted_lt,   bin_edges_lt,   devs_lt)
    width_err  = generate_error_path(start_llsb, n_steps, slope_llsb, intercept_llsb, x_sorted_llsb, bin_edges_llsb, devs_llsb)

    centerline = cam_path + lt_path
    widths     = nominal_width_mm + width_err

    sim_top    = centerline + 0.5 * widths
    sim_bottom = centerline - 0.5 * widths
    x_mm       = np.linspace(0, tow_length_mm, n_steps)

    return x_mm, sim_top, sim_bottom, centerline, widths

def single_sided_fft(signal: np.ndarray, sampling_rate: float, pad_factor: int = 1):
    """
    Returns positive-frequency single-sided amplitude spectrum.
    sampling_rate: samples per mm (i.e., 1/Δx)
    """
    n = len(signal)
    n_pad = int(pad_factor * n)
    padded = np.pad(signal, (0, n_pad - n), mode="constant") if n_pad > n else signal
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

# ----------------- Progress helper -----------------

def _print_progress(done: int, total: int):
    """Inline textual progress bar + percent."""
    if total <= 0:
        return
    pct = int(100 * done / total)
    bar_len = 28
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\rProgress: |{bar}| {pct:3d}%  ({done}/{total})")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")
        sys.stdout.flush()

# ----------------- Main optimizer -----------------

def find_best_nsteps_and_bins_edges(
    tow_range=range(2, 31),
    nsteps_candidates=None,
    bin_candidates=None,
    n_repeats=10,
    zero_padding_factor: int = 2,
    resample_uniform_real: bool = True,
    sim_length_mm: float = 1000.0,
    nominal_width_mm: float = 6.35,
    show_progress: bool = True,   # <-- new
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
        nsteps_candidates = list(range(100, 1500, 100))
        
    if bin_candidates is None:
        bin_candidates = list(range(30, 1031, 200))

    mse_surface = np.zeros((len(bin_candidates), len(nsteps_candidates)), dtype=float)

    # Precompute total work units for progress: each repeat of each (tow, bins, steps)
    total_units = len(list(tow_range)) * len(bin_candidates) * len(nsteps_candidates) * n_repeats
    done_units = 0
    if show_progress:
        _print_progress(done_units, total_units)

    for tow in tow_range:
        print(f"[INFO] Processing Tow {tow} ...")

        # -------- REAL EDGES (Traverse) --------
        x_real, left_real, right_real = build_real_traverse_edges_like_A(
            tow=tow,
            resample_uniform=resample_uniform_real,
            target_steps=None,               # keep native resolution
            per_edge_normalize=True,
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

                    # progress tick per repeat
                    done_units += 1
                    if show_progress:
                        _print_progress(done_units, total_units)

                # accumulate over tows
                mse_surface[b_idx, s_idx] += total_mse

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
    optimal_bins  = bin_candidates[min_idx[0]]
    optimal_steps = nsteps_candidates[min_idx[1]]
    best_val      = mse_surface[min_idx]
    print(f"Optimal → n_steps: {optimal_steps}, num_bins: {optimal_bins}, Total MSE: {best_val:.6f}")

    return mse_surface, optimal_steps, optimal_bins

def find_best_bins_fft_mse_real_vs_sim(tow: int,bins_min: int = 20,bins_max: int = 500,bins_step: int = 20,zero_padding_factor: int = 2,tow_length_mm: float = 1000.0,show_plot: bool = True):
    """
    Sweep Consecutive_Error_Bins from bins_min to bins_max (inclusive, in bins_step increments),
    compute FFT MSE between real and simulated tows, and find the optimal bin count.

    After finding the optimal bin count, this function:
        • Recomputes FFT for both real and simulated tows
        • Displays amplitude and phase spectra side by side.

    Returns
    -------
    best_bins : int
        Bin count that yields the lowest FFT MSE.
    mse_values : dict
        Mapping {num_bins: mse_value}.
    """

    global Consecutive_Error_Bins  # use the global setting from your model

    # Build the candidate bin list
    bin_candidates = list(range(bins_min, bins_max + 1, bins_step))
    mse_values = {}

    print(f"[INFO] Evaluating Tow {tow} over bin range {bins_min}–{bins_max} (step={bins_step})")
    print(f"[INFO] Total {len(bin_candidates)} candidate values.\n")

    # --- Sweep over candidate bin values ---
    for num_bins in bin_candidates:
        Consecutive_Error_Bins = num_bins  # temporarily override global value

        # --- Generate real & simulated data ---
        real_data, sim_data = plot_simulated_vs_real_tow(
            tow=tow,
            tow_length_mm=tow_length_mm,
            plot=False,
            force_steps=True)

        # --- Extract centerlines ---
        real_y = real_data["centerline"].to_numpy()
        sim_y = sim_data["centerline"].to_numpy()

        # Align array lengths
        n = min(len(real_y), len(sim_y))
        real_y = real_y[:n]
        sim_y = sim_y[:n]

        dx = tow_length_mm / (n - 1)
        fs = 1.0 / dx  # samples per mm

        # --- FFTs ---
        freq_real, amp_real = single_sided_fft(real_y, fs, pad_factor=zero_padding_factor)
        freq_sim, amp_sim = single_sided_fft(sim_y, fs, pad_factor=zero_padding_factor)

        # --- Interpolate to common frequency range and compute MSE ---
        fmax = min(freq_real.max(), freq_sim.max())
        mask = freq_real <= fmax
        fr = freq_real[mask]
        Ar = amp_real[mask]
        As = np.interp(fr, freq_sim, amp_sim)
        mse = mean_squared_error(Ar, As)

        mse_values[num_bins] = mse
        print(f"  → Bins={num_bins:<5d} | MSE={mse:.6e}")

    # --- Determine best bin count ---
    best_bins = min(mse_values, key=mse_values.get)
    best_val = mse_values[best_bins]
    print(f"\n✅ Optimal Consecutive_Error_Bins = {best_bins}  (MSE={best_val:.6e})")

    # --- Optional Plot: MSE vs Bin Count ---
    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(list(mse_values.keys()), list(mse_values.values()), "o-", lw=2)
        plt.xlabel("Consecutive_Error_Bins")
        plt.ylabel("Spectral MSE (Real vs Simulated Centerline)")
        plt.title(f"Tow {tow} — FFT MSE vs Bin Count")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- Recompute FFTs for best bin setting (for visualization) ---
    Consecutive_Error_Bins = best_bins
    real_data, sim_data = plot_simulated_vs_real_tow(
        tow=tow,
        tow_length_mm=tow_length_mm,
        plot=False,
        force_steps=True)

    real_y = real_data["centerline"].to_numpy()
    sim_y = sim_data["centerline"].to_numpy()
    n = min(len(real_y), len(sim_y))
    real_y = real_y[:n]
    sim_y = sim_y[:n]

    dx = tow_length_mm / (n - 1)
    fs = 1.0 / dx

    # --- Compute complex FFTs for amplitude + phase ---
    def full_fft(signal, fs, pad_factor):
        n = len(signal)
        n_pad = int(pad_factor * n)
        padded = np.pad(signal, (0, n_pad - n), mode="constant") if n_pad > n else signal
        freq = np.fft.fftfreq(len(padded), d=1.0 / fs)
        fft_vals = np.fft.fft(padded)
        pos = freq > 0
        return freq[pos], fft_vals[pos]

    freq_real, fft_real = full_fft(real_y, fs, zero_padding_factor)
    freq_sim, fft_sim = full_fft(sim_y, fs, zero_padding_factor)

    amp_real = 2.0 * np.abs(fft_real) / len(real_y)
    amp_sim  = 2.0 * np.abs(fft_sim) / len(sim_y)
    phase_real = np.angle(fft_real)
    phase_sim  = np.angle(fft_sim)

    # --- Plot amplitude + phase spectra ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Amplitude
    ax1.plot(freq_real, amp_real, color="blue", label="Real Tow")
    ax1.plot(freq_sim, amp_sim, color="gold", label="Simulated Tow", alpha=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Tow {tow} — FFT Spectra at Optimal Bins = {best_bins}")
    ax1.legend()
    ax1.grid(True)

    # Phase
    ax2.plot(freq_real, phase_real, color="blue", label="Real Tow Phase")
    ax2.plot(freq_sim, phase_sim, color="gold", label="Sim Tow Phase", alpha=0.8)
    ax2.set_xlabel("Frequency [cycles/mm]")
    ax2.set_ylabel("Phase [radians]")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return best_bins, mse_values

def analyze_all_tows_best_bins_fft_mse(tow_range=range(2, 31),bins_min: int = 20,bins_max: int = 500,bins_step: int = 20,zero_padding_factor: int = 2,tow_length_mm: float = 1000.0):
    """
    Loops over all tows, finds the best Consecutive_Error_Bins for each based on FFT MSE,
    computes mean and std of best bin counts, and plots a representative FFT amplitude & phase
    comparison using the mean optimal bin count.

    Parameters
    ----------
    tow_range : iterable
        Range or list of tow indices to evaluate (default = range(1, 32)).
    bins_min, bins_max, bins_step : int
        Bin sweep parameters.
    zero_padding_factor : int
        FFT zero-padding factor.
    tow_length_mm : float
        Tow length (for sampling rate).

    Returns
    -------
    results : dict
        Contains:
            - "best_bins_per_tow": dict of {tow: best_bins}
            - "mean_best_bins": float
            - "std_best_bins": float
    """

    global Consecutive_Error_Bins

    best_bins_dict = {}
    print(f"[INFO] Analyzing {len(tow_range)} tows...\n")

    # --- Loop through all tows ---
    for tow in tow_range:
        print(f"→ Tow {tow}...")
        best_bins, _ = find_best_bins_fft_mse_real_vs_sim(
            tow=tow,
            bins_min=bins_min,
            bins_max=bins_max,
            bins_step=bins_step,
            zero_padding_factor=zero_padding_factor,
            tow_length_mm=tow_length_mm,
            show_plot=False)  # Disable per-tow plotting
        
        best_bins_dict[tow] = best_bins
        print(f"   ✅ Best bins for Tow {tow}: {best_bins}\n")

    # --- Compute stats ---
    best_bins_array = np.array(list(best_bins_dict.values()))
    mean_bins = np.mean(best_bins_array)
    std_bins = np.std(best_bins_array)
    print("========================================================")
    print(f"✅ Mean Optimal Consecutive_Error_Bins: {mean_bins:.2f}")
    print(f"✅ Std Dev of Optimal Bins: {std_bins:.2f}")
    print("========================================================\n")

    # --- Plot example FFT spectra using mean bin count ---
    example_tow = tow_range[len(tow_range)//2]  # pick a midrange tow as representative
    Consecutive_Error_Bins = int(round(mean_bins))

    print(f"[INFO] Plotting example FFT for Tow {example_tow} using mean bins = {Consecutive_Error_Bins}")

    real_data, sim_data = plot_simulated_vs_real_tow(
        tow=example_tow,
        tow_length_mm=tow_length_mm,
        plot=False,
        force_steps=True)

    real_y = real_data["centerline"].to_numpy()
    sim_y = sim_data["centerline"].to_numpy()
    n = min(len(real_y), len(sim_y))
    real_y = real_y[:n]
    sim_y = sim_y[:n]

    dx = tow_length_mm / (n - 1)
    fs = 1.0 / dx

    # --- Compute complex FFTs ---
    def full_fft(signal, fs, pad_factor):
        n = len(signal)
        n_pad = int(pad_factor * n)
        padded = np.pad(signal, (0, n_pad - n), mode="constant") if n_pad > n else signal
        freq = np.fft.fftfreq(len(padded), d=1.0 / fs)
        fft_vals = np.fft.fft(padded)
        pos = freq > 0
        return freq[pos], fft_vals[pos]

    freq_real, fft_real = full_fft(real_y, fs, zero_padding_factor)
    freq_sim, fft_sim = full_fft(sim_y, fs, zero_padding_factor)

    amp_real = 2.0 * np.abs(fft_real) / len(real_y)
    amp_sim = 2.0 * np.abs(fft_sim) / len(sim_y)
    phase_real = np.angle(fft_real)
    phase_sim = np.angle(fft_sim)

    # --- Plot amplitude + phase spectra ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Example Tow {example_tow} — FFT Spectra @ Mean Bins = {int(round(mean_bins))}", fontsize=13)

    # Amplitude plot
    ax1.plot(freq_real, amp_real, label="Real Tow", color="blue")
    ax1.plot(freq_sim, amp_sim, label="Simulated Tow", color="gold", alpha=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)

    # Phase plot
    ax2.plot(freq_real, phase_real, label="Real Tow Phase", color="blue")
    ax2.plot(freq_sim, phase_sim, label="Sim Tow Phase", color="gold", alpha=0.8)
    ax2.set_xlabel("Frequency [cycles/mm]")
    ax2.set_ylabel("Phase [radians]")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # --- Return results ---
    return {
        "best_bins_per_tow": best_bins_dict,
        "mean_best_bins": mean_bins,
        "std_best_bins": std_bins}

def lenghts_consecutive_error_bins_mse(
    real_hist_bins=350,
    sim_hist_bins=100,
    tow_length_mm=1000,
    num_tows=30,
    bin_start=5,
    bin_end=750,
    bin_step=10,
    verbose=True,
    plot=True,
    force_steps=True):
    """
    Optimize Consecutive_Error_Bins using MSE, testing bins in steps and plotting MSE vs bins.
    
    Args:
        real_hist_bins: histogram bins for real traverse tow data
        sim_hist_bins: histogram bins for simulated multi-tow data
        tow_length_mm: length of simulated tows
        num_tows: number of simulated tows
        bin_start: starting number of consecutive error bins
        bin_end: maximum number of consecutive error bins (exclusive)
        bin_step: step size for bin search
        verbose: print progress
        plot: whether to plot MSE vs bins
    
    Returns:
        best_bin: optimal number of consecutive error bins
        results: dict with MSEs for each tested bin count
    """

    # --- Get real traverse tow distributions ---
    gap_real, overlap_real, _, _ = traverse_tow_gaps_and_overlaps_lengths(plot=False, histogram_bins=real_hist_bins, force_steps=force_steps)

    results = {}
    tested_bins = []

    for bins in range(bin_start, bin_end, bin_step):
        # --- Simulate multi-tow ---
        _, gap_sim, overlap_sim, _, _ = generate_multitow_layout_lengths(
            num_tows=num_tows,
            tow_length_mm=tow_length_mm,
            plot=False,
            histogram_bins=sim_hist_bins,
            num_bins=bins)

        # --- Compute MSE for gaps ---
        min_gap = min(gap_real.min(), gap_sim.min())
        max_gap = max(gap_real.max(), gap_sim.max())
        gap_edges = np.linspace(min_gap, max_gap, real_hist_bins + 1)
        hist_gap_real, _ = np.histogram(gap_real, bins=gap_edges)
        hist_gap_sim, _ = np.histogram(gap_sim, bins=gap_edges)
        mse_gap = np.mean((hist_gap_real - hist_gap_sim) ** 2)

        # --- Compute MSE for overlaps ---
        min_overlap = min(overlap_real.min(), overlap_sim.min())
        max_overlap = max(overlap_real.max(), overlap_sim.max())
        overlap_edges = np.linspace(min_overlap, max_overlap, real_hist_bins + 1)
        hist_overlap_real, _ = np.histogram(overlap_real, bins=overlap_edges)
        hist_overlap_sim, _ = np.histogram(overlap_sim, bins=overlap_edges)
        mse_overlap = np.mean((hist_overlap_real - hist_overlap_sim) ** 2)

        total_mse = mse_gap + mse_overlap
        results[bins] = {"mse_gap": mse_gap, "mse_overlap": mse_overlap, "total_mse": total_mse}
        tested_bins.append(bins)

        if verbose:
            print(f"Bins={bins}, MSE_gap={mse_gap:.3f}, MSE_overlap={mse_overlap:.3f}, Total={total_mse:.3f}")

    # --- Find best bin ---
    best_bin = min(results, key=lambda k: results[k]["total_mse"])
    if verbose:
        print(f"\nOptimal Consecutive_Error_Bins (MSE): {best_bin} with total MSE={results[best_bin]['total_mse']:.3f}")

    # --- Plot MSE vs bins ---
    if plot:
        total_mse_values = [results[b]["total_mse"] for b in tested_bins]
        mse_gap_values = [results[b]["mse_gap"] for b in tested_bins]
        mse_overlap_values = [results[b]["mse_overlap"] for b in tested_bins]

        plt.figure(figsize=(9, 6))
        plt.plot(tested_bins, mse_gap_values, 'o-', color='green', label='MSE (Gaps)')
        plt.plot(tested_bins, mse_overlap_values, 's--', color='orange', label='MSE (Overlaps)')
        plt.plot(tested_bins, total_mse_values, 'd-', color='blue', label='Total MSE')

        plt.axvline(best_bin, color='red', linestyle='--', label=f"Optimal bins={best_bin}")
        plt.xlabel("Consecutive Error Bins")
        plt.ylabel("Mean Squared Error")
        plt.title("MSE vs Consecutive Error Bins (Gaps, Overlaps, and Total)")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.show()

    return best_bin, results

##############################################################################################################
"""Run this file"""

def main():
    # find_best_nsteps_and_bins_edges()

    # best_bins, mse_curve = find_best_bins_fft_mse_real_vs_sim(tow=7,bins_min=2,bins_max=3000,bins_step=20,zero_padding_factor=2)
    # results = analyze_all_tows_best_bins_fft_mse(tow_range=range(2, 31),bins_min=20,bins_max=500,bins_step=5,zero_padding_factor= 2)
    lenghts_consecutive_error_bins_mse()

if __name__ == "__main__":
    main()
