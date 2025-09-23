# -*- coding: utf-8 -*-
"""This file is used to find the optimum number of steps and bins — EDGES version.

What it does
------------
Grid-search over (n_steps, num_bins) by comparing **edge spectra** (FFT magnitudes):

  • Simulated edges are built EXACTLY like the visualizer:
        centerline = CAM + LT
        width      = NOMINAL_WIDTH_MM + LLS_B_error
        top_edge    = centerline + 0.5 * width
        bottom_edge = centerline - 0.5 * width

  • Real edges come from traverse via:
        Model_ALL_Validation_Tow_Visualiser.plot_real_tow(tow, plot=False)
    We align to a common x-range, resample to the candidate n_steps grid,
    and (optionally) remove DC to avoid offset bias in FFT magnitude comparison.

Objective
---------
Average the FFT magnitude MSE over the two edges:
    0.5 * [ MSE( |FFT(SimTop)|, |FFT(RealLeft)| ) + MSE( |FFT(SimBottom)|, |FFT(RealRight)| ) ]
"""

##############################################################################################################
# External imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import random

##############################################################################################################
# Internal imports
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path
from Model_ALL_Validation_Tow_Visualiser import plot_real_tow
try:
    from constants import NOMINAL_WIDTH_MM
except Exception:
    NOMINAL_WIDTH_MM = 6.35  # Fallback if not defined

##############################################################################################################
"""Functions"""

# ----------------- Helpers -----------------

def uniform_resample(x, y, n_steps: int):
    """
    Resample (x, y) to 'n_steps' uniformly spaced points between x[0] and x[-1].
    Returns (x_uni, y_uni).
    """
    x0, x1 = float(x[0]), float(x[-1])
    x_uni = np.linspace(x0, x1, int(n_steps), endpoint=True)
    y_uni = np.interp(x_uni, x, y)
    return x_uni, y_uni


def single_sided_rfft_mag(signal: np.ndarray, pad_factor: int = 1):
    """
    Real FFT (rfft) single-sided magnitude, including DC..Nyquist.
    Returns magnitude array (no frequency vector).
    Normalized by input length for comparability across lengths.
    """
    s = np.asarray(signal, dtype=float)
    n = s.size
    n_fft = int(max(n, pad_factor * n))
    mag = np.abs(np.fft.rfft(s, n=n_fft)) / n
    return mag


def build_real_edges_from_visualizer(tow: int, target_steps: int, remove_dc: bool = True):
    """
    Fetch traverse edges exactly like the visualizer via:
        plot_real_tow(tow, plot=False)
    Steps:
      - Align left/right to common overlapping x-range
      - Put right onto left's x via interpolation
      - Resample both to 'target_steps' on the same uniform grid
      - Optionally remove DC (mean)

    Returns
    -------
    x_uni : (target_steps,)
    y_left_uni : (target_steps,)
    y_right_uni : (target_steps,)
    """
    real_df = plot_real_tow(tow, plot=False)

    x_r = real_df["x_right"].to_numpy()
    y_r = real_df["y_right"].to_numpy()
    x_l = real_df["x_left"].to_numpy()
    y_l = real_df["y_left"].to_numpy()

    # Align to common x-overlap
    x_min = max(x_l[0], x_r[0])
    x_max = min(x_l[-1], x_r[-1])
    mask_l = (x_l >= x_min) & (x_l <= x_max)
    mask_r = (x_r >= x_min) & (x_r <= x_max)
    x_l, y_l = x_l[mask_l], y_l[mask_l]
    x_r, y_r = x_r[mask_r], y_r[mask_r]

    # Put right on left's x, then resample both to a shared uniform grid
    y_r_on_l = np.interp(x_l, x_r, y_r)
    x_uni, y_left_uni = uniform_resample(x_l, y_l, target_steps)
    _,     y_right_uni = uniform_resample(x_l, y_r_on_l, target_steps)

    if remove_dc:
        y_left_uni  = y_left_uni  - np.mean(y_left_uni)
        y_right_uni = y_right_uni - np.mean(y_right_uni)

    return x_uni, y_left_uni, y_right_uni


def simulate_edges_like_visualizer(n_steps: int,
                                   tow_length_mm: float,
                                   num_bins: int,
                                   use_seed: bool = False,
                                   random_seed: int = 0):
    """
    Simulate edges IDENTICALLY to the visualizer:

        centerline = CAM + LT
        width      = NOMINAL_WIDTH_MM + LLS_B_error
        top_edge    = centerline + 0.5 * width
        bottom_edge = centerline - 0.5 * width

    Returns
    -------
    x_sim : (n_steps,)
    top_edge : (n_steps,)
    bottom_edge : (n_steps,)
    """
    if use_seed:
        np_rng_state = np.random.get_state()
        py_rng_state = random.getstate()
        np.random.seed(int(random_seed))
        random.seed(int(random_seed))

    try:
        # Fit models with the candidate num_bins
        _, sc, ic, _, _, _, xsc, bec, dvc = consecutive_error(
            "CAM", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
            random_state=random.randint(0, 10_000)
        )
        _, sl, il, _, _, _, xsl, bel, dvl = consecutive_error(
            "LT", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
            random_state=random.randint(0, 10_000)
        )
        _, sw, iw, _, _, _, xsw, bew, dvw = consecutive_error(
            "LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
            random_state=random.randint(0, 10_000)
        )

        # Start ranges (same as visualizer)
        start_cam  = random.uniform(-0.75,  0.75)
        start_lt   = random.uniform(-0.90, -0.70)
        start_llsb = random.uniform(-0.21, -0.02)

        cam_path = generate_error_path(start_cam,  n_steps, sc,  ic,  xsc,  bec, dvc)
        lt_path  = generate_error_path(start_lt,   n_steps, sl,  il,  xsl,  bel, dvl)
        w_err    = generate_error_path(start_llsb, n_steps, sw,  iw,  xsw,  bew, dvw)

        centerline = cam_path + lt_path
        width = float(NOMINAL_WIDTH_MM) + w_err

        top_edge    = centerline + 0.5 * width
        bottom_edge = centerline - 0.5 * width

        x_sim = np.linspace(0.0, tow_length_mm, int(n_steps), endpoint=True)
        return x_sim, top_edge, bottom_edge

    finally:
        if use_seed:
            np.random.set_state(np_rng_state)
            random.setstate(py_rng_state)


# ----------------- Main optimizer (EDGES) -----------------

def find_best_nsteps_and_bins(
    tow_range=range(2, 8),
    nsteps_candidates=None,
    bin_candidates=None,
    n_repeats: int = 6,
    length_tow_mm: float = 1000.0,
    zero_padding_factor: int = 2,
    use_seed: bool = False,
    random_seed: int = 0,
    remove_dc_real: bool = True,
    plot_surface_3d: bool = True,
):
    """
    Grid-search optimizer that matches **edge spectra** (SimTop/SimBottom vs RealLeft/RealRight).

    Parameters
    ----------
    tow_range : iterable
        Set of tows to include.
    nsteps_candidates : list[int]
        Candidate n_steps values (uniform resampling + sim length discretization).
    bin_candidates : list[int]
        Candidate num_bins values for the consecutive-error models.
    n_repeats : int
        Number of random simulations per grid point to average out randomness.
    length_tow_mm : float
        Physical tow length for the simulated x-grid.
    zero_padding_factor : int
        rFFT zero-padding factor for spectrum smoothness.
    use_seed : bool
        If True, seeds RNG for reproducibility per repeat.
    random_seed : int
        Base seed when use_seed=True.
    remove_dc_real : bool
        Remove mean from real edges before FFT magnitude.
    plot_surface_3d : bool
        Whether to show the 3D surface.

    Returns
    -------
    mse_surface : np.ndarray, shape (len(bin_candidates), len(nsteps_candidates))
    optimal_steps : int
    optimal_bins : int
    """
    if nsteps_candidates is None:
        nsteps_candidates = list(range(180, 541, 20))
    if bin_candidates is None:
        bin_candidates = list(range(60, 241, 20))

    # Precompute REAL edge spectra per (tow, n_steps)
    real_cache = {}  # key: (tow, n_steps) -> (mag_left, mag_right)
    for tow in tow_range:
        for n_steps in nsteps_candidates:
            _, yL_real, yR_real = build_real_edges_from_visualizer(
                tow, target_steps=n_steps, remove_dc=remove_dc_real
            )
            mag_L = single_sided_rfft_mag(yL_real, pad_factor=zero_padding_factor)
            mag_R = single_sided_rfft_mag(yR_real, pad_factor=zero_padding_factor)
            real_cache[(tow, n_steps)] = (mag_L, mag_R)

    mse_surface = np.zeros((len(bin_candidates), len(nsteps_candidates)), dtype=float)

    # Grid search
    for b_idx, num_bins in enumerate(bin_candidates):
        for s_idx, n_steps in enumerate(nsteps_candidates):
            mse_accum = 0.0
            for tow in tow_range:
                mag_L_real, mag_R_real = real_cache[(tow, n_steps)]
                for rep in range(n_repeats):
                    _, yTop_sim, yBot_sim = simulate_edges_like_visualizer(
                        n_steps=n_steps,
                        tow_length_mm=length_tow_mm,
                        num_bins=num_bins,
                        use_seed=use_seed,
                        random_seed=(random_seed + rep) if use_seed else 0
                    )
                    mag_L_sim = single_sided_rfft_mag(yTop_sim, pad_factor=zero_padding_factor)  # SimTop ↔ RealLeft
                    mag_R_sim = single_sided_rfft_mag(yBot_sim, pad_factor=zero_padding_factor)  # SimBottom ↔ RealRight

                    mse_L = mean_squared_error(mag_L_real, mag_L_sim)
                    mse_R = mean_squared_error(mag_R_real, mag_R_sim)
                    mse_accum += 0.5 * (mse_L + mse_R)

            mse_surface[b_idx, s_idx] = mse_accum / float(len(tow_range) * n_repeats)

    # ---- Plot the surface (same style as your first file) ----
    if plot_surface_3d:
        X, Y = np.meshgrid(nsteps_candidates, bin_candidates)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, mse_surface, cmap='viridis')
        ax.set_xlabel("number of steps")
        ax.set_ylabel("number of bins")
        ax.set_zlabel("Total MSE (mean over tows & repeats)")
        plt.tight_layout()
        plt.show()

    # ---- Best configuration (same return signature) ----
    min_idx = np.unravel_index(np.argmin(mse_surface), mse_surface.shape)
    optimal_bins = bin_candidates[min_idx[0]]
    optimal_steps = nsteps_candidates[min_idx[1]]
    best_val = mse_surface[min_idx]
    print(f"Optimal -> n_steps: {optimal_steps}, num_bins: {optimal_bins}, Total MSE: {best_val:.6f}")

    return mse_surface, optimal_steps, optimal_bins


##############################################################################################################
"""Run this file"""

def main():
    find_best_nsteps_and_bins(
        tow_range=range(2, 8),
        nsteps_candidates=list(range(180, 541, 20)),
        bin_candidates=list(range(60, 241, 20)),
        n_repeats=6,
        length_tow_mm=1000.0,
        zero_padding_factor=2,
        use_seed=False,
        random_seed=0,
        remove_dc_real=True,
        plot_surface_3d=True,
    )

if __name__ == "__main__":
    main()

