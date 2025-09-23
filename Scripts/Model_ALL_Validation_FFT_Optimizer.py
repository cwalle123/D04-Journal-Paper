# -*- coding: utf-8 -*-
"""
Model_ALL_Validation_FFT_Optimizer — EDGES version (safe rFFT)

What it does
------------
Grid-search over (n_steps, num_bins) by comparing **edge spectra** (FFT magnitudes):

  • Simulated edges are built EXACTLY like the visualizer:
        centerline = CAM + LT
        width      = NOMINAL_WIDTH_MM + LLS_B_error
        sim_top    = centerline + 0.5 * width
        sim_bottom = centerline - 0.5 * width

  • Experimental edges come from traverse via:
        df = traverse_tow_constructor(tow)
        real_left(x), real_right(x)

We align experimental edges to a uniform grid of n_steps over [0, L_mm],
(optionally) remove DC, compute one-sided |rFFT| (dropping DC only),
and minimize the average magnitude MSE:

    0.5 * [ MSE( |FFT(sim_top)|, |FFT(real_left)| )
          + MSE( |FFT(sim_bottom)|, |FFT(real_right)| ) ]

Notes
-----
- Pairing matches the visualizer convention: sim_top ↔ real_left, sim_bottom ↔ real_right.
- Uses rFFT and trims to the common length to avoid off-by-one (Nyquist) mismatches.
- Light local cache for model params per `num_bins` (no changes to your global cache system).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error
import random
from typing import Dict, Tuple, Any

# --- Project imports (non-standalone, as in your repo) ---
from Data_ALL_traverse import traverse_tow_constructor
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path

# ----------------- Config -----------------
NOMINAL_WIDTH_MM = 6.35

# ----------------- Local cache -----------------
# Cache model parameters per num_bins to avoid recomputation in the grid search
_model_cache: Dict[int, Tuple[Any, ...]] = {}

def _one_sided_fft_mag(signal_1d: np.ndarray) -> np.ndarray:
    """
    Consistent one-sided magnitude spectrum for MSE comparison.
    - Uses rFFT (real FFT) → length n//2+1 (even) or (n+1)//2 (odd)
    - Normalizes by n
    - Drops DC (index 0), keeps the rest (including Nyquist if present)
    """
    y = np.asarray(signal_1d, dtype=float)
    n = y.size
    if n < 2:
        return np.zeros(0, dtype=float)

    spec = np.fft.rfft(y)
    mag = np.abs(spec) / n
    return mag[1:]  # drop DC only

def _edge_fft_mse(sim_edge: np.ndarray, real_edge: np.ndarray) -> float:
    """
    MSE between one-sided FFT magnitudes of two aligned signals.
    Trims to common length to avoid off-by-one issues.
    """
    m_sim = _one_sided_fft_mag(sim_edge)
    m_real = _one_sided_fft_mag(real_edge)
    k = min(m_sim.size, m_real.size)
    if k == 0:
        return float("inf")
    return mean_squared_error(m_sim[:k], m_real[:k])

def _interp_to_uniform(x_src: np.ndarray, y_src: np.ndarray, n_steps: int, L_mm: float):
    """
    Interpolate y(x) defined on x_src to a uniform grid of n_steps spanning [0, L_mm].
    """
    x_uniform = np.linspace(0.0, L_mm, n_steps)
    y_uniform = np.interp(x_uniform, x_src, y_src)
    return x_uniform, y_uniform

def _build_real_edges_like_visualizer(tow: int, n_steps: int, L_mm: float, remove_dc: bool):
    """
    Real (Traverse) edges → align both to the same uniform grid.
    """
    df = traverse_tow_constructor(tow)
    x_left  = df["x_left"].to_numpy()
    y_left  = df["y_left"].to_numpy()
    x_right = df["x_right"].to_numpy()
    y_right = df["y_right"].to_numpy()

    x_u, left_u  = _interp_to_uniform(x_left,  y_left,  n_steps, L_mm)
    _,   right_u = _interp_to_uniform(x_right, y_right, n_steps, L_mm)

    if remove_dc:
        left_u  = left_u  - np.mean(left_u)
        right_u = right_u - np.mean(right_u)

    # Optional sanitization (no behavioral change for normal data)
    left_u  = np.nan_to_num(left_u,  nan=0.0, posinf=0.0, neginf=0.0)
    right_u = np.nan_to_num(right_u, nan=0.0, posinf=0.0, neginf=0.0)

    return x_u, left_u, right_u

def _get_models_for_bins(num_bins: int, rng: random.Random):
    """
    Build model parameters for CAM, LT, LLS_B using consecutive_error
    (same inputs/flags as used in the visualizer path).
    Cached per `num_bins` to avoid recompute across grid.
    """
    if num_bins in _model_cache:
        # Light-touch message; avoids changing your existing cache prints.
        # print(f"[CACHE] Using model params for num_bins={num_bins} from local cache")
        return _model_cache[num_bins]

    # Random seeds mimic the visualizer’s randomized training split
    rs_cam   = rng.randint(0, 10000)
    rs_lt    = rng.randint(0, 10000)
    rs_llsb  = rng.randint(0, 10000)

    # Keep arguments aligned with your project conventions
    _, slope_cam,   intercept_cam,   _, _, _, x_cam,   bin_edges_cam,   devs_cam   = consecutive_error(
        "CAM",   test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False, random_state=rs_cam)
    _, slope_lt,    intercept_lt,    _, _, _, x_lt,    bin_edges_lt,    devs_lt    = consecutive_error(
        "LT",    test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False, random_state=rs_lt)
    _, slope_llsb,  intercept_llsb,  _, _, _, x_llsb,  bin_edges_llsb,  devs_llsb  = consecutive_error(
        "LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False, random_state=rs_llsb)

    params = (slope_cam, intercept_cam, x_cam, bin_edges_cam, devs_cam,
              slope_lt,  intercept_lt,  x_lt,  bin_edges_lt,  devs_lt,
              slope_llsb, intercept_llsb, x_llsb, bin_edges_llsb, devs_llsb)
    _model_cache[num_bins] = params
    return params

def _simulate_edges_like_visualizer(n_steps: int, L_mm: float, model_params, rng: random.Random, remove_dc: bool):
    """
    EXACTLY the visualizer method to make simulated edges:
      centerline = CAM + LT
      width      = NOMINAL_WIDTH_MM + LLS_B_error
      top        = centerline + 0.5*width
      bottom     = centerline - 0.5*width
    """
    (slope_cam, intercept_cam, x_cam, bin_edges_cam, devs_cam,
     slope_lt,  intercept_lt,  x_lt,  bin_edges_lt,  devs_lt,
     slope_llsb, intercept_llsb, x_llsb, bin_edges_llsb, devs_llsb) = model_params

    # Visualizer-style random starts
    start_cam  = rng.uniform(-0.75,  0.75)
    start_lt   = rng.uniform(-0.90, -0.70)
    start_llsb = rng.uniform(-0.21, -0.02)

    cam_path = generate_error_path(start_cam,  n_steps, slope_cam,   intercept_cam,   x_cam,   bin_edges_cam,   devs_cam)
    lt_path  = generate_error_path(start_lt,   n_steps, slope_lt,    intercept_lt,    x_lt,    bin_edges_lt,    devs_lt)
    width_e  = generate_error_path(start_llsb, n_steps, slope_llsb,  intercept_llsb,  x_llsb,  bin_edges_llsb,  devs_llsb)

    centerline = cam_path + lt_path
    width      = NOMINAL_WIDTH_MM + width_e

    sim_top    = centerline + 0.5 * width
    sim_bottom = centerline - 0.5 * width

    if remove_dc:
        sim_top    = sim_top    - np.mean(sim_top)
        sim_bottom = sim_bottom - np.mean(sim_bottom)

    # Optional sanitization
    sim_top    = np.nan_to_num(sim_top,    nan=0.0, posinf=0.0, neginf=0.0)
    sim_bottom = np.nan_to_num(sim_bottom, nan=0.0, posinf=0.0, neginf=0.0)

    x_u = np.linspace(0.0, L_mm, n_steps)
    return x_u, sim_top, sim_bottom

def evaluate_pair_edges_MSE(tow: int, n_steps: int, num_bins: int, L_mm: float = 1000.0,
                            remove_dc: bool = True, rng_seed: int | None = None) -> float:
    """
    Build experimental edges (traverse) and simulated edges (visualizer method),
    and return average magnitude-spectrum MSE across TOP/LEFT and BOTTOM/RIGHT pairs.
    """
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

    # Cache-able model params for this num_bins
    model_params = _get_models_for_bins(num_bins, rng)

    # Real edges (interpolated to uniform grid)
    _, real_left, real_right = _build_real_edges_like_visualizer(tow, n_steps, L_mm, remove_dc=remove_dc)

    # Sim edges (exact visualizer construction)
    _, sim_top, sim_bottom   = _simulate_edges_like_visualizer(n_steps, L_mm, model_params, rng, remove_dc=remove_dc)

    # Pairing: sim_top ↔ real_left, sim_bottom ↔ real_right
    mse_top    = _edge_fft_mse(sim_top,    real_left)
    mse_bottom = _edge_fft_mse(sim_bottom, real_right)
    return 0.5 * (mse_top + mse_bottom)

def find_best_nsteps_and_bins_edges(
    tow_range=range(2, 8),                       # experimental tows to include
    nsteps_candidates=(512, 768, 1024, 1536),    # candidate n_steps
    bin_candidates=(20, 30, 40, 50),             # candidate num_bins
    repeats=3,                                   # repeats per (n_steps, num_bins)
    L_mm: float = 1000.0,
    remove_dc: bool = True,
    base_seed: int = 0
):
    """
    Evaluate average edge-spectrum MSE over a set of tows, candidates, and repeats.
    Returns: dict(best=..., surface=..., steps_list=..., bins_list=...)
    """
    steps_list = list(nsteps_candidates)
    bins_list  = list(bin_candidates)
    surface    = np.zeros((len(steps_list), len(bins_list)), dtype=float)

    # Reproducible-but-diverse trials across the grid
    def seed_for(i_s, i_b, rep):
        return base_seed + 10007 * i_s + 7907 * i_b + 53 * rep

    for i_s, n_steps in enumerate(steps_list):
        for i_b, nb in enumerate(bins_list):
            mselist = []
            for rep in range(repeats):
                rng_seed = seed_for(i_s, i_b, rep)
                tow_mses = []
                for tow in tow_range:
                    mse = evaluate_pair_edges_MSE(
                        tow=tow, n_steps=n_steps, num_bins=nb,
                        L_mm=L_mm, remove_dc=remove_dc, rng_seed=rng_seed
                    )
                    tow_mses.append(mse)
                mselist.append(float(np.mean(tow_mses)))
            surface[i_s, i_b] = float(np.mean(mselist))

    # Find argmin
    idx = np.unravel_index(np.argmin(surface), surface.shape)
    best = {
        "n_steps": steps_list[idx[0]],
        "num_bins": bins_list[idx[1]],
        "mse": float(surface[idx]),
    }
    return {"best": best, "surface": surface, "steps_list": steps_list, "bins_list": bins_list}

# ----------------- Example run -----------------
if __name__ == "__main__":
    result = find_best_nsteps_and_bins_edges(
        tow_range=range(2, 8),
        nsteps_candidates=(512, 768, 1024, 1536, 2048),
        bin_candidates=(20, 30, 40, 50, 60),
        repeats=4,
        L_mm=1000.0,
        remove_dc=True,
        base_seed=0,
    )
    print("Best params:", result["best"])
