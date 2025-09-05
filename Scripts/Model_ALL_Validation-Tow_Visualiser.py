"""This file is used to generate a figure of a simulated tow vs an experimental tow.
   This file is currently not being used for anything except plotting
   Written by: Manuel Cruz"""

##############################################################################################################

# External imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from scipy.fft import fft, fftfreq

#Internal imports
from Handling_ALL_Functions import get_synced_data
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path
from Model_ALL_Simulation import generate_multitow_layout

##############################################################################################################
"""Functions"""

def plot_real_tow(tow: int, tow_length_mm=1000, plot=False):
    """
    Plot a real tow profile using Traverse data
    """

    real_df = get_synced_data(tow, "TRAVERSE")

    # Extract columns
    x_pos_lower = real_df.iloc[:, 0].to_numpy()
    x_pos_upper = real_df.iloc[:,2].to_numpy()
    upper_edge = real_df.iloc[:, 3].to_numpy()
    lower_edge = real_df.iloc[:, 1].to_numpy()

    '''x-coordinate will be the average of the measurements for 
    the bottom edge x-position and  upper edge x-position'''

    x_pos = (x_pos_lower+x_pos_upper)/2 * 1000  # It is currently in meters, we need in mm
    centerline = (lower_edge + upper_edge)/2
    width = abs(lower_edge - upper_edge)
    mean_center = np.mean(centerline)

    # Because the y-position of a tow is not around y=0, we have to 'normalize'
    # the edges so that the experimental tow is around y=0
    lower_edge = lower_edge - mean_center
    upper_edge = upper_edge - mean_center
    centerline = (lower_edge + upper_edge)/2

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(x_pos, centerline, "--", linewidth=1.5, label="Centerline")
        plt.plot(x_pos, lower_edge, "-", linewidth=2.0, label="Left edge")
        plt.plot(x_pos, upper_edge, "-", linewidth=2.0, label="Right edge")
        plt.xlabel("Tow length (mm)")
        plt.ylabel("Position (mm)")
        plt.title(f"Real Tow {tow}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return pd.DataFrame({
        "x_mm": x_pos,
        "centerline": centerline,
        "left_edge": upper_edge,
        "right_edge": lower_edge,
        "width": width})

def plot_simulated_vs_real_tow(tow: int, tow_length_mm=1000, scaled: bool = False, plot: bool = True):
    """
    Overlay a simulated tow on a real tow.
    Real tow is built from Traverse Data
    Simulated tow is generated with generate_multitow_layout.

    Parameters
    ----------
    tow : int
        Tow index.
    tow_length_mm : int, optional
        Tow length in mm (default 1000).
    scaled : bool, optional
        If True, plot is shown with 1:1 scale (equal aspect ratio).
    """

    # --- Get real tow ---
    real_df = plot_real_tow(tow, tow_length_mm=tow_length_mm)

    # --- Generate one simulated tow (no plot, just data) ---
    gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent = generate_multitow_layout(
        num_tows=1, tow_length_mm=tow_length_mm, plot=False)

    # --- For single tow, regenerate its geometry ---
    num_bins = 80
    n_steps = int(tow_length_mm * 340 / 1000)

    # Load error model fits
    bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = consecutive_error(
        "CAM", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = consecutive_error(
        "LT", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_llsb, slope_llsb, intercept_llsb, _, _, _, x_sorted_llsb, bin_edges_llsb, devs_llsb = consecutive_error(
        "LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))

    # Generate simulated paths
    start_cam = random.uniform(-0.75, 0.75)
    start_lt = random.uniform(-0.9, -0.7)
    start_llsb = random.uniform(-0.21, -0.02)

    cam_path = generate_error_path(start_cam, n_steps, slope_cam, intercept_cam,
                                   x_sorted_cam, bin_edges_cam, devs_cam)
    lt_path = generate_error_path(start_lt, n_steps, slope_lt, intercept_lt,
                                  x_sorted_lt, bin_edges_lt, devs_lt)
    tow_centerline_sim = cam_path + lt_path

    width_error = generate_error_path(start_llsb, n_steps, slope_llsb, intercept_llsb,
                                      x_sorted_llsb, bin_edges_llsb, devs_llsb)
    tow_widths_sim = 6.35 + width_error  # nominal width + error

    sim_top = tow_centerline_sim + 0.5 * tow_widths_sim
    sim_bottom = tow_centerline_sim - 0.5 * tow_widths_sim
    sim_x = np.linspace(0, tow_length_mm, len(tow_centerline_sim))

    # --- Plot both ---
    if plot == True:
        plt.figure(figsize=(10, 6))

        # Real tow
        plt.plot(real_df["x_mm"], real_df["centerline"], "--", color="blue", label="Real centerline")
        plt.plot(real_df["x_mm"], real_df["left_edge"], "-", color="blue", alpha=0.6, label="Real edges")
        plt.plot(real_df["x_mm"], real_df["right_edge"], "-", color="blue", alpha=0.6)

        # Simulated tow
        plt.plot(sim_x, tow_centerline_sim, "--", color="red", label="Sim centerline")
        plt.plot(sim_x, sim_top, "-", color="red", alpha=0.6, label="Sim edges")
        plt.plot(sim_x, sim_bottom, "-", color="red", alpha=0.6)

        plt.xlabel("Tow length (mm)", fontsize=14)
        plt.ylabel("Position (mm)", fontsize=14)
        plt.title(f"Tow {tow}: Real vs Simulated", fontsize=16)
        plt.legend()
        plt.grid(True)

        # Apply 1:1 scale if requested
        if scaled:
            plt.axis("equal")  # ensures X and Y have same scale

        plt.tight_layout()
        plt.show()

    return real_df, pd.DataFrame({
        "x_mm": sim_x,
        "centerline": tow_centerline_sim,
        "top_edge": sim_top,
        "bottom_edge": sim_bottom,
        "width": tow_widths_sim})

def compare_simulated_vs_real_tow(tow: int, tow_length_mm=1000, plot: bool = True):
    """
    Compare simulated and real tow edges by calculating average lateral error.

    Parameters
    ----------
    tow : int
        Tow index.
    tow_length_mm : int, optional
        Tow length in mm (default 1000).

    Returns
    -------
    errors : dict
        Dictionary with average errors between real and simulated tow edges.
        Includes mean absolute error (MAE) and root mean square error (RMSE).
    """

    # --- Get real tow ---
    real_df = plot_real_tow(tow, tow_length_mm=tow_length_mm)

    # --- Generate one simulated tow (no plot, just data) ---
    _, sim_df = plot_simulated_vs_real_tow(tow, tow_length_mm=tow_length_mm, scaled=False, plot=plot)

    # --- Interpolate real edges to simulated x positions ---
    real_left_interp = np.interp(sim_df["x_mm"], real_df["x_mm"], real_df["left_edge"])
    real_right_interp = np.interp(sim_df["x_mm"], real_df["x_mm"], real_df["right_edge"])

    # --- Compute edge errors (corrected mapping) ---
    err_top = sim_df["top_edge"] - real_left_interp      # Sim top vs Real left
    err_bottom = sim_df["bottom_edge"] - real_right_interp  # Sim bottom vs Real right

    # --- Error metrics ---
    mae_top = float(np.mean(np.abs(err_top)))
    mae_bottom = float(np.mean(np.abs(err_bottom)))
    rmse_top = float(np.sqrt(np.mean(err_top**2)))
    rmse_bottom = float(np.sqrt(np.mean(err_bottom**2)))

    errors = {
        "MAE_top_edge (SimTop vs RealLeft)": mae_top,
        "MAE_bottom_edge (SimBottom vs RealRight)": mae_bottom,
        "RMSE_top_edge (SimTop vs RealLeft)": rmse_top,
        "RMSE_bottom_edge (SimBottom vs RealRight)": rmse_bottom}

    print("MAE_top_edge (SimTop vs RealLeft)", mae_top, "mm")
    print("MAE_bottom_edge (SimBottom vs RealRight)", mae_bottom, "mm")
    print("RMSE_top_edge (SimTop vs RealLeft)", rmse_top, "mm")
    print("RMSE_bottom_edge (SimBottom vs RealRight)", rmse_bottom, "mm")

    return errors

def compare_multiple_simulations(tow: int, n_simulations: int = 50, tow_length_mm: int = 1000):
    """
    Run multiple simulated vs real tow comparisons and return error statistics,
    plus plot histograms with Gaussian fits of the error distributions.

    Parameters
    ----------
    tow : int
        Tow index.
    n_simulations : int, optional
        Number of simulation runs (default = 50).
    tow_length_mm : int, optional
        Tow length in mm (default 1000).

    Returns
    -------
    stats : dict
        Dictionary with mean and std deviation of the errors across runs.
    """

    results = []

    for i in range(n_simulations):
        errors = compare_simulated_vs_real_tow(tow, tow_length_mm=tow_length_mm, plot=False)
        results.append(errors)

    # Convert to DataFrame for easier aggregation
    df = pd.DataFrame(results)

    stats = {}
    for col in df.columns:
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std()
        }

    print("\n=== Error Statistics over", n_simulations, "simulations ===")
    for col in df.columns:
        print(f"{col}: mean = {stats[col]['mean']:.3f} mm, std = {stats[col]['std']:.3f} mm")

    # --- Plot histograms with Gaussian fit ---
    num_metrics = len(df.columns)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 4), constrained_layout=True)

    if num_metrics == 1:
        axes = [axes]

    for ax, col in zip(axes, df.columns):
        data = df[col].values
        mu, sigma = stats[col]["mean"], stats[col]["std"]

        # Histogram
        count, bins, _ = ax.hist(data, bins=50, density=True, alpha=0.6,
                                 color="steelblue", edgecolor="black")

        # Gaussian curve
        x = np.linspace(min(bins), max(bins), 200)
        pdf = norm.pdf(x, mu, sigma)
        ax.plot(x, pdf, "r--", linewidth=2, label=f"N({mu:.2f}, {sigma:.2f}²)")

        ax.set_title(col, fontsize=12)
        ax.set_xlabel("Error (mm)")
        ax.set_ylabel("Density")
        ax.legend()

    plt.show()

    return stats

##############################################################################################################
"""TEST functions to check FFT""" # Use Model_ALL_Validation-FFT.py instead of this function!

def compare_fft_real_vs_sim(real_df: pd.DataFrame, sim_df: pd.DataFrame, tow_length_mm=1000):
    """
    Perform FFT on real vs simulated tow centerlines and compare spectra.
    
    Parameters
    ----------
    real_df : DataFrame
        Output from plot_real_tow(...) containing centerline.
    sim_df : DataFrame
        Output from plot_simulated_vs_real_tow(...) containing centerline.
    tow_length_mm : float
        Length of the tow in mm (default 1000).
    """

    # --- Extract centerlines ---
    real = real_df["centerline"].to_numpy()
    sim = sim_df["centerline"].to_numpy()

    # --- Align lengths (truncate to shortest) ---
    min_len = min(len(real), len(sim))
    real = real[:min_len]
    sim = sim[:min_len]

    # --- Sampling step (assume uniform spacing in mm) ---
    dx = tow_length_mm / min_len

    # --- FFT ---
    freqs = fftfreq(min_len, d=dx)[:min_len // 2]  # positive freqs only
    fft_real = np.abs(fft(real)[:min_len // 2])
    fft_sim = np.abs(fft(sim)[:min_len // 2])

    # --- Normalize (optional for comparison) ---
    fft_real /= np.max(fft_real)
    fft_sim /= np.max(fft_sim)

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_real, label="Real Tow FFT", color="blue")
    plt.plot(freqs, fft_sim, label="Simulated Tow FFT", color="red", alpha=0.7)
    plt.xlabel("Spatial frequency (1/mm)", fontsize=14)
    plt.ylabel("Normalized magnitude", fontsize=14)
    plt.title("FFT Comparison: Real vs Simulated Tow", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return freqs, fft_real, fft_sim

##############################################################################################################
"""Run this file"""

def main():
    
    plot_simulated_vs_real_tow(8)
    # compare_fft_real_vs_sim(real_df, sim_df)

    #compare_simulated_vs_real_tow(8)
    # compare_multiple_simulations(8, 50)

if __name__ == "__main__":
    main()