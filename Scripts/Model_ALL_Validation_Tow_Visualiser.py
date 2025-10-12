"""This file is used to generate a figure of a simulated tow vs an experimental tow.
   Written by: Martijn van der Voort, Clifton-John Walle and Manuel Cruz"""

##############################################################################################################

# External imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import norm

#Internal imports
from Data_ALL_importer import LLS_B_excel_to_array, CAM_excel_to_array, LT_x_excel_to_array, LT_y_normalized_excel_to_array
from Handling_ALL_Functions import get_synced_data
from Data_ALL_traverse import traverse_tow_constructor, traverse_tow_gaps_and_overlaps
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path
from Model_ALL_Simulation import generate_multitow_layout
from Model_ALL_RandomWalk import plot_RW_tows, generate_RW_multitow
from constants import number_of_steps, Consecutive_Error_Bins, y_offset_traverse, y_increment_traverse

##############################################################################################################
"""Functions"""

def plot_real_tow(tow: int, tow_length_mm=1000, plot: bool = True, force_steps: bool = False):
    """
    Plot a real tow profile using Traverse interpolated data
    """

    real_df = traverse_tow_constructor(tow)
    x_right = real_df["x_right"].to_numpy()
    y_right = real_df["y_right"].to_numpy()
    x_left = real_df["x_left"].to_numpy()
    y_left = real_df["y_left"].to_numpy()
    x_centerline  = real_df["x_centerline"].to_numpy()
    y_centerline  = real_df["y_centerline"].to_numpy()

    if force_steps:
        target_points = 340
        n_points = len(x_centerline)
        if n_points > target_points:
            indices = np.linspace(0, n_points - 1, target_points, dtype=int)
            x_right = x_right[indices]
            y_right = y_right[indices]
            x_left = x_left[indices]
            y_left = y_left[indices]
            x_centerline = x_centerline[indices]
            y_centerline = y_centerline[indices]
            real_df = real_df.iloc[indices].reset_index(drop=True)

    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(x_right, y_right, "-", linewidth=2.0, label="Right edge")
        plt.plot(x_left, y_left, "-", linewidth=2.0, label="Left edge")
        plt.plot(x_centerline, y_centerline, "--", linewidth=2.0, label="Centerline")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.title(f"Real tow {tow} from traverse interpolated data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return real_df

def plot_simulated_vs_real_tow(tow: int, tow_length_mm=1000, scaled: bool = False, plot: bool = True, force_steps: bool = False):
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
    real_df = plot_real_tow(tow, tow_length_mm=tow_length_mm, plot=False, force_steps=force_steps)

    # Extract edges
    real_x_right = real_df["x_right"].to_numpy()
    real_y_right = real_df["y_right"].to_numpy()
    real_x_left = real_df["x_left"].to_numpy()
    real_y_left = real_df["y_left"].to_numpy()

    '''real_x = real_df["x_right"].to_numpy()
    real_y_right = real_df["y_right"].to_numpy()
    real_y_left = real_df["y_left"].to_numpy()'''

    # Compute centerline
    real_centerline = 0.5 * (real_y_left + real_y_right)
    x_centerline = 0.5 * (real_x_right + real_x_left)

    # Translate tow to start around where tow 1 would be for comparison!
    real_offset = 112 + (tow - 1)*y_increment_traverse
    real_centerline = real_centerline - real_offset
    real_y_left = real_y_left - real_offset
    real_y_right = real_y_right - real_offset

    real_data = pd.DataFrame({
        "x_right_edge": real_x_right,
        "x_left_edge": real_x_left,
        "centerline": real_centerline,
        "x_centerline": x_centerline,
        "left_edge": real_y_left,
        "right_edge": real_y_right,
        "width": real_y_left - real_y_right})

    # --- Generate simulated tow ---
    gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent = generate_multitow_layout(
        num_tows=1, tow_length_mm=tow_length_mm, plot=False)

    num_bins = Consecutive_Error_Bins
    n_steps = number_of_steps

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

    sim_data = pd.DataFrame({
        "x_mm": sim_x,
        "centerline": tow_centerline_sim,
        "top_edge": sim_top,
        "bottom_edge": sim_bottom,
        "width": tow_widths_sim})

    # --- Plot both ---
    if plot:
        plt.figure(figsize=(10, 6))

        # Real tow
        plt.plot(real_data["x_centerline"], real_data["centerline"], "--", color="blue", label="Real centerline")
        plt.plot(real_data["x_left_edge"], real_data["left_edge"], "-", color="blue", alpha=1, label="Real edges")
        plt.plot(real_data["x_right_edge"], real_data["right_edge"], "-", color="blue", alpha=1)

        # Simulated tow
        plt.plot(sim_data["x_mm"], sim_data["centerline"], "--", color="gold", label="Sim centerline")
        plt.plot(sim_data["x_mm"], sim_data["top_edge"], "-", color="gold", alpha=1, label="Sim edges")
        plt.plot(sim_data["x_mm"], sim_data["bottom_edge"], "-", color="gold", alpha=1)

        plt.xlabel("Tow length (mm)", fontsize=14)
        plt.ylabel("Lateral Position (mm)", fontsize=14)
        plt.legend()
        plt.grid(True)

        if scaled:
            plt.axis("equal")

        plt.tight_layout()
        plt.show()

    return real_data, sim_data

def plot_simulated_vs_RW_tow(tow: int, tow_length_mm=1000, scaled: bool = False, plot: bool = True, force_steps: bool = False):
    """
    Overlay a simulated tow on a real tow.
    Real tow is built from Traverse Data
    Simulated tow is generated with RW data.

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
    real_df = plot_real_tow(tow, tow_length_mm=tow_length_mm, plot=False, force_steps=force_steps)

    # Extract edges
    real_x_right = real_df["x_right"].to_numpy()
    real_y_right = real_df["y_right"].to_numpy()
    real_x_left = real_df["x_left"].to_numpy()
    real_y_left = real_df["y_left"].to_numpy()

    '''real_x = real_df["x_right"].to_numpy()
    real_y_right = real_df["y_right"].to_numpy()
    real_y_left = real_df["y_left"].to_numpy()'''

    # Compute centerline
    real_centerline = 0.5 * (real_y_left + real_y_right)
    x_centerline = 0.5 * (real_x_right + real_x_left)

    # Translate tow to start around where tow 1 would be for comparison!
    real_offset = 111.7 + (tow - 1)*y_increment_traverse
    real_centerline = real_centerline - real_offset
    real_y_left = real_y_left - real_offset
    real_y_right = real_y_right - real_offset

    real_data = pd.DataFrame({
        "x_right_edge": real_x_right,
        "x_left_edge": real_x_left,
        "centerline": real_centerline,
        "x_centerline": x_centerline,
        "left_edge": real_y_left,
        "right_edge": real_y_right,
        "width": real_y_left - real_y_right})

    sim_data = plot_RW_tows()

    sim_centerline = sim_data["y"].to_numpy() + sim_data["center_CAM"].to_numpy()
    sim_top_edge = sim_centerline + 0.5*sim_data["width_LLS_B"].to_numpy()
    sim_bottom_edge = sim_centerline - 0.5*sim_data["width_LLS_B"].to_numpy()
    sim_x = sim_data["x"].to_numpy()

    sim_data = pd.DataFrame({
        "x_mm": sim_x,
        "centerline": sim_centerline,
        "top_edge": sim_top_edge,
        "bottom_edge": sim_bottom_edge,
        "width": sim_top_edge - sim_bottom_edge})

    # --- Plot both ---
    if plot:
        plt.figure(figsize=(10, 6))

        # Real tow
        plt.plot(real_data["x_centerline"], real_data["centerline"], "--", color="blue", label="Real centerline")
        plt.plot(real_data["x_left_edge"], real_data["left_edge"], "-", color="blue", alpha=1, label="Real edges")
        plt.plot(real_data["x_right_edge"], real_data["right_edge"], "-", color="blue", alpha=1)

        # Simulated tow
        plt.plot(sim_data["x_mm"], sim_data["centerline"], "--", color="gold", label="Sim centerline")
        plt.plot(sim_data["x_mm"], sim_data["top_edge"], "-", color="gold", alpha=1, label="Sim edges")
        plt.plot(sim_data["x_mm"], sim_data["bottom_edge"], "-", color="gold", alpha=1)

        plt.xlabel("Tow length (mm)", fontsize=14)
        plt.ylabel("Lateral Position (mm)", fontsize=14)
        plt.legend()
        plt.grid(True)

        if scaled:
            plt.axis("equal")

        plt.tight_layout()
        plt.show()

    return real_data, sim_data

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

def compare_real_vs_simulated_gaps_overlaps(tow_length_mm=1000):
    """
    Compare real traverse tow layout gap/overlap percentages 
    with simulated tow layout percentages (no plotting).

    Parameters
    ----------
    tow_length_mm : int, optional
        Tow length in mm (default 1000).
    """

    # Get real gap/overlap data from traverse layout
    _, _, _, real_gap_percent, real_overlap_percent = traverse_tow_gaps_and_overlaps(plot=False)

    # Generate a full simulated layout (like multitow layout generation)
    print("=== Calculating Simulated Percentages (May take 3-5 minutes) ===")
    _, _, _, sim_gap_percent, sim_overlap_percent = generate_multitow_layout(num_tows=30, tow_length_mm=tow_length_mm, plot=False)

    # --- Print comparison ---
    print("\n=== Comparison Summary ===")
    print(f"Real   Gap Percentage:      {real_gap_percent:.2f}%")
    print(f"Simulated Gap Percentage:   {sim_gap_percent:.2f}%")
    print(f"Real   Overlap Percentage:  {real_overlap_percent:.2f}%")
    print(f"Simulated Overlap Percentage: {sim_overlap_percent:.2f}%")

    # --- Return structured data for further analysis ---
    return {
        "real_gap_percent": real_gap_percent,
        "real_overlap_percent": real_overlap_percent,
        "sim_gap_percent": sim_gap_percent,
        "sim_overlap_percent": sim_overlap_percent}

def compare_real_vs_RW_gaps_overlaps():
    """
    Compare real traverse tow layout gap/overlap percentages 
    with simulated tow layout percentages (no plotting).

    Parameters
    ----------
    tow_length_mm : int, optional
        Tow length in mm (default 1000).
    """

    # Get real gap/overlap data from traverse layout
    _, _, _, real_gap_percent, real_overlap_percent = traverse_tow_gaps_and_overlaps(plot=False)

    # Generate a full simulated layout (like multitow layout generation)
    print("=== Calculating Simulated Percentages (May take 3-5 minutes) ===")
    _, _, _, sim_gap_percent, sim_overlap_percent = generate_RW_multitow(num_tows=30)

    # --- Print comparison ---
    print("\n=== Comparison Summary ===")
    print(f"Real   Gap Percentage:      {real_gap_percent:.2f}%")
    print(f"Simulated Gap Percentage:   {sim_gap_percent:.2f}%")
    print(f"Real   Overlap Percentage:  {real_overlap_percent:.2f}%")
    print(f"Simulated Overlap Percentage: {sim_overlap_percent:.2f}%")

    # --- Return structured data for further analysis ---
    return {
        "real_gap_percent": real_gap_percent,
        "real_overlap_percent": real_overlap_percent,
        "sim_gap_percent": sim_gap_percent,
        "sim_overlap_percent": sim_overlap_percent}

##############################################################################################################
"""Run this file"""

def main():
    # plot_real_tow(6)

    real_df, sim_df = plot_simulated_vs_real_tow(6, plot = True, force_steps = True)
    compare_real_vs_simulated_gaps_overlaps()

    real_df, sim_df = plot_simulated_vs_RW_tow(6, plot = True, force_steps = True)
    compare_real_vs_RW_gaps_overlaps()

    # compare_simulated_vs_real_tow(8)
    # compare_multiple_simulations(8, 50)

if __name__ == "__main__":
    main()