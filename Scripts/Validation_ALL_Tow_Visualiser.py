"""This file is used to generate a figures of simulated tows vs experimental tows.
   Written by: Martijn van der Voort, Clifton-John Walle and Manuel Cruz"""

##############################################################################################################

# External imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import random
from scipy.stats import norm, pareto
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

#Internal imports
from Data_ALL_importer import LLS_B_excel_to_array, CAM_excel_to_array, LT_x_excel_to_array, LT_y_normalized_excel_to_array
from Handling_ALL_Functions import get_synced_data
from Data_ALL_traverse import traverse_tow_constructor, traverse_tow_gaps_and_overlaps, traverse_tow_gaps_and_overlaps_lengths
from D04_Model.Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path
from D04_Model.Model_ALL_Simulation import generate_multitow_layout, generate_multitow_layout_lengths
from Model_ALL_RandomWalk import plot_RW_tows, generate_RW_multitow, generate_RW_multitow_layout_lengths
from Model_ALL_RandomSampling import generate_RS_multitow, generate_RS_multitow_layout_lengths
from constants import number_of_steps, Consecutive_Error_Bins, y_offset_traverse, y_increment_traverse, y_increment_programmed, font_extra_small, font_small, font_medium, font_large, font_extra_large

##############################################################################################################
"""Functions"""

# Functions to plot traverse tows
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
        target_points = number_of_steps
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

# Functions to plot model tow comparisons
def plot_real_vs_D04_tow(tow: int, tow_length_mm=1000, scaled: bool = False, plot: bool = True, force_steps: bool = False):
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
    real_data = plot_real_tow(tow, tow_length_mm=tow_length_mm, plot=False, force_steps=force_steps)

    # Extract real data
    real_x_right = real_data["x_right"].to_numpy()
    real_y_right = real_data["y_right"].to_numpy()
    real_x_left = real_data["x_left"].to_numpy()
    real_y_left = real_data["y_left"].to_numpy()
    real_x_centerline  = real_data["x_centerline"].to_numpy()
    real_y_centerline  = real_data["y_centerline"].to_numpy()

    # Translate tow to start around where tow 1 would be for comparison!
    real_offset = 112 + (tow - 1)*y_increment_traverse
    real_y_centerline = real_y_centerline - real_offset
    real_y_left = real_y_left - real_offset
    real_y_right = real_y_right - real_offset

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

    cam_path = generate_error_path(start_cam, n_steps, bin_stats_cam, slope_cam, intercept_cam,
                                   x_sorted_cam, bin_edges_cam, devs_cam)
    lt_path = generate_error_path(start_lt, n_steps, bin_stats_lt, slope_lt, intercept_lt,
                                  x_sorted_lt, bin_edges_lt, devs_lt)
    tow_centerline_sim = cam_path + lt_path

    width_error = generate_error_path(start_llsb, n_steps, bin_stats_llsb, slope_llsb, intercept_llsb,
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
        plt.plot(real_x_centerline, real_y_centerline, "--", color="blue", alpha=0.4, label="Real centerline")
        plt.plot(      real_x_left,       real_y_left,  "-", color="blue",   alpha=1, label="Real edges")
        plt.plot(     real_x_right,      real_y_right,  "-", color="blue",   alpha=1)

        # Simulated tow
        plt.plot(sim_x, sim_data["centerline"], "--", color="gold", label="D04 tow centerline")
        plt.plot(sim_x, sim_data["top_edge"], "-", color="gold", alpha=1, label="D04 tow edges")
        plt.plot(sim_x, sim_data["bottom_edge"], "-", color="gold", alpha=1)

        plt.xlabel("Tow length (mm)", fontsize=14)
        plt.ylabel("Lateral Position (mm)", fontsize=14)
        plt.legend()
        plt.grid(True)

        if scaled:
            plt.axis("equal")

        plt.tight_layout()
        plt.show()

    return real_data, sim_data

def plot_real_vs_RW_tow(tow: int, tow_length_mm=1000, scaled: bool = False, plot: bool = True, force_steps: bool = False):
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
    real_data = plot_real_tow(tow, tow_length_mm=tow_length_mm, plot=False, force_steps=force_steps)

    # Extract real data
    real_x_right = real_data["x_right"].to_numpy()
    real_y_right = real_data["y_right"].to_numpy()
    real_x_left = real_data["x_left"].to_numpy()
    real_y_left = real_data["y_left"].to_numpy()
    real_x_centerline  = real_data["x_centerline"].to_numpy()
    real_y_centerline  = real_data["y_centerline"].to_numpy()

    # Translate tow to start around where tow 1 would be for comparison!
    real_offset = 111.7 + (tow - 1)*y_increment_traverse
    real_y_centerline = real_y_centerline - real_offset
    real_y_left = real_y_left - real_offset
    real_y_right = real_y_right - real_offset

    # --- Get RW tow ---
    gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent, RW_all_tows_data = generate_RW_multitow(num_tows=1)
    RW_data = RW_all_tows_data[0]

    # Extract RW data
    RW_centerline = RW_data["centerline"].to_numpy()
    RW_top_edge = RW_data["top_edge"].to_numpy()
    RW_bottom_edge = RW_data["bottom_edge"].to_numpy()
    RW_x = RW_data["x_mm"].to_numpy()

    # --- Plot both tows ---
    if plot:
        plt.figure(figsize=(10, 6))

        # Real tow
        plt.plot(real_x_centerline, real_y_centerline, "--", color="blue", alpha=0.4, label="Real centerline")
        plt.plot(      real_x_left,       real_y_left,  "-", color="blue",   alpha=1, label="Real edges")
        plt.plot(     real_x_right,      real_y_right,  "-", color="blue",   alpha=1)

        # Simulated tow
        plt.plot(RW_x, RW_centerline, "--", color="gold", alpha=0.4, label="RW centerline")
        plt.plot(RW_x, RW_top_edge,    "-", color="gold",   alpha=1, label="RW edges")
        plt.plot(RW_x, RW_bottom_edge, "-", color="gold",   alpha=1)

        plt.xlabel("Tow length (mm)", fontsize=14)
        plt.ylabel("Lateral Position (mm)", fontsize=14)
        plt.legend()
        plt.grid(True)

        if scaled:
            plt.axis("equal")

        plt.tight_layout()
        plt.show()

    return real_data, RW_data

def plot_real_vs_D04_vs_RW_vs_RS_tow(tow: int, tow_length_mm=1000, force_steps: bool = False, offset: float=y_increment_programmed, save_PDF=False):
    """
    Make a figure with tows below each other obtained from the 4 different methods for visual comparison.
    """
    
    # Extract real tow
    real_df = traverse_tow_constructor(tow, normalize=True)
    print(real_df)
    x_real_right = real_df["x_right"].to_numpy()
    y_real_right = real_df["y_right"].to_numpy() + 2 * y_increment_programmed
    x_real_left = real_df["x_left"].to_numpy()
    y_real_left = real_df["y_left"].to_numpy() + 2 * y_increment_programmed
    x_real_y_centerline  = real_df["x_centerline"].to_numpy()
    y_real_y_centerline  = real_df["y_centerline"].to_numpy() + 2 * y_increment_programmed

    #Uniformly drop datapoints to get to target_points steps per meter of tow
    if force_steps:
        target_points = 370
        n_points = len(x_real_y_centerline)
        if n_points > target_points:
            indices = np.linspace(0, n_points - 1, target_points, dtype=int)
            x_real_right = x_real_right[indices]
            y_real_right = y_real_right[indices]
            x_real_left = x_real_left[indices]
            y_real_left = y_real_left[indices]
            x_real_y_centerline = x_real_y_centerline[indices]
            y_real_y_centerline = y_real_y_centerline[indices]
            real_df = real_df.iloc[indices].reset_index(drop=True)

    # Extract D04 tow
    _, sim_df = plot_real_vs_D04_tow(tow, tow_length_mm=tow_length_mm, scaled=False, plot=False)
    print(sim_df)
    x_D04_right = sim_df["x_mm"].to_numpy()
    y_D04_right = sim_df["bottom_edge"].to_numpy() + 2 * y_increment_programmed
    x_D04_left = sim_df["x_mm"].to_numpy()
    y_D04_left = sim_df["top_edge"].to_numpy() + 2 * y_increment_programmed
    x_D04_centerline  = sim_df["x_mm"].to_numpy()
    y_D04_centerline  = sim_df["centerline"].to_numpy() + 2 * y_increment_programmed

    # Extract RW tow
    _, _, _, _, _, RW_df_list = generate_RW_multitow(num_tows=1)
    print(RW_df_list)
    RW_df = RW_df_list[0]
    x_RW_right = RW_df["x_mm"].to_numpy() 
    y_RW_right = RW_df["bottom_edge"].to_numpy() + y_increment_programmed
    x_RW_left = RW_df["x_mm"].to_numpy()
    y_RW_left = RW_df["top_edge"].to_numpy() + y_increment_programmed
    x_RW_centerline  = RW_df["x_mm"].to_numpy()
    y_RW_centerline  = RW_df["centerline"].to_numpy() + y_increment_programmed

    # Extract RS tow
    _, RS_df_list, _, _ = generate_RS_multitow(num_tows=1, n_steps=370)
    print(RS_df_list)
    RS_df = RS_df_list[0]
    x_RS_right = RS_df["x_mm"].to_numpy()
    y_RS_right = RS_df["bottom_edge"].to_numpy()
    x_RS_left = RS_df["x_mm"].to_numpy()
    y_RS_left = RS_df["top_edge"].to_numpy()
    x_RS_centerline  = RS_df["x_mm"].to_numpy()
    y_RS_centerline  = RS_df["centerline"].to_numpy()

    plt.figure(figsize=(10,6))
    # Real tow
    plt.plot(x_real_y_centerline, y_real_y_centerline, "--", color="blue")
    plt.plot(x_real_left, y_real_left, "-", color="blue", label="Experimental")
    plt.plot(x_real_right, y_real_right, "-", color="blue")

    # Simulated tow
    #plt.plot(x_D04_centerline, y_D04_centerline, "--", color="orange", label="D04 centerline")
    #plt.plot(x_D04_left, y_D04_left, "-", color="orange", label="D04 edges")
    #plt.plot(x_D04_right, y_D04_right, "-", color="orange")

    # Random Walk tow
    plt.plot(x_RW_centerline, y_RW_centerline, "--", color="green")
    plt.plot(x_RW_left, y_RW_left, "-", color="green", label="MCMC simulation")
    plt.plot(x_RW_right, y_RW_right, "-", color="green")

    # Random sampling tow
    plt.plot(x_RS_centerline, y_RS_centerline, "--", color="orange")
    plt.plot(x_RS_left, y_RS_left, "-", color="orange", label="MC simulation")
    plt.plot(x_RS_right, y_RS_right, "-", color="orange")

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    plt.xlabel("Tow Length (mm)", fontsize=12, fontname='Times New Roman')
    plt.ylabel("Position (mm)", fontsize=12, fontname='Times New Roman')
    plt.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncols=1, 
                frameon=True,              
                edgecolor="black",         
                fancybox=False,            
                framealpha=1.0         
)
    
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_edgecolor('black')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(top=True, bottom=True, left=True, right=True, direction='in', length=8, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(10)
    plt.tight_layout()
    if save_PDF == True:
        plt.savefig("Tow comparison of 3 methods.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    #plt.legend()
    #plt.xlabel("X (mm)", fontsize=font_large)
    #plt.ylabel("Y (mm)", fontsize=font_large)
    #plt.grid()
    #plt.tight_layout()
    #plt.show()

# Functions to compare tow values and model parameters
def compare_real_vs_D04_tow(tow: int, tow_length_mm=1000, plot: bool = True):
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
    _, sim_df = plot_real_vs_D04_tow(tow, tow_length_mm=tow_length_mm, scaled=False, plot=plot)

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

def compare_real_vs_D04_tow_multiple_simulations(tow: int, n_simulations: int = 50, tow_length_mm: int = 1000):
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
        errors = compare_real_vs_D04_tow(tow, tow_length_mm=tow_length_mm, plot=False)
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

# Functions to compare gap and overlaps for each model
def compare_real_vs_D04_gaps_overlaps(tow_length_mm=1000):
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
    _, _, _, sim_gap_percent, sim_overlap_percent, _ = generate_RW_multitow(num_tows=30)

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

def compare_real_vs_D04_gaps_overlaps_lengths(histogram_bins=100, force_steps=False):
    """
    Compare gaps and overlaps between traverse tows and simulated multi-tows.
    Generates two plots: one for gaps, one for overlaps.
    Ensures histograms use identical bin edges so they align visually.
    """

    # --- Run traverse tow analysis ---
    gap_traverse, overlap_traverse, gap_fit_traverse, overlap_fit_traverse = traverse_tow_gaps_and_overlaps_lengths(plot=False, histogram_bins=histogram_bins, force_steps=force_steps)

    # --- Run simulated multi-tow analysis ---
    _, gap_sim, overlap_sim, gap_fit_sim, overlap_fit_sim = generate_multitow_layout_lengths(num_tows=30, plot=False, histogram_bins=histogram_bins)

    # --- Create figure with 2 subplots ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # --- Define shared bin edges for fair comparison ---
    # GAPS
    all_gaps = np.concatenate([gap_traverse, gap_sim])
    gap_min, gap_max = np.min(all_gaps), np.max(all_gaps)
    shared_gap_bins = np.linspace(gap_min, gap_max, histogram_bins + 1)

    # OVERLAPS
    all_overlaps = np.concatenate([overlap_traverse, overlap_sim])
    overlap_min, overlap_max = np.min(all_overlaps), np.max(all_overlaps)
    shared_overlap_bins = np.linspace(overlap_min, overlap_max, histogram_bins + 1)

    # --- Helper to overlay Pareto fit ---
    def plot_pareto_overlay(ax, data, fit, bins, color, label):
        if len(data) == 0:
            return
        counts, bin_edges, _ = ax.hist(
            data,
            bins=bins,
            alpha=0.5,
            color=color,
            density=False,
            label=label,
            edgecolor="black")

        # Overlay Pareto fit
        x = np.linspace(min(data), max(data), 400)
        pdf = pareto.pdf(x, fit["shape"], loc=fit["loc"], scale=fit["scale"])
        bin_width = bin_edges[1] - bin_edges[0]
        pdf_scaled = pdf * len(data) * bin_width
        ax.plot(x, pdf_scaled, color=color, linestyle='--', linewidth=2,
                label=f"{label} Pareto α={fit['shape']:.2f}")
        ax.axvline(fit["mean"], color=color, linestyle='--', linewidth=1.5,
                   label=f"{label} mean={fit['mean']:.2f}")

    # --- Gaps subplot ---
    plot_pareto_overlay(ax[0], gap_traverse, gap_fit_traverse, shared_gap_bins, 'blue', 'Traverse Tows')
    plot_pareto_overlay(ax[0], gap_sim, gap_fit_sim, shared_gap_bins, 'orange', 'Simulated Multi-Tows')
    ax[0].set_xlabel("Gap Length (mm)")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Gap Length Comparison with Pareto Fits")
    ax[0].legend(fontsize=9)
    ax[0].grid(True, linestyle=":")

    # --- Overlaps subplot ---
    plot_pareto_overlay(ax[1], overlap_traverse, overlap_fit_traverse, shared_overlap_bins, 'blue', 'Traverse Tows')
    plot_pareto_overlay(ax[1], overlap_sim, overlap_fit_sim, shared_overlap_bins, 'orange', 'Simulated Multi-Tows')
    ax[1].set_xlabel("Overlap Length (mm)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Overlap Length Comparison with Pareto Fits")
    ax[1].legend(fontsize=9)
    ax[1].grid(True, linestyle=":")

    plt.tight_layout()
    plt.show()

    # --- Print summaries ---
    print("--- Traverse Tows ---")
    print(f"Gaps: N={len(gap_traverse)}, mean={gap_fit_traverse['mean']:.2f} mm, std={gap_fit_traverse['std']:.2f} mm")
    print(f"Overlaps: N={len(overlap_traverse)}, mean={overlap_fit_traverse['mean']:.2f} mm, std={overlap_fit_traverse['std']:.2f} mm\n")

    print("--- Simulated Multi-Tows ---")
    print(f"Gaps: N={len(gap_sim)}, mean={gap_fit_sim['mean']:.2f} mm, std={gap_fit_sim['std']:.2f} mm")
    print(f"Overlaps: N={len(overlap_sim)}, mean={overlap_fit_sim['mean']:.2f} mm, std={overlap_fit_sim['std']:.2f} mm\n")

def compare_real_vs_RW_simulated_gaps_overlaps_lengths(
    histogram_bins=300,
    force_steps=True,
    num_tows_sim=29,
    proposal_type="RWM",
    override=False,
    starting_mods=[None, 1, 1],
    alternate_start=[None, "params"]):
    """
    Compare gap and overlap length distributions between real traverse tows
    and RW-simulated multi-tows.

    Generates two plots: one for gaps, one for overlaps,
    with aligned histogram bins (no Pareto fitting).

    Args:
        histogram_bins : int
            Number of bins for histograms (default=300)
        force_steps : bool
            Passed to traverse_tow_gaps_and_overlaps_lengths
        num_tows_sim : int
            Number of simulated tows (default=30)
        proposal_type : str
            RW proposal type (default="RWM")
        override : bool
            If True, generates ideal tows instead of random-walked ones
        starting_mods, alternate_start : list
            Passed to generate_RW_multitow_layout_lengths

    Returns:
        None
        (Plots and prints basic statistics)
    """

    # --- Get real traverse tow data ---
    gap_traverse, overlap_traverse, *_ = traverse_tow_gaps_and_overlaps_lengths(
        plot=False,
        histogram_bins=histogram_bins,
        force_steps=force_steps)

    # --- Get RW simulated tow data ---
    _, gap_sim, overlap_sim, hist_data_sim = generate_RW_multitow_layout_lengths(
        num_tows=num_tows_sim,
        proposal_type=proposal_type,
        plot=False,
        override=override,
        histogram_bins=histogram_bins,
        starting_mods=starting_mods,
        alternate_start=alternate_start)

    # --- Shared histogram bins for fair comparison ---
    def shared_bins(data1, data2):
        all_data = np.concatenate([data1, data2]) if len(data1) and len(data2) else np.array([])
        if len(all_data):
            return np.linspace(np.min(all_data), np.max(all_data), histogram_bins + 1)
        else:
            return histogram_bins

    shared_gap_bins = shared_bins(gap_traverse, gap_sim)
    shared_overlap_bins = shared_bins(overlap_traverse, overlap_sim)

    # --- Plot histograms ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 3))

    # Gaps
    ax[0].hist(gap_traverse, bins=shared_gap_bins, color="blue", alpha=0.5,
               label="Traverse Tows", edgecolor="black")
    ax[0].hist(gap_sim, bins=shared_gap_bins, color="orange", alpha=0.5,
               label="RW Simulated", edgecolor="black")
    ax[0].set_xlim(0, 150)
    ax[0].set_ylim(0, 70)
    ax[0].set_xlabel("Gap Length (mm)")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Gap Length Comparison (Traverse vs RW Simulation)")
    ax[0].legend(fontsize=9)
    ax[0].grid(True, linestyle=":")

    # Overlaps
    ax[1].hist(overlap_traverse, bins=shared_overlap_bins, color="blue", alpha=0.5,
               label="Traverse Tows", edgecolor="black")
    ax[1].hist(overlap_sim, bins=shared_overlap_bins, color="orange", alpha=0.5,
               label="RW Simulated", edgecolor="black")
    ax[1].set_xlim(0, 150)
    ax[1].set_ylim(0, 70)
    ax[1].set_xlabel("Overlap Length (mm)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Overlap Length Comparison (Traverse vs RW Simulation)")
    ax[1].legend(fontsize=9)
    ax[1].grid(True, linestyle=":")

    plt.tight_layout()
    plt.show()

    # --- Print summaries ---
    def summarize(label, data):
        if len(data):
            print(f"{label}: N={len(data)}, mean={np.mean(data):.2f} mm, std={np.std(data):.2f} mm")
        else:
            print(f"{label}: No data")

    print("\n--- Traverse Tows ---")
    summarize("Gaps", gap_traverse)
    summarize("Overlaps", overlap_traverse)

    print("\n--- RW Simulated Multi-Tows ---")
    summarize("Gaps", gap_sim)
    summarize("Overlaps", overlap_sim)

def compare_real_vs_RS_simulated_gaps_overlaps_lengths(
    histogram_bins=300,
    force_steps=True,
    num_tows_sim=30,
    method="Sidd",
    print_statement=False):
    """
    Compare gap and overlap length distributions between real traverse tows
    and RS-simulated multi-tows.

    Generates two plots: one for gaps, one for overlaps,
    with aligned histogram bins (no Pareto fitting).

    Args:
        histogram_bins : int
            Number of bins for histograms (default=300)
        force_steps : bool
            Passed to traverse_tow_gaps_and_overlaps_lengths
        num_tows_sim : int
            Number of simulated tows (default=30)
        method : str
            RS width generation method ("Sidd" or "Random")
        print_statement : bool
            Whether to print layout summary (passed to RS generator)

    Returns:
        None
        (Plots and prints statistical summaries)
    """

    # --- Get real traverse tow data ---
    gap_traverse, overlap_traverse, *_ = traverse_tow_gaps_and_overlaps_lengths(
        plot=False,
        histogram_bins=histogram_bins,
        force_steps=force_steps
    )

    # --- Get RS simulated tow data ---
    _, gap_sim, overlap_sim, hist_data_sim = generate_RS_multitow_layout_lengths(
        num_tows=num_tows_sim,
        method=method,
        plot=False,
        histogram_bins=histogram_bins,
        print_statement=print_statement
    )

    # --- Define shared histogram bins for fair comparison ---
    def shared_bins(data1, data2):
        all_data = np.concatenate([data1, data2]) if len(data1) and len(data2) else np.array([])
        if len(all_data):
            return np.linspace(np.min(all_data), np.max(all_data), histogram_bins + 1)
        else:
            return histogram_bins

    shared_gap_bins = shared_bins(gap_traverse, gap_sim)
    shared_overlap_bins = shared_bins(overlap_traverse, overlap_sim)

    # --- Plot histograms ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 3))

    # Gaps
    ax[0].hist(gap_traverse, bins=shared_gap_bins, color="blue", alpha=0.5,
               label="Traverse Tows", edgecolor="black")
    ax[0].hist(gap_sim, bins=shared_gap_bins, color="green", alpha=0.5,
               label="RS Simulated", edgecolor="black")
    ax[0].set_xlim(0, 150)
    ax[0].set_ylim(0, 200)
    ax[0].set_xlabel("Gap Length (mm)")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Gap Length Comparison (Traverse vs RS Simulation)")
    ax[0].legend(fontsize=9)
    ax[0].grid(True, linestyle=":")

    # Overlaps
    ax[1].hist(overlap_traverse, bins=shared_overlap_bins, color="blue", alpha=0.5,
               label="Traverse Tows", edgecolor="black")
    ax[1].hist(overlap_sim, bins=shared_overlap_bins, color="green", alpha=0.5,
               label="RS Simulated", edgecolor="black")
    ax[1].set_xlim(0, 150)
    ax[1].set_ylim(0, 200)
    ax[1].set_xlabel("Overlap Length (mm)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Overlap Length Comparison (Traverse vs RS Simulation)")
    ax[1].legend(fontsize=9)
    ax[1].grid(True, linestyle=":")

    plt.tight_layout()
    plt.show()

    # --- Print summaries ---
    def summarize(label, data):
        if len(data):
            print(f"{label}: N={len(data)}, mean={np.mean(data):.2f} mm, std={np.std(data):.2f} mm")
        else:
            print(f"{label}: No data")

    print("\n--- Traverse Tows ---")
    summarize("Gaps", gap_traverse)
    summarize("Overlaps", overlap_traverse)

    print("\n--- RS Simulated Multi-Tows ---")
    summarize("Gaps", gap_sim)
    summarize("Overlaps", overlap_sim)

def compare_real_vs_RS_RW_gap_length_distributions(
    histogram_bins=300,
    force_steps=True,
    num_tows_sim=30,
    proposal_type="RWM",
    override=False,
    starting_mods=[None, 1, 1],
    alternate_start=[None, "params"],
    method="Sidd",
    print_statement=False,
    xlim=(0, 140),
    ylim=(0, 108),
    ylim_break=(1065, 1090),
    stack_graphs=False):
    
    from brokenaxes import brokenaxes

    # --- Real data ---
    gap_traverse, _, *_ = traverse_tow_gaps_and_overlaps_lengths(
        plot=False, histogram_bins=histogram_bins, force_steps=force_steps)

    # --- RW simulated ---
    _, gap_RW, _, _ = generate_RW_multitow_layout_lengths(
        num_tows=num_tows_sim,
        proposal_type=proposal_type,
        plot=False,
        override=override,
        histogram_bins=histogram_bins,
        starting_mods=starting_mods,
        alternate_start=alternate_start)

    # --- RS simulated ---
    _, gap_RS, _, _ = generate_RS_multitow_layout_lengths(
        num_tows=num_tows_sim,
        method=method,
        plot=False,
        histogram_bins=histogram_bins,
        print_statement=print_statement)

    # --- Compute shared bins ---
    all_data = np.concatenate([gap_traverse, gap_RW, gap_RS])
    shared_bins = np.linspace(np.min(all_data), np.max(all_data), histogram_bins + 1)

    # --- Plot ---
    if stack_graphs:
        fig = plt.figure(figsize=(12, 8))
        spec = fig.add_gridspec(2, 1)
    else:
        fig = plt.figure(figsize=(14, 4))
        spec = fig.add_gridspec(1, 2)

    font_size = 23
    tick_size = font_size - 10

    # --- Traverse vs RW ---
    ax0 = fig.add_subplot(spec[0])
    ax0.hist(gap_traverse, bins=shared_bins, color="blue", alpha=0.5, edgecolor="black", label="Traverse Tows")
    ax0.hist(gap_RW, bins=shared_bins, color="green", alpha=0.5, edgecolor="black", label="RW Simulated")
    ax0.set_xlim(*xlim)
    ax0.set_ylim(0, 140)
    ax0.set_xlabel("Gap Length (mm)", fontsize=font_size, fontname="Times New Roman")
    ax0.set_ylabel("Count", fontsize=font_size, fontname="Times New Roman")
    ax0.tick_params(axis='both', labelsize=tick_size)
    ax0.legend(prop={"family": "Times New Roman", "size": font_size - 5})

    # --- Traverse vs RS (broken y-axis) ---
    bax1 = brokenaxes(
        ylims=[ylim, ylim_break],
        hspace=.1,
        subplot_spec=spec[1])
    bax1.hist(gap_traverse, bins=shared_bins, color="blue", alpha=0.5, edgecolor="black", label="Traverse Tows")
    bax1.hist(gap_RS, bins=shared_bins, color="orange", alpha=0.5, edgecolor="black", label="RS Simulated")
    bax1.set_xlim(*xlim)
    bax1.set_xlabel("Gap Length (mm)", fontsize=font_size, fontname="Times New Roman", labelpad=25)
    bax1.set_ylabel("Count", fontsize=font_size, fontname="Times New Roman", labelpad=40)
    for ax in bax1.axs:
        ax.tick_params(axis='both', labelsize=tick_size)
    bax1.legend(prop={"family": "Times New Roman", "size": font_size - 5})

    for ax in bax1.axs:
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1)

    bax1.axs[0].spines['bottom'].set_visible(False)
    bax1.axs[1].spines['top'].set_visible(False)

    plt.tight_layout()
    fig.savefig("gap_length_comparison.svg", format="svg", dpi=300)
    plt.show()

    # --- Summaries ---
    def summarize(name, data):
        if len(data):
            print(f"{name}: N={len(data)}, Mean={np.mean(data):.2f} mm, Std={np.std(data):.2f} mm")
        else:
            print(f"{name}: No data")

    print("\n--- Summary Statistics ---")
    summarize("Traverse Gaps", gap_traverse)
    summarize("RW Simulated Gaps", gap_RW)
    summarize("RS Simulated Gaps", gap_RS)

##############################################################################################################
"""Run this file"""

def main():
    # plot_real_tow(6)

    # real_df, sim_df = plot_real_vs_RW_tow(6, plot = True, force_steps = False)
    # compare_real_vs_RW_gaps_overlaps()

    # compare_simulated_vs_real_tow(8)
    #compare_multiple_simulations(8, 50)
    #plot_real_vs_D04_vs_RW_vs_RS_tow(2, save_PDF=True)
    compare_real_vs_RW_gaps_overlaps()
    #compare_real_vs_RW_simulated_gaps_overlaps_lengths(histogram_bins=300)
    # compare_real_vs_RS_simulated_gaps_overlaps_lengths(histogram_bins=300)
    #compare_real_vs_RS_RW_gap_length_distributions(histogram_bins=300, stack_graphs=True)

if __name__ == "__main__":
    main()


    