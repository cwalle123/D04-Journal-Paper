"""This file deals with generating simulated tows using the model"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
from scipy.fft import fft, fftfreq

# Internal imports
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path
from Handling_ALL_Functions import get_synced_data

##############################################################################################################
"""Functions for generating simulated tows"""

#starting error distribution can be found here, but is assumed to be uniform based on these graphs ranges of values
def fit_starting_error_distribution(sensor: str, plot=True):
    """
    Fits a normal distribution to the first non-NaN values of a sensor's error
    across all tows (2-31). Returns mean, std, and list of first values.
    """
    # Map to correct error column in new get_synced_data
    column_map = {
        "CAM": "error_CAM",
        "LT": "error_LT",
        "LLS_A": "error_LLS_A",
        "LLS_B": "error_LLS_B"
    }

    if sensor not in column_map:
        raise KeyError(f"Unknown sensor '{sensor}'")

    col_name = column_map[sensor]
    first_values = []

    for tow in range(2, 32):  # tows 2-31
        df = get_synced_data(tow, sensor_type=sensor, overwrite=False, helper=False)

        if col_name in df.columns and not df[col_name].isna().all():
            # Take first non-NaN value
            value = df[col_name].dropna().values[0]
            first_values.append(value)

    if len(first_values) == 0:
        raise ValueError(f"No valid first values found for sensor '{sensor}'")

    # Fit normal distribution
    mu, sigma = stats.norm.fit(first_values)

    if plot:
        plt.figure(figsize=(8, 5))
        count, bins, _ = plt.hist(first_values, bins=len(first_values), density=True,
                                  edgecolor="black", alpha=0.7, label="Start Values")
        x = np.linspace(min(bins), max(bins), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', label=f"Fit: μ={mu:.2f}, σ={sigma:.2f}")
        plt.title(f"Start Error Distribution - {sensor}")
        plt.xlabel("Start Error [mm]")
        plt.ylabel("Density")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return mu, sigma, first_values

def generate_multitow_layout(num_tows=5, tow_spacing_mm=6.35, tow_width_mm=6.35, tow_length_mm=1000, cam_start_range=(-0.75, 0.75), lt_start_range=(-0.9, -0.7), llsb_start_range=(-0.21, -0.02), plot=False):
    """
    Generate a multi-tow layout using real error models (CAM, LT, LLS_B).
    Returns gap/overlap DataFrames and percentages.
    """

    num_bins = 80
    n_steps = int(tow_length_mm * 340 / 1000)  # base step count

    # --- Load error model fits ---
    bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = consecutive_error(
        "CAM", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = consecutive_error(
        "LT", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_llsb, slope_llsb, intercept_llsb, _, _, _, x_sorted_llsb, bin_edges_llsb, devs_llsb = consecutive_error(
        "LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))

    # --- Perfect offsets for tow centerlines ---
    tow_offsets = np.linspace(-(num_tows - 1) / 2, (num_tows - 1) / 2, num_tows) * tow_spacing_mm

    top_edge_paths, bottom_edge_paths = [], []

    # --- Simulate each tow ---
    for tow_index, offset in enumerate(tow_offsets):

        # Pick random start values
        start_cam = random.uniform(*cam_start_range)
        start_lt = random.uniform(*lt_start_range)
        start_llsb = random.uniform(*llsb_start_range)

        # Generate centerline errors
        cam_path = generate_error_path(start_cam, n_steps, slope_cam, intercept_cam,
                                       x_sorted_cam, bin_edges_cam, devs_cam)
        lt_path = generate_error_path(start_lt, n_steps, slope_lt, intercept_lt,
                                      x_sorted_lt, bin_edges_lt, devs_lt)
        tow_centerline = offset + cam_path + lt_path

        # Generate width errors
        width_error = generate_error_path(start_llsb, n_steps, slope_llsb, intercept_llsb,
                                          x_sorted_llsb, bin_edges_llsb, devs_llsb)
        tow_widths = tow_width_mm + width_error

        tow_top_edge = tow_centerline + 0.5 * tow_widths
        tow_bottom_edge = tow_centerline - 0.5 * tow_widths

        top_edge_paths.append(tow_top_edge)
        bottom_edge_paths.append(tow_bottom_edge)

        # --- Align x-values and programmed offsets to path length ---
        path_len = len(tow_centerline)
        x_vals = np.linspace(0, tow_length_mm, path_len)
        offset_array = np.full(path_len, offset)

        # --- Plotting ---
        if plot:
            color = plt.get_cmap("tab10")(tow_index % 10)

            # Programmed reference path
            plt.plot(x_vals, offset_array, ":", color="black",
                     linewidth=1, label="Programmed paths" if tow_index == 0 else "_nolegend_")

            # Simulated centerline and edges
            plt.plot(x_vals, tow_centerline, "--", color=color,
                     linewidth=1.5, label="Tow centerlines" if tow_index == 0 else "_nolegend_")
            plt.plot(x_vals, tow_top_edge, "-", color=color,
                     linewidth=2.0, label="Tow edges" if tow_index == 0 else "_nolegend_")
            plt.plot(x_vals, tow_bottom_edge, "-", color=color,
                     linewidth=2.0, label="_nolegend_")

    # --- Gap/Overlap analysis ---
    gap_overlap_dict = {
        f"Gap/overlap_Tow{tow_index+1}_Tow{tow_index+2}": bottom_edge_paths[tow_index+1] - top_edge_paths[tow_index]
        for tow_index in range(num_tows - 1)
    }
    gap_overlap_df = pd.DataFrame(gap_overlap_dict)

    gap_df = gap_overlap_df.where(gap_overlap_df > 0)
    overlap_df = gap_overlap_df.where(gap_overlap_df < 0)

    # --- Area calculations ---
    highest_tow_edge = top_edge_paths[-1]
    lowest_tow_edge = bottom_edge_paths[0]
    total_layout_area = np.trapezoid(highest_tow_edge - lowest_tow_edge, x_vals)

    total_gap_area = sum(np.trapezoid(np.clip(values, 0, None), x_vals) for values in gap_overlap_df.values.T)
    total_overlap_area = sum(np.trapezoid(np.clip(-values, 0, None), x_vals) for values in gap_overlap_df.values.T)

    gap_percent = (total_gap_area / total_layout_area) * 100 if total_layout_area > 0 else 0
    overlap_percent = (total_overlap_area / total_layout_area) * 100 if total_layout_area > 0 else 0

    # --- Plot cosmetics ---
    if plot:
        plt.xlabel("Tow length (mm)", fontsize=14)
        plt.ylabel("Tow position (mm)", fontsize=14)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=10, frameon=True)
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    # --- Print summary ---
    print(f"\nTotal layout area: {total_layout_area:.2f} mm²")
    print(f"Gap area: {total_gap_area:.2f} mm² ({gap_percent:.2f}%)")
    print(f"Overlap area: {total_overlap_area:.2f} mm² ({overlap_percent:.2f}%)")

    return gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent

##############################################################################################################
"""Functions to be used in Model_ALL_ConsecutiveModeler.py"""

def generate_single_tow_error(sensor, tow_length_mm=1000, cam_start_range=(-0.75, 0.75), lt_start_range=(-0.9, -0.7), llsb_start_range=(-0.21, -0.02), llsa_start_range=(-0.43, -0.16)):
    """
    Generate error path for a single tow and a chosen sensor (LT, CAM, LLS_A, LLS_B).

    Returns:
        np.array: error path values (length ~ tow_length_mm*0.34)
    """
    assert sensor in ["LT", "CAM", "LLS_A", "LLS_B"], "Sensor must be one of: LT, CAM, LLS_A, LLS_B"

    num_bins = 80
    n_steps = int(tow_length_mm * 340 / 1000)  # base step count

    # --- Load error model fits for the chosen sensor ---
    bin_stats, slope, intercept, _, _, _, x_sorted, bin_edges, devs = consecutive_error(
        sensor, test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))

    # --- Pick start value depending on sensor ---
    if sensor == "CAM":
        start_val = random.uniform(*cam_start_range)
    elif sensor == "LT":
        start_val = random.uniform(*lt_start_range)
    elif sensor == "LLS_B":
        start_val = random.uniform(*llsb_start_range)
    elif sensor == "LLS_A":
        start_val = random.uniform(*llsa_start_range)

    # --- Generate error path ---
    error_path = generate_error_path(start_val, n_steps, slope, intercept, x_sorted, bin_edges, devs)

    return error_path

def plot_sensor_error_histograms(num_tows=10, tow_length_mm=1000, bins=50):
    sensors = ["LT", "CAM", "LLS_A", "LLS_B"]
    plt.figure(figsize=(14, 6))  # wider since we now have 4 subplots

    for i, sensor in enumerate(sensors, 1):
        all_errors = []

        # generate multiple tows for this sensor
        for _ in range(num_tows):
            err_path = generate_single_tow_error(sensor, tow_length_mm=tow_length_mm)
            all_errors.extend(err_path)

        all_errors = np.array(all_errors)

        # histogram subplot (probability density)
        plt.subplot(1, 4, i)
        plt.hist(all_errors, bins=bins, alpha=0.7, color="tab:blue",
                 edgecolor="black", density=True)
        plt.title(f"{sensor} Error Distribution", fontsize=12)
        plt.xlabel("Error (mm)")
        plt.ylabel("Probability Density")

    plt.tight_layout()
    plt.show()

##############################################################################################################
"""Functions for checking values of the experimental data"""
# (These apperently delete a lot of data!, only use as indicator for percentage of gap overlap)

def calculate_real_gap_overlap_percentages(num_tows=5, tow_spacing_mm=6.35):
    offsets = np.linspace(-(num_tows - 1) / 2, (num_tows - 1) / 2, num_tows) * tow_spacing_mm
    top_lines = []
    bottom_lines = []

    for tow in range(2, 2 + num_tows):
        df = get_synced_data(tow, spacesynced=True)

        cam = df["center_CAM"].dropna().values
        lt = df["error_LT"].dropna().values
        width = df["width_LLS_B"].dropna().values

        min_len = min(len(cam), len(lt), len(width))
        cam = cam[:min_len]
        lt = lt[:min_len]
        width = width[:min_len]

        centerline = cam + lt
        top = centerline + 0.5 * width + offsets[tow - 2]
        bottom = centerline - 0.5 * width + offsets[tow - 2]

        top_lines.append(top)
        bottom_lines.append(bottom)

    # Compute gaps/overlaps only on valid shared ranges
    gap_overlap_data = {}
    total_gap_area = 0.0
    total_overlap_area = 0.0

    for i in range(num_tows - 1):
        top_i = top_lines[i]
        bottom_next = bottom_lines[i + 1]
        common_len = min(len(top_i), len(bottom_next))

        top_i = top_i[:common_len]
        bottom_next = bottom_next[:common_len]

        gap_overlap = bottom_next - top_i
        col_name = f"Gap/overlap_Tow{i+1}_Tow{i+2}"
        gap_overlap_data[col_name] = gap_overlap

        gaps = np.where(gap_overlap > 0, gap_overlap, 0)
        overlaps = np.where(gap_overlap < 0, -gap_overlap, 0)

        total_gap_area += np.trapezoid(gaps)
        total_overlap_area += np.trapezoid(overlaps)

    #Total layout area between outermost top and bottom lines
    topmost = top_lines[-1]
    bottommost = bottom_lines[0]
    common_len_total = min(len(topmost), len(bottommost))
    total_area = np.trapezoid(topmost[:common_len_total] - bottommost[:common_len_total])

    gap_percent = (total_gap_area / total_area) * 100 if total_area > 0 else 0
    overlap_percent = (total_overlap_area / total_area) * 100 if total_area > 0 else 0

    print(f"\n[REAL] Total layout area (unitless): {total_area:.2f}")
    print(f"[REAL] Gap area: {total_gap_area:.2f} ({gap_percent:.2f}%)")
    print(f"[REAL] Overlap area: {total_overlap_area:.2f} ({overlap_percent:.2f}%)")

    return gap_overlap_data, gap_percent, overlap_percent

def simulation_verification(num_simulations=100):
    """
    Run multiple simulations of the multitow layout and compute average gap/overlap percentages.
    Also calculates standard deviation and plots normal distributions + histograms.
    """

    gap_percents = []
    overlap_percents = []

    for i in range(num_simulations):
        _, _, _, gap_percent, overlap_percent = generate_multitow_layout(
            num_tows=5,
            tow_spacing_mm=6.35,
            tow_width_mm=6.35,
            tow_length_mm=1000,
            cam_start_range=(-0.75, 0.75),
            lt_start_range=(-0.9, -0.7),
            llsb_start_range=(-0.21, -0.02),
            plot=False)

        gap_percents.append(gap_percent)
        overlap_percents.append(overlap_percent)

    # --- Stats ---
    avg_gap = np.mean(gap_percents)
    std_gap = np.std(gap_percents)

    avg_overlap = np.mean(overlap_percents)
    std_overlap = np.std(overlap_percents)

    # --- Print results ---
    print(f"\n\nAfter {num_simulations} simulations of 5-tow layout:")
    print(f"Average Gap Percentage: {avg_gap:.2f}% (std: {std_gap:.2f}%)")
    print(f"Average Overlap Percentage: {avg_overlap:.2f}% (std: {std_overlap:.2f}%)")

    # --- Plot distributions + histograms ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms (normalized to probability density)
    ax.hist(gap_percents, bins=20, density=True, alpha=0.5, color="blue", label="Gap % Histogram")
    ax.hist(overlap_percents, bins=20, density=True, alpha=0.5, color="red", label="Overlap % Histogram")

    # Range for plotting normal curves
    if std_gap > 0:
        x_gap = np.linspace(min(gap_percents), max(gap_percents), 200)
        ax.plot(x_gap, stats.norm.pdf(x_gap, avg_gap, std_gap), "b-", linewidth=2, label="Gap Normal Fit")
    if std_overlap > 0:
        x_overlap = np.linspace(min(overlap_percents), max(overlap_percents), 200)
        ax.plot(x_overlap, stats.norm.pdf(x_overlap, avg_overlap, std_overlap), "r-", linewidth=2, label="Overlap Normal Fit")

    ax.set_title(f"Gap/Overlap Distributions ({num_simulations} simulations)")
    ax.set_xlabel("Percentage")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()

    return avg_gap, avg_overlap, std_gap, std_overlap

##############################################################################################################
"""Test functions (NOT TRAVERSE DATA!)"""

def plot_real_tow(tow: int, tow_length_mm=1000, plot=False):
    """
    Plot a real tow profile using synced data:
    centerline = LT_y + CAM
    edges = centerline ± 0.5 * LLS_B_width
    """

    # --- Load synced data ---
    lt_df = get_synced_data(tow, "LT")
    cam_df = get_synced_data(tow, "CAM")
    llsb_df = get_synced_data(tow, "LLS_B")

    # --- Extract needed columns ---
    # LT has X, Y, and error_LT → grab just Y normalized
    lt_y = lt_df.filter(like="y").iloc[:, 0].to_numpy()
    cam_pos = cam_df.iloc[:, 0].to_numpy()
    llsb_width = llsb_df.iloc[:, 0].to_numpy()

    # --- Align lengths (truncate to shortest dataset) ---
    min_len = min(len(lt_y), len(cam_pos), len(llsb_width))
    lt_y = lt_y[:min_len]
    cam_pos = cam_pos[:min_len]
    llsb_width = llsb_width[:min_len]

    # --- Compute real tow geometry ---
    centerline = lt_y + cam_pos
    top_edge = centerline + 0.5 * llsb_width
    bottom_edge = centerline - 0.5 * llsb_width

    # --- X axis (tow length) ---
    x_vals = np.linspace(0, tow_length_mm, min_len)

    # --- Plot ---
    if plot == True:
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, centerline, "--", color="blue", linewidth=1.5, label="Centerline")
        plt.plot(x_vals, top_edge, "-", color="red", linewidth=2.0, label="Top edge")
        plt.plot(x_vals, bottom_edge, "-", color="red", linewidth=2.0, label="Bottom edge")

        plt.xlabel("Tow length (mm)", fontsize=14)
        plt.ylabel("Position (mm)", fontsize=14)
        plt.title(f"Real Tow {tow}", fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return pd.DataFrame({
        "x_mm": x_vals,
        "centerline": centerline,
        "top_edge": top_edge,
        "bottom_edge": bottom_edge,
        "width": llsb_width})

def plot_simulated_vs_real_tow(tow: int, tow_length_mm=1000):
    """
    Overlay a simulated tow on a real tow.
    Real tow is built from synced LT_y + CAM ± 0.5*LLS_B.
    Simulated tow is generated with generate_multitow_layout.
    """

    # --- Get real tow ---
    real_df = plot_real_tow(tow, tow_length_mm=tow_length_mm)

    # --- Generate one simulated tow (no plot, just data) ---
    gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent = generate_multitow_layout(
        num_tows=1, tow_length_mm=tow_length_mm, plot=False
    )

    # Simulated geometry (from generate_multitow_layout internals)
    # For a single tow, centerline = (top_edge + bottom_edge) / 2
    sim_centerline = (gap_overlap_df.index.to_numpy())  # placeholder X
    sim_top = None
    sim_bottom = None

    # --- For single tow, we can directly regenerate its geometry like in generate_multitow_layout ---
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
    plt.figure(figsize=(10, 6))

    # Real tow
    plt.plot(real_df["x_mm"], real_df["centerline"], "--", color="blue", label="Real centerline")
    plt.plot(real_df["x_mm"], real_df["top_edge"], "-", color="blue", alpha=0.6, label="Real edges")
    plt.plot(real_df["x_mm"], real_df["bottom_edge"], "-", color="blue", alpha=0.6)

    # Simulated tow
    plt.plot(sim_x, tow_centerline_sim, "--", color="red", label="Sim centerline")
    plt.plot(sim_x, sim_top, "-", color="red", alpha=0.6, label="Sim edges")
    plt.plot(sim_x, sim_bottom, "-", color="red", alpha=0.6)

    plt.xlabel("Tow length (mm)", fontsize=14)
    plt.ylabel("Position (mm)", fontsize=14)
    plt.title(f"Tow {tow}: Real vs Simulated", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return real_df, pd.DataFrame({
        "x_mm": sim_x,
        "centerline": tow_centerline_sim,
        "top_edge": sim_top,
        "bottom_edge": sim_bottom,
        "width": tow_widths_sim
    })

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
    generate_multitow_layout(3, plot=True)

    #plot_simulated_vs_real_tow(3)

    #simulation_verification(20)

    #mean, std, start_values = fit_starting_error_distribution("LLS_A")
    # start_values = np.array(start_values)
    #print(mean)

    #plot_sensor_error_histograms(num_tows=10, tow_length_mm=1000, bins=60)

    #real_df, sim_df = plot_simulated_vs_real_tow(3)
    #compare_fft_real_vs_sim(real_df, sim_df)

    
if __name__ == "__main__":
    main()
