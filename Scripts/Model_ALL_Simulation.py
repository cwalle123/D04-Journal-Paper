"""This file deals with generating simulated tows using the model.
   Written by: """

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
from scipy.stats import pareto

# Internal imports
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path
from Handling_ALL_Functions import get_synced_data
from constants import number_of_steps, Consecutive_Error_Bins

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

    # Fit uniform distributrion - dy
    a, b = stats.uniform.fit(first_values)

    # KS test
    ks_norm = stats.kstest(first_values, "norm", args=(mu, sigma))
    ks_uniform = stats.kstest(first_values, "uniform", args=(a, b-a))

    print("Normal KS:", ks_norm)
    print("Uniform KS:", ks_uniform)

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

    return mu, sigma, a, b, first_values

def generate_multitow_layout(num_tows=5, tow_spacing_mm=6.35, tow_width_mm=6.35, tow_length_mm=1000, cam_start_range=(-0.75, 0.75), lt_start_range=(-0.9, -0.7), llsb_start_range=(-0.21, -0.02), plot=False, scaled=False):
    """
    Generate a multi-tow layout using real error models (CAM, LT, LLS_B).
    Returns gap/overlap DataFrames and percentages.
    """

    num_bins = Consecutive_Error_Bins
    n_steps = int(tow_length_mm * 340 / 1000)  # base step count
    n_steps = number_of_steps

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
        if scaled:
            plt.axis("equal")
        plt.tight_layout()
        plt.show()

    # --- Print summary ---
    print(f"\nTotal layout area: {total_layout_area:.2f} mm²")
    print(f"Gap area: {total_gap_area:.2f} mm² ({gap_percent:.2f}%)")
    print(f"Overlap area: {total_overlap_area:.2f} mm² ({overlap_percent:.2f}%)")

    return gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent

def generate_multitow_layout_lengths(
    num_tows=5,
    tow_spacing_mm=6.35,
    tow_width_mm=6.35,
    tow_length_mm=1000,
    cam_start_range=(-0.75, 0.75),
    lt_start_range=(-0.9, -0.7),
    llsb_start_range=(-0.21, -0.02),
    plot=False,
    scaled=False,
    histogram_bins=30,
    num_bins=Consecutive_Error_Bins):
    """
    Generate a multi-tow layout using real error models (CAM, LT, LLS_B),
    compute *lengths* of gaps and overlaps, and fit Pareto distributions.

    Returns:
        gap_overlap_df : DataFrame of pointwise gap/overlap distances
        gap_lengths, overlap_lengths : arrays of segment lengths
        gap_fit, overlap_fit : dicts with Pareto parameters and stats
    """

    # --- Setup ---
    n_steps = number_of_steps

    # --- Load error models ---
    bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = consecutive_error(
        "CAM", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = consecutive_error(
        "LT", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_llsb, slope_llsb, intercept_llsb, _, _, _, x_sorted_llsb, bin_edges_llsb, devs_llsb = consecutive_error(
        "LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))

    tow_offsets = np.linspace(-(num_tows - 1) / 2, (num_tows - 1) / 2, num_tows) * tow_spacing_mm
    top_edge_paths, bottom_edge_paths = [], []

    # --- Generate tow edges ---
    for tow_index, offset in enumerate(tow_offsets):
        start_cam = random.uniform(*cam_start_range)
        start_lt = random.uniform(*lt_start_range)
        start_llsb = random.uniform(*llsb_start_range)

        cam_path = generate_error_path(start_cam, n_steps, slope_cam, intercept_cam,
                                       x_sorted_cam, bin_edges_cam, devs_cam)
        lt_path = generate_error_path(start_lt, n_steps, slope_lt, intercept_lt,
                                      x_sorted_lt, bin_edges_lt, devs_lt)
        width_error = generate_error_path(start_llsb, n_steps, slope_llsb, intercept_llsb,
                                          x_sorted_llsb, bin_edges_llsb, devs_llsb)

        tow_centerline = offset + cam_path + lt_path
        tow_widths = tow_width_mm + width_error

        top_edge_paths.append(tow_centerline + 0.5 * tow_widths)
        bottom_edge_paths.append(tow_centerline - 0.5 * tow_widths)

    # --- Compute x-values ---
    path_len = len(top_edge_paths[0])
    x_vals = np.linspace(0, tow_length_mm, path_len)

    # --- Compute gap/overlap arrays between adjacent tows ---
    gap_overlap_dict = {
        f"Tow{tow_index+1}_Tow{tow_index+2}": bottom_edge_paths[tow_index+1] - top_edge_paths[tow_index]
        for tow_index in range(num_tows - 1)
    }
    gap_overlap_df = pd.DataFrame(gap_overlap_dict, index=x_vals)

    # --- Identify continuous gap/overlap segments ---
    def extract_segment_lengths(series, positive=True):
        """Return lengths (in mm) of continuous gap/overlap regions."""
        values = series.values
        mask = values > 0 if positive else values < 0
        lengths = []
        run_length = 0
        for i in range(len(mask)):
            if mask[i]:
                run_length += 1
            elif run_length > 0:
                lengths.append(run_length)
                run_length = 0
        if run_length > 0:
            lengths.append(run_length)
        dx = series.index[1] - series.index[0]
        return np.array(lengths) * dx

    # --- Compute segment lengths for all pairs ---
    gap_lengths, overlap_lengths = [], []
    for col in gap_overlap_df.columns:
        gap_lengths.extend(extract_segment_lengths(gap_overlap_df[col], positive=True))
        overlap_lengths.extend(extract_segment_lengths(gap_overlap_df[col], positive=False))

    gap_lengths = np.array(gap_lengths)
    overlap_lengths = np.array(overlap_lengths)

    # --- Pareto fit helper ---
    def fit_pareto(data):
        if len(data) == 0:
            return {"shape": 0, "loc": 0, "scale": 0, "mean": 0, "std": 0}
        shape, loc, scale = pareto.fit(data, floc=0)  # fix loc=0
        mean = pareto.mean(shape, loc=loc, scale=scale)
        std = pareto.std(shape, loc=loc, scale=scale)
        return {"shape": shape, "loc": loc, "scale": scale, "mean": mean, "std": std}

    gap_fit = fit_pareto(gap_lengths)
    overlap_fit = fit_pareto(overlap_lengths)

    # -------------------------------------------------------------------------
    # --- Plotting Section ----------------------------------------------------
    # -------------------------------------------------------------------------
    if plot:
        # --- 1. Tow layout plot ---
        plt.figure(figsize=(10, 5))
        for tow_index, offset in enumerate(tow_offsets):
            color = plt.get_cmap("tab10")(tow_index % 10)
            tow_centerline = (top_edge_paths[tow_index] + bottom_edge_paths[tow_index]) / 2
            plt.plot(x_vals, tow_centerline, "--", color=color,
                     linewidth=1.2, label="Tow centerline" if tow_index == 0 else "_nolegend_")
            plt.plot(x_vals, top_edge_paths[tow_index], "-", color=color,
                     linewidth=1.8, label="Tow edges" if tow_index == 0 else "_nolegend_")
            plt.plot(x_vals, bottom_edge_paths[tow_index], "-", color=color, linewidth=1.8)
        # Programmed reference lines
        for offset in tow_offsets:
            plt.plot(x_vals, np.full_like(x_vals, offset), ":", color="black", linewidth=1)
        plt.xlabel("Tow length (mm)", fontsize=12)
        plt.ylabel("Tow position (mm)", fontsize=12)
        plt.title("Simulated Multi-Tow Layout")
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
        if scaled:
            plt.axis("equal")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

        # --- 2. Histograms with Pareto overlay (counts) ---
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for i, (data, title, fit) in enumerate([
            (gap_lengths, "Gap Lengths (Pareto Fit)", gap_fit),
            (overlap_lengths, "Overlap Lengths (Pareto Fit)", overlap_fit)
        ]):
            if len(data):
                counts, bins, _ = ax[i].hist(
                    data,
                    bins=histogram_bins,
                    density=False,
                    alpha=0.7,
                    edgecolor="black",
                    label="Empirical counts"
                )

                # Scale Pareto PDF to match histogram counts
                x = np.linspace(min(data), max(data), 400)
                pdf = pareto.pdf(x, fit["shape"], loc=fit["loc"], scale=fit["scale"])
                bin_width = bins[1] - bins[0]
                pdf_scaled = pdf * len(data) * bin_width
                ax[i].plot(x, pdf_scaled, "r-", linewidth=2,
                           label=f"Pareto α={fit['shape']:.2f}")

                # Mean line
                ax[i].axvline(fit["mean"], color="blue", linestyle="--", linewidth=1.5,
                              label=f"Mean={fit['mean']:.2f} mm")

                ax[i].set_xlabel("Length (mm)")
                ax[i].set_ylabel("Count")
                ax[i].set_title(title)
                ax[i].legend(fontsize=9)
                ax[i].grid(True, linestyle=":")

        plt.tight_layout()
        plt.show()

    # --- Print summary ---
    print("\n--- Gap Lengths ---")
    print(f"  N={len(gap_lengths)}, α={gap_fit['shape']:.3f}, scale={gap_fit['scale']:.3f}")
    print(f"  Mean={gap_fit['mean']:.3f} mm, Std={gap_fit['std']:.3f} mm")

    print("\n--- Overlap Lengths ---")
    print(f"  N={len(overlap_lengths)}, α={overlap_fit['shape']:.3f}, scale={overlap_fit['scale']:.3f}")
    print(f"  Mean={overlap_fit['mean']:.3f} mm, Std={overlap_fit['std']:.3f} mm")

    return gap_overlap_df, gap_lengths, overlap_lengths, gap_fit, overlap_fit

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
"""Run this file"""

def main():
    #!ATTENTION: DO NOT USE TOW 1 FOR ANY OF THE FUNCTIONS BELOW

    # generate_multitow_layout(5, plot=True)

    generate_multitow_layout_lengths(50, plot=True, histogram_bins=100)

    #simulation_verification(20)

    #mean, std, start_values = fit_starting_error_distribution("LLS_A")
    # start_values = np.array(start_values)
    #print(mean)

if __name__ == "__main__":
    main()
