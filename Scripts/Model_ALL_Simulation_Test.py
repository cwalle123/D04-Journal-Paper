"""This file deals with generating simulated tows using the model"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Internal imports
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path

##############################################################################################################
"""Functions"""

def generate_multitow_layout(num_tows=5, tow_spacing_mm=6.35, tow_width_mm=6.35, tow_length_mm=1000, cam_start_range=(-0.6, 0.4), lt_start_range=(-1, -0.8), llsb_start_range=(-0.15, -0.02), plot=False):
    """
    Generate a multi-tow layout using real error models (CAM, LT, LLS_B).
    Returns gap/overlap DataFrames and percentages.
    """

    n_steps = int(tow_length_mm * 340 / 1000)  # base step count

    # --- Load error model fits ---
    bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = consecutive_error(
        "CAM", test_ratio=0.5, num_bins=180, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = consecutive_error(
        "LT", test_ratio=0.5, num_bins=180, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_llsb, slope_llsb, intercept_llsb, _, _, _, x_sorted_llsb, bin_edges_llsb, devs_llsb = consecutive_error(
        "LLS_B", test_ratio=0.5, num_bins=180, bins_show=False, plot_fit=False,
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

####################################################################################################
"""Run this file"""

def main():
    generate_multitow_layout(10, plot=True)

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
