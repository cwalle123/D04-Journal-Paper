"""This file deals with generating simulated tows using the model"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import random

# Internal imports
# from Model_ALL_ConsecutiveErrorTheo import *

##############################################################################################################
"""Functions"""

def generate_tow_error_path(source, n_steps, random_state=None):
    """
    Placeholder for tow error simulation.
    Replace this with actual call to your tow generator.
    Example: return generate_error_path(...)
    """
    rng = np.random.default_rng(random_state)
    return rng.normal(0, 0.1, size=n_steps)  # dummy path for now

def generate_multitow_layout(num_tows=5, tow_spacing_mm=6.35, tow_width_mm=6.35, tow_length_mm=1000, plot=False):
    """
    Generate a multi-tow layout with simulated centerline and width errors.
    Returns gap/overlap DataFrames and percentages.
    """

    # Steps calculation based on 340 steps = 1000 mm
    n_steps = int(tow_length_mm * 340 / 1000)

    # Offsets for each tow centerline (symmetrically placed around 0)
    tow_offsets = np.linspace(-(num_tows - 1) / 2, (num_tows - 1) / 2, num_tows) * tow_spacing_mm
    tow_positions_mm = np.linspace(0, tow_length_mm, n_steps)

    top_edge_paths, bottom_edge_paths = [], []

    # --- Simulate each tow ---
    for tow_index, offset_mm in enumerate(tow_offsets):
        # Placeholder: replace with your real tow simulation
        centerline_error = generate_tow_error_path("center", n_steps, random_state=random.randint(0, 10000))
        width_error     = generate_tow_error_path("width",  n_steps, random_state=random.randint(0, 10000))

        tow_centerline = offset_mm + centerline_error
        tow_widths = tow_width_mm + width_error

        tow_top_edge = tow_centerline + 0.5 * tow_widths
        tow_bottom_edge = tow_centerline - 0.5 * tow_widths

        top_edge_paths.append(tow_top_edge)
        bottom_edge_paths.append(tow_bottom_edge)

        # --- Plotting ---
        if plot:
            color = plt.get_cmap("tab10")(tow_index % 10)
            plt.plot(tow_positions_mm, tow_centerline, "--", color=color, linewidth=1.5,
                     label="Tow centerline" if tow_index == 0 else "_nolegend_")
            plt.plot(tow_positions_mm, tow_top_edge, "-", color=color, linewidth=2.0,
                     label="Tow edges" if tow_index == 0 else "_nolegend_")
            plt.plot(tow_positions_mm, tow_bottom_edge, "-", color=color, linewidth=2.0,
                     label="_nolegend_")

    # --- Gap/Overlap analysis ---
    gap_overlap_dict = {
        f"Gap/overlap_Tow{tow_index+1}_Tow{tow_index+2}": bottom_edge_paths[tow_index+1] - top_edge_paths[tow_index]
        for tow_index in range(num_tows - 1)
    }
    gap_overlap_df = pd.DataFrame(gap_overlap_dict)

    # Separate gaps and overlaps
    gap_df = gap_overlap_df.where(gap_overlap_df > 0)
    overlap_df = gap_overlap_df.where(gap_overlap_df < 0)

    # --- Area calculations ---
    highest_tow_edge = top_edge_paths[-1]
    lowest_tow_edge = bottom_edge_paths[0]
    total_layout_area = np.trapezoid(highest_tow_edge - lowest_tow_edge, tow_positions_mm)

    total_gap_area = sum(np.trapezoid(np.clip(values, 0, None), tow_positions_mm) for values in gap_overlap_df.values.T)
    total_overlap_area = sum(np.trapezoid(np.clip(-values, 0, None), tow_positions_mm) for values in gap_overlap_df.values.T)

    gap_percent = (total_gap_area / total_layout_area) * 100 if total_layout_area > 0 else 0
    overlap_percent = (total_overlap_area / total_layout_area) * 100 if total_layout_area > 0 else 0

    # --- Plot cosmetics ---
    plt.xlabel("Tow length (mm)", fontsize=14)
    plt.ylabel("Tow position (mm)", fontsize=14)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=10, frameon=True)
    plt.grid(False)
    plt.tight_layout()
    if plot:
        plt.show()

    # --- Print summary ---
    print(f"\nTotal layout area: {total_layout_area:.2f} mm²")
    print(f"Gap area: {total_gap_area:.2f} mm² ({gap_percent:.2f}%)")
    print(f"Overlap area: {total_overlap_area:.2f} mm² ({overlap_percent:.2f}%)")

    return gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent

####################################################################################################
"""Run this file"""

def main():
    generate_multitow_layout(2, plot=True)

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
