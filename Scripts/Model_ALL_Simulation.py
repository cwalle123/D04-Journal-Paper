from Model_ALL_ConsecutiveErrorTheo import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from Handling_ALL_Functions import get_synced_data
import random
import pandas as pd
import constants

steps_per_mm = 360 / 1000       # Keep consistent with User_Interface

#starting error distribution can be found here, but is assumed to be uniform based on these graphs ranges of values
def fit_starting_error_distribution(sensor: str, plot=True):
    column_map = {
        "CAM": "center_CAM",
        "LT": "error_LT",
        "LLS_A": "width error_LLS_A",
        "LLS_B": "width error_LLS_B"
    }

    col_name = column_map[sensor]
    first_values = []

    for tow in range(2, 32):
        df = get_synced_data(tow, spacesynced=True)

        if col_name in df.columns and not df[col_name].isna().all():
            value = df[col_name].dropna().values[0]  # get first non-NaN value
            first_values.append(value)
 

    mu, sigma = stats.norm.fit(first_values)

    if plot:
        plt.figure(figsize=(8, 5))
        count, bins, _ = plt.hist(first_values, bins=len(first_values), density=True, edgecolor="black", alpha=0.7, label="Start Values")
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

def generate_multitow_layout(num_tows=5, tow_spacing_mm=6.35, tow_width_mm=6.35, n_steps=360, cam_start_range=(-0.6, 0.4), lt_start_range=(-1, -0.8), llsb_start_range=(-0.15, -0.02),plot=True):
    # Get binned models
    bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = consecutive_error(
        "CAM", test_ratio=0.5, num_bins=180, bins_show=False, plot_fit=False, random_state=random.randint(0, 10000))
    bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = consecutive_error(
        "LT", test_ratio=0.5, num_bins=180, bins_show=False, plot_fit=False, random_state=random.randint(0, 10000))
    bin_stats_llsb, slope_llsb, intercept_llsb, _, _, _, x_sorted_llsb, bin_edges_llsb, devs_llsb = consecutive_error(
        "LLS_B", test_ratio=0.5, num_bins=180, bins_show=False, plot_fit=False, random_state=random.randint(0, 10000))
    #get perfect offsets
    offsets = np.linspace(-(num_tows - 1) / 2, (num_tows - 1) / 2, num_tows) * tow_spacing_mm
    plt.figure(figsize=(12, 8))
    # plt.axis("equal") # ADDED HERE !!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #for coloring properly (chatgpt did the plotting)
    cmap = plt.get_cmap("tab10")
    x_vals = np.arange(n_steps) / steps_per_mm  # convert step indices to mm

    top_lines = []
    bottom_lines = []


    tow_colors = [
    '#FFD700',   # gold
    '#FF8C00',   # darkorange
    ]   

    for i, offset in enumerate(offsets):

        '''Uncomment this if you want to visualize the 10 virtual tows with different colors, 
        and comment the line that says:
        color = tow_colors[i % len(tow_colors)]  
        which is (circa) line 111'''
        #color = tow_colors[i % len(tow_colors)]
        
        start_cam = random.uniform(*cam_start_range)
        start_lt = random.uniform(*lt_start_range)
        start_llsb = random.uniform(*llsb_start_range)

        cam_path = generate_error_path(start_cam, n_steps, slope_cam, intercept_cam,
                                       x_sorted_cam, bin_edges_cam, devs_cam)
        lt_path = generate_error_path(start_lt, n_steps, slope_lt, intercept_lt,
                                      x_sorted_lt, bin_edges_lt, devs_lt)
        centerline = offset + cam_path + lt_path

        width_error = generate_error_path(start_llsb, n_steps, slope_llsb, intercept_llsb,
                                          x_sorted_llsb, bin_edges_llsb, devs_llsb)
        width = width_error + tow_width_mm

        top_line = centerline + 0.5 * width
        bottom_line = centerline - 0.5 * width

        top_lines.append(top_line)
        bottom_lines.append(bottom_line)

        # --- Plot ---

        # Make titles Times new Roman
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif']  = ['Times New Roman']

        # if you use math text and want matching font:
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm']      = 'Times New Roman'

        # Sort the colors for the two tows
        color = tow_colors[i % len(tow_colors)]
        
        '''I did a little tweak, just for the label box below the figure looks nicer'''
        plt.plot(
        x_vals,
        [offset]*n_steps,
        linestyle=":",
        color="black",
        linewidth=1,
        label="Programmed paths" if i == 0 else "_nolegend_")

        # centrelines — only label once
        plt.plot(
        x_vals,
        centerline[:n_steps],
        "--",
        color=color,
        linewidth=1.5,
        label="Tow centerlines" if i == 0 else "_nolegend_")

        # edges — only label once
        plt.plot(
        x_vals,
        top_line[:n_steps],
        "-",
        color=color,
        linewidth=2.5,
        label="Tow edges" if i == 0 else "_nolegend_")

        # bottom edge same style, no legend
        plt.plot(
        x_vals,
        bottom_line[:n_steps],
        "-",
        color=color,
        linewidth=2.5,
        label="_nolegend_")

    plt.legend(loc='lower center',           # centered below the axes
                    bbox_to_anchor=(0.5, -0.18),  # change the y-coord so it goes underneath the x-axis
                    ncol=3,                       # three entries side-by-side
                    frameon=True,                 
                    fontsize=12)
        
    # Set x-axis limits
    plt.xlim(0, 1000)


    if plot == True:
        plt.xlabel("Tow length (mm)", fontsize=25)
        plt.ylabel("Tow width (mm)", fontsize=25)
        #plt.title(f"Simulated {num_tows}-Tow Layout with Random Start Errors", fontsize=20)
        '''Uncomment the two following lines if you want a legend in the 2 virtual tow figure'''
        #if num_tows <= 50:
            #plt.legend(loc="upper right", ncol=2, fontsize=15)

        plt.grid(False)
        plt.tight_layout()
        # The next line is commented for User_Interface. If you need to show the graph for a bit, make sure to revert it back after.
        # plt.show()

    #gaps and overlaps calculation
    #Compute vertical gaps between adjacent tows
    gap_overlap_data = {}

    for i in range(num_tows - 1):
        gap_overlap = bottom_lines[i+1]-top_lines[i]  # vertical space between adjacent tows
        col_name = f"Gap/overlap_Tow{i+1}_Tow{i+2}"
        gap_overlap_data[col_name] = gap_overlap  # shape

    gap_overlap_df = pd.DataFrame(gap_overlap_data)
    gap_df = gap_overlap_df.where(gap_overlap_df > 0)
    overlap_df = gap_overlap_df.where(gap_overlap_df < 0)

    #Area Calculations (unitless)
    topmost_line = top_lines[-1]      # Top edge of highest-numbered tow
    bottommost_line = bottom_lines[0] # Bottom edge of lowest-numbered tow
    total_area = np.trapezoid(topmost_line - bottommost_line)

    total_gap_area = 0.0
    total_overlap_area = 0.0

    for col in gap_overlap_df.columns:
        gap_vals = gap_overlap_df[col].values
        gaps = np.where(gap_vals > 0, gap_vals, 0)
        overlaps = np.where(gap_vals < 0, -gap_vals, 0)  # flip sign for integration
        total_gap_area += np.trapezoid(gaps)
        total_overlap_area += np.trapezoid(overlaps)

    gap_percent = (total_gap_area / total_area) * 100 
    overlap_percent = (total_overlap_area / total_area) * 100 

    print(f"\nTotal layout area (unitless): {total_area:.2f}")
    print(f"Gap area: {total_gap_area:.2f} ({gap_percent:.2f}%)")
    print(f"Overlap area: {total_overlap_area:.2f} ({overlap_percent:.2f}%)")

    return gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent

# DONT REMOVE THE NEXT LINE. It is required for generate_multitow_layout_wrapped in User_Interface!
gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent = generate_multitow_layout(num_tows=15)

#REAL DATA (!deletes a lot of data!, only use as indicator for percentage of gap overlap)

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

def simulation_verificatoin(num_simulations):
    gap_percent_list = []
    overlap_percent_list = []
    for i in range(num_simulations):
        print(f"Running simulation {i+1}/{num_simulations}", end="\r")
        _, _, _, gap_pct, overlap_pct = generate_multitow_layout(num_tows=25,plot=False)
        gap_percent_list.append(gap_pct)
        overlap_percent_list.append(overlap_pct)

    avg_gap = np.mean(gap_percent_list)
    avg_overlap = np.mean(overlap_percent_list)

    print(f"\n\nAfter {num_simulations} simulations of x-tow layout:")
    print(f"Average Gap Percentage: {avg_gap:.2f}%")
    print(f"Average Overlap Percentage: {avg_overlap:.2f}%")

def main():
    gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent = generate_multitow_layout(num_tows=2)

    # Plot the gap(s) over steps (not time)
    x_vals_mm = gap_overlap_df.index / steps_per_mm  # convert index (steps) to mm

    plt.figure(figsize=(12, 5))
    for column in gap_overlap_df.columns:
        plt.plot(x_vals_mm, gap_overlap_df[column], label=column)

    plt.xlabel("Position (mm)", fontsize=20)
    plt.ylabel("Distance between tows (mm)", 
               fontsize=20)
    
    '''plt.title(f"Vertical Gaps Between Adjacent Tows Over Distance\n"
            f"Gap Area: {gap_percent:.2f}%      Overlap Area: {overlap_percent:.2f}%",
            fontsize=constants.font_large)'''
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
