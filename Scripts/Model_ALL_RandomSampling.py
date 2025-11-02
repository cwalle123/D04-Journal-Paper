import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import functools
import matplotlib.colors as mcolors
from matplotlib import cm
from tqdm import tqdm

from dataclasses import dataclass
from scipy.stats import norm, logistic, gamma, beta, expon, lognorm, skewnorm, gumbel_r, gumbel_l, genextreme
from Handling_ALL_Functions import get_synced_data
from constants import tow_width_specified
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path, generate_starting_error, get_data_pairs
from Data_ALL_statistics import plot_histograms_separated, best_fit_distribution
from Model_ALL_RandomWalk import get_data

def generate_random_sampling_data(sensor: str, steps: int=400, tows: int=1, plot_histogram=False):
    # setting up the distribution from which we are sampling
    data, weights = get_data(sensor)

    best = best_fit_distribution(np.array(data), weights=np.array(weights))
    dist, params = best['dist'], best['params']
    distribution = lambda x: dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

    generated_tows, generated_data = [], []

    for tow in range(tows):

        generated_path = []
        for step in range(steps):
            value = dist.rvs(*params[:-2], loc=params[-2], scale=params[-1])
            generated_path.append(value)
            generated_data.append(value)

        generated_tows.append(generated_path)
    generated_tows = np.array(generated_tows)


    if plot_histogram:
        # making the pdf's
        x_pdf = np.linspace(-1.2, 1.2, 100)
        y_pdf = distribution(x_pdf)

        # making the histogram
        plt.plot(x_pdf, y_pdf, label='probability-distribution')
        plt.hist(generated_data, density=True, bins=30, label='generated data')
        plt.title("Histogram of generated path w. probability-distribution" + sensor)
        plt.xlim(-1.2, 1.2)
        plt.legend()
        plt.show()

    return generated_tows

def generate_RS_multitow(
    num_tows: int,
    n_steps: int = 400,
    tow_spacing_mm: float = 6.35,
    tow_width_mm: float = 6.35,
    tow_length_mm: float = 1000,
    method: str = "Sidd",
    print_statement: bool = False
):
    """
    Generate random sampling (RS) multitow layout and compute gap/overlap areas and percentages.

    Args:
        num_tows (int): Number of tows.
        n_steps (int): Number of discretization steps along the tow length.
        tow_spacing_mm (float): Center-to-center spacing between adjacent tows.
        tow_width_mm (float): Nominal tow width.
        tow_length_mm (float): Length of each tow.
        method (str): Width generation method ("Sidd" or "Random").
        print_statement (bool): Print summary information.

    Returns:
        gap_overlap_df (pd.DataFrame): Gap/overlap distances between adjacent tows.
        RS_all_tows_data (list[pd.DataFrame]): Each tow’s centerline/top/bottom edge data.
        gap_percent (float): % of total layout area that is gap.
        overlap_percent (float): % of total layout area that is overlap.
    """

    top_edge_paths, bottom_edge_paths = [], []
    RS_all_tows_data = []

    x_vals = np.linspace(0, tow_length_mm, n_steps)

    # --- Generate random sampling data for each parameter ---
    LT_RS_data = generate_random_sampling_data("LT", steps=n_steps, tows=num_tows)
    CAM_RS_data = generate_random_sampling_data("CAM", steps=n_steps, tows=num_tows)

    if method == "Sidd":
        LLSB_RS_data = generate_siddharth_width(steps=n_steps, tows=num_tows)
    else:
        LLSB_RS_data = generate_random_sampling_data("LLS_B", steps=n_steps, tows=num_tows)

    # --- Construct each tow geometry ---
    tow_offset = 0
    for tow in range(num_tows):
        tow_centerline_data = tow_offset + np.array(CAM_RS_data[tow, :]) + np.array(LT_RS_data[tow, :])
        tow_width_data = tow_width_mm + np.array(LLSB_RS_data[tow, :])

        tow_top_edge = tow_centerline_data + 0.5 * tow_width_data
        tow_bottom_edge = tow_centerline_data - 0.5 * tow_width_data

        top_edge_paths.append(tow_top_edge)
        bottom_edge_paths.append(tow_bottom_edge)
        tow_offset += tow_spacing_mm

        RS_data = pd.DataFrame({
            "x_mm": x_vals,
            "centerline": tow_centerline_data,
            "top_edge": tow_top_edge,
            "bottom_edge": tow_bottom_edge
        })
        RS_all_tows_data.append(RS_data)

    # --- Compute gap/overlap distances ---
    gap_overlap_dict = {
        f"Gap/Overlap_Tow{t+1}_Tow{t+2}": bottom_edge_paths[t+1] - top_edge_paths[t]
        for t in range(num_tows - 1)
    }
    gap_overlap_df = pd.DataFrame(gap_overlap_dict)

    # --- Compute area-based gap and overlap percentages ---
    highest_tow_edge = top_edge_paths[-1]
    lowest_tow_edge = bottom_edge_paths[0]
    total_layout_area = np.trapezoid(highest_tow_edge - lowest_tow_edge, x_vals)

    total_gap_area = sum(np.trapezoid(np.clip(values, 0, None), x_vals) for values in gap_overlap_df.values.T)
    total_overlap_area = sum(np.trapezoid(np.clip(-values, 0, None), x_vals) for values in gap_overlap_df.values.T)

    gap_percent = (total_gap_area / total_layout_area) * 100 if total_layout_area > 0 else 0
    overlap_percent = (total_overlap_area / total_layout_area) * 100 if total_layout_area > 0 else 0

    # --- Optional printout ---
    if print_statement:
        print(f"\nTotal layout area: {total_layout_area:.2f} mm²")
        print(f"Gap area: {total_gap_area:.2f} mm² ({gap_percent:.2f}%)")
        print(f"Overlap area: {total_overlap_area:.2f} mm² ({overlap_percent:.2f}%)")

    return gap_overlap_df, RS_all_tows_data, gap_percent, overlap_percent

def generate_siddharth_width(steps: int=400, tows: int=1, plot_histogram=False):
    LLS_A_data = generate_random_sampling_data("LLS_A", steps=steps, tows=tows)
    modified_data = np.zeros_like(LLS_A_data)

    data, weights = get_data("LLS_B")
    best = best_fit_distribution(np.array(data), weights=np.array(weights))
    dist, params = best['dist'], best['params']
    distribution = lambda x: dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

    for tow in range(tows):
        for step in range(steps):
            value_bigger = False
            while value_bigger == False:
                new_value = dist.rvs(*params[:-2], loc=params[-2], scale=params[-1])
                if new_value > LLS_A_data[tow, step]:
                    modified_data[tow, step] = new_value
                    value_bigger = True

    if plot_histogram:
        LLS_B_raw_data = generate_random_sampling_data("LLS_B", steps=steps, tows=tows)
        LLS_A_list, modified_list, LLS_B_raw_list = [], [], []
        for tow in range(tows):
            LLS_A_list += list(LLS_A_data[tow])
            modified_list += list(modified_data[tow])
            LLS_B_raw_list += list(LLS_B_raw_data[tow])


        x_pdf = np.linspace(-1.2, 1.2, 100)
        y_pdf = distribution(x_pdf)

        # making the histogram
        plt.plot(x_pdf, y_pdf, label='probability-distribution')
        plt.hist(modified_list, density=True, bins=30, alpha=0.5, label='Siddharth method width data')
        plt.hist(LLS_B_raw_list, density=True, bins=30, alpha=0.5, label='raw LLS_B generated data')
        plt.hist(LLS_A_list, density=True, bins=30, alpha=0.2, label='LLS_A generated data')
        plt.title("Histogram of generated paths w. probability-distribution")
        plt.xlim(-0.7, 0.3)
        plt.legend()
        plt.show()

    return modified_data

def run_multiple_RS_simulations_for_gaps_and_overlap_percentages(
    n_simulations: int = 100,
    num_tows: int = 5,
    n_steps: int = 370,
    tow_spacing_mm: float = 6.35,
    tow_width_mm: float = 6.35,
    tow_length_mm: float = 1000,
    method: str = "Sidd",
    verbose: bool = True,
    progress_bar: bool = True):
    """
    Run multiple RS multitow simulations and compute statistics.

    Returns:
        summary_df (pd.DataFrame): gap/overlap % per simulation
        stats (dict): mean and std of gap and overlap percentages
    """

    gap_percents, overlap_percents = [], []

    iterator = tqdm(range(n_simulations), desc="Running RS simulations") if progress_bar else range(n_simulations)

    for i in iterator:
        try:
            _, _, gap_percent, overlap_percent = generate_RS_multitow(
                num_tows=num_tows,
                n_steps=n_steps,
                tow_spacing_mm=tow_spacing_mm,
                tow_width_mm=tow_width_mm,
                tow_length_mm=tow_length_mm,
                method=method,
                print_statement=False
            )

            gap_percents.append(gap_percent)
            overlap_percents.append(overlap_percent)
            
            # compute running averages
            mean_gap = np.mean(gap_percents)
            mean_overlap = np.mean(overlap_percents)
            std_gap = np.std(gap_percents)
            std_overlap = np.std(overlap_percents)

            # verbose print per sim
            if verbose:
                print(f"\nSimulation {i+1}/{n_simulations}")
                print(f"  Gap: {gap_percent:.2f}% | Overlap: {overlap_percent:.2f}%")
                print(f"  Running Avg → Gap: {mean_gap:.3f}% ± {std_gap:.3f}%, Overlap: {mean_overlap:.3f}% ± {std_overlap:.3f}%")

        except Exception as e:
            print(f"Simulation {i+1} failed: {e}")
            continue

    summary_df = pd.DataFrame({
        "Simulation": np.arange(1, len(gap_percents) + 1),
        "Gap_%": gap_percents,
        "Overlap_%": overlap_percents
    })

    stats = {
        "Mean_Gap_%": np.mean(gap_percents),
        "Std_Gap_%": np.std(gap_percents),
        "Mean_Overlap_%": np.mean(overlap_percents),
        "Std_Overlap_%": np.std(overlap_percents)
    }

    print("\n=== RS Simulation Summary ===")
    print(f"Average Gap: {stats['Mean_Gap_%']:.3f}% ± {stats['Std_Gap_%']:.3f}%")
    print(f"Average Overlap: {stats['Mean_Overlap_%']:.3f}% ± {stats['Std_Overlap_%']:.3f}%")

    return summary_df, stats

def main():
    #generate_random_sampling_data("LLS_B", steps=400, tows=300, plot_histogram=True)
    # gap_overlap_df, RS_data = generate_RS_multitow(31, n_steps=400, tow_spacing_mm=12.5, tow_width_mm=6.35)
    # print(gap_overlap_df, RS_data)
    #print(gap_overlap_df)
    #generate_siddharth_width(tows=30, plot_histogram=True)
    run_multiple_RS_simulations_for_gaps_and_overlap_percentages(n_simulations=50,num_tows=31)


if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else