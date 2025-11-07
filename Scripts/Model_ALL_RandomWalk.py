"""This file contains the Random Walk model"""

##############################################################################################################

# External imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import functools
import matplotlib.colors as mcolors
from matplotlib import cm
from dataclasses import dataclass
from scipy.stats import norm, logistic, gamma, beta, expon, lognorm, skewnorm, gumbel_r, gumbel_l, genextreme
from tqdm import tqdm
import os
from scipy.stats import pareto

# Internal imports
from Handling_ALL_Functions import get_synced_data, get_data
from constants import tow_width_specified
from D04_Model.Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path, generate_starting_error, get_data_pairs
from Data_ALL_statistics import plot_histograms_separated, best_fit_distribution

##############################################################################################################
"""Functions"""

def get_n_steps(sensor):
    """This function gets the number of steps, which is the number of data points in a one meter tow"""
    data, weights = get_data(sensor, format='separated')

    lengths = []
    for i in range(len(data)):
        lengths.append(len(data[i][:]))

    return int(np.average(lengths))

def propose_new_RWM_value(x_current, dist_std):    # random walk metropolis (RWM), using normal dist???
    mean = 0
    proposal = x_current + np.random.normal(mean, dist_std)  # this uses std to recreate real tow 'waviness'
    # update_states()     # comment this if not initialising from StartVariations
    return proposal

def fit_random_walk(sensor: str, bins=40):
    n_steps = get_n_steps(sensor)
    data, weights = get_data(sensor, format='merged')
    if sensor != "CAM":
        best = best_fit_distribution(np.array(data), bins=bins, weights=np.array(weights))
    elif sensor == "CAM":
        best = best_fit_distribution(np.array(data), weights=np.array(weights), use_all_dist=True, shrink_scale_factor=0.9)
    dist, params = best['dist'], best['params']
    target_distribution = lambda x: dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

    proposal_std = get_proposal_distribution(sensor)

    return n_steps, proposal_std, target_distribution, dist, params

def generate_random_walk(sensor: str, n_steps: int, proposal_std:  float, target_dist, dist, params, proposal_type: str="RWM",
                         plot_histogram: bool=False, plot_path: bool=False, comparison: bool=False, return_pdf: bool=False,
                         burn_in_period: int=0, plot_covergence_params: bool=False, print_statement: bool=False, start_value_override=None):
    '''
    This function generates a random walk according to the specified sensor.
    '''

    # adding the burn in period to the #steps
    actual_steps = n_steps + burn_in_period

    if start_value_override is not None:
        start_value = start_value_override
    else:
        start_value = dist.rvs(*params[:-2], loc=params[-2], scale=params[-1])
        # update_states()     # comment this if not initialising from StartVariations
    generated_path = []
    x_current = start_value

    accepted, rejected = 0, 0
    for step in range(actual_steps-1):

        if proposal_type == "RWM":
            x_proposal = propose_new_RWM_value(x_current, proposal_std)
        else:
         print("Error, invalid proposal type")

        # code to accept or reject the proposed new value
        alpha = target_dist(x_proposal) / target_dist(x_current)      # alpha = acceptance probability
        U = random.uniform(0, 1)
        # update_states()     # comment this if not initialising from StartVariations

        if alpha >= U:  # we accept the proposed value
            x_next = x_proposal
            accepted += 1
        else:           # we reject the proposed value
            rejected += 1
            x_next = x_current

        # only taking samples after the burn-in period is over
        if step >= burn_in_period:
            generated_path.append(x_current)
        x_current = x_next

        if plot_covergence_params == True and len(generated_path)%10 == 0:
            mean_list = np.average(generated_path)
            std_list = np.std(generated_path)

    generated_path.append(x_next)

    if print_statement == True:
        print("acceptance rate =", accepted/(accepted + rejected))

    # plot-plot-plot #
    x_pdf = np.linspace(-1.2, 1.2, 100)
    y_pdf = target_dist(x_pdf)

    if plot_histogram:
        plt.plot(x_pdf, y_pdf, label='probability-distribution')
        plt.hist(generated_path, density=True, bins=30, label='generated path')
        #plt.hist(data, density=True, bins=50, label='data')
        plt.title("Histogram of generated path w. probability-distribution of "+sensor)
        plt.xlim(-1.2, 1.2)
        plt.legend()
        plt.show()

    if plot_covergence_params:
        # making the parameter data that we need to plot
        real_mean = np.average(generated_path[(int(n_steps*(3/4))):])
        real_std = np.std(generated_path[(int(n_steps*(3/4))):])
        print(real_mean, real_std)

        mean_list, std_list, step_list = [], [], []
        for i in range(len(generated_path)):
            if i !=0 and i%10 == 0:
                mean_list.append(np.average(generated_path[:i]))
                std_list.append(np.std(generated_path[:i]))
                step_list.append(len(generated_path[:i]))

        break_mean, break_std = False, False
        for i in range(len(step_list)):
            if i != 0:
                if abs(mean_list[-i]) >= abs(real_mean)+abs(real_std*0.05):
                    mean_convergence = step_list[-i]
                    break_mean = True
                if abs(std_list[-i]) >= abs(real_std)+abs(real_std*0.05):
                    std_convergence = step_list[-i]
                    break_std = True
                if break_mean == True and break_std == True:
                    break

        print("Mean has converged within 5% of final STD after" + str(mean_convergence) + "steps")
        print("STD has converged within 5% of final STD after" + str(std_convergence) + "steps")

        plt.plot(mean_list, step_list, label='Mean')
        plt.plot(std_list, step_list, label='STD')
        plt.legend()
        plt.show()

    if plot_path:
        # random walk plotting
        fig, ax = plt.subplots(figsize=(8, 2.5))
        x = np.linspace(0, 1000, n_steps)
        plt.plot(x, generated_path, label='generated path')
        plt.title("plot of generated path" + sensor)

        #actual data plotting
        if comparison:
            real_data_uncut = get_synced_data(5, sensor)
            if sensor == "LLS_A":   real_data = real_data_uncut["error_LLS_A"]
            elif sensor == "LLS_B": real_data = real_data_uncut["error_LLS_B"]
            elif sensor == "CAM":   real_data = real_data_uncut["center_CAM"]
            elif sensor == "LT":
                real_data = real_data_uncut["error_LT"]
                weight_data = np.array(real_data_uncut["Weights"])[:-1, 0]
            # getting the x-position for the real data
            if sensor != "LT": weight_data = np.array(real_data_uncut["Weights"])[:-1]
            velocity_factor = 1/sum(weight_data)
            real_x, x = [0], 0
            for i in range(len(real_data)-1):
                x += velocity_factor*weight_data[i]*1000
                real_x.append(x)

            plt.plot(real_x, real_data, label='real path')

        plt.tight_layout()
        plt.legend()
        plt.show()

    if return_pdf: return generated_path, x_pdf, y_pdf
    return generated_path

def get_proposal_distribution(sensor, plot: bool=False):
    data_pairs = get_data_pairs(sensor)
    data, weights = [], []
    for i in range(len(data_pairs)):
        diff = data_pairs[i, 0] - data_pairs[i, 1]
        weight = data_pairs[i, 2]
        data.append(diff)
        weights.append(weight)

    # determining the normal distribution which fits:
    mean = np.average(data, weights= weights)
    variance = np.average((data-mean)**2, weights=weights)
    std = np.sqrt(variance)

    if plot:
        #x = np.linspace(min(data), max(data), 200)
        x = np.linspace(mean-3*std, mean+3*std, 200)

        distribution = lambda x: norm.pdf(x, loc=mean, scale=std)

        # plotting
        plt.plot(x, distribution(x), label='proposal distribution')
        plt.hist(data, weights=weights, density=True, label='step-size data', bins=300)
        plt.xlim(mean-3*std, mean+3*std)
        plt.title('step size distribution for ' + sensor)
        plt.legend()
        plt.show()

    return std

def interpolate(data, new_steps):
    data = np.array(data)
    old_indices = np.linspace(0, 1, num=len(data))
    new_indices = np.linspace(0, 1, num=new_steps)
    return np.interp(new_indices, old_indices, data)

def generate_RW_multitow(num_tows: int=5, tow_spacing_mm: float=6.35, tow_width_mm: float=6.35, tow_length_mm: float=1000,
                         proposal_type: str="RWM", print_statement: bool=False, starting_mods: list=[None, 1, 1], alternate_start: list=[None, "params"], override: bool=False):
    """This function generate a multitow layout using RW"""

    # fitting random walk to experimental data
    LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params = fit_random_walk("LT")
    CAM_steps, CAM_proposal_std, CAM_target_dist, CAM_dist, CAM_params = fit_random_walk("CAM")
    LLS_B_steps, LLS_B_proposal_std, LLS_B_target_dist, LLS_B_dist, LLS_B_params = fit_random_walk("LLS_B")
    LLS_A_steps, LLS_A_proposal_std, LLS_A_target_dist, LLS_A_dist, LLS_A_params = fit_random_walk("LLS_A")
    if print_statement == True:
        print("LT_steps = ", LT_steps, "CAM_steps = ", CAM_steps, "LLS_B_steps = ", LLS_B_steps)

    # This seciton modifies the starting distributions used by the model, which is needed for Model_ALL_StartVariations.
    if starting_mods != [None, 1, 1]:
        if starting_mods[0] != None:  # this changes the starting distribution type if necessary
            CAM_dist = starting_mods[0]

        # these are the factors by which the mean and std are changed
        loc_factor, scale_factor = starting_mods[1], starting_mods[2]

        # code used for start value: start_value = dist.rvs(*params[:-2], loc=params[-2], scale=params[-1])
        #LT_params, LLS_B_params = list(LT_params), list(LLS_B_params)
        CAM_params = list(CAM_params)
        #LT_params[-2] *= loc_factor
        CAM_params[-2] *= loc_factor
        #LLS_B_params[-2] *= loc_factor
        #LT_params[-1] *= scale_factor      # TODO: make sure the scale parameters equally affect the different distribution types
        CAM_params[-1] *= scale_factor
        #LLS_B_params[-1] *= scale_factor
        #LT_params, LLS_B_params = tuple(LT_params), tuple(LLS_B_params)
        CAM_params = tuple(CAM_params)

    if alternate_start[0] != None:
        print("alternate starting distribution was used.")
        CAM_dist = alternate_start[0]
        CAM_params = alternate_start[1]

    tow_offset = 0
    RW_all_tows_data, top_edge_paths, bottom_edge_paths = [], [], []
    for n in range(num_tows):

        # generating random walk data
        LT_walk_data = generate_random_walk("LT", LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params, proposal_type=proposal_type)
        CAM_walk_data = generate_random_walk("CAM", CAM_steps, CAM_proposal_std, CAM_target_dist, CAM_dist, CAM_params, proposal_type=proposal_type)
        LLSB_walk_data = generate_random_walk("LLS_B", LLS_B_steps, LLS_B_proposal_std, LLS_B_target_dist, LLS_B_dist, LLS_B_params, proposal_type=proposal_type)
        LLSA_walk_data = generate_random_walk("LLS_A", LLS_A_steps, LLS_A_proposal_std, LLS_A_target_dist, LLS_A_dist, LLS_A_params, proposal_type=proposal_type)

        # determine what the smallest number of steps is for the errors and use this is the global number of steps
        n_steps = min(LT_steps, CAM_steps, LLS_B_steps)
        if n_steps != CAM_steps: print('Note: CAM data length was NOT used!')
        x_walk_data = np.linspace(0, tow_length_mm, n_steps)

        # interpolate only datasets that are longer than the reference
        if n_steps != LT_steps:
            LT_walk_data = interpolate(LT_walk_data, n_steps)
        if n_steps != CAM_steps:
            CAM_walk_data = interpolate(CAM_walk_data, n_steps)
        if n_steps != LLS_B_steps:
            LLSB_walk_data = interpolate(LLSB_walk_data, n_steps)
        if n_steps != LLS_A_steps:
            LLSA_walk_data = interpolate(LLSA_walk_data, n_steps)

        compaction_error = -(LLSB_walk_data - LLSA_walk_data)
        for i in range(len(LLSB_walk_data)):
            if compaction_error[i] > 0:
                compaction_error[i] = 0

        # getting it into centerline and width format
        tow_centerline_data = tow_offset + np.array(CAM_walk_data) + np.array(LT_walk_data)
        tow_width_data = tow_width_mm + np.array(LLSB_walk_data)

        #print(f'Tow centerline: {tow_centerline_data}')
        #print(f'Tow width: {tow_width_data}')

        # --- Override for flat/square tows (no random walk) ---
        if override == True:
            # create perfectly straight, flat tows
            tow_centerline_data = np.full_like(x_walk_data, tow_offset)
            tow_width_data = np.full_like(x_walk_data, tow_width_mm)

        tow_top_edge = tow_centerline_data + 0.5 * tow_width_data
        tow_bottom_edge = tow_centerline_data - 0.5 * tow_width_data

        top_edge_paths.append(tow_top_edge)
        bottom_edge_paths.append(tow_bottom_edge)
        tow_offset += tow_spacing_mm

        #print(f'Length of x: {len(x_walk_data)}')
        #print(f'Length of centerline: {len(tow_centerline_data)}')
        #print(f'Length of top: {len(tow_top_edge)}')
        #print(f'Length of bottom: {len(tow_bottom_edge)}')

        RW_data = pd.DataFrame({
            "x_mm": x_walk_data,
            "centerline": tow_centerline_data,
            "top_edge": tow_top_edge,
            "bottom_edge": tow_bottom_edge})
        
        RW_all_tows_data.append(RW_data)

    # creating the gap_overlap_data
    gap_overlap_dict = {
        f"Gap/overlap_Tow{tow_index + 1}_Tow{tow_index + 2}": bottom_edge_paths[tow_index + 1] - top_edge_paths[tow_index]
        for tow_index in range(num_tows - 1)}
    gap_overlap_df = pd.DataFrame(gap_overlap_dict)

    x_vals = x_walk_data
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

    # --- Print summary ---
    if print_statement == True:
        print(f"\nTotal layout area: {total_layout_area:.2f} mm²")
        print(f"Gap area: {total_gap_area:.2f} mm² ({gap_percent:.2f}%)")
        print(f"Overlap area: {total_overlap_area:.2f} mm² ({overlap_percent:.2f}%)")

    return gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent, RW_all_tows_data

def plot_RW_tows(num_tows: int = 1, proposal_type: str = "RWM", plot_individual_histograms: bool = False):
    """
    Plots a comparison of random-walk-generated tows, including top, bottom, and centerline.

    Parameters
    ----------
    num_tows : int
        Number of tows to generate and plot.
    proposal_type : str
        Proposal type used in random walk generation.
    plot_individual_histograms : bool
        If True, plots histograms of tow widths as well.
    """

    # --- Generate the random-walk tow data ---
    gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent, RW_all_tows_data = generate_RW_multitow(num_tows=num_tows, proposal_type=proposal_type)

    # --- Create the main figure ---
    plt.figure(figsize=(10, 6))
    plt.title(f"Random Walk Tow Layout ({proposal_type})")
    plt.xlabel("Tow Length (mm)")
    plt.ylabel("Position (mm)")

    # --- Use a color map to give each tow a unique but consistent color ---
    colors = plt.cm.tab10(np.linspace(0, 1, num_tows))

    for i, (tow_df, color) in enumerate(zip(RW_all_tows_data, colors)):
        x = tow_df["x_mm"]
        plt.plot(x, tow_df["top_edge"], color=color, lw=1.2, label=f"Tow {i+1} Top/Bottom/Center")
        plt.plot(x, tow_df["bottom_edge"], color=color, lw=1.2)
        plt.plot(x, tow_df["centerline"], color=color, lw=1.5, linestyle="--")

    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, linestyle="--", alpha=1)
    plt.tight_layout()

    # --- Optional: plot histograms of tow widths ---
    if plot_individual_histograms:
        plt.figure(figsize=(8, 4))
        for i, (tow_df, color) in enumerate(zip(RW_all_tows_data, colors)):
            widths = tow_df["top_edge"] - tow_df["bottom_edge"]
            plt.hist(widths, bins=30, alpha=0.5, color=color, label=f"Tow {i+1}")
        plt.title("Tow Width Distributions")
        plt.xlabel("Width (mm)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

    # --- Print summary info ---
    print(f"Gap %: {gap_percent:.2f}% | Overlap %: {overlap_percent:.2f}%")

    plt.show()

    return RW_all_tows_data

def plot_animated_walk_hist(sensor: str, n_tows: int, proposal_type: str='RWM', tow_length: float=1000):

    n_steps = get_n_steps(sensor)
    step_size = tow_length/n_steps

    # Setting up a random number generator with a fixed state for reproducibility.
    rng = np.random.default_rng(seed=19680801)
    # Fixing bin edges.
    HIST_BINS = np.linspace(-1.2, 1.2, 100)

    # Histogram our data with numpy.
    n_steps, proposal_std, target_dist, dist, params = fit_random_walk(sensor)
    data, x_pdf, y_pdf = generate_random_walk(sensor, n_steps, proposal_std, target_dist, dist, params, proposal_type=proposal_type, return_pdf=True)
    n, _ = np.histogram(data, HIST_BINS, density=True)

    # To animate the histogram, we need an ``animate`` function, which generates
    # a random set of numbers and updates the heights of rectangles. The ``animate``
    # function updates the `.Rectangle` patches on an instance of `.BarContainer`.

    def animate(frame_number, bar_container):
        nonlocal data
        # Simulate new data coming in.
        data += generate_random_walk(sensor, n_steps, proposal_std, target_dist, dist, params, proposal_type=proposal_type, return_pdf=False)
        n, _ = np.histogram(data, HIST_BINS, density=True)
        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)

        return bar_container.patches

    # Using :func:`~matplotlib.pyplot.hist` allows us to get an instance of
    # `.BarContainer`, which is a collection of `.Rectangle` instances.  Since
    # `.FuncAnimation` will only pass the frame number parameter to the animation
    # function, we use `functools.partial` to fix the ``bar_container`` parameter.

    # Output generated via `matplotlib.animation.Animation.to_jshtml`.

    fig, ax = plt.subplots()
    _, _, bar_container = ax.hist(data, HIST_BINS, lw=1, ec="yellow", fc="green", alpha=0.5)

    # Plot the static function once (does not change during animation)
    ax.plot(x_pdf, y_pdf, 'r-', lw=1.5, label="Reference PDF", alpha=0.5)
    ax.set_ylim(top=max(y_pdf)+1)  # set safe limit to ensure that all data is visible.

    anim = functools.partial(animate, bar_container=bar_container)
    ani = animation.FuncAnimation(fig, anim, n_tows, repeat=False, blit=True)
    #plt.show()

    FFwriter = animation.HTMLWriter(fps=10)
    ani.save('animation.html', writer=FFwriter)

def plot_LLS_hist():
    """This function is used to check the LLS histograms"""
    data, weights = get_data('LLS_B', format='merged')
    print(data)
    print(max(data))
    plt.hist(data, bins=200, density=True)
    plt.show()

def analyze_tow_spacing_effect(
    spacing_values_mm: list = None,
    num_simulations: int = 10,
    num_tows_per_simulation: int = 2,
    tow_width_mm: float = 6.35,
    tow_length_mm: float = 1000,
    proposal_type: str = "RWM",
    print_progress: bool = True,
    existing_data: pd.DataFrame | str | None = None):
    """
    Analyzes the effect of tow spacing on gap and overlap percentage.

    If `existing_data` is None, the function runs simulations, saves the results as a CSV 
    in the "Cached Data" folder (including intersection info), and plots the data.
    If `existing_data` is a DataFrame or CSV path, it will plot that data directly.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with average gap and overlap percentages vs tow spacing,
        including intersection spacing and value columns.
    """

    # --- CASE 1: Data provided (plot only) ---
    if existing_data is not None:
        if isinstance(existing_data, str):
            results_df = pd.read_csv(existing_data)
        elif isinstance(existing_data, pd.DataFrame):
            results_df = existing_data.copy()
        else:
            raise ValueError("`existing_data` must be a pandas DataFrame or a CSV file path.")

    # --- CASE 2: Run simulations and save ---
    else:
        if spacing_values_mm is None:
            spacing_values_mm = np.linspace(5.0, 7.5, 9)

        avg_gap_percentages = []
        avg_overlap_percentages = []

        for spacing in tqdm(spacing_values_mm, desc="Start Values"):
            if print_progress:
                print(f"\n--- Simulating for tow spacing = {spacing:.2f} mm ---")

            gap_results = []
            overlap_results = []

            for sim in tqdm(range(num_simulations), desc=f"Spacing={spacing:.2f}", leave=False):
                _, _, _, gap_percent, overlap_percent, _ = generate_RW_multitow(
                    num_tows=num_tows_per_simulation,
                    tow_spacing_mm=spacing,
                    tow_width_mm=tow_width_mm,
                    tow_length_mm=tow_length_mm,
                    proposal_type=proposal_type,
                    print_statement=False)

                gap_results.append(gap_percent)
                overlap_results.append(overlap_percent)

            avg_gap = np.mean(gap_results)
            avg_overlap = np.mean(overlap_results)
            avg_gap_percentages.append(avg_gap)
            avg_overlap_percentages.append(avg_overlap)

            if print_progress:
                print(f"  → Average gap: {avg_gap:.3f}% | Average overlap: {avg_overlap:.3f}%")

        # Create DataFrame
        results_df = pd.DataFrame({
            "Tow Spacing (mm)": spacing_values_mm,
            "Average Gap (%)": avg_gap_percentages,
            "Average Overlap (%)": avg_overlap_percentages})

    # --- Find intersection (where gap = overlap) ---
    gap_arr = np.array(results_df["Average Gap (%)"])
    overlap_arr = np.array(results_df["Average Overlap (%)"])
    spacing_values_mm = np.array(results_df["Tow Spacing (mm)"])
    diff = gap_arr - overlap_arr

    intersection_spacing = None
    intersection_gap_value = None

    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            x1, x2 = spacing_values_mm[i], spacing_values_mm[i + 1]
            y1, y2 = diff[i], diff[i + 1]
            intersection_spacing = x1 - y1 * (x2 - x1) / (y2 - y1)

            g1, g2 = gap_arr[i], gap_arr[i + 1]
            intersection_gap_value = g1 + (g2 - g1) * ((intersection_spacing - x1) / (x2 - x1))
            break

    # --- Add intersection info as new columns ---
    results_df["Intersection Spacing (mm)"] = intersection_spacing
    results_df["Intersection Gap/Overlap (%)"] = intersection_gap_value

    # --- Save updated CSV if data was freshly generated ---
    if existing_data is None:
        os.makedirs("Cached Data", exist_ok=True)
        csv_path = os.path.join(
            "Cached Data",
            f"Tow_spacing_effect_{proposal_type}_with_{num_simulations}_simulations_of_a_{num_tows_per_simulation}_tow_laminate.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n✅ Results (including intersection columns) saved to: {csv_path}")

    # --- Plot ---
    plt.figure(figsize=(9.25, 2.90))
    ax = plt.gca()

    plt.plot(spacing_values_mm, gap_arr, color="blue", label="Gap", linewidth=1)
    plt.plot(spacing_values_mm, overlap_arr, color="red", label="Overlap", linewidth=1)

    # Labels (Times New Roman)
    plt.xlabel("Programmed shift (mm)", fontname="Times New Roman", fontsize=15)
    plt.ylabel("Defect area (%)", fontname="Times New Roman", fontsize=15)
    plt.title("")

    # Remove grid
    plt.grid(False)

    # Box border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("black")

    plt.xticks(fontname="Times New Roman", fontsize=15)
    plt.yticks(fontname="Times New Roman", fontsize=15)
    plt.legend(prop={"family": "Times New Roman", "size": 15})
    plt.tight_layout()
    
    ax.tick_params(top=True, bottom=True, left=True, right=True,
                direction='in',  # Ticks point inward (optional, for a clean boxed look)
                length=6, width=1)  # Adjust tick size and thickness

    xmin, xmax = 5, 7.5
    ymin, ymax = 0, 10

    ax.set_xlim(xmin - 0.02*(xmax-xmin), xmax + 0.02*(xmax-xmin))
    ax.set_ylim(ymin - 0.1*(ymax-ymin), ymax + 0.1*(ymax-ymin))

    plt.show()

    if intersection_spacing is not None:
        print(f"\n🔴 Intersection point at {intersection_spacing:.3f} mm "
              f"(Gap ≈ Overlap = {intersection_gap_value:.3f}%)")
    else:
        print("\n⚠️ No intersection found between gap and overlap curves.")

    return results_df

def generate_RW_multitow_with_local_percent(
    num_tows: int=5, 
    tow_spacing_mm: float=6.35, 
    tow_width_mm: float=6.35, 
    tow_length_mm: float=1000, 
    proposal_type: str="RWM", 
    print_statement: bool=False):
    """
    Generate a multitow layout using random walks, and calculate both total and local (per-x) gap/overlap percentages.
    """

    # --- Run same base function as before ---
    gap_overlap_df, gap_df, overlap_df, total_gap_percent, total_overlap_percent, RW_all_tows_data = \
        generate_RW_multitow(
            num_tows=num_tows,
            tow_spacing_mm=tow_spacing_mm,
            tow_width_mm=tow_width_mm,
            tow_length_mm=tow_length_mm,
            proposal_type=proposal_type,
            print_statement=print_statement)

    # --- Local % computation ---
    x_vals = RW_all_tows_data[0]["x_mm"].values
    top_edges = np.array([tow["top_edge"].values for tow in RW_all_tows_data])
    bottom_edges = np.array([tow["bottom_edge"].values for tow in RW_all_tows_data])

    # The total height of the stack at each x position
    total_height = top_edges[-1, :] - bottom_edges[0, :]

    # Extract gap/overlap widths at each x (positive=gap, negative=overlap)
    gap_widths = np.clip(gap_overlap_df.values, 0, None)
    overlap_widths = np.clip(-gap_overlap_df.values, 0, None)

    # Sum all gaps and overlaps at each x
    local_gap_sum = np.sum(gap_widths, axis=1)
    local_overlap_sum = np.sum(overlap_widths, axis=1)

    # Compute local percentages relative to total height
    local_gap_percent = (local_gap_sum / total_height) * 100
    local_overlap_percent = (local_overlap_sum / total_height) * 100

    if print_statement:
        print(f"Average local gap percent: {np.mean(local_gap_percent):.2f}%")
        print(f"Average local overlap percent: {np.mean(local_overlap_percent):.2f}%")

    # Return both total and local data
    return {
        "x_vals": x_vals,
        "local_gap_percent": local_gap_percent,
        "local_overlap_percent": local_overlap_percent,
        "total_gap_percent": total_gap_percent,
        "total_overlap_percent": total_overlap_percent,
        "RW_all_tows_data": RW_all_tows_data,
        "gap_overlap_df": gap_overlap_df}

def generate_RW_multitow_layout_lengths(
    num_tows: int = 5,
    tow_spacing_mm: float = 6.35,
    tow_width_mm: float = 6.35,
    tow_length_mm: float = 1000,
    proposal_type: str = "RWM",
    print_statement: bool = False,
    starting_mods: list = None,
    alternate_start: list = None,
    override: bool = False,
    plot: bool = False,
    scaled: bool = False,
    histogram_bins: int = 30
):
    """
    Wrapper that calls generate_RW_multitow(...) to create RW tows, then:
      - builds a gap/overlap DataFrame (indexed by x in mm),
      - extracts continuous gap / overlap segment lengths (in mm),
      - shows histogram data (no distribution fitting).

    Returns:
        gap_overlap_df : DataFrame of pointwise gap/overlap distances (indexed by x_mm)
        gap_lengths     : numpy array of continuous gap segment lengths (mm)
        overlap_lengths : numpy array of continuous overlap segment lengths (mm)
        hist_data       : dict with histogram counts and bin edges for gaps/overlaps
    """

    if starting_mods is None:
        starting_mods = [None, 1, 1]
    if alternate_start is None:
        alternate_start = [None, "params"]

    try:
        gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent, RW_all_tows_data = generate_RW_multitow(
            num_tows=num_tows,
            tow_spacing_mm=tow_spacing_mm,
            tow_width_mm=tow_width_mm,
            tow_length_mm=tow_length_mm,
            proposal_type=proposal_type,
            print_statement=print_statement,
            starting_mods=starting_mods,
            alternate_start=alternate_start,
            override=override
        )
    except NameError:
        raise RuntimeError("generate_RW_multitow must be defined and importable in the current namespace.")

    # Ensure we have an x index to use (x_mm from RW_all_tows_data[0])
    if len(RW_all_tows_data) == 0:
        raise RuntimeError("RW_all_tows_data returned empty; cannot compute segment lengths.")
    x_vals = np.array(RW_all_tows_data[0]["x_mm"])
    if not np.allclose(gap_overlap_df.index.values.astype(float), x_vals.astype(float)):
        gap_overlap_df = gap_overlap_df.copy()
        gap_overlap_df.index = x_vals

    # --- Helper to extract segment lengths (in mm) from a series indexed by x (mm) ---
    def _extract_segment_lengths(series, positive=True):
        values = series.values
        if len(values) == 0:
            return np.array([], dtype=float)
        mask = values > 0 if positive else values < 0
        lengths = []
        run_length = 0
        for val in mask:
            if val:
                run_length += 1
            elif run_length > 0:
                lengths.append(run_length)
                run_length = 0
        if run_length > 0:
            lengths.append(run_length)
        if len(lengths) == 0:
            return np.array([], dtype=float)
        dx = series.index[1] - series.index[0]
        return np.array(lengths, dtype=float) * dx

    # Aggregate segment lengths across adjacent tow pairs
    gap_lengths_list = []
    overlap_lengths_list = []

    for col in gap_overlap_df.columns:
        series = gap_overlap_df[col]
        gap_lengths_list.extend(_extract_segment_lengths(series, positive=True).tolist())
        overlap_lengths_list.extend(_extract_segment_lengths(series, positive=False).tolist())

    gap_lengths = np.array(gap_lengths_list, dtype=float)
    overlap_lengths = np.array(overlap_lengths_list, dtype=float)

    # Compute histogram data (no fitting)
    gap_counts, gap_bins = np.histogram(gap_lengths, bins=histogram_bins)
    overlap_counts, overlap_bins = np.histogram(overlap_lengths, bins=histogram_bins)

    hist_data = {
        "gap": {"counts": gap_counts, "bins": gap_bins},
        "overlap": {"counts": overlap_counts, "bins": overlap_bins}
    }

    # --- Optional plotting ---
    if plot:
        # 1) Layout plot (centerlines + top/bottom edges)
        plt.figure(figsize=(10, 5))
        for tow_index, tow_df in enumerate(RW_all_tows_data):
            x = tow_df["x_mm"]
            center = tow_df["centerline"]
            top = tow_df["top_edge"]
            bottom = tow_df["bottom_edge"]
            color = plt.get_cmap("tab10")(tow_index % 10)
            plt.plot(x, center, "--", color=color, linewidth=1.2, label="Tow centerline" if tow_index == 0 else "_nolegend_")
            plt.plot(x, top, "-", color=color, linewidth=1.5, label="Tow edges" if tow_index == 0 else "_nolegend_")
            plt.plot(x, bottom, "-", color=color, linewidth=1.5)
        offsets = np.arange(num_tows) * tow_spacing_mm
        for offset in offsets:
            plt.plot(x_vals, np.full_like(x_vals, offset), ":", color="black", linewidth=1)
        plt.xlabel("Tow length (mm)")
        plt.ylabel("Tow position (mm)")
        plt.title("RW Simulated Multi-Tow Layout")
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
        if scaled:
            plt.axis("equal")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

        # 2) Histograms (gap and overlap lengths)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(gap_lengths, bins=histogram_bins, color="skyblue", edgecolor="black")
        ax[0].set_xlabel("Gap length (mm)")
        ax[0].set_ylabel("Count")
        ax[0].set_title("Gap Length Distribution")
        ax[0].grid(True, linestyle=":")

        ax[1].hist(overlap_lengths, bins=histogram_bins, color="salmon", edgecolor="black")
        ax[1].set_xlabel("Overlap length (mm)")
        ax[1].set_ylabel("Count")
        ax[1].set_title("Overlap Length Distribution")
        ax[1].grid(True, linestyle=":")

        plt.tight_layout()
        plt.show()

    # --- Summary printout ---
    print("\n--- Gap Lengths ---")
    print(f"  N = {len(gap_lengths)}")
    if len(gap_lengths):
        print(f"  Mean = {np.mean(gap_lengths):.3f} mm, Std = {np.std(gap_lengths):.3f} mm")

    print("\n--- Overlap Lengths ---")
    print(f"  N = {len(overlap_lengths)}")
    if len(overlap_lengths):
        print(f"  Mean = {np.mean(overlap_lengths):.3f} mm, Std = {np.std(overlap_lengths):.3f} mm")

    return gap_overlap_df, gap_lengths, overlap_lengths, hist_data

def run_multiple_RW_simulations_for_gaps_and_overlap_percentages(
    n_simulations: int = 100,
    num_tows: int = 5,
    tow_spacing_mm: float = 6.35,
    tow_width_mm: float = 6.35,
    tow_length_mm: float = 1000,
    proposal_type: str = "RWM",
    starting_mods: list = [None, 1, 1],
    alternate_start: list = [None, "params"],
    override: bool = False,
    verbose: bool = False,
    progress_bar: bool = True):
    """
    Runs multiple random walk multitow simulations and computes the average
    and standard deviation of the gap and overlap percentages.

    Returns:
        summary_df (pd.DataFrame): simulation results table
        stats (dict): mean and std of gap and overlap percentages
    """
    gap_percents = []
    overlap_percents = []

    iterator = tqdm(range(n_simulations), desc="Running simulations") if progress_bar else range(n_simulations)

    for i in iterator:
        try:
            _, _, _, gap_percent, overlap_percent, _ = generate_RW_multitow(
                num_tows=num_tows,
                tow_spacing_mm=tow_spacing_mm,
                tow_width_mm=tow_width_mm,
                tow_length_mm=tow_length_mm,
                proposal_type=proposal_type,
                print_statement=False,
                starting_mods=starting_mods,
                alternate_start=alternate_start,
                override=override)

            gap_percents.append(gap_percent)
            overlap_percents.append(overlap_percent)

            if verbose:
                print(f"Sim {i+1}: Gap={gap_percent:.2f}%, Overlap={overlap_percent:.2f}%")

        except Exception as e:
            print(f"Simulation {i+1} failed: {e}")
            continue

    # compile results
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

    print("\n=== Simulation Summary ===")
    print(f"Average Gap: {stats['Mean_Gap_%']:.3f}% ± {stats['Std_Gap_%']:.3f}%")
    print(f"Average Overlap: {stats['Mean_Overlap_%']:.3f}% ± {stats['Std_Overlap_%']:.3f}%")

    return summary_df, stats

def update_states():
    #print(random.getstate())
    state_data.append(random.getstate())

def initiate_state_data():
    global state_data
    state_data = []

def check_state_data():
    print("checking states...")
    print(len(state_data))
    duplicate_states = False

    for state in state_data:
        n_state = state_data.count(state)
        if n_state != 1:
            print(f"a state has occured {n_state} times, it is: {state}")
            duplicate_states = True

    if duplicate_states: print("duplicate states detected")
    else: print("no duplicate states detected")

##############################################################################################################
"""Run this file"""

def main():
    # LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params = fit_random_walk("CAM")
    #LT_walk_data = generate_random_walk("LT", LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params,
    #                            proposal_type="RWM", plot_histogram=True, plot_path=True)
    #plot_RW_tows(proposal_type="RWM", plot_individual_histograms=True)
    #std = get_proposal_distribution("CAM", plot=True)
    #plot_animated_walk_hist("CAM", 100)
    #print(get_n_steps("CAM"))
    #plot_LLS_hist()
    #check_burn_in("CAM", 23200, 5)
    #generate_random_walk("CAM", n_steps=30000, burn_in_period=0, proposal_type="RWM", plot_covergence_params=True, plot_histogram=True, plot_path=True)

    # generate_RW_multitow(num_tows=10)
    #plot_RW_tows(2, plot_individual_histograms=True)
    # analyze_tow_spacing_effect(spacing_values_mm = np.linspace(5.0, 7.5, 99), num_simulations = 100, num_tows_per_simulation = 29) # Takes 16 hours
    analyze_tow_spacing_effect(existing_data="Cached Data/tow_spacing_effect_RWM_with_100_simulations_of_a_29_tow_laminate.csv") # Only plots data
    # run_multiple_RW_simulations_for_gaps_and_overlap_percentages(n_simulations=500,num_tows=31)
    #plot_LLS_hist()

    # generate_RW_multitow_layout_lengths(num_tows=30, plot=True, histogram_bins = 300)

    # generate_random_walk(sensor='CAM', n_steps=LT_steps, proposal_std=LT_proposal_std, target_dist=LT_target_dist, dist=LT_dist, params=LT_params, proposal_type='RWM', plot_histogram=True, return_pdf=True)

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
