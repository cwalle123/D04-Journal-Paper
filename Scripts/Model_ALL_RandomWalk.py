"""Explanation of file:..."""

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

# Internal imports
from Handling_ALL_Functions import get_synced_data
from constants import tow_width_specified
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path, generate_starting_error, get_data_pairs
from Data_ALL_statistics import plot_histograms_separated, best_fit_distribution

##############################################################################################################
"""Functions"""

def get_data(sensor: str, tows: list = list(np.arange(2, 32, 1)), format: str = "merged"):
    """
    This function gets the required data for the specified sensor.
    """
    # Wrong sensor error message
    if not sensor == "LT" and not sensor == "CAM" and not sensor == "LLS_A" and not sensor == "LLS_B":
        raise ValueError("Invalid sensor type. Possible values are 'LT', 'CAM', 'LLS_A', and 'LLS_B'.")

    data, weights = [], []
    # loops through each tow that is specified and get data
    for tow_num in tows:
        tow_data = get_synced_data(tow_num, sensor)

        if sensor == "LLS_A":
            tow_data = tow_data[["error_LLS_A", "Weights"]]
        elif sensor == "LLS_B":
            tow_data = tow_data[["error_LLS_B", "Weights"]]
        elif sensor == "CAM":
            tow_data = tow_data[["error_CAM", "Weights"]]
        elif sensor == "LT":
            tow_data = tow_data[(tow_data["x"] >= 0) & (tow_data["x"] <= 1000)]     # TODO: check if this is correct
            tow_data = tow_data[["error_LT", "Weights"]]
        tow_data = np.array(tow_data)

        if format == "merged":
            # put data into correct format (pairs)
            for i in range(len(tow_data[:, 0])):
                data.append(float(tow_data[i, 0]))
                weights.append(float(tow_data[i, 1]))

        elif format == "separated":
            data.append(tow_data[:, 0])
            weights.append(tow_data[:, 1])

        else: print('Invalid format. Possible values are "merged" and "separated".')
    return data, weights

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
    return proposal

def fit_random_walk(sensor: str):
    n_steps = get_n_steps(sensor)
    data, weights = get_data(sensor, format='merged')

    best = best_fit_distribution(np.array(data), weights=np.array(weights))
    dist, params = best['dist'], best['params']
    target_distribution = lambda x: dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

    proposal_std = get_proposal_distribution(sensor)

    return n_steps, proposal_std, target_distribution, dist, params

def generate_random_walk(sensor: str, n_steps: int, proposal_std:  float, target_dist, dist, params, proposal_type: str="RWM",
                         plot_histogram: bool=False, plot_path: bool=False, comparison: bool=False, return_pdf: bool=False,
                         burn_in_period: int=0, plot_covergence_params: bool=False, print_statement: bool=False):
    '''
    This function generates a random walk according to the specified sensor.
    '''

    # adding the burn in period to the #steps
    actual_steps = n_steps + burn_in_period

    start_value = dist.rvs(*params[:-2], loc=params[-2], scale=params[-1])
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
                         proposal_type: str="RWM", print_statement: bool=False, starting_mods: list=[None, 1, 1]):
    """This function generate a multitow layout using RW"""

    # fitting random walk to experimental data
    LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params = fit_random_walk("LT")
    CAM_steps, CAM_proposal_std, CAM_target_dist, CAM_dist, CAM_params = fit_random_walk("CAM")
    LLS_B_steps, LLS_B_proposal_std, LLS_B_target_dist, LLS_B_dist, LLS_B_params = fit_random_walk("LLS_B")
    if print_statement == True:
        print("LT_steps = ", LT_steps, "CAM_steps = ", CAM_steps, "LLS_B_steps = ", LLS_B_steps)

    # This seciton modifies the starting distributions used by the model, which is needed for Model_ALL_StartVariations.
    if starting_mods != [None, 1, 1]:
        if starting_mods[0] != None:  # this changes the starting distribution type if necessary
            dist = starting_mods[0]

        # these are the factors by which the mean and std are changed
        loc_factor, scale_factor = starting_mods[1], starting_mods[2]

        # code used for start value: start_value = dist.rvs(*params[:-2], loc=params[-2], scale=params[-1])
        LT_params, CAM_params, LLS_B_params = list(LT_params), list(CAM_params), list(LLS_B_params)
        LT_params[-2] *= loc_factor
        CAM_params[-2] *= loc_factor
        LLS_B_params[-2] *= loc_factor
        LT_params[-1] *= scale_factor      # TODO: make sure the scale parameters equally affect the different distribution types
        CAM_params[-1] *= scale_factor
        LLS_B_params[-1] *= scale_factor
        LT_params, CAM_params, LLS_B_params = tuple(LT_params), tuple(CAM_params), tuple(LLS_B_params)

    tow_offset = 0
    RW_all_tows_data, top_edge_paths, bottom_edge_paths = [], [], []
    for n in range(num_tows):

        # generating random walk data
        LT_walk_data = generate_random_walk("LT", LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params, proposal_type=proposal_type)
        CAM_walk_data = generate_random_walk("CAM", CAM_steps, CAM_proposal_std, CAM_target_dist, CAM_dist, CAM_params, proposal_type=proposal_type)
        LLSB_walk_data = generate_random_walk("LLS_B", LLS_B_steps, LLS_B_proposal_std, LLS_B_target_dist, LLS_B_dist, LLS_B_params, proposal_type=proposal_type)

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

        # getting it into centerline and width format
        tow_centerline_data = tow_offset + np.array(CAM_walk_data) + np.array(LT_walk_data)
        tow_width_data = tow_width_mm + np.array(LLSB_walk_data)

        #print(f'Tow centerline: {tow_centerline_data}')
        #print(f'Tow width: {tow_width_data}')

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
    print_progress: bool = True):
    """
    Runs multiple simulations of generate_RW_multitow() over a range of tow spacings
    and computes the average gap and overlap percentages for each spacing.
    Also plots the point where gap and overlap percentages intersect.

    Parameters
    ----------
    spacing_values_mm : list, optional
        List of tow spacing values (in mm) to test.
        Defaults to np.linspace(5, 7.5, 9) mm.
    num_simulations : int
        Number of random simulations per spacing value.
    num_tows : int
        Number of tows per simulation.
    tow_width_mm : float
        Nominal width of each tow.
    tow_length_mm : float
        Length of each tow in mm.
    proposal_type : str
        Type of random walk proposal ("RWM", "MALA", etc.).
    print_progress : bool
        Whether to print progress updates.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with average gap and overlap percentages vs tow spacing.
    """

    if spacing_values_mm is None:
        spacing_values_mm = np.linspace(5.0, 7.5, 9)

    avg_gap_percentages = []
    avg_overlap_percentages = []

    for spacing in spacing_values_mm:
        if print_progress:
            print(f"\n--- Simulating for tow spacing = {spacing:.2f} mm ---")

        gap_results = []
        overlap_results = []

        for sim in range(num_simulations):
            _, _, _, gap_percent, overlap_percent, _ = generate_RW_multitow(
                num_tows=num_tows_per_simulation,
                tow_spacing_mm=spacing,
                tow_width_mm=tow_width_mm,
                tow_length_mm=tow_length_mm,
                proposal_type=proposal_type,
                print_statement=False,
            )
            gap_results.append(gap_percent)
            overlap_results.append(overlap_percent)

            if print_progress and num_simulations >= 10 and (sim + 1) % (num_simulations // 100) == 0:
                print(f"  Completed {sim + 1}/{num_simulations} simulations")

        avg_gap = np.mean(gap_results)
        avg_overlap = np.mean(overlap_results)
        avg_gap_percentages.append(avg_gap)
        avg_overlap_percentages.append(avg_overlap)

        if print_progress:
            print(f"  → Average gap: {avg_gap:.3f}% | Average overlap: {avg_overlap:.3f}%")

    # Compile results into DataFrame
    results_df = pd.DataFrame({
        "Tow Spacing (mm)": spacing_values_mm,
        "Average Gap (%)": avg_gap_percentages,
        "Average Overlap (%)": avg_overlap_percentages,
    })

    # --- Find intersection (where gap = overlap) ---
    gap_arr = np.array(avg_gap_percentages)
    overlap_arr = np.array(avg_overlap_percentages)
    diff = gap_arr - overlap_arr

    intersection_spacing = None
    intersection_gap_value = None

    # Find where sign changes (i.e., where curves cross)
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            # Linear interpolation for more precise intersection
            x1, x2 = spacing_values_mm[i], spacing_values_mm[i + 1]
            y1, y2 = diff[i], diff[i + 1]
            intersection_spacing = x1 - y1 * (x2 - x1) / (y2 - y1)

            # Corresponding gap (≈ overlap) value at intersection
            g1, g2 = gap_arr[i], gap_arr[i + 1]
            intersection_gap_value = g1 + (g2 - g1) * ((intersection_spacing - x1) / (x2 - x1))
            break

    # --- Plot results ---
    plt.figure(figsize=(8, 5))
    plt.plot(spacing_values_mm, avg_gap_percentages, marker="o", label="Average Gap %")
    plt.plot(spacing_values_mm, avg_overlap_percentages, marker="s", label="Average Overlap %")

    if intersection_spacing is not None:
        plt.axvline(intersection_spacing, color="red", linestyle="--", alpha=0.6)
        plt.scatter(intersection_spacing, intersection_gap_value, color="red", s=80, zorder=5)
        plt.text(intersection_spacing, intersection_gap_value + 0.3,
                 f"  Intersection = {intersection_spacing:.2f} mm",
                 color="red", fontsize=9, va="bottom")

    plt.title(f"Effect of Tow Spacing on Gap/Overlap Percentage ({proposal_type})")
    plt.xlabel("Programmed Shift (mm)")
    plt.ylabel("Defect Area (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if intersection_spacing is not None:
        print(f"\n🔴 Intersection point at {intersection_spacing:.3f} mm "
              f"(Gap ≈ Overlap = {intersection_gap_value:.3f}%)")
    else:
        print("\n⚠️ No intersection found between gap and overlap curves.")

    return results_df


##############################################################################################################
"""Run this file"""

def main():
    #LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params = fit_random_walk("LT")
    #LT_walk_data = generate_random_walk("LT", LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params,
                                proposal_type="RWM", plot_histogram=True, plot_path=True)
    #plot_RW_tows(proposal_type="RWM", plot_individual_histograms=True)
    #std = get_proposal_distribution("CAM", plot=True)
    #plot_animated_walk_hist("CAM", 100)
    #print(get_n_steps("CAM"))
    #plot_LLS_hist()
    #check_burn_in("CAM", 23200, 5)
    #generate_random_walk("CAM", n_steps=30000, burn_in_period=0, proposal_type="RWM", plot_covergence_params=True, plot_histogram=True, plot_path=True)
    # generate_RW_multitow(num_tows=10)
    # plot_RW_tows(2)
    analyze_tow_spacing_effect(spacing_values_mm = np.linspace(5.0, 7.5, 99), num_simulations = 100, num_tows_per_simulation = 2)


if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else