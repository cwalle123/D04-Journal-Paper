'''
    Way the model works: We input our current error, which we will call x
    A regression model is made by binning the data, calculate the mean of each bin and find a regression line
    we calculate the mean of the next error from the regression model which we call y
    the value of x, the previous error, corresponds to a certain bin which contains a normal curve the randomness in the deviation of y 
    we extract a random point from the normal curve, and we add this value to the before calculated mean
    
    Note: we cant create a value of the mean or the histogram/normal curve of the devation for a certain data point of x(previous error),
    because we don’t have enough data points at that precise point. This is why bins have been created: 
    t
    his works, but will obtain a slight bias, because the deviation normal curve does not
    correspond to the exact value of x, but only to the values around it
'''

##############################################################################################################

#ideas for improvement:
#Find optimum number of bins
#only extract value of LLS B width if width_LLSB>width_LLSA
#to smoothen out curve if too much waviness
#increase resolution of predicting curve and taking mean of predicted points around real datapoint: more realistic dynamics, smoother paths
#use relation between errors to improve model

# EXternal imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from scipy.stats import linregress
import math
from scipy.stats import truncnorm
import time
import statsmodels.api as sm

# Internal imports
import constants
from Handling_ALL_Functions import get_synced_data

##############################################################################################################
"""Functions"""

def weighted_linregress(x, y, weights):
    """
    Perform weighted linear regression and return
    slope, intercept, r_value, p_value, stderr
    matching scipy.stats.linregress signature.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(weights)

    # Design matrix (add constant for intercept)
    X = sm.add_constant(x)

    # Fit WLS model
    model = sm.WLS(y, X, weights=w)
    res   = model.fit()

    intercept, slope = res.params
    stderr_slope      = res.bse[1]
    p_value_slope     = res.pvalues[1]

    # r_value is the signed sqrt of R²
    r_squared = res.rsquared
    r_value   = np.sign(slope) * np.sqrt(r_squared)

    return slope, intercept, r_value, p_value_slope, stderr_slope

def consecutive_error(sensor, test_ratio=0.2, num_bins = 20, random_state=None, bins_show = False, plot_fit=True, fourPlots = False, axs = None, noTitle = True, return_plot_data=False):
    """
        Analyze consecutive error pairs and their distributions from processed sensor data.

        Parameters
        ----------
        sensor : str
            The type of sensor data to process. This determines which column of data
            is used for analysis.
        test_ratio : float
            Proportion of data to use for testing, ranging from 0.0 to 1.0 (e.g., 0.2 uses
            20% of the data for testing and 80% for training).
        random_state : int
            Is used for the test/train split. Ensures reproducible splits of the data.
            Change it to another integer for a different split, or set it to None for random behavior.

    """
    # Wrong sensor error message
    if not sensor == "LT" and not sensor == "CAM" and not sensor == "LLS_A" and not sensor == "LLS_B":
        raise ValueError("Invalid sensor type. Possible values are 'LT', 'CAM', 'LLS_A', and 'LLS_B'.")

    # Takes care of which column to use
    if sensor == "CAM" or sensor == "LT":
        column = -2
    else:
        column = -1


    # Prepare an empty list to store (x_n, x_{n+1}) pairs for each tow as well as other lists
    all_pairs, time_pairs, x_pairs, vel = [], [], [], []

    # Loop through tow numbers from 1 to 31
    for tow_number in range(2, 32):
        # Get processed data for the current tow and sensor type
        tow_data_bef = get_synced_data(tow_number,spacesynced=True)

        if sensor == "LT":
            tow_data = tow_data_bef[["time", "x", "y", "z", "error_LT", "z error"]]
            x_lable = "Error robot position (mm)"
        if sensor == "LLS_A":
            tow_data = tow_data_bef[["time", "width_LLS_A", "center_LLS_A", "width error_LLS_A"]]
            x_lable = "Error tape width before compaction (mm)"
        if sensor == "LLS_B":
            tow_data = tow_data_bef[["time", "width_LLS_B", "center_LLS_B", "width error_LLS_B"]]
            x_lable = "Error tape width after compaction (mm)"
        if sensor == "CAM":
            tow_data = tow_data_bef[["time", "width_CAM", "center_CAM", "error_CAM"]]
            x_lable = "Error tape lateral movement (mm)"
        velocity_data = tow_data_bef[["time", "x"]]

        # Ensure that the returned object is a dataframe
        if not tow_data.empty and tow_data.shape[1] > 1:  # Ensure there are at least two columns
            # Extract the last or second-to-last column (based on sensor type)
            second_to_last_column = tow_data.iloc[:, column].values  # Convert to numpy array

            # Create (x_n, x_{n+1}) pairs for the current tow
            x_values = second_to_last_column[:-1]
            y_values = second_to_last_column[1:]

            # Extract x and time and convert to np
            velocity_data_x = velocity_data.iloc[:, -1].values
            velocity_data_time = velocity_data.iloc[:, -2].values
            time_values_i = velocity_data_time[:-1]
            time_values_i2 = velocity_data_time[1:]
            x_values_i = velocity_data_x[:-1]
            x_values_i2 = velocity_data_x[1:]

            # Append pairs as a list of tuples
            all_pairs.extend(zip(x_values, y_values))
            time_pairs.extend(zip(time_values_i, time_values_i2))
            x_pairs.extend(zip(x_values_i, x_values_i2))

        # After processing all tows, convert collected pairs into numpy arrays
    all_pairs = np.array(all_pairs)
    x_values = all_pairs[:, 0]
    y_values = all_pairs[:, 1]

    time_pairs = np.array(time_pairs)
    time_i = time_pairs[:, 0]
    time_i2 = time_pairs[:, 1]
    time_gaps = time_i2 - time_i

    x_pairs = np.array(x_pairs)
    x_i = x_pairs[:, 0]
    x_i2 = x_pairs[:, 1]
    x_gaps = x_i2 - x_i

    vel = x_gaps / time_gaps

    # Train-Test Split

    # Split into training and testing (test_ratio * 100)% of data is used.
    x_train, x_test, y_train, y_test, vel_train, vel_test = train_test_split(
        x_values, y_values, vel, test_size=test_ratio, random_state=random_state
    )
    # NOTE: random_state ensures reproducible splits of the data;
    # change it to another integer for a different split, or set it to None for random behavior.

    # Sort training x-values and reorder y-values accordingly
    sorted_indices = np.argsort(x_train)
    x_sorted = x_train[sorted_indices]
    y_sorted = y_train[sorted_indices]
    vel_sorted = vel_train[sorted_indices]

    # Equal-count bin edges
    bin_edges = np.linspace(0, len(x_sorted), num_bins + 1, dtype=int)

    # Compute bin-wise averages
    x_binned = [np.mean(x_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]
    y_binned = [np.mean(y_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]
    vel_binned = [np.mean(vel_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]

    # scatter Plot with Binned Averages and regression model
    # slope, intercept, r_value, p_value, std_err = linregress(x_binned, y_binned)
    slope, intercept, r_value, p_value, std_err = weighted_linregress(x_binned, y_binned, vel_binned)
    # print(r_value)

    # Define error label
    error_labels = {"LT": "y error", "CAM": "position error", "LLS_A": "width error", "LLS_B": "width error"}
    error_label = error_labels[sensor]

    # Plot scatter + binned fit
    if plot_fit:                    #TODO: fix these plots
        plt.figure(figsize=(5, 4))    #(6.5, 5)
        plt.scatter(x_train, y_train, alpha=0.2, marker='o', s=10, edgecolors='k', label="Training Set")
        plt.scatter(x_binned, y_binned, alpha=1, color='red', marker='s', s=10, label="Binned Averages")
        plt.plot(x_binned, np.array(x_binned) * slope + intercept, color='red', label='Linear Fit')

        plt.xlabel(x_lable, fontsize=constants.font_medium)  #"$ε_{i}$ (mm)"
        plt.ylabel("subsequent error (mm)", fontsize=constants.font_medium)    #"$ε_{i+1}$ (mm)"

        if not noTitle:
            plt.title(f"{sensor} {error_label} : Consecutive Error Correlation (Training set)",
                      fontsize=constants.font_small)

        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xticks(np.linspace(-1.2, 1.2, 9))
        plt.yticks(np.linspace(-1.2, 1.2, 9))
        plt.legend(fontsize=constants.font_small)
        #plt.grid(True)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

    if return_plot_data:
        return x_train, y_train, x_binned, y_binned, slope, intercept

    # Compute Deviations per Bin

    deviations_per_bin, velocities_per_bin = [], []

    for i in range(num_bins):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]

        # Get x and y values in this bin
        bin_x_values = x_sorted[bin_start:bin_end]
        bin_y_values = y_sorted[bin_start:bin_end]
        bin_vel_values = vel_sorted[bin_start:bin_end]

        # Predict y-values using regression model
        predicted_y_values = slope * bin_x_values + intercept

        # Compute deviation (residuals) at each point
        deviations = bin_y_values - predicted_y_values
        deviations_per_bin.append(deviations)
        velocities_per_bin.append(bin_vel_values)

    # Paginate histogram grids
    rows, cols = 4, 5
    plots_per_page = rows * cols
    total_bins = num_bins
    total_pages = math.ceil(total_bins / plots_per_page)

    if bins_show:
        for page in range(total_pages):
            start = page * plots_per_page
            end = min(start + plots_per_page, total_bins)

            fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
            fig.suptitle(f"{sensor} {error_label} : Histograms of Deviations per Bin (Page {page + 1}/{total_pages})",
                         fontsize=16)
            axes_flat = axes.flatten()

            for idx_plot in range(start, end):
                bin_idx = idx_plot
                ax = axes_flat[idx_plot - start]
                devs = deviations_per_bin[bin_idx]
                xs = x_sorted[bin_edges[bin_idx]:bin_edges[bin_idx + 1]]
                vels = velocities_per_bin[bin_idx]

                # Histogram and normal fit
                counts, bins_hist, _ = ax.hist(devs, bins=30, edgecolor='black', density=True)
                # mu, std = stats.norm.fit(devs)
                mu = np.average(devs, weights=vels)
                std = math.sqrt(np.average((devs - mu) ** 2, weights=vels))
                x_fit = np.linspace(devs.min(), devs.max(), 100)
                p_fit = stats.norm.pdf(x_fit, mu, std)
                ax.plot(x_fit, p_fit, 'r', linewidth=2)

                # Annotation
                annotation = f"x ∈ [{xs.min():.2f}, {xs.max():.2f}]\nμ = {mu:.4f}\nσ = {std:.4f}"
                ax.text(0.95, 0.95, annotation, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='right', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8))

                ax.set_title(f"Bin {bin_idx}")
                ax.set_xlabel("Deviation [mm]")
                ax.set_ylabel("Density")
                ax.grid(True)

            # Turn off unused subplots on last page
            for unused in range(end - start, plots_per_page):
                axes_flat[unused].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    # -------------------------
    # summarize all data
    # -------------------------

    bin_stats = []

    for i in range(num_bins):
        bin_devs = deviations_per_bin[i]
        vels = velocities_per_bin[i]
        x_mean = x_binned[i]
        y_mean = y_binned[i]
        # mu, std = stats.norm.fit(bin_devs)
        mu = np.average(bin_devs, weights=vels)
        std = math.sqrt(np.average((bin_devs - mu) ** 2, weights=vels))
        variance = std ** 2

        bin_stats.append({
            "x_mean": x_mean,
            "y_mean": y_mean,
            "deviation_mean": mu,
            "deviation_variance": variance
        })

    # Convert to DataFrame for easy viewing
    bin_stats_df = pd.DataFrame(bin_stats)

    # Display the table
    # print(bin_stats_df)

    return bin_stats_df, slope, intercept, r_value, p_value, std_err, x_sorted, bin_edges, deviations_per_bin

def generate_error_path(start_error, n_steps, slope, intercept, x_sorted, bin_edges, deviations_per_bin,use_truncnorm=False):
    np.random.seed()
    error_path = [start_error]
    x_current = start_error

    for _ in range(n_steps):
        # Predict mean of next error
        y_pred = slope * x_current + intercept

        # Find correct bin
        bin_index = None
        for i in range(len(bin_edges) - 1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_x_min = x_sorted[bin_start]
            bin_x_max = x_sorted[bin_end - 1]
            if bin_x_min <= x_current <= bin_x_max:
                bin_index = i
                break
        # Use edge bin if out of range
        if bin_index is None:
            bin_index = 0 if x_current < x_sorted[0] else len(bin_edges) - 2

        # Get deviation stats and sample a deviation
        deviations = deviations_per_bin[bin_index]
        mu, sigma = stats.norm.fit(deviations)
        if use_truncnorm:
            # Use truncated normal within ±2σ
            from scipy.stats import truncnorm
            a, b = -2, 2
            sampled_deviation = truncnorm(a, b, loc=mu, scale=sigma).rvs()
        else:
            # Use regular normal distribution
            sampled_deviation = np.random.normal(mu, sigma)
        # Next error
        next_error = y_pred + sampled_deviation
        error_path.append(next_error)
        x_current = next_error

    return np.array(error_path)

def generate_simulated_VS_real(n_real_tow=1, rdm_seed=0, test_ratio=0.2, errorCor_show=False, bins_show=False, num_bins=100, peak_plots = False, sim_plot = False):
    # Get binned models from historical data
    bin_stats_df, slope, intercept, r_value, p_value, std_err, x_sorted, bin_edges, deviations_per_bin = consecutive_error(
        "CAM", test_ratio=test_ratio, num_bins=num_bins, bins_show=bins_show, plot_fit=errorCor_show)
    bin_stats_df1, slope1, intercept1, r_value1, p_value1, std_err1, x_sorted1, bin_edges1, deviations_per_bin1 = consecutive_error(
        "LT", test_ratio=test_ratio, num_bins=num_bins, bins_show=bins_show, plot_fit=errorCor_show)
    bin_stats_df2, slope2, intercept2, r_value2, p_value2, std_err2, x_sorted2, bin_edges2, deviations_per_bin2 = consecutive_error(
        "LLS_B", test_ratio=test_ratio, num_bins=num_bins, bins_show=bins_show, plot_fit=errorCor_show)
    bin_stats_df3, slope3, intercept3, r_value3, p_value3, std_err3, x_sorted3, bin_edges3, deviations_per_bin3 = consecutive_error(
        "LLS_A", test_ratio=test_ratio, num_bins=num_bins, bins_show=bins_show, plot_fit=errorCor_show)


    # Load real tow data
    synced_data_tow_1 = get_synced_data(tow=n_real_tow,spacesynced=True)

    # --- CAM Error ---
    synced_data_cam_tow_1 = synced_data_tow_1["center_CAM"].values
    start_error = synced_data_cam_tow_1[0]
    n_steps = len(synced_data_cam_tow_1) - 1
    simulated_tow_path_cam = generate_error_path(
        start_error, n_steps, slope, intercept, x_sorted, bin_edges, deviations_per_bin
    )

    # --- LT Error ---
    synced_data_LT_tow_1 = synced_data_tow_1["error_LT"].values
    start_error1 = synced_data_LT_tow_1[0]
    simulated_tow_path_LT = generate_error_path(
        start_error1, n_steps, slope1, intercept1, x_sorted1, bin_edges1, deviations_per_bin1
    )

    # --- Centerline Offset ---
    simulated_total_offset_centerline = simulated_tow_path_LT + simulated_tow_path_cam
    total_offset_real = synced_data_LT_tow_1 + synced_data_cam_tow_1


    # --- LLS_B Width ---
    synced_data_LLS_B_tow_error = synced_data_tow_1["width error_LLS_B"].values
    synced_data_LLS_B_tow_width = synced_data_tow_1["width_LLS_B"].values
    start_error2 = synced_data_LLS_B_tow_error[0]
    simulated_tow_width_LLS_B = generate_error_path(
        start_error2, n_steps, slope2, intercept2, x_sorted2, bin_edges2, deviations_per_bin2
    )+6.35


    # --- LLS_A Width ---
    synced_data_LLS_A_tow_error = synced_data_tow_1["width error_LLS_A"].values
    synced_data_LLS_A_tow_width = synced_data_tow_1["width_LLS_A"].values
    start_error2_LLS_A = synced_data_LLS_A_tow_error[0]
    simulated_tow_width_LLS_A = generate_error_path(
        start_error2_LLS_A, n_steps, slope2, intercept2, x_sorted2, bin_edges2, deviations_per_bin2
    )+6.35

    # --- Compute Boundaries ---
    simulated_upper_boundary = simulated_total_offset_centerline + 0.5 * simulated_tow_width_LLS_B
    simulated_lower_boundary = simulated_total_offset_centerline - 0.5 * simulated_tow_width_LLS_B
    real_upper_boundary = total_offset_real + 0.5 * synced_data_LLS_B_tow_width
    real_lower_boundary = total_offset_real - 0.5 * synced_data_LLS_B_tow_width



    # 1. CAM Error

    sw_cam_real = -1  # sw=-1 means going down and sw=1 means going up
    peaks_cam_real = 0
    peak_cam_real_list = []
    peak_cam_real_idxs = []

    for i in range(len(synced_data_cam_tow_1)-1):
        if sw_cam_real == -1:
            if synced_data_cam_tow_1[i+1] > synced_data_cam_tow_1[i]:
                peaks_cam_real += 1
                sw_cam_real = 1
                peak_cam_real_idxs.append(i)
                peak_cam_real_list.append(synced_data_cam_tow_1[i])
        if sw_cam_real == 1:
            if synced_data_cam_tow_1[i+1] < synced_data_cam_tow_1[i]:
                peaks_cam_real += 1
                sw_cam_real = -1
                peak_cam_real_idxs.append(i)
                peak_cam_real_list.append(synced_data_cam_tow_1[i])
    #print('peaks_cam_real',peaks_cam_real)

    sw_cam_sim = -1  # sw=-1 means going down and sw=1 means going up
    peaks_cam_sim = 0
    peak_cam_sim_list = []
    peak_cam_sim_idxs = []

    for i in range(len(simulated_tow_path_cam) - 1):
        if sw_cam_sim == -1:
            if simulated_tow_path_cam[i + 1] > simulated_tow_path_cam[i]:
                peaks_cam_sim += 1
                sw_cam_sim = 1
                peak_cam_sim_idxs.append(i)
                peak_cam_sim_list.append(simulated_tow_path_cam[i])
        if sw_cam_sim == 1:
            if simulated_tow_path_cam[i + 1] < simulated_tow_path_cam[i]:
                peaks_cam_sim += 1
                sw_cam_sim = -1
                peak_cam_sim_idxs.append(i)
                peak_cam_sim_list.append(simulated_tow_path_cam[i])
    #print('peaks_cam_sim',peaks_cam_sim)



    # --- Plot 4 Separate Real vs Simulated Error Paths ---
    if peak_plots:
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    if peak_plots:
        axs[0].plot(synced_data_cam_tow_1, label="Real CAM Error", color="red")
        axs[0].plot(simulated_tow_path_cam, label="Simulated CAM Error", color="blue")
        axs[0].scatter(peak_cam_real_idxs, peak_cam_real_list, color="red", marker='o', s=50, label=f"Peaks Real ({peaks_cam_real})")
        axs[0].scatter(peak_cam_sim_idxs, peak_cam_sim_list, color="blue", marker='o', s=50, label=f"Peaks Sim ({peaks_cam_sim})")
        axs[0].set_ylabel("Error [mm]")
        axs[0].set_title("CAM Center Error")
        axs[0].legend()
        axs[0].grid(True)

    #print('total number of data points CAM', len(synced_data_cam_tow_1))


    # 2. LT Error

    sw_LT_real = -1  # sw=-1 means going down and sw=1 means going up
    peaks_LT_real = 0
    peak_LT_real_list = []
    peak_LT_real_idxs = []

    for i in range(len(synced_data_LT_tow_1) - 1):
        if sw_LT_real == -1:
            if synced_data_LT_tow_1[i + 1] > synced_data_LT_tow_1[i]:
                peaks_LT_real += 1
                sw_LT_real = 1
                peak_LT_real_idxs.append(i)
                peak_LT_real_list.append(synced_data_LT_tow_1[i])
        if sw_LT_real == 1:
            if synced_data_LT_tow_1[i + 1] < synced_data_LT_tow_1[i]:
                peaks_LT_real += 1
                sw_LT_real = -1
                peak_LT_real_idxs.append(i)
                peak_LT_real_list.append(synced_data_LT_tow_1[i])
    #print("peaks_LT_real", peaks_LT_real)

    sw_LT_sim = -1  # sw=-1 means going down and sw=1 means going up
    peaks_LT_sim = 0
    peak_LT_sim_list = []
    peak_LT_sim_idxs = []

    for i in range(len(simulated_tow_path_LT) - 1):
        if sw_LT_sim == -1:
            if simulated_tow_path_LT[i + 1] > simulated_tow_path_LT[i]:
                peaks_LT_sim += 1
                sw_LT_sim = 1
                peak_LT_sim_idxs.append(i)
                peak_LT_sim_list.append(simulated_tow_path_LT[i])
        if sw_LT_sim == 1:
            if simulated_tow_path_LT[i + 1] < simulated_tow_path_LT[i]:
                peaks_LT_sim += 1
                sw_LT_sim = -1
                peak_LT_sim_idxs.append(i)
                peak_LT_sim_list.append(simulated_tow_path_LT[i])
    #print('peaks_LT_sim',peaks_LT_sim)

    if peak_plots:
        axs[1].plot(synced_data_LT_tow_1, label="Real LT Error", color="red")
        axs[1].plot(simulated_tow_path_LT, label="Simulated LT Error", color="blue")
        axs[1].scatter(peak_LT_real_idxs, peak_LT_real_list, color="red", marker='o', s=50, label=f"Peaks Real ({peaks_LT_real})")
        axs[1].scatter(peak_LT_sim_idxs, peak_LT_sim_list, color="blue", marker='o', s=50, label=f"Peaks Sim ({peaks_LT_real})")
        axs[1].set_ylabel("Error [mm]")
        axs[1].set_title("LT y Error")
        axs[1].legend()
        axs[1].grid(True)


    #print('total number of data points LT', len(synced_data_LT_tow_1))
    # 3. Centerline Offset

    sw_offset_real = -1  # sw=-1 means going down and sw=1 means going up
    peaks_offset_real = 0
    peak_offset_real_list = []
    peak_offset_real_idxs = []

    for i in range(len(total_offset_real) - 1):
        if sw_offset_real == -1:
            if total_offset_real[i + 1] > total_offset_real[i]:
                peaks_offset_real += 1
                sw_offset_real = 1
                peak_offset_real_idxs.append(i)
                peak_offset_real_list.append(total_offset_real[i])
        if sw_offset_real == 1:
            if total_offset_real[i + 1] < total_offset_real[i]:
                peaks_offset_real += 1
                sw_offset_real = -1
                peak_offset_real_idxs.append(i)
                peak_offset_real_list.append(total_offset_real[i])
    #print("peaks_offset_real", peaks_offset_real)

    sw_offset_sim = -1  # sw=-1 means going down and sw=1 means going up
    peaks_offset_sim = 0
    peak_offset_sim_list = []
    peak_offset_sim_idxs = []

    for i in range(len(simulated_total_offset_centerline) - 1):
        if sw_offset_sim == -1:
            if simulated_total_offset_centerline[i + 1] > simulated_total_offset_centerline[i]:
                peaks_offset_sim += 1
                sw_offset_sim = 1
                peak_offset_sim_idxs.append(i)
                peak_offset_sim_list.append(simulated_total_offset_centerline[i])
        if sw_offset_sim == 1:
            if simulated_total_offset_centerline[i + 1] < simulated_total_offset_centerline[i]:
                peaks_offset_sim += 1
                sw_offset_sim = -1
                peak_offset_sim_idxs.append(i)
                peak_offset_sim_list.append(simulated_total_offset_centerline[i])
    #print('peaks_offset_sim',peaks_offset_sim)

    if peak_plots:
        axs[2].plot(total_offset_real, label="Real Total Offset", color="red")
        axs[2].plot(simulated_total_offset_centerline, label="Simulated Total Offset", color="blue")
        axs[2].scatter(peak_offset_real_idxs, peak_offset_real_list, color="red", marker='o', s=50, label=f"Peaks Real ({peaks_offset_real})")
        axs[2].scatter(peak_offset_sim_idxs, peak_offset_sim_list, color="blue", marker='o', s=50, label=f"Peaks Sim ({peaks_offset_sim})")
        axs[2].set_ylabel("Offset [mm]")
        axs[2].set_title("Offset from Centerline")
        axs[2].legend()
        axs[2].grid(True)



    # 4. Width LLS_B

    sw_LLS_B_real = -1  # sw=-1 means going down and sw=1 means going up
    peaks_LLS_B_real = 0
    peak_LLS_B_real_list = []
    peak_LLS_B_real_idxs = []

    for i in range(len(synced_data_LLS_B_tow_width) - 1):
        if sw_LLS_B_real == -1:
            if synced_data_LLS_B_tow_width[i + 1] > synced_data_LLS_B_tow_width[i]:
                peaks_LLS_B_real += 1
                sw_LLS_B_real = 1
                peak_LLS_B_real_idxs.append(i)
                peak_LLS_B_real_list.append(synced_data_LLS_B_tow_width[i])
        if sw_LLS_B_real == 1:
            if synced_data_LLS_B_tow_width[i + 1] < synced_data_LLS_B_tow_width[i]:
                peaks_LLS_B_real += 1
                sw_LLS_B_real = -1
                peak_LLS_B_real_idxs.append(i)
                peak_LLS_B_real_list.append(synced_data_LLS_B_tow_width[i])

    sw_LLS_B_sim = -1  # sw=-1 means going down and sw=1 means going up
    peaks_LLS_B_sim = 0
    peak_LLS_B_sim_list = []
    peak_LLS_B_sim_idxs = []

    for i in range(len(simulated_tow_width_LLS_B) - 1):
        if sw_LLS_B_sim == -1:
            if simulated_tow_width_LLS_B[i + 1] > simulated_tow_width_LLS_B[i]:
                peaks_LLS_B_sim += 1
                sw_LLS_B_sim = 1
                peak_LLS_B_sim_idxs.append(i)
                peak_LLS_B_sim_list.append(simulated_tow_width_LLS_B[i])
        if sw_LLS_B_sim == 1:
            if simulated_tow_width_LLS_B[i + 1] < simulated_tow_width_LLS_B[i]:
                peaks_LLS_B_sim += 1
                sw_LLS_B_sim = -1
                peak_LLS_B_sim_idxs.append(i)
                peak_LLS_B_sim_list.append(simulated_tow_width_LLS_B[i])
    #print('peaks_LLS_B_sim', peaks_LLS_B_sim)

    if peak_plots:
        axs[3].plot(synced_data_LLS_B_tow_width, label="Real Width LLS_B", color="red")
        axs[3].plot(simulated_tow_width_LLS_B, label="Simulated Width LLS_B", color="blue")
        axs[3].scatter(peak_LLS_B_real_idxs, peak_LLS_B_real_list, color="red", marker='o', s=50, label=f"Peaks Real ({peaks_LLS_B_real})")
        axs[3].scatter(peak_LLS_B_sim_idxs, peak_LLS_B_sim_list, color="blue", marker='o', s=50, label=f"Peaks Sim ({peaks_LLS_B_sim})")
        axs[3].set_ylabel("Width [mm]")
        axs[3].set_xlabel("Step")
        axs[3].set_title("Tow Width (LLS_B)")
        axs[3].legend()
        axs[3].grid(True)

        fig.suptitle(f'Simulated vs Real (tow {n_real_tow}) with {len(synced_data_LT_tow_1)} datapoints', fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    # LLS_A
    # if peak_plots:
    #     axs[4].plot(synced_data_LLS_A_tow_width, label="Real Width LLS_B", color="red")
    #     axs[4].plot(simulated_tow_width_LLS_A, label="Simulated Width LLS_B", color="blue")
    #     axs[4].set_ylabel("Width [mm]")
    #     axs[4].set_xlabel("Step")
    #     axs[4].set_title("Tow Width (LLS_A)")
    #     axs[4].legend()
    #     axs[4].grid(True)


    #print('total number of data points LLS_B', len(synced_data_LLS_B_tow_width))



    if sim_plot:
        # Convert steps to length (mm)
        x_sign = np.linspace(0, get_delta_x(n_real_tow), len(simulated_total_offset_centerline))

        print(get_delta_x(n_real_tow))
        # --- Overlay Plot with Upper/Lower Boundaries ---
        plt.figure(figsize=(8, 5))
        plt.plot(x_sign, simulated_total_offset_centerline, label="model's total offset simulated", c="orange", linestyle='--')
        plt.plot(x_sign, simulated_upper_boundary, label="model's lower/upper edges", c="orange")
        plt.plot(x_sign, simulated_lower_boundary, c="orange")
        plt.plot(x_sign, total_offset_real, label="experimental total offset", c="b", linestyle='--')
        plt.plot(x_sign, real_upper_boundary, label="experimental lower/upper edges", c="b")
        plt.plot(x_sign, real_lower_boundary, c="b")
        plt.xlabel("X position (mm)", fontsize=constants.font_small)
        plt.ylabel("Displacement (mm)", fontsize=constants.font_small)
        # plt.title("Simulated Machine Error Path Over Time")
        plt.grid(True)
        plt.legend(
            fontsize=constants.font_extra_small,
            loc='upper right',  # anchor point of the legend box
            bbox_to_anchor=(1, 0.85)  # x, y coordinates outside or inside the axes (1 is right edge)
        )
        plt.show()

    return peaks_cam_sim, peaks_LT_sim, peaks_offset_sim, peaks_LLS_B_sim, peaks_cam_real, peaks_LT_real, peaks_offset_real, peaks_LLS_B_real, synced_data_cam_tow_1, synced_data_LT_tow_1, total_offset_real, synced_data_LLS_B_tow_width

def peakMeanRealTows():

    cam_lst, LT_lst, offset_lst, LLS_B_lst = [], [], [], []
    final_real_means = []

    for i in range(2,31):
        (peaks_cam_sim, peaks_LT_sim, peaks_offset_sim, peaks_LLS_B_sim, peaks_cam_real, peaks_LT_real, peaks_offset_real,
         peaks_LLS_B_real, synced_data_cam_tow_1, synced_data_LT_tow_1, total_offset_real, synced_data_LLS_B_tow_width) = (
            generate_simulated_VS_real(n_real_tow=i, rdm_seed=0, errorCor_show=False, bins_show=False,
                                       num_bins=10, peak_plots=False, sim_plot=False, test_ratio=0.2))
        cam_lst.append(peaks_cam_real / len(synced_data_cam_tow_1))
        LT_lst.append(peaks_LT_real / len(synced_data_LT_tow_1))
        offset_lst.append(peaks_offset_real / len(total_offset_real))
        LLS_B_lst.append(peaks_LLS_B_real / len(synced_data_LLS_B_tow_width))

    final_real_means.append(float(np.mean(cam_lst)))
    final_real_means.append(float(np.mean(LT_lst)))
    final_real_means.append(float(np.mean(offset_lst)))
    final_real_means.append(float(np.mean(LLS_B_lst)))

    return final_real_means

def peaksVSbins(bins, nb_sim):
    start_time = time.time() # ETA stuff
    iteration = 0  # ETA stuff
    total = len(bins)  # ETA stuff

    means_cam_sim = []
    means_LT_sim = []
    means_offset_sim = []
    means_LLS_B_sim = []
    for j in bins:
        iteration += 1
        iter_start = time.time()  # ETA stuff

        peaks_cam_list = []
        peaks_LT_list = []
        peaks_offset_list = []
        peaks_LLS_B_list = []
        for i in range(nb_sim):
            (peaks_cam_sim, peaks_LT_sim, peaks_offset_sim, peaks_LLS_B_sim, peaks_cam_real, peaks_LT_real,
             peaks_offset_real, peaks_LLS_B_real, synced_data_cam_tow_1, synced_data_LT_tow_1, total_offset_real, synced_data_LLS_B_tow_width) \
                = generate_simulated_VS_real(n_real_tow=3,
                                            errorCor_show=False, bins_show=False, sim_plot=False, num_bins=j,
                                             peak_plots = False, test_ratio=0.2)
            peaks_cam_list.append(peaks_cam_sim)
            peaks_LT_list.append(peaks_LT_sim)
            peaks_offset_list.append(peaks_offset_sim)
            peaks_LLS_B_list.append(peaks_LLS_B_sim)

        means_cam_sim.append(float(np.mean(peaks_cam_list)))
        means_LT_sim.append(float(np.mean(peaks_LT_list)))
        means_offset_sim.append(float(np.mean(peaks_offset_list)))
        means_LLS_B_sim.append(float(np.mean(peaks_LLS_B_list)))

        # timing
        if iteration == 1 or iteration % 10 == 0 or iteration == total:
            iter_elapsed = time.time() - iter_start
            overall_elapsed = time.time() - start_time
            avg_time = overall_elapsed / iteration
            remaining = total - iteration
            eta = time.time() + remaining * avg_time
            eta_str = time.strftime("%H:%M", time.localtime(eta))

            print(f"Completed {iteration}/{total} bins (j={j}, took {iter_elapsed:.2f}s). ETA: {eta_str}")

    return means_cam_sim, means_LT_sim, means_offset_sim, means_LLS_B_sim

def GlobalValidation(nb_bins=101, nb_sim=20):
    bin_list = list(range(10, nb_bins + 1))
    print('To test for', len(bin_list), 'bins, and',nb_sim, 'simulations per bin, we have to simulate', len(bin_list)*nb_sim,'tows.')
    means_cam_sim, means_LT_sim, means_offset_sim, means_LLS_B_sim = peaksVSbins(bin_list, nb_sim=nb_sim)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # helper to draw series + regression
    def plot_with_reg(ax, x, y, label, color=None):
        ax.plot(x, y, 'o', label=label, color=color)
        # fit a line
        # m, b = np.polyfit(x, y, 1)
        # ax.plot(x, m * np.array(x) + b, '--', label=f"{label} fit", color=color)
        ax.set_title(label)
        ax.grid(True)
        # ax.legend()

    # CAM
    plot_with_reg(axes[0, 0], bin_list, means_cam_sim, "CAM")

    # LT
    plot_with_reg(axes[0, 1], bin_list, means_LT_sim, "LT", color='tab:orange')

    # Offset
    plot_with_reg(axes[1, 0], bin_list, means_offset_sim, "Centerline Offset", color='tab:green')


    # LLS_B
    plot_with_reg(axes[1, 1], bin_list, means_LLS_B_sim, "LLS_B", color='tab:red')


    # Shared X-label
    fig.text(0.5, 0.02, 'Number of bins', ha='center', va='center', rotation='horizontal')

    # Shared Y-label
    fig.text(0.04, 0.5, 'Mean number of peaks', va='center', rotation='vertical')

    # Overall title and layout
    #fig.suptitle("Mean Peaks vs. Number of Bins", fontsize=16)
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.95])
    plt.show()

def fourPlots():  # NOT WORKING RIGHT NOW
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    test_ratio = 0.2
    random_state = None
    num_bins = 90


    sensor = 'CAM'
    # Takes care of which column to use
    if sensor == "CAM" or sensor == "LT":
        column = -2
    else:
        column = -1

    # Prepare an empty list to store (x_n, x_{n+1}) pairs for each tow as well as other lists
    all_pairs, time_pairs, x_pairs, vel = [], [], [], []

    # Loop through tow numbers from 1 to 31
    for tow_number in range(2, 32):
        # Get processed data for the current tow and sensor type
        tow_data_bef = get_synced_data(tow_number, spacesynced=True)

        if sensor == "LT":
            tow_data = tow_data_bef[["time", "x", "y", "z", "error_LT", "z error"]]
        if sensor == "LLS_A":
            tow_data = tow_data_bef[["time", "width_LLS_A", "center_LLS_A", "width error_LLS_A"]]
        if sensor == "LLS_B":
            tow_data = tow_data_bef[["time", "width_LLS_B", "center_LLS_B", "width error_LLS_B"]]
        if sensor == "CAM":
            tow_data = tow_data_bef[["time", "width_CAM", "center_CAM", "error_CAM"]]
        velocity_data = tow_data_bef[["time", "x"]]

        # Ensure that the returned object is a dataframe
        if not tow_data.empty and tow_data.shape[1] > 1:  # Ensure there are at least two columns
            # Extract the last or second-to-last column (based on sensor type)
            second_to_last_column = tow_data.iloc[:, column].values  # Convert to numpy array

            # Create (x_n, x_{n+1}) pairs for the current tow
            x_values = second_to_last_column[:-1]
            y_values = second_to_last_column[1:]

            # Extract x and time and convert to np
            velocity_data_x = velocity_data.iloc[:, -1].values
            velocity_data_time = velocity_data.iloc[:, -2].values
            time_values_i = velocity_data_time[:-1]
            time_values_i2 = velocity_data_time[1:]
            x_values_i = velocity_data_x[:-1]
            x_values_i2 = velocity_data_x[1:]

            # Append pairs as a list of tuples
            all_pairs.extend(zip(x_values, y_values))
            time_pairs.extend(zip(time_values_i, time_values_i2))
            x_pairs.extend(zip(x_values_i, x_values_i2))

        # After processing all tows, convert collected pairs into numpy arrays
    all_pairs = np.array(all_pairs)
    x_values = all_pairs[:, 0]
    y_values = all_pairs[:, 1]

    time_pairs = np.array(time_pairs)
    time_i = time_pairs[:, 0]
    time_i2 = time_pairs[:, 1]
    time_gaps = time_i2 - time_i

    x_pairs = np.array(x_pairs)
    x_i = x_pairs[:, 0]
    x_i2 = x_pairs[:, 1]
    x_gaps = x_i2 - x_i

    vel = x_gaps / time_gaps

    # Train-Test Split

    # Split into training and testing (test_ratio * 100)% of data is used.
    x_train, x_test, y_train, y_test, vel_train, vel_test = train_test_split(
        x_values, y_values, vel, test_size=test_ratio, random_state=random_state
    )
    # NOTE: random_state ensures reproducible splits of the data;
    # change it to another integer for a different split, or set it to None for random behavior.

    # Sort training x-values and reorder y-values accordingly
    sorted_indices = np.argsort(x_train)
    x_sorted = x_train[sorted_indices]
    y_sorted = y_train[sorted_indices]
    vel_sorted = vel_train[sorted_indices]

    # Equal-count bin edges
    bin_edges = np.linspace(0, len(x_sorted), num_bins + 1, dtype=int)

    # Compute bin-wise averages
    x_binned = [np.mean(x_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]
    y_binned = [np.mean(y_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]
    vel_binned = [np.mean(vel_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]

    # scatter Plot with Binned Averages and regression model
    # slope, intercept, r_value, p_value, std_err = linregress(x_binned, y_binned)
    slope, intercept, r_value, p_value, std_err = weighted_linregress(x_binned, y_binned, vel_binned)
    # print(r_value)

    # Define error label
    error_labels = {"LT": "y error", "CAM": "position error", "LLS_A": "width error", "LLS_B": "width error"}
    error_label = error_labels[sensor]

    # Plot scatter + binned fit
    if sensor == "CAM":
        i, j = 0, 0
    elif sensor == "LT":
        i, j = 0, 1
    elif sensor == "LLS_A":
        i, j = 1, 0
    else:
        i, j = 1, 1

    axs[i, j].scatter(x_train, y_train, alpha=0.5, marker='o', edgecolors='k', label="Training Set")
    axs[i, j].scatter(x_binned, y_binned, color='red', marker='s', label="Binned Averages")
    axs[i, j].plot(x_binned, np.array(x_binned) * slope + intercept, color='red', label='Linear Fit')
    axs[i, j].set_xlabel("$ε_{i}$ [mm]")
    axs[i, j].set_ylabel("$ε_{i+1}$ [mm]")
    axs[i, j].set_title(f"{sensor} {error_label} : Consecutive Error Correlation (Training set)")
    axs[i, j].legend()
    axs[i, j].grid(True)


    ####################

    sensor = 'LT'
    # Takes care of which column to use
    if sensor == "CAM" or sensor == "LT":
        column = -2
    else:
        column = -1

    # Prepare an empty list to store (x_n, x_{n+1}) pairs for each tow as well as other lists
    all_pairs, time_pairs, x_pairs, vel = [], [], [], []

    # Loop through tow numbers from 1 to 31
    for tow_number in range(2, 32):
        # Get processed data for the current tow and sensor type
        tow_data_bef = get_synced_data(tow_number, spacesynced=True)

        if sensor == "LT":
            tow_data = tow_data_bef[["time", "x", "y", "z", "error_LT", "z error"]]
        if sensor == "LLS_A":
            tow_data = tow_data_bef[["time", "width_LLS_A", "center_LLS_A", "width error_LLS_A"]]
        if sensor == "LLS_B":
            tow_data = tow_data_bef[["time", "width_LLS_B", "center_LLS_B", "width error_LLS_B"]]
        if sensor == "CAM":
            tow_data = tow_data_bef[["time", "width_CAM", "center_CAM", "error_CAM"]]
        velocity_data = tow_data_bef[["time", "x"]]

        # Ensure that the returned object is a dataframe
        if not tow_data.empty and tow_data.shape[1] > 1:  # Ensure there are at least two columns
            # Extract the last or second-to-last column (based on sensor type)
            second_to_last_column = tow_data.iloc[:, column].values  # Convert to numpy array

            # Create (x_n, x_{n+1}) pairs for the current tow
            x_values = second_to_last_column[:-1]
            y_values = second_to_last_column[1:]

            # Extract x and time and convert to np
            velocity_data_x = velocity_data.iloc[:, -1].values
            velocity_data_time = velocity_data.iloc[:, -2].values
            time_values_i = velocity_data_time[:-1]
            time_values_i2 = velocity_data_time[1:]
            x_values_i = velocity_data_x[:-1]
            x_values_i2 = velocity_data_x[1:]

            # Append pairs as a list of tuples
            all_pairs.extend(zip(x_values, y_values))
            time_pairs.extend(zip(time_values_i, time_values_i2))
            x_pairs.extend(zip(x_values_i, x_values_i2))

        # After processing all tows, convert collected pairs into numpy arrays
    all_pairs = np.array(all_pairs)
    x_values = all_pairs[:, 0]
    y_values = all_pairs[:, 1]

    time_pairs = np.array(time_pairs)
    time_i = time_pairs[:, 0]
    time_i2 = time_pairs[:, 1]
    time_gaps = time_i2 - time_i

    x_pairs = np.array(x_pairs)
    x_i = x_pairs[:, 0]
    x_i2 = x_pairs[:, 1]
    x_gaps = x_i2 - x_i

    vel = x_gaps / time_gaps

    # Train-Test Split

    # Split into training and testing (test_ratio * 100)% of data is used.
    x_train, x_test, y_train, y_test, vel_train, vel_test = train_test_split(
        x_values, y_values, vel, test_size=test_ratio, random_state=random_state
    )
    # NOTE: random_state ensures reproducible splits of the data;
    # change it to another integer for a different split, or set it to None for random behavior.

    # Sort training x-values and reorder y-values accordingly
    sorted_indices = np.argsort(x_train)
    x_sorted = x_train[sorted_indices]
    y_sorted = y_train[sorted_indices]
    vel_sorted = vel_train[sorted_indices]

    # Equal-count bin edges
    bin_edges = np.linspace(0, len(x_sorted), num_bins + 1, dtype=int)

    # Compute bin-wise averages
    x_binned = [np.mean(x_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]
    y_binned = [np.mean(y_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]
    vel_binned = [np.mean(vel_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]

    # scatter Plot with Binned Averages and regression model
    # slope, intercept, r_value, p_value, std_err = linregress(x_binned, y_binned)
    slope, intercept, r_value, p_value, std_err = weighted_linregress(x_binned, y_binned, vel_binned)
    # print(r_value)

    # Define error label
    error_labels = {"LT": "y error", "CAM": "position error", "LLS_A": "width error", "LLS_B": "width error"}
    error_label = error_labels[sensor]

    # Plot scatter + binned fit
    if sensor == "CAM":
        i, j = 0, 0
    elif sensor == "LT":
        i, j = 0, 1
    elif sensor == "LLS_A":
        i, j = 1, 0
    else:
        i, j = 1, 1

    axs[i, j].scatter(x_train, y_train, alpha=0.5, marker='o', edgecolors='k', label="Training Set")
    axs[i, j].scatter(x_binned, y_binned, color='red', marker='s', label="Binned Averages")
    axs[i, j].plot(x_binned, np.array(x_binned) * slope + intercept, color='red', label='Linear Fit')
    axs[i, j].set_xlabel("$ε_{i}$ [mm]")
    axs[i, j].set_ylabel("$ε_{i+1}$ [mm]")
    axs[i, j].set_title(f"{sensor} {error_label} : Consecutive Error Correlation (Training set)")
    axs[i, j].legend()
    axs[i, j].grid(True)


    #######################

    sensor = 'LSS_A'
    # Takes care of which column to use
    if sensor == "CAM" or sensor == "LT":
        column = -2
    else:
        column = -1

    # Prepare an empty list to store (x_n, x_{n+1}) pairs for each tow as well as other lists
    all_pairs, time_pairs, x_pairs, vel = [], [], [], []

    # Loop through tow numbers from 1 to 31
    for tow_number in range(2, 32):
        # Get processed data for the current tow and sensor type
        tow_data_bef = get_synced_data(tow_number, spacesynced=True)

        if sensor == "LT":
            tow_data = tow_data_bef[["time", "x", "y", "z", "error_LT", "z error"]]
        if sensor == "LLS_A":
            tow_data = tow_data_bef[["time", "width_LLS_A", "center_LLS_A", "width error_LLS_A"]]
        if sensor == "LLS_B":
            tow_data = tow_data_bef[["time", "width_LLS_B", "center_LLS_B", "width error_LLS_B"]]
        if sensor == "CAM":
            tow_data = tow_data_bef[["time", "width_CAM", "center_CAM", "error_CAM"]]
        velocity_data = tow_data_bef[["time", "x"]]

        # Ensure that the returned object is a dataframe
        if not tow_data.empty and tow_data.shape[1] > 1:  # Ensure there are at least two columns
            # Extract the last or second-to-last column (based on sensor type)
            second_to_last_column = tow_data.iloc[:, column].values  # Convert to numpy array

            # Create (x_n, x_{n+1}) pairs for the current tow
            x_values = second_to_last_column[:-1]
            y_values = second_to_last_column[1:]

            # Extract x and time and convert to np
            velocity_data_x = velocity_data.iloc[:, -1].values
            velocity_data_time = velocity_data.iloc[:, -2].values
            time_values_i = velocity_data_time[:-1]
            time_values_i2 = velocity_data_time[1:]
            x_values_i = velocity_data_x[:-1]
            x_values_i2 = velocity_data_x[1:]

            # Append pairs as a list of tuples
            all_pairs.extend(zip(x_values, y_values))
            time_pairs.extend(zip(time_values_i, time_values_i2))
            x_pairs.extend(zip(x_values_i, x_values_i2))

        # After processing all tows, convert collected pairs into numpy arrays
    all_pairs = np.array(all_pairs)
    x_values = all_pairs[:, 0]
    y_values = all_pairs[:, 1]

    time_pairs = np.array(time_pairs)
    time_i = time_pairs[:, 0]
    time_i2 = time_pairs[:, 1]
    time_gaps = time_i2 - time_i

    x_pairs = np.array(x_pairs)
    x_i = x_pairs[:, 0]
    x_i2 = x_pairs[:, 1]
    x_gaps = x_i2 - x_i

    vel = x_gaps / time_gaps

    # Train-Test Split

    # Split into training and testing (test_ratio * 100)% of data is used.
    x_train, x_test, y_train, y_test, vel_train, vel_test = train_test_split(
        x_values, y_values, vel, test_size=test_ratio, random_state=random_state
    )
    # NOTE: random_state ensures reproducible splits of the data;
    # change it to another integer for a different split, or set it to None for random behavior.

    # Sort training x-values and reorder y-values accordingly
    sorted_indices = np.argsort(x_train)
    x_sorted = x_train[sorted_indices]
    y_sorted = y_train[sorted_indices]
    vel_sorted = vel_train[sorted_indices]

    # Equal-count bin edges
    bin_edges = np.linspace(0, len(x_sorted), num_bins + 1, dtype=int)

    # Compute bin-wise averages
    x_binned = [np.mean(x_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]
    y_binned = [np.mean(y_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]
    vel_binned = [np.mean(vel_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]

    # scatter Plot with Binned Averages and regression model
    # slope, intercept, r_value, p_value, std_err = linregress(x_binned, y_binned)
    slope, intercept, r_value, p_value, std_err = weighted_linregress(x_binned, y_binned, vel_binned)
    # print(r_value)

    # Define error label
    error_labels = {"LT": "y error", "CAM": "position error", "LLS_A": "width error", "LLS_B": "width error"}
    error_label = error_labels[sensor]

    # Plot scatter + binned fit
    if sensor == "CAM":
        i, j = 0, 0
    elif sensor == "LT":
        i, j = 0, 1
    elif sensor == "LLS_A":
        i, j = 1, 0
    else:
        i, j = 1, 1

    axs[i, j].scatter(x_train, y_train, alpha=0.5, marker='o', edgecolors='k', label="Training Set")
    axs[i, j].scatter(x_binned, y_binned, color='red', marker='s', label="Binned Averages")
    axs[i, j].plot(x_binned, np.array(x_binned) * slope + intercept, color='red', label='Linear Fit')
    axs[i, j].set_xlabel("$ε_{i}$ [mm]")
    axs[i, j].set_ylabel("$ε_{i+1}$ [mm]")
    axs[i, j].set_title(f"{sensor} {error_label} : Consecutive Error Correlation (Training set)")
    axs[i, j].legend()
    axs[i, j].grid(True)


    ###############################

    sensor = 'LLS_B'
    # Takes care of which column to use
    if sensor == "CAM" or sensor == "LT":
        column = -2
    else:
        column = -1

    # Prepare an empty list to store (x_n, x_{n+1}) pairs for each tow as well as other lists
    all_pairs, time_pairs, x_pairs, vel = [], [], [], []

    # Loop through tow numbers from 1 to 31
    for tow_number in range(2, 32):
        # Get processed data for the current tow and sensor type
        tow_data_bef = get_synced_data(tow_number, spacesynced=True)

        if sensor == "LT":
            tow_data = tow_data_bef[["time", "x", "y", "z", "error_LT", "z error"]]
        if sensor == "LLS_A":
            tow_data = tow_data_bef[["time", "width_LLS_A", "center_LLS_A", "width error_LLS_A"]]
        if sensor == "LLS_B":
            tow_data = tow_data_bef[["time", "width_LLS_B", "center_LLS_B", "width error_LLS_B"]]
        if sensor == "CAM":
            tow_data = tow_data_bef[["time", "width_CAM", "center_CAM", "error_CAM"]]
        velocity_data = tow_data_bef[["time", "x"]]

        # Ensure that the returned object is a dataframe
        if not tow_data.empty and tow_data.shape[1] > 1:  # Ensure there are at least two columns
            # Extract the last or second-to-last column (based on sensor type)
            second_to_last_column = tow_data.iloc[:, column].values  # Convert to numpy array

            # Create (x_n, x_{n+1}) pairs for the current tow
            x_values = second_to_last_column[:-1]
            y_values = second_to_last_column[1:]

            # Extract x and time and convert to np
            velocity_data_x = velocity_data.iloc[:, -1].values
            velocity_data_time = velocity_data.iloc[:, -2].values
            time_values_i = velocity_data_time[:-1]
            time_values_i2 = velocity_data_time[1:]
            x_values_i = velocity_data_x[:-1]
            x_values_i2 = velocity_data_x[1:]

            # Append pairs as a list of tuples
            all_pairs.extend(zip(x_values, y_values))
            time_pairs.extend(zip(time_values_i, time_values_i2))
            x_pairs.extend(zip(x_values_i, x_values_i2))

        # After processing all tows, convert collected pairs into numpy arrays
    all_pairs = np.array(all_pairs)
    x_values = all_pairs[:, 0]
    y_values = all_pairs[:, 1]

    time_pairs = np.array(time_pairs)
    time_i = time_pairs[:, 0]
    time_i2 = time_pairs[:, 1]
    time_gaps = time_i2 - time_i

    x_pairs = np.array(x_pairs)
    x_i = x_pairs[:, 0]
    x_i2 = x_pairs[:, 1]
    x_gaps = x_i2 - x_i

    vel = x_gaps / time_gaps

    # Train-Test Split

    # Split into training and testing (test_ratio * 100)% of data is used.
    x_train, x_test, y_train, y_test, vel_train, vel_test = train_test_split(
        x_values, y_values, vel, test_size=test_ratio, random_state=random_state
    )
    # NOTE: random_state ensures reproducible splits of the data;
    # change it to another integer for a different split, or set it to None for random behavior.

    # Sort training x-values and reorder y-values accordingly
    sorted_indices = np.argsort(x_train)
    x_sorted = x_train[sorted_indices]
    y_sorted = y_train[sorted_indices]
    vel_sorted = vel_train[sorted_indices]

    # Equal-count bin edges
    bin_edges = np.linspace(0, len(x_sorted), num_bins + 1, dtype=int)

    # Compute bin-wise averages
    x_binned = [np.mean(x_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]
    y_binned = [np.mean(y_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]
    vel_binned = [np.mean(vel_sorted[bin_edges[i]:bin_edges[i + 1]]) for i in range(num_bins)]

    # scatter Plot with Binned Averages and regression model
    # slope, intercept, r_value, p_value, std_err = linregress(x_binned, y_binned)
    slope, intercept, r_value, p_value, std_err = weighted_linregress(x_binned, y_binned, vel_binned)
    # print(r_value)

    # Define error label
    error_labels = {"LT": "y error", "CAM": "position error", "LLS_A": "width error", "LLS_B": "width error"}
    error_label = error_labels[sensor]

    # Plot scatter + binned fit
    if sensor == "CAM":
        i, j = 0, 0
    elif sensor == "LT":
        i, j = 0, 1
    elif sensor == "LLS_A":
        i, j = 1, 0
    else:
        i, j = 1, 1

    axs[i, j].scatter(x_train, y_train, alpha=0.5, marker='o', edgecolors='k', label="Training Set")
    axs[i, j].scatter(x_binned, y_binned, color='red', marker='s', label="Binned Averages")
    axs[i, j].plot(x_binned, np.array(x_binned) * slope + intercept, color='red', label='Linear Fit')
    axs[i, j].set_xlabel("$ε_{i}$ [mm]")
    axs[i, j].set_ylabel("$ε_{i+1}$ [mm]")
    axs[i, j].set_title(f"{sensor} {error_label} : Consecutive Error Correlation (Training set)")
    axs[i, j].legend()
    axs[i, j].grid(True)


    plt.tight_layout()
    plt.show()

def get_delta_x(tow_number):
    tow_data_bef = get_synced_data(tow_number, spacesynced=True)
    delta_t = tow_data_bef["x"].iloc[-1] - tow_data_bef["x"].iloc[0]
    return delta_t

def plot_blobs():
    # some parameters for the plots
    test_ratio = 0.0001
    num_bins = 15
    bins_show = False
    errorCor_show = True
    x_label = 'Current error'
    y_label = 'Subsequent error'
    csfont = {'fontname': 'Times New Roman'}

    # gathering the data for the plots from the consecutive error function
    X_CAM, Y_CAM, X_CAM_binned, Y_CAM_binned, slope_CAM, intercept_CAM = consecutive_error(
        "CAM", test_ratio=test_ratio, num_bins=num_bins, bins_show=bins_show, plot_fit=errorCor_show, return_plot_data=True)
    X_LT, Y_LT, X_LT_binned, Y_LT_binned, slope_LT, intercept_LT = consecutive_error(
        "LT", test_ratio=test_ratio, num_bins=num_bins, bins_show=bins_show, plot_fit=errorCor_show, return_plot_data=True)
    X_LLSB, Y_LLSB, X_LLSB_binned, Y_LLSB_binned, slope_LLSB, intercept_LLSB = consecutive_error(
        "LLS_B", test_ratio=test_ratio, num_bins=num_bins, bins_show=bins_show, plot_fit=errorCor_show, return_plot_data=True)
    X_LLSA, Y_LLSA, X_LLSA_binned, Y_LLSA_binned, slope_LLSA, intercept_LLSA = consecutive_error(
        "LLS_A", test_ratio=test_ratio, num_bins=num_bins, bins_show=bins_show, plot_fit=errorCor_show, return_plot_data=True)

    # --------PLotting----------
    plt.rc('font', family='Times New Roman')
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    #CAM plot
    axs[0, 0].scatter(X_CAM, Y_CAM, alpha=0.2, marker='o', s=8, edgecolors='k', label="Training Set")
    axs[0, 0].scatter(X_CAM_binned, Y_CAM_binned, alpha=1, color='red', marker='s', s=13, label="Binned Averages")
    axs[0, 0].plot(X_CAM_binned, np.array(X_CAM_binned) * slope_CAM + intercept_CAM, color='red', label='Linear Fit')

    #axs[0, 0].set_xlabel(x_label, fontsize=constants.font_medium, **csfont)  #"$ε_{i}$ (mm)"
    axs[0, 0].set_ylabel(y_label, fontsize=constants.font_medium, **csfont)    #"$ε_{i+1}$ (mm)"
    axs[0, 0].set_title('Tape lateral movement', fontsize=constants.font_medium, **csfont)
    axs[0, 0].set_xlim(-0.6, 0.9)
    axs[0, 0].set_ylim(-0.6, 0.9)
    axs[0, 0].set_xticks(np.linspace(-0.6, 0.9, 6))
    axs[0, 0].set_yticks(np.linspace(-0.6, 0.9, 6))
    axs[0, 0].grid(True)

    #LT plot
    axs[0, 1].scatter(X_LT, Y_LT, alpha=0.2, marker='o', s=8, edgecolors='k')
    axs[0, 1].scatter(X_LT_binned, Y_LT_binned, alpha=1, color='red', marker='s', s=13)
    axs[0, 1].plot(X_LT_binned, np.array(X_LT_binned) * slope_LT + intercept_LT, color='red', linewidth=2)

    #axs[0, 1].set_xlabel(x_label, fontsize=constants.font_medium, **csfont)
    #axs[0, 1].set_ylabel(y_label, fontsize=constants.font_medium, **csfont)
    axs[0, 1].set_title('Robot position', fontsize=constants.font_medium, **csfont)
    axs[0, 1].set_xlim(-1.2, -0.6)
    axs[0, 1].set_ylim(-1.2, -0.6)
    axs[0, 1].set_xticks(np.linspace(-1.2, -0.6, 3))
    axs[0, 1].set_yticks(np.linspace(-1.2, -0.6, 3))
    axs[0, 1].grid(True)

    #LLSB plot
    axs[1,0].scatter(X_LLSB, Y_LLSB, alpha=0.2, marker='o', s=8, edgecolors='k')
    axs[1,0].scatter(X_LLSB_binned, Y_LLSB_binned, alpha=1, color='red', marker='s', s=13)
    axs[1,0].plot(X_LLSB_binned, np.array(X_LLSB_binned) * slope_LLSB + intercept_LLSB, color='red', linewidth=2)

    axs[1,0].set_xlabel(x_label, fontsize=constants.font_medium, **csfont)
    axs[1,0].set_ylabel(y_label, fontsize=constants.font_medium, **csfont)
    axs[1, 0].set_title('Tape width after compaction', fontsize=constants.font_medium, **csfont)
    axs[1,0].set_xlim(-0.6, 0.3)
    axs[1,0].set_ylim(-0.6, 0.3)
    axs[1,0].set_xticks(np.linspace(-0.6, 0.3, 4))
    axs[1,0].set_yticks(np.linspace(-0.6, 0.3, 4))
    axs[1,0].grid(True)

    #LLSA plot
    axs[1, 1].scatter(X_LLSA, Y_LLSA, alpha=0.2, marker='o', s=8, edgecolors='k')
    axs[1, 1].scatter(X_LLSA_binned, Y_LLSA_binned, alpha=1, color='red', marker='s', s=13)
    axs[1, 1].plot(X_LLSA_binned, np.array(X_LLSA_binned) * slope_LLSA + intercept_LLSA, color='red', linewidth=2)

    axs[1, 1].set_xlabel(x_label, fontsize=constants.font_medium, **csfont)
    axs[1, 1].set_title('Tape width before compaction', fontsize=constants.font_medium, **csfont)
    axs[1, 1].set_xlim(-0.6, 0.3)
    axs[1, 1].set_ylim(-0.6, 0.3)
    axs[1, 1].set_xticks(np.linspace(-0.6, 0.3, 4))
    axs[1, 1].set_yticks(np.linspace(-0.6, 0.3, 4))
    axs[1, 1].grid(True)


    #fig.subplots_adjust(bottom=0.2)
    lgd = fig.legend(fontsize=constants.font_small, loc='lower center',
          fancybox=True, shadow=True, ncol=5)
    #leg.set_in_layout(True)
    #plt.grid(True)
    #plt.savefig('samplefigure', bbox_extra_artists=(lgd), bbox_inches='tight')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()

##############################################################################################################
""""Run this file"""

def main():
    #start_time = time.perf_counter()
    #generate_simulated_VS_real(n_real_tow=6, rdm_seed=75, test_ratio=0.2, errorCor_show=False, bins_show=False,
    #                           num_bins=100, peak_plots = False, sim_plot = True)
    #end_time = time.perf_counter()
    #elapsed_time = end_time - start_time
    #print(f"Elapsed time: {round(elapsed_time,2)} seconds")
    plot_blobs()

if __name__ == "__main__":
    main()

