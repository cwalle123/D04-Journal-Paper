# External imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
import seaborn as sns

# Internal imports
from constants import tow_width_specified, font_extra_small, font_small, font_medium, font_large, font_extra_large
from Handling_ALL_Functions import get_synced_data
from Data_ALL_statistics import main as real_hist, plot_histograms_separated, best_fit_distribution
from Model_ALL_RandomWalk import fit_random_walk, generate_random_walk, generate_RW_multitow
from Model_ALL_Simulation import fit_starting_error_distribution
from scipy.stats import norm, logistic, gamma, beta, expon, lognorm, skewnorm, gumbel_r, gumbel_l, genextreme



def generate_multilaminate_layout(n_laminates: int, tows_per_laminate: int=31, starting_mods: list=[None, 1, 1]):
    # generating and combining data for all laminates
    gap_overlap_dict = {}
    all_RW_data = []
    for laminate in range(n_laminates):
        temp_gap_overlap_df, gap_df, overlap_df, total_gap_percent, total_overlap_percent, RW_data = generate_RW_multitow(
            num_tows=tows_per_laminate, starting_mods=starting_mods)
        temp_gap_overlap_df = np.array(temp_gap_overlap_df)

        temp_gap_overlap_dict = {
            f"Gap/overlap_Laminate{laminate}_Tow{tow + 1}-{tow + 2}": temp_gap_overlap_df[:, tow]
            for tow in range(tows_per_laminate - 1)}
        gap_overlap_dict.update(temp_gap_overlap_dict)
        all_RW_data.append(RW_data)

    gap_overlap_df = pd.DataFrame(gap_overlap_dict)
    return gap_overlap_df, all_RW_data          # TODO: find a way to calc (defect) areas for multiple laminates... );

def calc_lengthwise_defect_percent(tows: int, num_divisions: int=31, starting_mods: list=[None, 1, 1]):
    gap_overlap_df, gap_df, overlap_df, total_gap_percent, total_overlap_percent, RW_data = generate_RW_multitow(num_tows=tows, starting_mods=starting_mods)

    # some info for area calculations:
    highest_tow_edge = np.array(RW_data[-1]["top_edge"])
    lowest_tow_edge = np.array(RW_data[0]["bottom_edge"])

    defect_percent_list, gap_percent_list, overlap_percent_list, x_list = [], [], [], []
    x_tow = np.linspace(0, 1000, len(gap_overlap_df)+1)
    divisions = np.linspace(0, len(gap_overlap_df), num_divisions+1)
    for i in range(num_divisions):
        start, end = divisions[i], divisions[i+1]
        data = gap_overlap_df[int(start):int(end)]
        x_vals = x_tow[int(start):int(end)]

        total_layout_area = np.trapezoid(highest_tow_edge[int(start):int(end)] - lowest_tow_edge[int(start):int(end)], x_vals)
        total_gap_area = sum(np.trapezoid(np.clip(values, 0, None), x_vals) for values in data.values.T)
        total_overlap_area = sum(np.trapezoid(np.clip(-values, 0, None), x_vals) for values in data.values.T)
        print(total_layout_area, total_gap_area, total_overlap_area, len(x_vals), len(lowest_tow_edge[int(start):int(end)]))

        gap_percent = (total_gap_area / total_layout_area) * 100 if total_layout_area > 0 else 0
        overlap_percent = (total_overlap_area / total_layout_area) * 100 if total_layout_area > 0 else 0
        defect_percent = gap_percent + overlap_percent  # TODO: is this correct????

        defect_percent_list.append(defect_percent)
        gap_percent_list.append(gap_percent)
        overlap_percent_list.append(overlap_percent)
        x_list.append(np.average(x_vals))


    defect_data = pd.DataFrame(
        {"x": x_list,
         "defect_percent": defect_percent_list,
         "gap_percent": gap_percent_list,
         "overlap_percent": overlap_percent_list})
    print(defect_data)
    return defect_data

def plot_lengthwise_defect_percent(defect_data_original: pd.DataFrame, defect_data_modified: pd.DataFrame):
    # original data
    #plt.scatter(defect_data_original["x"], defect_data_original["defect_percent"], color="blue", marker='o')
    #plt.plot(defect_data_original["x"], defect_data_original["defect_percent"], color="blue", linestyle='solid', label="Defect Percentage, normal")
    plt.scatter(defect_data_original["x"], defect_data_original["gap_percent"], color="green", marker='o')
    plt.plot(defect_data_original["x"], defect_data_original["gap_percent"], color="green", linestyle='solid', label="Gap Percentage, normal")
    plt.scatter(defect_data_original["x"], defect_data_original["overlap_percent"], color="red", marker='o')
    plt.plot(defect_data_original["x"], defect_data_original["overlap_percent"], color="red", linestyle='solid', label="Overlap Percentage, normal")
    # modified data
    #plt.scatter(defect_data_modified["x"], defect_data_modified["defect_percent"], color="blue", marker='^')
    #plt.plot(defect_data_modified["x"], defect_data_modified["defect_percent"], color="blue", linestyle='dashed', label="Defect Percentage, modified")
    plt.scatter(defect_data_modified["x"], defect_data_modified["gap_percent"], color="green", marker='^')
    plt.plot(defect_data_modified["x"], defect_data_modified["gap_percent"], color="green", linestyle='dashed', label="Gap Percentage, modified")
    plt.scatter(defect_data_modified["x"], defect_data_modified["overlap_percent"], color="red", marker='^')
    plt.plot(defect_data_modified["x"], defect_data_modified["overlap_percent"], color="red", linestyle='dashed', label="Overlap Percentage, modified")

    plt.xlabel("x (mm)")
    plt.ylabel("Defect Percentage (%)")
    plt.legend()
    plt.show()

def calc_lengthwise_defect_percent_exp(tows: int, num_divisions: int=31, starting_mods: list=[None, 1, 1]):
    # TODO: Find gap, overlap, etc data for experimental, not for RW
    gap_overlap_df, gap_df, overlap_df, total_gap_percent, total_overlap_percent, RW_data = generate_RW_multitow(num_tows=tows, starting_mods=starting_mods)
    #1) get the experimental data tows
    #2) get top edge of tow 31 and bottom edge of tow 2 to get total width of layup
    #3) find the gap distance at every x-position
    #4) find the total width of layup at every x-position
    #5) gap percentage at each x-location will be #3/#4 * 100 
    '''percentage of defect at a certain x-position is total distance of defect at that location
    divided by the total length of the layup at that x-location'''


def analyze_starting_variation_effects(starting_mods: list=[None, 1, 1], proposal_type="RWM"):
    LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params = fit_random_walk("LT")
    CAM_steps, CAM_proposal_std, CAM_target_dist, CAM_dist, CAM_params = fit_random_walk("CAM")
    LLS_B_steps, LLS_B_proposal_std, LLS_B_target_dist, LLS_B_dist, LLS_B_params = fit_random_walk("LLS_B")
    print(f"distributions type per error:\n LT:{LT_dist}\n CAM:{CAM_dist}\nLLS_B:{LLS_B_dist}")

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
        LT_params[
            -1] *= scale_factor  # TODO: make sure the scale parameters equally affect the different distribution types
        CAM_params[-1] *= scale_factor
        LLS_B_params[-1] *= scale_factor
        LT_params, CAM_params, LLS_B_params = tuple(LT_params), tuple(CAM_params), tuple(LLS_B_params)

    # generating random walk data
    LT_walk_data = generate_random_walk("LT", LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params,
                                        proposal_type=proposal_type, plot_path=True, plot_histogram=True)
    CAM_walk_data = generate_random_walk("CAM", CAM_steps, CAM_proposal_std, CAM_target_dist, CAM_dist, CAM_params,
                                         proposal_type=proposal_type, plot_path=True, plot_histogram=True)
    LLSB_walk_data = generate_random_walk("LLS_B", LLS_B_steps, LLS_B_proposal_std, LLS_B_target_dist, LLS_B_dist,
                                          LLS_B_params, proposal_type=proposal_type, plot_path=True, plot_histogram=True)


def plot_target_vs_start(target_mods: list=[None, 1, 1], starting_mods: list=[1, 1]):
    x_pdf = np.linspace(-1.2, 1.2, 100)     # this is for the plotting later on.

    ### Code for the target distributions ###
    CAM_steps, CAM_proposal_std, CAM_target_dist, CAM_dist, CAM_params = fit_random_walk("CAM")
    y_target = CAM_target_dist(x_pdf)

    # This section modifies the starting distributions used by the model, which is needed for Model_ALL_StartVariations.
    if target_mods != [None, 1, 1]:
        if target_mods[0] != None:  # this changes the starting distribution type if necessary
            CAM_dist_modded = target_mods[0]
        else: CAM_dist_modded = CAM_dist
        # these are the factors by which the mean and std are changed
        loc_factor, scale_factor = target_mods[1], target_mods[2]
        # code used for start value: start_value = dist.rvs(*params[:-2], loc=params[-2], scale=params[-1])
        CAM_params_modded = list(CAM_params)
        CAM_params_modded[-2] *= loc_factor
        CAM_params_modded[-1] *= scale_factor  # TODO: make sure the scale parameters equally affect the different distribution types
        CAM_params_modded = tuple(CAM_params_modded)
        CAM_target_dist_modded = lambda x: CAM_dist_modded.pdf(x, *CAM_params_modded[:-2], loc=CAM_params_modded[-2], scale=CAM_params_modded[-1])
        y_target_modded = CAM_target_dist_modded(x_pdf)
        plt.plot(x_pdf, y_target_modded, label='modified target distribution')

    ### Code for the starting value distributions ###
    mean, std, _, _, first_values = fit_starting_error_distribution("CAM", plot=False)
    CAM_starting_dist = lambda x: norm.pdf(x, loc=mean, scale=std)
    y_starting = CAM_starting_dist(x_pdf)

    # This section modifies the starting distributions used by the model, which is needed for Model_ALL_StartVariations.
    if starting_mods != [1, 1]:
        # these are the factors by which the mean and std are changed
        loc_factor, scale_factor = starting_mods[0], starting_mods[1]
        mean_modded = mean * loc_factor
        std_modded = std * scale_factor
        CAM_starting_dist_modded = lambda x: norm.pdf(x, loc=mean_modded, scale=std_modded)
        y_starting_modded = CAM_starting_dist_modded(x_pdf)
        plt.plot(x_pdf, y_starting_modded, label='modified target distribution')


    plt.hist(first_values, bins=10, density=True, alpha=0.3, label='experimental starting values')
    plt.plot(x_pdf, y_target, label='target distribution')
    plt.plot(x_pdf, y_starting, label='starting values distribution')
    plt.title("starting and target distributions")
    plt.xlim(-1.2, 1.2)
    plt.legend()
    plt.show()


def main():
    #analyze_starting_variation_effects(starting_mods = [None, 1, 2], proposal_type = "RWM")
    #plot_target_vs_start()      # target_mods=[None, 1, 1.5], starting_mods=[1, 1.5]

    defect_data_original = calc_lengthwise_defect_percent(3, tows_per_laminate=5, num_divisions=31, starting_mods=[None, 1, 1])
    #defect_data_modified = calc_lengthwise_defect_percent(3, tows_per_laminate=5, num_divisions=31, starting_mods=[None, 1, 1.5])
    #plot_lengthwise_defect_percent(defect_data_original, defect_data_modified)

if __name__ == "__main__":
    main()