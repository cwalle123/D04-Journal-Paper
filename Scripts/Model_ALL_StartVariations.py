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
from Data_ALL_traverse import traverse_tow_gaps_and_overlaps, traverse_tow_gaps_and_overlaps_lengths, traverse_tow_constructor


def generate_multilaminate_layout(n_laminates: int, tows_per_laminate: int=29, starting_mods: list=[None, 1, 1], alternate_start: list=[None, "params"]):
    # generating and combining data for all laminates
    gap_overlap_dict = {}
    top_edges, bottom_edges = [], []
    for laminate in range(n_laminates):
        temp_gap_overlap_df, gap_df, overlap_df, total_gap_percent, total_overlap_percent, RW_data = generate_RW_multitow(
            num_tows=tows_per_laminate, starting_mods=starting_mods, alternate_start=alternate_start)
        temp_gap_overlap_df = np.array(temp_gap_overlap_df)

        temp_gap_overlap_dict = {
            f"Gap/overlap_Laminate{laminate}_Tow{tow + 1}-{tow + 2}": temp_gap_overlap_df[:, tow]
            for tow in range(tows_per_laminate - 1)}
        gap_overlap_dict.update(temp_gap_overlap_dict)

        highest_tow_edge = np.array(RW_data[-1]["top_edge"])
        lowest_tow_edge = np.array(RW_data[0]["bottom_edge"])
        top_edges.append(highest_tow_edge)
        bottom_edges.append(lowest_tow_edge)

    top_edges = np.array(top_edges)
    bottom_edges = np.array(bottom_edges)
    gap_overlap_df = pd.DataFrame(gap_overlap_dict)
    return gap_overlap_df, top_edges, bottom_edges

def calc_lengthwise_defect_percent(n_laminates, tows_per_laminate: int=29, num_divisions: int=100, starting_mods: list=[None, 1, 1], alternate_start: list=[None, "params"]):
    # gap_overlap_df, gap_df, overlap_df, total_gap_percent, total_overlap_percent, RW_data = generate_RW_multitow(num_tows=tows, starting_mods=starting_mods)
    gap_overlap_df, top_edges, bottom_edges = generate_multilaminate_layout(n_laminates, tows_per_laminate=tows_per_laminate,
                                                                            starting_mods=starting_mods, alternate_start=alternate_start)

    gap_percent_list, overlap_percent_list, x_list = [], [], []
    x_tow = np.linspace(0, 1000, len(gap_overlap_df)+1)
    divisions = np.linspace(0, len(gap_overlap_df), num_divisions+1)
    for i in range(num_divisions):
        start, end = divisions[i], divisions[i+1]
        data = gap_overlap_df[int(start):int(end)]
        x_vals = x_tow[int(start):int(end)]

        # old code: total_layout_area = np.trapezoid(highest_tow_edge[int(start):int(end)] - lowest_tow_edge[int(start):int(end)], x_vals)
        total_layout_area = 0
        for lam in range(n_laminates):
            lam_layout_area = np.trapezoid(top_edges[lam, int(start):int(end)] - bottom_edges[lam, int(start):int(end)], x_vals)
            total_layout_area += lam_layout_area

        total_gap_area = sum(np.trapezoid(np.clip(values, 0, None), x_vals) for values in data.values.T)
        total_overlap_area = sum(np.trapezoid(np.clip(-values, 0, None), x_vals) for values in data.values.T)

        gap_percent = (total_gap_area / total_layout_area) * 100 if total_layout_area > 0 else 0
        overlap_percent = (total_overlap_area / total_layout_area) * 100 if total_layout_area > 0 else 0

        gap_percent_list.append(gap_percent)
        overlap_percent_list.append(overlap_percent)
        x_list.append(np.average(x_vals))


    defect_data = pd.DataFrame(
        {"x": x_list,
         "gap_percent": gap_percent_list,
         "overlap_percent": overlap_percent_list})
    print(defect_data)
    return defect_data

def plot_lengthwise_defect_percent(defect_data_original: pd.DataFrame, defect_data_modified: pd.DataFrame):
    # original data
    #plt.scatraverse_tower(defect_data_original["x"], defect_data_original["defect_percent"], color="blue", marker='o')
    #plt.plot(defect_data_original["x"], defect_data_original["defect_percent"], color="blue", linestyle='solid', label="Defect Percentage, normal")
    plt.scatraverse_tower(defect_data_original["x"], defect_data_original["gap_percent"], color="green", marker='o')
    plt.plot(defect_data_original["x"], defect_data_original["gap_percent"], color="green", linestyle='solid', label="Gap Percentage, normal")
    plt.scatraverse_tower(defect_data_original["x"], defect_data_original["overlap_percent"], color="red", marker='o')
    plt.plot(defect_data_original["x"], defect_data_original["overlap_percent"], color="red", linestyle='solid', label="Overlap Percentage, normal")
    # modified data
    #plt.scatraverse_tower(defect_data_modified["x"], defect_data_modified["defect_percent"], color="blue", marker='^')
    #plt.plot(defect_data_modified["x"], defect_data_modified["defect_percent"], color="blue", linestyle='dashed', label="Defect Percentage, modified")
    plt.scatraverse_tower(defect_data_modified["x"], defect_data_modified["gap_percent"], color="green", marker='^')
    plt.plot(defect_data_modified["x"], defect_data_modified["gap_percent"], color="green", linestyle='dashed', label="Gap Percentage, modified")
    plt.scatraverse_tower(defect_data_modified["x"], defect_data_modified["overlap_percent"], color="red", marker='^')
    plt.plot(defect_data_modified["x"], defect_data_modified["overlap_percent"], color="red", linestyle='dashed', label="Overlap Percentage, modified")

    plt.xlabel("x (mm)")
    plt.ylabel("Defect Percentage (%)")
    plt.legend()
    plt.show()

def calc_lengthwise_defect_percent_exp(nbins=100, length_mm=1000.0):
    #! Algorithm:
    #1) get the experimental data tows
    #2) get top edge of tow 31 and botraverse_towom edge of tow 2 to get total width of layup
    #3) Separate the dataset into bins of x-positions 10mm each
    #3) find the gap area at each bin
    #4) find the total area of layup at every bin
    #5) gap percentage at each bin will be #3/#4 * 100


    # 1) get the experimental data tows(2,30)
    gap_overlap_df, gap_df, overlap_df, _, _ = traverse_tow_gaps_and_overlaps(plot=False)
    x_go = gap_overlap_df.index.to_numpy()  # x for gap/overlap arrays (already truncated to common length)

    # 2) Rebuild top & bottom edges vs x to get layup height(x) = top - bottom
    top_edge_paths, bottom_edge_paths = [], []
    x_vals = None
    for tow in range(2, 31):
        traverse_tow = traverse_tow_constructor(tow, normalize=True)
        if traverse_tow is None:
            continue
        offset_mm = (tow - 2) * tow_width_specified
        x = traverse_tow["x_centerline"].to_numpy()
        top = traverse_tow["y_left"].to_numpy() + offset_mm
        bot = traverse_tow["y_right"].to_numpy() + offset_mm

        if x_vals is None:
            x_vals = x
        # keep everything truncated to a common min length (same behavior as your function)
        min_len = min(len(x_vals), len(x))
        x_vals = x_vals[:min_len]
        top_edge_paths.append(top[:min_len])
        bottom_edge_paths.append(bot[:min_len])

    # Highest top edge (Tow 30) and lowest bottom edge (Tow 2) across x
    highest_tow_edge = top_edge_paths[-1]
    lowest_tow_edge  = bottom_edge_paths[0]
    height = (highest_tow_edge - lowest_tow_edge).astype(float)   # layup height vs x (mm)
    x_h = x_vals

    # 3) Build bins, 10 mm each
    bins = np.linspace(0.0, float(length_mm), nbins + 1)

    def integrate_series_per_bin(x, y, bins):
        """
        Integrate y each bin using trapezoids.
        Returns an array of areas per bin (same units as y * mm).
        """
        x = x.astype(float)
        y = y.astype(float)
        out = np.zeros(len(bins) - 1, dtype=float)
        x_min, x_max = x.min(), x.max()

        for i in range(len(bins) - 1):
            x0, x1 = bins[i], bins[i+1]
            if x1 <= x_min or x0 >= x_max:
                # bin completely outside sampled x-range
                out[i] = 0.0
                continue

            # Clip the bin to data range
            xa, xb = max(x0, x_min), min(x1, x_max)

            # Points inside [xa, xb]
            mask = (x >= xa) & (x <= xb)
            if not mask.any():
                # No interior points but still inside global range -> integrate straight line
                y0 = np.interp(xa, x, y)
                y1 = np.interp(xb, x, y)
                out[i] = 0.5 * (y0 + y1) * (xb - xa)
                continue

            xs = x[mask]
            ys = y[mask]

            # Ensure endpoints present
            if xs[0] > xa:
                ys0 = np.interp(xa, x, y)
                xs = np.insert(xs, 0, xa)
                ys = np.insert(ys, 0, ys0)
            if xs[-1] < xb:
                ys1 = np.interp(xb, x, y)
                xs = np.append(xs, xb)
                ys = np.append(ys, ys1)

            out[i] = np.trapezoid(ys, xs)
        return out

    # 4) Integrate gaps/overlaps per bin across all adjacent-pair columns
    gap_area_bins = np.zeros(len(bins) - 1, dtype=float)
    overlap_area_bins = np.zeros(len(bins) - 1, dtype=float)

    for col in gap_overlap_df.columns:
        y = gap_overlap_df[col].to_numpy()
        # Positive (gaps)
        gap_area_bins     += integrate_series_per_bin(x_go, np.clip(y,  0, None), bins)
        # Negative (overlaps) — take magnitude
        overlap_area_bins += integrate_series_per_bin(x_go, np.clip(-y, 0, None), bins)

    # 5) Layup area per bin = integral of height(x) over each bin
    layup_area_bins = integrate_series_per_bin(x_h, height, bins)

    # 6) Assemble result
    with np.errstate(divide='ignore', invalid='ignore'):
        gap_pct     = 100.0 * np.divide(gap_area_bins, layup_area_bins, out=np.zeros_like(gap_area_bins), where=layup_area_bins > 0)
        overlap_pct = 100.0 * np.divide(overlap_area_bins, layup_area_bins, out=np.zeros_like(overlap_area_bins), where=layup_area_bins > 0)

    result = pd.DataFrame({
        "bin_left_mm":       bins[:-1],
        "bin_right_mm":      bins[1:],
        "gap_area_mm2":      gap_area_bins,
        "overlap_area_mm2":  overlap_area_bins,
        "layup_area_mm2":    layup_area_bins,
        "gap_percent_bin":     gap_pct,
        "overlap_percent_bin": overlap_pct,
    })
    return result


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
    x_pdf = np.linspace(-2, 2, 100)     # this is for the plotting later on.

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
    print(f"std={std}, mean={mean}")
    y_starting = CAM_starting_dist(x_pdf)

    # This section modifies the starting distributions used by the model, which is needed for Model_ALL_StartVariations.
    if starting_mods != [1, 1]:
        # these are the factors by which the mean and std are changed
        loc_factor, scale_factor = starting_mods[0], starting_mods[1]
        mean_modded = mean * loc_factor
        std_modded = std * scale_factor
        print(f"std_modded={std_modded}, mean_modded={mean_modded}")
        CAM_starting_dist_modded = lambda x: norm.pdf(x, loc=mean_modded, scale=std_modded)
        y_starting_modded = CAM_starting_dist_modded(x_pdf)
        plt.plot(x_pdf, y_starting_modded, label='modified target distribution')


    plt.hist(first_values, bins=10, density=True, alpha=0.3, label='experimental starting values')
    plt.plot(x_pdf, y_target, label='target distribution')
    plt.plot(x_pdf, y_starting, label='starting values distribution')
    plt.title("starting and target distributions")
    plt.xlim(-2, 2)
    plt.legend()
    plt.show()


def main():
    #analyze_starting_variation_effects(starting_mods = [None, 1, 2], proposal_type = "RWM")
    #plot_target_vs_start(starting_mods=[1, 1.5])      # target_mods=[None, 1, 1.5], starting_mods=[1, 1.5]

    defect_data_original = calc_lengthwise_defect_percent(10, tows_per_laminate=29, num_divisions=31, alternate_start=[norm, [0.01221346, 0.3]]) #0.48016
    defect_data_modified = calc_lengthwise_defect_percent(10, tows_per_laminate=29, num_divisions=31, alternate_start=[norm, [0.01221346, 0.45]])
    plot_lengthwise_defect_percent(defect_data_original, defect_data_modified)
    calc_lengthwise_defect_percent_exp()

if __name__ == "__main__":
    main()