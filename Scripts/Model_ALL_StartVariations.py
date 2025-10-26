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
from Model_ALL_RandomWalk import fit_random_walk, generate_random_walk, generate_RW_multitow, initiate_state_data, update_states, check_state_data
from Model_ALL_RandomSampling import generate_RS_multitow
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

    #print(f'Top edges are: {top_edges}')
    #print(f'Bottom edges are: {bottom_edges}')
    #print(f'Dataframe is: {gap_overlap_df}')
    return gap_overlap_df, top_edges, bottom_edges

def generate_multilaminate_layout_RS(n_laminates: int, tows_per_laminate: int=29, starting_mods: list=[None, 1, 1], alternate_start: list=[None, "params"]):
    # generating and combining data for all laminates
    gap_overlap_dict = {}
    top_edges, bottom_edges = [], []
    for laminate in range(n_laminates):
        temp_gap_overlap_df, RS_data = generate_RS_multitow(
            num_tows=tows_per_laminate, starting_mods=starting_mods, alternate_start=alternate_start)
        temp_gap_overlap_df = np.array(temp_gap_overlap_df)

        temp_gap_overlap_dict = {
            f"Gap/overlap_Laminate{laminate}_Tow{tow + 1}-{tow + 2}": temp_gap_overlap_df[:, tow]
            for tow in range(tows_per_laminate - 1)}
        gap_overlap_dict.update(temp_gap_overlap_dict)

        highest_tow_edge = np.array(RS_data[-1]["top_edge"])
        lowest_tow_edge = np.array(RS_data[0]["bottom_edge"])
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
    x_tow = np.linspace(0, 1000, len(gap_overlap_df))
    divisions = np.linspace(0, len(gap_overlap_df), num_divisions+1)
    for i in range(num_divisions):
        start, end = divisions[i], divisions[i+1]
        data = gap_overlap_df[int(start):int(end)+1]
        x_vals = x_tow[int(start):int(end)+1]

        # old code: total_layout_area = np.trapezoid(highest_tow_edge[int(start):int(end)] - lowest_tow_edge[int(start):int(end)], x_vals)
        total_layout_area = 0
        for lam in range(n_laminates):
            lam_layout_area = np.trapezoid(top_edges[lam, int(start):int(end)+1] - bottom_edges[lam, int(start):int(end)+1], x_vals)
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

def plot_lengthwise_defect_percent(defect_data_original: pd.DataFrame, defect_data_modified: pd.DataFrame, mean_label, std_label, name: str=None):
    plt.figure(figsize=(10, 6))

    # original data
    # plt.scatter(defect_data_original["x"], defect_data_original["gap_percent"], color="green", marker='.')
    plt.plot(defect_data_original["x"], defect_data_original["gap_percent"], color="green", linestyle='dashed',
             label="Gap: mean="+str(mean_label[0])+", std="+str(std_label[0]))
    # plt.scatter(defect_data_original["x"], defect_data_original["overlap_percent"], color="green", marker='.')
    plt.plot(defect_data_original["x"], defect_data_original["overlap_percent"], color="green", linestyle='solid',
             label="Overlap: mean="+str(mean_label[0])+", std="+str(std_label[0]))

    # modified data
    # plt.scatter(defect_data_modified["x"], defect_data_modified["gap_percent"], color="red", marker='.')
    plt.plot(defect_data_modified["x"], defect_data_modified["gap_percent"], color="red", linestyle='dashed',
             label="Gap: mean="+str(mean_label[1])+", std="+str(std_label[1]))
    # plt.scatter(defect_data_modified["x"], defect_data_modified["overlap_percent"], color="red", marker='.')
    plt.plot(defect_data_modified["x"], defect_data_modified["overlap_percent"], color="red", linestyle='solid',
             label="Overlap: mean="+str(mean_label[1])+", std="+str(std_label[1]))

    plt.xlim(0, 1000)
    plt.xticks(np.linspace(0, 1000, 11), fontsize=10)
    plt.ylim(1, 5)
    plt.yticks(np.linspace(1, 5, 5))
    plt.grid(True, axis='y')
    plt.xlabel("x (mm)", fontsize=12)
    plt.ylabel("Defect Percentage (%)", fontsize=12)
    lgd = plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fancybox=True, shadow=False, ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    if name != None:
        plt.savefig(name, format="pdf", bbox_inches="tight")
    plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_lengthwise_defect_percent_exp(bin_size_mm: float, plot: bool = True):
    """
    Compute lengthwise gap/overlap percentage per bin for traverse data (tows 2–30),
    and also plot per-x percentages for each adjacent tow pair (29 lines per plot).

    Returns
    -------
    df_bins : pd.DataFrame
        Columns:
          - bin_start, bin_end, bin_center
          - layup_area     (envelope area per bin)
          - gap_area, overlap_area
          - gap_pct, overlap_pct
    totals : dict
        Overall totals across full length:
          - layup_area, gap_area, overlap_area, gap_pct, overlap_pct
    """

    # ---------- helpers ----------
    def _assert_strictly_increasing(x, name="x"):
        x = np.asarray(x)
        if not np.all(np.diff(x) > 0):
            raise ValueError(f"{name} must be strictly increasing.")

    def _integrate_on_interval(x, y, a, b):
        """
        Edge-aware trapezoid: integrate y(x) from a to b (a <= b) given
        samples on a strictly-increasing x. Interpolates y at [a,b] and
        uses any interior points strictly between (a, b).
        """
        if b <= a:
            return 0.0
        x = np.asarray(x); y = np.asarray(y)
        _assert_strictly_increasing(x, "x")
        inside = (x > a) & (x < b)

        if not np.any(inside):
            ya = float(np.interp(a, x, y))
            yb = float(np.interp(b, x, y))
            return 0.5 * (ya + yb) * (b - a)

        xs = np.concatenate(([a], x[inside], [b]))
        ys = np.concatenate(([np.interp(a, x, y)], y[inside], [np.interp(b, x, y)]))
        return float(np.trapezoid(ys, xs))

    # ---------- gather raw tows ----------
    top_edge_x, top_edge_y = [], []     # each tow's x and y_left (top)
    bot_edge_x, bot_edge_y = [], []     # each tow's x and y_right (bottom)
    for tow in range(2, 31):
        traverse_tow = traverse_tow_constructor(tow, normalize=True)
        if traverse_tow is None:
            continue

        offset_mm = (tow - 2) * tow_width_specified

        x = traverse_tow["x_centerline"].to_numpy()
        y_top = traverse_tow["y_left"].to_numpy() + offset_mm      # "left" = top edge
        y_bot = traverse_tow["y_right"].to_numpy() + offset_mm     # "right" = bottom edge

        _assert_strictly_increasing(x, name=f"x (tow {tow})")

        top_edge_x.append(x);  top_edge_y.append(y_top)
        bot_edge_x.append(x);  bot_edge_y.append(y_bot)

    if len(top_edge_x) < 2:
        raise RuntimeError("Not enough tows found to compute gaps/overlaps (need at least tows 2 and 3).")

    # ---------- build common x-grid (overlapping domain across all tows) ----------
    x_min_common = max(arr[0] for arr in top_edge_x)
    x_max_common = min(arr[-1] for arr in top_edge_x)
    x_min_common = max(x_min_common, max(arr[0] for arr in bot_edge_x))
    x_max_common = min(x_max_common, min(arr[-1] for arr in bot_edge_x))
    if not (x_max_common > x_min_common):
        raise RuntimeError("No overlapping x-range across tows; cannot form a common x-grid.")

    base_x_full = top_edge_x[0]
    mask_domain = (base_x_full >= x_min_common) & (base_x_full <= x_max_common)
    base_x = base_x_full[mask_domain]
    _assert_strictly_increasing(base_x, "base_x")
    if base_x.size < 3:
        raise RuntimeError("Common x-grid too small after intersection; need at least 3 points.")

    # ---------- interpolate all edges onto common x-grid ----------
    top_edges = [np.interp(base_x, x, y) for x, y in zip(top_edge_x, top_edge_y)]
    bot_edges = [np.interp(base_x, x, y) for x, y in zip(bot_edge_x, bot_edge_y)]

    # ---------- envelopes and pairwise diffs ----------
    highest_tow_edge = top_edges[-1]   # top of tow 30
    lowest_tow_edge  = bot_edges[0]    # bottom of tow 2
    env_height = highest_tow_edge - lowest_tow_edge
    env_height = np.where(env_height > 0, env_height, 0.0)  # clamp
    denom = np.where(env_height > 0, env_height, np.nan)    # for pct division

    # adjacent pair separation: bottom(i+1) - top(i)
    gap_overlap_stack = np.stack(
        [bot_edges[i+1] - top_edges[i] for i in range(len(top_edges) - 1)],
        axis=0
    )  # shape = (29, N)

    # positive = gap; negative magnitude = overlap
    gap_height_each = np.clip(gap_overlap_stack, 0, None)       # (29, N)
    ovl_height_each = np.clip(-gap_overlap_stack, 0, None)      # (29, N)

    # per-x totals (used for bin integration)
    pos_sum = np.sum(gap_height_each, axis=0)                    # (N,)
    neg_sum = np.sum(ovl_height_each, axis=0)                    # (N,)

    # ---------- binning with edge-aware integration ----------
    x_start, x_end = float(base_x[0]), float(base_x[-1])
    n_bins = max(1, int(np.ceil((x_end - x_start) / bin_size_mm)))
    edges = x_start + np.arange(n_bins + 1) * bin_size_mm
    edges[-1] = x_end  # land exactly on the end

    rows = []
    for b in range(n_bins):
        left, right = float(edges[b]), float(edges[b+1])

        env_area = _integrate_on_interval(base_x, env_height, left, right)
        gap_area = _integrate_on_interval(base_x, pos_sum,     left, right)
        ovl_area = _integrate_on_interval(base_x, neg_sum,     left, right)

        gap_pct = (gap_area / env_area * 100.0) if env_area > 0 else np.nan
        ovl_pct = (ovl_area / env_area * 100.0) if env_area > 0 else np.nan

        rows.append({
            "bin_start": left,
            "bin_end": right,
            "bin_center": 0.5*(left + right),
            "layup_area": env_area,        # envelope area per bin (kept name for compatibility)
            "gap_area": gap_area,
            "overlap_area": ovl_area,
            "gap_pct": gap_pct,
            "overlap_pct": ovl_pct
        })

    df_bins = pd.DataFrame(rows)

    # ---------- totals over full length ----------
    layup_area_tot   = _integrate_on_interval(base_x, env_height, x_start, x_end)
    gap_area_tot     = _integrate_on_interval(base_x, pos_sum,     x_start, x_end)
    overlap_area_tot = _integrate_on_interval(base_x, neg_sum,     x_start, x_end)
    gap_pct_tot      = (gap_area_tot / layup_area_tot * 100.0) if layup_area_tot > 0 else np.nan
    overlap_pct_tot  = (overlap_area_tot / layup_area_tot * 100.0) if layup_area_tot > 0 else np.nan

    totals = dict(
        layup_area=layup_area_tot,
        gap_area=gap_area_tot,
        overlap_area=overlap_area_tot,
        gap_pct=gap_pct_tot,
        overlap_pct=overlap_pct_tot
    )

    # ---------- PLOTTING ----------
    if plot:
        # (A) Per-bin % plot (your original)
        plt.figure(figsize=(9, 4.6))
        plt.plot(df_bins["bin_center"].values, df_bins["gap_pct"].values, marker="o", label="Gap %")
        plt.plot(df_bins["bin_center"].values, df_bins["overlap_pct"].values, marker="s", label="Overlap %")
        plt.xlabel("Distance (mm)")
        plt.ylabel("Percentage of envelope area (%)")
        plt.title(f"Gap & Overlap % per {bin_size_mm:.0f} mm bin")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Per tow gap percentage vs. x
        # gap_pct_each[i, :] = (gap_height_each[i, :] / env_height) * 100
        gap_pct_each = (gap_height_each / denom[None, :]) * 100.0
        plt.figure(figsize=(10, 5))
        for i in range(gap_pct_each.shape[0]):
            label=f"Tows {i+2}-{i+3}"  # uncomment to show legend (may be crowded)
            plt.plot(base_x, gap_pct_each[i, :], linewidth=1, alpha=0.9, label=label)
        plt.xlabel("Distance (mm)")
        plt.ylabel("Gap height / envelope (%)")
        plt.title("Gap % vs x for each adjacent tow pair")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=8)  
        plt.tight_layout()
        plt.show()

        # Per tow overlap percentage vs. x
        ovl_pct_each = (ovl_height_each / denom[None, :]) * 100.0
        plt.figure(figsize=(10, 5))
        for i in range(ovl_pct_each.shape[0]):
            label=f"Tows {i+2}-{i+3}"
            plt.plot(base_x, ovl_pct_each[i, :], linewidth=1, alpha=0.9, label=label)
        plt.xlabel("Distance (mm)")
        plt.ylabel("Overlap height / envelope (%)")
        plt.title("Overlap % vs x for each adjacent tow pair")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=8) 
        plt.tight_layout()
        plt.show()

    return df_bins, totals



def analyze_starting_variation_effects(starting_mods: list=[None, 1, 1], proposal_type="RWM"):
    LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params = fit_random_walk("LT")
    CAM_steps, CAM_proposal_std, CAM_target_dist, CAM_dist, CAM_params = fit_random_walk("CAM")
    LLS_B_steps, LLS_B_proposal_std, LLS_B_target_dist, LLS_B_dist, LLS_B_params = fit_random_walk("LLS_B")
    print(f"distributions type per error:\n LT:{LT_dist}\n CAM:{CAM_dist}\nLLS_B:{LLS_B_dist}")

    # This secton modifies the starting distributions used by the model, which is needed for Model_ALL_StartVariations.
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


def detect_duplicate_states():
    initiate_state_data()       # this initiates the global variable state_data within the RW code

    defect_data_original = calc_lengthwise_defect_percent(5, tows_per_laminate=5, num_divisions=31, alternate_start=[norm, [0.01221346, 0.3]]) #0.48016
    defect_data_modified = calc_lengthwise_defect_percent(5, tows_per_laminate=5, num_divisions=31, alternate_start=[norm, [0.01221346, 0.45]])
    plot_lengthwise_defect_percent(defect_data_original, defect_data_modified)

    check_state_data()      # this check for duplicate states

def main():
    #analyze_starting_variation_effects(starting_mods = [None, 1, 2], proposal_type = "RWM")
    #plot_target_vs_start(starting_mods=[1, 1.5])      # target_mods=[None, 1, 1.5], starting_mods=[1, 1.5]

    ### version 1 ###
    defect_data_original = calc_lengthwise_defect_percent(50, tows_per_laminate=29, num_divisions=100,
                                                          alternate_start=[norm, [0, 0.3]]) #0.01221346, 0.48016
    defect_data_modified = calc_lengthwise_defect_percent(50, tows_per_laminate=29, num_divisions=100,
                                                          alternate_start=[norm, [0, 0.45]])
    plot_lengthwise_defect_percent(defect_data_original, defect_data_modified, mean_label=[0, 0], std_label=[0.3, 0.45], name="StartVariations-version_1.pdf")

    ### version 2 ###
    #defect_data_original = calc_lengthwise_defect_percent(50, tows_per_laminate=29, num_divisions=100,
    #                                                      alternate_start=[norm, [0, 0.3]])  # 0.01221346, 0.48016
    #defect_data_modified = calc_lengthwise_defect_percent(50, tows_per_laminate=29, num_divisions=100,
    #                                                      alternate_start=[norm, [0.3, 0.3]])
    #plot_lengthwise_defect_percent(defect_data_original, defect_data_modified, mean_label=[0, 0.3], std_label=[0.3, 0.3], name="StartVariations-version_2.pdf")

    #calc_lengthwise_defect_percent_exp(bin_size_mm=10)
    #generate_multilaminate_layout(2)
    
if __name__ == "__main__":
    main()