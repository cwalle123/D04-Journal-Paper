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
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path, generate_starting_error
from Data_ALL_statistics import main as real_hist, plot_histograms_separated, best_fit_distribution
from Model_ALL_RandomWalk import generate_random_walk, generate_RW_multitow
from Model_ALL_Simulation import generate_multitow_layout
from Model_ALL_RandomSampling import generate_RS_multitow



def generate_RW_multitow_startvar():
    a=1


def calc_lengthwise_defect_percent(tows: int, num_divisions: int=31):
    # TODO: replace funciton below with a variable start function
    gap_overlap_df, gap_df, overlap_df, total_gap_percent, total_overlap_percent, RW_data = generate_RW_multitow(num_tows=tows)

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
        print(total_layout_area, total_gap_area, total_overlap_area)

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

def plot_lengthwise_defect_percent(defect_data: pd.DataFrame):
    plt.plot(defect_data["x"], defect_data["defect_percent"], label="Defect Percentage")
    plt.show()




def main():
    defect_data = calc_lengthwise_defect_percent(100)
    plot_lengthwise_defect_percent(defect_data)

if __name__ == "__main__":
    main()