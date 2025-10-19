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


def calc_lengthwise_defect_percent(tows: int):
    # TODO: replace funciton below with a variable start function
    gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent, RW_data = generate_RW_multitow(num_tows=tows)

    print(gap_overlap_df)



    defect_data = pd.DataFrame(
        {"x": x,
         "defect_percent": defect_percent,
         "gap_percent": gap_percent,
         "overlap_percent": overlap_percent})
    return defect_data

def plot_lengthwise_defect_percent(defect_data: pd.DataFrame):
    a=1




def main():
    calc_lengthwise_defect_percent(10)

if __name__ == "__main__":
    main()