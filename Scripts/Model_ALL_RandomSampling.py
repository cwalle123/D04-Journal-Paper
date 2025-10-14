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


def generate_RS_multitow(num_tows: int, n_steps: int=400, tow_spacing_mm: float=6.35, tow_width_mm: float=6.35, tow_length_mm: float=1000):
def generate_RS_multitow(num_tows: int, n_steps: int=400, tow_spacing_mm: float=6.35, tow_width_mm: float=6.35, method: str="Sidd"):
    tow_offset = 0
    top_edge_paths, bottom_edge_paths = [], []

    x_sampling_data = np.linspace(0, tow_length_mm, n_steps)

    LT_RS_data = generate_random_sampling_data("LT", steps=n_steps, tows=num_tows)
    CAM_RS_data = generate_random_sampling_data("CAM", steps=n_steps, tows=num_tows)

    if method == "Sidd":
        LLSB_RS_data = generate_siddharth_width(steps=n_steps, tows=num_tows)
    else:
        LLSB_RS_data = generate_random_sampling_data("LLS_B", steps=n_steps, tows=num_tows)

    for tow in range(num_tows):


        # getting it into centerline and width format
        tow_centerline_data = tow_offset + np.array(CAM_RS_data[tow, :]) + np.array(LT_RS_data[tow, :])
        tow_width_data = tow_width_mm + np.array(LLSB_RS_data[tow, :])

        tow_top_edge = tow_centerline_data + 0.5 * tow_width_data
        tow_bottom_edge = tow_centerline_data - 0.5 * tow_width_data

        top_edge_paths.append(tow_top_edge)
        bottom_edge_paths.append(tow_bottom_edge)
        tow_offset += tow_spacing_mm

        RS_data = pd.DataFrame({
            "x_mm": x_sampling_data,
            "centerline": tow_centerline_data,
            "top_edge": tow_top_edge,
            "bottom_edge": tow_bottom_edge,
        })

    # creating the gap_overlap_data
    gap_overlap_dict = {
        f"Gap/overlap_Tow{tow_index + 1}_Tow{tow_index + 2}": bottom_edge_paths[tow_index + 1] - top_edge_paths[
            tow_index]
        for tow_index in range(num_tows - 1)
    }
    gap_overlap_df = pd.DataFrame(gap_overlap_dict)

    return gap_overlap_df, RS_data


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


def main():
    #generate_random_sampling_data("LLS_B", steps=400, tows=300, plot_histogram=True)
    #gap_overlap_df = generate_RS_multitow(31, n_steps=400, tow_spacing_mm=12.5, tow_width_mm=6.35)
    #print(gap_overlap_df)
    generate_siddharth_width(tows=50, plot_histogram=True)


if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else