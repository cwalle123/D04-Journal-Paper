import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from Scripts.constants import tow_width_specified
from scipy.stats import norm, logistic, gamma, beta, expon, lognorm, skewnorm, gumbel_r, gumbel_l, genextreme

from Scripts.Handling_ALL_Functions import get_synced_data
import constants

from Scripts.Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path, generate_starting_error
from Scripts.Data_ALL_statistics import plot_histograms_separated, best_fit_distribution



def get_data(sensor: str, tows: list = list(np.arange(2, 32, 1))):
    """
    Gets the data for the specified sensors and puts it in the required form for the regression model.
    [[1st_data_point, 2nd, weight], [2nd, 3rd, weight], ...]
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
            tow_data = tow_data[["error_LT", "Weights"]]

        tow_data = np.array(tow_data)
        #print(tow_data)

        # put data into correct format (pairs)
        for i in range(len(tow_data[:, 0])):
            data.append(float(tow_data[i, 0]))
            weights.append(float(tow_data[i, 1]))

    return data, weights


def propose_new_RWM_value(x_current, dist_std):    # random walk metropolis (RWM), using normal dist???
    mean = 0
    std = 0.25
    proposal = x_current + np.random.normal(mean, dist_std*0.5)
    return proposal

def propose_new_MALA_value(x_current, dist_std):   # Metropolis-adjuster Langevian algorithm (MALA)

    #proposal = ???
    return proposal



def generate_random_walk(sensor: str, n_steps: int, proposal_type: str, plot_histogram: bool=False, plot_path: bool=False, return_path: bool = False):

    # setting up the distribution which is being mimicked
    data, weights = get_data(sensor)

    best = best_fit_distribution(       # TODO: write some proper code rather than this bs workaround...
        np.array([data, data, data, data]), weights=np.array([weights, weights, weights, weights])
        )
    dist, params = best['dist'], best['params']
    distribution = lambda x: dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

    start_value = -0.9              # TODO: get proper starting values
    generated_path = [start_value]
    x_current = start_value
    for step in range(n_steps-1):

        if proposal_type == "RWM":
            x_proposal = propose_new_RWM_value(x_current, float(params[-1]))
        elif proposal_type == "MALA":
            x_proposal = propose_new_MALA_value(x_current, float(params[-1]))
        else:
         print("Error, invalid type")

        # code to accept or reject the proposed new value
        alpha = distribution(x_proposal) / distribution(x_current)      # alpha = acceptance probability
        U = random.uniform(0, 1)

        if alpha >= U:  # we accept the proposed value
            x_next = x_proposal
            print("accepted")
        else:           # we reject the proposed value
            print("rejected")
            x_next = x_current

        generated_path.append(x_next)
        x_current = x_next




    if plot_histogram:
        #plotplotplot
        x = np.linspace(-1.5, 0.6, 200)
        pdf = distribution(x)
        plt.plot(x, pdf, label='probability-distribution')
        plt.hist(generated_path, density=True, bins=50, label='generated path')
        plt.legend()
        plt.show()

    if plot_path:
        x = np.linspace(0, n_steps , n_steps)
        plt.plot(x, generated_path, label='generated path')
        plt.show()

    if return_path:
        return generated_path



def main():
    generate_random_walk(sensor="LT", n_steps=2000, proposal_type="RWM", plot_histogram=True, plot_path=True)

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else