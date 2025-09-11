import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import functools

from dataclasses import dataclass
from scipy.stats import norm, logistic, gamma, beta, expon, lognorm, skewnorm, gumbel_r, gumbel_l, genextreme
from Handling_ALL_Functions import get_synced_data
from constants import tow_width_specified
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path, generate_starting_error, get_data_pairs
from Data_ALL_statistics import plot_histograms_separated, best_fit_distribution
from Model_ALL_ConsecutiveModeler import tow_visualizer



def get_data(sensor: str, tows: list = list(np.arange(2, 32, 1)), format: str = 'merged'):
    """
    Gets the required data for the specified sensors.
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

        if format == 'merged':
            # put data into correct format (pairs)
            for i in range(len(tow_data[:, 0])):
                data.append(float(tow_data[i, 0]))
                weights.append(float(tow_data[i, 1]))

        elif format == 'separated':
            data.append(tow_data[:, 0])
            weights.append(tow_data[:, 1])

        else: print("Invalid format. Possible values are 'merged' and 'separated'.")

    return data, weights

def get_n_steps(sensor):
    data, weights = get_data(sensor, format='separated')
    data = np.array(data)

    lengths = []
    for i in range(len(data)):
        lengths.append(len(data[i, :]))

    return int(np.average(lengths))

def propose_new_RWM_value(x_current, dist_std, sensor):    # random walk metropolis (RWM), using normal dist???
    mean = 0
    # setting the std for each sensor
    #if sensor == "LLS_A": std_factor = 0.7
    #elif sensor == "LLS_B": std_factor = 0.7
    #elif sensor == "CAM": std_factor = 0.15
    #elif sensor == "LT": std_factor = 0.35

    #std_factor = 2.4  # for optimal exploration of dist -> accepetance rate = 44%, NOT ACCURATE! only CAM & LLS

    proposal = x_current + np.random.normal(mean, dist_std)  # this uses std to recreate real tow 'waviness'
    return proposal

def propose_new_MALA_value(x_current, dist_std):   # Metropolis-adjuster Langevian algorithm (MALA)

    #proposal = ???
    return proposal


def generate_random_walk(sensor: str, proposal_type: str, n_steps: int=None, plot_histogram: bool=False,
                         plot_path: bool=False, comparison: bool=False, return_pdf: bool=False):
    '''
    This function generates a random walk according to the specified sensor.
    '''
    if n_steps is None:
        n_steps = get_n_steps(sensor)

    # setting up the distribution which is being mimicked
    data, weights = get_data(sensor)

    best = best_fit_distribution(np.array(data), weights=np.array(weights))
    dist, params = best['dist'], best['params']
    distribution = lambda x: dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

    start_value = generate_starting_error(sensor)
    generated_path = [start_value]
    x_current = start_value
    sensor_std = get_proposal_distribution(sensor)

    accepted, rejected = 0, 0
    for step in range(n_steps-1):

        if proposal_type == "RWM":
            x_proposal = propose_new_RWM_value(x_current, sensor_std, sensor)
        elif proposal_type == "MALA":
            x_proposal = propose_new_MALA_value(x_current, float(params[-1]))
        else:
         print("Error, invalid type")

        # code to accept or reject the proposed new value
        alpha = distribution(x_proposal) / distribution(x_current)      # alpha = acceptance probability
        U = random.uniform(0, 1)

        if alpha >= U:  # we accept the proposed value
            x_next = x_proposal
            accepted += 1
        else:           # we reject the proposed value
            rejected += 1
            x_next = x_current

        generated_path.append(x_next)
        x_current = x_next

    print("acceptance rate =", accepted/(accepted + rejected))

    # plot-plot-plot #
    x_pdf = np.linspace(-1.2, 1.2, 100)
    y_pdf = distribution(x_pdf)
    if plot_histogram:
        plt.plot(x_pdf, y_pdf, label='probability-distribution')
        plt.hist(generated_path, density=True, bins=30, label='generated path')
        plt.hist(data, density=True, bins=50, label='data')
        plt.title("Histogram of generated path w. probability-distribution" + sensor)
        plt.xlim(-1.2, 1.2)
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
            real_data_uncut = get_synced_data(2, sensor)
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
        x = np.linspace(min(data), max(data), 200)
        distribution = lambda x: norm.pdf(x, loc=mean, scale=std)

        # plotting
        plt.plot(x, distribution(x), label='proposal distribution')
        plt.hist(data, weights=weights, density=True, label='step-size data', bins=200)
        plt.title('step size distribution for ' + sensor)
        plt.legend()
        plt.show()

    return std


def get_actual_Dataframe(tow: int):
    '''
    Gets the tow-data and returns it as a dataframe in the required format for the tow-visualizer.
    '''
    LT_data = get_synced_data(tow, "LT")
    y = LT_data["error_LT"]
    x = LT_data["x"]
    CAM_data = get_synced_data(tow, "CAM")
    center = CAM_data["center_CAM"]
    LLSB_data = get_synced_data(tow, "LLS_B")
    width = LLSB_data["width_LLS_B"]
    # putting into dataframe which can be used by the tow visualizer
    dataframe = pd.DataFrame(
        {"y": y,
         "center_CAM": center,
         "width_LLS_B": width,
         "x": x})

    print(dataframe)
    return dataframe



def tow_visualizer_alt(tows: list[pd.DataFrame], y_intended: list, labels: list, ideal: bool):      # TODO: probably get rid of this.. . :(
    """
    This function takes a list of dataframes that contains features of a tows and plots the corresponding tows in one figure, as well as the ideal tow.
    The data it takes from that dataframe are
    the centerline, width and x-position. It is important that the columns in the dataframe are properly named.
    For this, check that the centerline column is named "center_CAM", the width after compaction column is named
    "width_LLS_B" and the x-position columns is called "x".

    Arguments are:
    tows: list[pd.DataFrame], a list with dataframes of the tows
    y_intended: list, a list of programmed centerline y-values of the tows, IMPORTANT: tows[i] HAS TO CORRESPOND WITH y_intended[i].
    name: str, the name of the operation that was done to obtain the dataframes of the tows, will be the title of the graph.
    ideal: bool, plots one ideal tow if true

    Author: Martijn
    """
    # Check if all elements are DataFrames
    if not all(isinstance(tow, pd.DataFrame) for tow in tows):
        raise TypeError("All elements in 'tows' must be pandas DataFrames.")

    # set figure size
    plt.figure(figsize=(15, 2))

    for i in range(len(y_intended)):
        CAM_centerline = tows[i]["center_CAM"]  # take the centerline from CAM
        LT_y = tows[i]["y"]  # take the y-position from LT
        intended_centerline = y_intended[i]  # take the programmed y-value for a straight line
        centerline = CAM_centerline + LT_y + intended_centerline  # calculate centerline in space by combining datatypes
        width = tows[i]["width_LLS_B"]  # take the width from LLS B
        x = tows[i]["x"]  # take the x-position from LT
        name = labels[i]

        # make the plots
        plt.plot(x, centerline, label=name+"centerline", linestyle='dashed', color='grey')  # plots the centerline
        plt.plot(x, centerline + 0.5 * width, label=name+"tow", linestyle='solid',
                 color='black')  # plots the top edge
        plt.plot(x, centerline - 0.5 * width, linestyle='solid', color='black')  # plots the bottom edge

        # plots the start end endlines of the tow
        plt.plot([x.iloc[0], x.iloc[0]],
                 [centerline.iloc[0] - 0.5 * width.iloc[0], centerline.iloc[0] + 0.5 * width.iloc[0]],
                 linestyle='solid', color='black')
        plt.plot([x.iloc[-1], x.iloc[-1]],
                 [centerline.iloc[-1] - 0.5 * width.iloc[-1], centerline.iloc[-1] + 0.5 * width.iloc[-1]],
                 linestyle='solid', color='black')

    if ideal == True:
        # plot the ideal tow (just a rectangle)
        plt.plot([0, 1000], [tow_width_specified * 0.5, tow_width_specified * 0.5], color='green', label='ideal tow')
        plt.plot([0, 1000], [-tow_width_specified * 0.5, -tow_width_specified * 0.5], color='green')
        plt.plot([0, 0], [tow_width_specified * 0.5, -tow_width_specified * 0.5], color='green')
        plt.plot([1000, 1000], [tow_width_specified * 0.5, -tow_width_specified * 0.5], color='green')
        plt.plot([0, 1000], [0, 0], color='green', linestyle='dashed', label='ideal centerline')

    # calculate the dimensions of the plots
    #x_min = min(min(tow["x"].min() for tow in tows) - 50, -50)
    #x_max = max(max(tow["x"].max() for tow in tows) + 50, 1050)
    #y_min = min(min(tow["y"].min() for tow in tows) - 100, -50)
    #y_max = max(max(tow["y"].max() for tow in tows) + 50, 1050)

    # plot info
    plt.xlabel("x-position [mm]")
    plt.ylabel("y-position [mm]")
    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
    plt.grid()
    plt.title("random walk comparison... or something like that")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()


def plot_tow_comparison(n_steps: int, step_size: float, proposal_type: str, plot_individual_histograms: bool=False):
    '''
    This function is intended to make a comparison of a full tow between random walk and actual data.
    '''
    # getting randomn walk data
    LT_walk_data = generate_random_walk("LT", proposal_type, n_steps=n_steps, plot_histogram=plot_individual_histograms, plot_path=True, comparison=True)
    CAM_walk_data = generate_random_walk("CAM", proposal_type, n_steps=n_steps, plot_histogram=plot_individual_histograms, plot_path=True, comparison=True)
    LLSB_walk_data = generate_random_walk("LLS_B", proposal_type, n_steps=n_steps, plot_histogram=plot_individual_histograms, plot_path=True, comparison=True)
    LLSB_walk_data = [x + tow_width_specified for x in LLSB_walk_data]
    x_walk_data = np.linspace(0, n_steps*step_size, n_steps)

    # putting into dataframe which can be used by the tow visualizer
    walk_dataframe = pd.DataFrame(
        {"y": LT_walk_data,
         "center_CAM": CAM_walk_data,
         "width_LLS_B": LLSB_walk_data,
         "x": x_walk_data})

    real_data = get_actual_Dataframe(2)

    tow_visualizer_alt([walk_dataframe, real_data], [0, 0], ["random walk", "Real"], False)


def plot_animated_walk_hist(sensor: str, n_tows: int, proposal_type: str='RWM', tow_length: float=1000):

    n_steps = get_n_steps(sensor)
    step_size = tow_length/n_steps

    # Setting up a random number generator with a fixed state for reproducibility.
    rng = np.random.default_rng(seed=19680801)
    # Fixing bin edges.
    HIST_BINS = np.linspace(-1.2, 1.2, 100)

    # Histogram our data with numpy.
    data, x_pdf, y_pdf = generate_random_walk(sensor=sensor, n_steps=n_steps, proposal_type=proposal_type, return_pdf=True)
    n, _ = np.histogram(data, HIST_BINS, density=True)

    # %%
    # To animate the histogram, we need an ``animate`` function, which generates
    # a random set of numbers and updates the heights of rectangles. The ``animate``
    # function updates the `.Rectangle` patches on an instance of `.BarContainer`.

    def animate(frame_number, bar_container):
        nonlocal data
        # Simulate new data coming in.
        data += generate_random_walk(sensor=sensor, n_steps=n_steps, proposal_type=proposal_type)
        n, _ = np.histogram(data, HIST_BINS, density=True)
        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)

        return bar_container.patches

    # %%
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
    plt.show()


def main():
    #generate_random_walk(sensor="CAM", proposal_type="RWM", n_steps=None, plot_histogram=True, plot_path=True, comparison=True)
    #plot_tow_comparison(n_steps=400, step_size=2.5, proposal_type="RWM", plot_individual_histograms=True)
    #std = get_proposal_distribution("CAM")
    #plot_animated_walk_hist("LT", 31)
    #get_n_steps("CAM")

    data = get_data("LT", tows=[2])

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else