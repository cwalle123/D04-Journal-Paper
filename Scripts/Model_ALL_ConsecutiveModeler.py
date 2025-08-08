'''This is meant to run a large number of simulations at a time.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from Scripts.constants import tow_width_specified

from Scripts.Handling_ALL_Functions import get_synced_data

from Scripts.Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path
from Scripts.Data_ALL_statistics import main as real_hist, statistical_values, plot_histograms_separated, best_fit_distribution
import random

def save_distribution_data():
    def export_data(data_table: pd.DataFrame, short_name):
        '''This function saves a pandas dataframe as
            a .pkl, it will be saved with the short name,
            use that to access it.
            It was never used and isn't currently functioning I think'''

        _save_path = "Script\\"

        data_table.to_pickle(_save_path + short_name + ".pkl")
        # note! this does not save headers or indexes. might need to change that depending on how we do
        return

# def save_all_distribution_data(_save_path, LT_short_name, CAM_short_name, LLSA_short_name, LLSB_short_name):
#     '''This function saves all the data of the distributions generated of the consecutive data.
#         It was never used and isn't currently functioning I think'''
#     LT_dist = consecutive_error('LT')
#     CAM_dist = consecutive_error('CAM')
#     LLSA_dist = consecutive_error('LLS_A')
#     LLSB_dist = consecutive_error('LLS_B')

#     LT_dist.columns = ["LT_mean", "LT_std"]
#     #LT_dist.columns = ["LT_mean", "LT_std"]
#     #LT_dist.columns = ["LT_mean", "LT_std"]
#     #LT_dist.columns = ["LT_mean", "LT_std"]

#     save_distribution_data(data, LT_short_name)
#     save_distribution_data(data, CAM_short_name)
#     save_distribution_data(data, LLSA_short_name)
#     save_distribution_data(data, LLSB_short_name)

def plot_histograms(real_data: pd.DataFrame, sim_data: list, title: str, bin_widths: list[float] = None):
    '''This function plots a histogram of real and simulated data
        for each of the sensors separately.'''

    distribution_labels = {
        'norm': 'Normal Distribution',
        'logistic': 'Logistic Distribution',
        'skewnorm': 'Skew-Normal Distribution',
        'genextreme': 'Generalized Extreme Value'}

    #fig, ax = plt.subplots(figsize=(10, 8))
    #fig.suptitle(title)
    errors = [
        real_data['width error_LLS_A'],
        real_data['width error_LLS_B'],
        real_data['error_LT'],
        real_data['center_CAM']]

    names = ['error_LLS_A', 'error_LLS_B', 'error_LT', 'error_CAM']

    titles = [
        'Error Tape width before compaction',
        'Error Tape width after compaction',
        'Error robot position',
        'Error tape lateral movement']

    bin_widths = [0.005, 0.005, 0.005, 0.008]
    if bin_widths is None:
        bin_widths = [None] * 4

    for i, vals in enumerate(errors):
        fig, ax = plt.subplots(figsize=(8, 2.5))
        # print(f'TESTTEST: i={i}, vals={vals} #######################')
        row, col = divmod(i, 2)
        clean = vals.dropna().to_numpy()
        mn, mx = clean.min(), clean.max()
        bw = bin_widths[i]
        bins = 40 if bw is None else np.arange(mn, mx + bw, bw)

        ax.hist(clean, bins=bins, alpha=0.5, density=True, label='Experimental')
        ax.hist(sim_data[i], bins=bins, alpha=0.5, density=True, label='Model')
        best = best_fit_distribution(clean, bins=len(bins) - 1)
        dist, params = best['dist'], best['params']
        friendly = distribution_labels.get(dist.name, dist.name)




        # print(f"{names[i]} best fit: {friendly}")

        #x = np.linspace(mn, mx, 200)
        #pdf = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])

        #ax[row, col].plot(x, pdf, '-', lw=2, label=friendly)
        #ax[row, col].text(0.02, 0.95, friendly, transform=ax[row, col].transAxes,
        #                  va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


        ################
        #ax[row, col].hist(sim_data[i], bins=bins, alpha=0.5, density=True, label='simulated', color='red')
        #best = best_fit_distribution(sim_data[i], bins=len(bins) - 1)
        #dist, params = best['dist'], best['params']
        #friendly = distribution_labels.get(dist.name, dist.name)
#
        #ax[row, col].hist(sim_data[i], bins=bins, alpha=0.6, density=True, label='simulated')
#
        ## print(f"{names[i]} best fit: {friendly}")
#
        #x = np.linspace(mn, mx, 200)
        #pdf = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
#
        #ax[row, col].plot(x, pdf, '-', lw=2, label=friendly)
        #ax[row, col].text(0.02, 0.95, friendly, transform=ax[row, col].transAxes,
        #                  va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ################

        # Fix limits for individual plots for better visualization
        #if i == 1:
        #    ax.set_xlim(-0.4, 0.2)  #0.4
        #elif i == 2:
        #    ax.set_xlim(-1.2, -0.75)
        #elif i == 3:
        #    ax.set_xlim(-0.5, 1)
        ax.set_xlim(-1.2, 1.2)

        mean_val = clean.mean()
        std_val = clean.std()
        sim_mean = np.array(sim_data[i]).mean()
        sim_std = np.array(sim_data[i]).std()

        print(f'{titles[i]} Experimental mean/std = {mean_val}/{std_val}')
        print(f'{titles[i]} Model mean/std = {sim_mean}/{sim_std}')

        ax.axvline(mean_val, color='purple', linestyle='-', label='Experimental Mean', linewidth=1)
        ax.axvline(sim_mean, color='red', linestyle='-', label='Model Mean', linewidth=1)     # + '\n' + rf'$\sigma$ = {sim_std:.2f}'
        ax.axvline(0, color='black', linestyle='dashed')

        #ax[row, col].set_title(titles[i])
        ax.set_xlabel(titles[i], fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter('{x:0<3.1f}')

        plt.xticks(np.linspace(-1.2, 1.2, 9))
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()



def run_model(save_data: bool=False, use_saved: bool=False, generate_varying_bin_plots: bool=False, return_data: bool=True):
    '''This function executes all steps needed to plot the
        histograms for each sensor, including experimental and real.
        This involves first getting the data and then plotting.'''

    real_data = pd.concat((get_synced_data(t, spacesynced=True) for t in range(3, 32, 2)), ignore_index=True)

    cam_start_range = (-0.4, 0.6)
    lt_start_range = (-1, -0.8)
    llsb_start_range = (-0.15, -0.02)

    if use_saved:
        _save_path = "Script\\"
        LT_short_name = 'LT_Dist_Data'
        CAM_short_name = 'CAM_Dist_Data'
        LLSA_short_name = 'LLSA_Dist_Data'
        LLSB_short_name = 'LLSB_Dist_Data'

        if save_data:
            save_distribution_data(_save_path, LT_short_name, CAM_short_name, LLSA_short_name, LLSB_short_name)
        LT_dist = pd.read_pickle(_save_path + LT_short_name + ".pkl")
        CAM_dist = pd.read_pickle(_save_path + CAM_short_name + ".pkl")
        LLSA_dist = pd.read_pickle(_save_path + LLSA_short_name + ".pkl")
        LLSB_dist = pd.read_pickle(_save_path + LLSB_short_name + ".pkl")


    LT_generated_bins_mean_var = []
    CAM_generated_bins_mean_var = []
    LLSA_generated_bins_mean_var = []
    LLSB_generated_bins_mean_var = []
    generated_bins_mean_var = [LT_generated_bins_mean_var, CAM_generated_bins_mean_var, LLSA_generated_bins_mean_var, LLSB_generated_bins_mean_var]
    for num_bins in range(130, 131, 5):
        #num_bins = 30
        rs = 42
        LT_dist = consecutive_error('LT', random_state=rs, num_bins=num_bins, test_ratio=0.00001, plot_fit=False)
        CAM_dist = consecutive_error('CAM', random_state=rs, num_bins=num_bins, test_ratio=0.00001, plot_fit=False)
        LLSA_dist = consecutive_error('LLS_A', random_state=rs, num_bins=num_bins, test_ratio=0.00001, plot_fit=False)
        LLSB_dist = consecutive_error('LLS_B', random_state=rs, num_bins=num_bins, test_ratio=0.00001, plot_fit=False)

        # ------ This section generates the simulated data used for the comparison ------
        n_runs = 200
        total_data = []
        n_steps = 289
        total_error = [[], [], [], []]
        for run in range(n_runs):
            # starting position data
            start_cam = random.uniform(*cam_start_range)
            start_lt = random.uniform(*lt_start_range)
            start_llsb = random.uniform(*llsb_start_range)

            # generating data
            LT_error_list = generate_error_path(start_lt, n_steps, LT_dist[1], LT_dist[2], LT_dist[-3], LT_dist[-2],
                                                LT_dist[-1])
            CAM_error_list = generate_error_path(start_cam, n_steps, CAM_dist[1], CAM_dist[2], CAM_dist[-3], CAM_dist[-2],
                                                 CAM_dist[-1])
            LLSA_error_list = generate_error_path(-0.25, n_steps, LLSA_dist[1], LLSA_dist[2], LLSA_dist[-3],
                                                  LLSA_dist[-2], LLSA_dist[-1])
            LLSB_error_list = generate_error_path(start_llsb, n_steps, LLSB_dist[1], LLSB_dist[2], LLSB_dist[-3],
                                                  LLSB_dist[-2], LLSB_dist[-1])


            total_error[0] = (total_error[0] + list(LT_error_list))
            total_error[1] = (total_error[1] + list(CAM_error_list))
            total_error[2] = (total_error[2] + list(LLSA_error_list))
            total_error[3] = (total_error[3] + list(LLSB_error_list))


            #generated_data = []
            #x = 0
            #for i in range(len(LT_error_list)):
            #    centerline_error = LT_error_list[i] + CAM_error_list[i]
            #    width_error = LLSB_error_list[i]
            #    x +=dx
            #    generated_data.append([x, centerline_error, width_error])
            #
            #generated_data = pd.DataFrame(generated_data, columns = ['x', 'error'])

        plot_histograms(
            real_data,
            [total_error[2], total_error[3], total_error[0], total_error[1]],
            title="Sensor Error Histograms (ALL TOWS BUT NOT SPACE SYNCED), num_bins=" + str(num_bins),
            bin_widths=[0.01, 0.01, 0.005, 0.03]
        )

        for i in range(4):
            mean = float(np.array(total_error[i]).mean())
            variance = float(np.array(total_error[i]).std())
            generated_bins_mean_var[i].append([num_bins, mean, variance])

    print(f'LT errors: {LT_generated_bins_mean_var}')
    print(f'CAM errors: {CAM_generated_bins_mean_var}')
    print(f'LLSA errors: {LLSA_generated_bins_mean_var}')
    print(f'LLSB errors: {LLSB_generated_bins_mean_var}')

    print('total number of data points = ', len(total_error[0]))
    plt.subplot(223)
    plt.hist(total_error[0], bins=50)
    plt.title('LT')
    plt.subplot(224)
    plt.hist(total_error[1], bins=50)
    plt.title('CAM')
    plt.subplot(221)
    plt.hist(total_error[2], bins=50)
    plt.title('LLSA')
    plt.subplot(222)
    plt.hist(total_error[3], bins=50)
    plt.title('LLSB')

    plt.tight_layout()
    plt.show()

    real_hist()

    # This generates plots to determine optimal bin number based on global mean and variance.
    if generate_varying_bin_plots:
        LT_generated_bins_mean_var = np.array(LT_generated_bins_mean_var)
        CAM_generated_bins_mean_var = np.array(CAM_generated_bins_mean_var)
        LLSA_generated_bins_mean_var = np.array(LLSA_generated_bins_mean_var)
        LLSB_generated_bins_mean_var = np.array(LLSB_generated_bins_mean_var)

        # plotting mean
        plt.subplot(223)
        plt.plot(LT_generated_bins_mean_var[:, 0], LT_generated_bins_mean_var[:, 1], label='mean')
        plt.hlines(y=[-0.94], xmin=0, xmax=100, linestyle='dotted')
        plt.title('LT')
        # plt.ylim((min(LT_generated_bins_mean_var[:, 0]), max(LT_generated_bins_mean_var[:, 0])))
        plt.legend()

        plt.subplot(224)
        plt.plot(CAM_generated_bins_mean_var[:, 0], CAM_generated_bins_mean_var[:, 1], label='mean')
        plt.hlines(y=[0.32], xmin=0, xmax=100, linestyle='dotted')
        plt.title('CAM')
        # plt.ylim((min(CAM_generated_bins_mean_var[:, 1]), max(CAM_generated_bins_mean_var[:, 1])))
        plt.legend()

        plt.subplot(221)
        plt.plot(LLSA_generated_bins_mean_var[:, 0], LLSA_generated_bins_mean_var[:, 1], label='mean')
        plt.hlines(y=[-0.25], xmin=0, xmax=100, linestyle='dotted')
        plt.title('LLSA')
        # plt.ylim((min(LLSA_generated_bins_mean_var[:, 1]), max(LLSA_generated_bins_mean_var[:, 1])))
        plt.legend()

        plt.subplot(222)
        plt.plot(LLSB_generated_bins_mean_var[:, 0], LLSB_generated_bins_mean_var[:, 1], label='mean')
        plt.hlines(y=[-0.08], xmin=0, xmax=100, linestyle='dotted')
        plt.title('LLSB')
        # plt.ylim((min(LLSB_generated_bins_mean_var[:, 1]), max(LLSB_generated_bins_mean_var[:, 1])))
        plt.legend()

        plt.tight_layout()
        plt.show()

        ###############
        # plotting variance
        plt.subplot(223)
        plt.plot(LT_generated_bins_mean_var[:, 0], LT_generated_bins_mean_var[:, 2], label='variance')
        plt.hlines(y=[0.05], xmin=0, xmax=100, linestyle='dotted')
        plt.title('LT')
        plt.legend()

        plt.subplot(224)
        plt.plot(CAM_generated_bins_mean_var[:, 0], CAM_generated_bins_mean_var[:, 2], label='variance')
        plt.hlines(y=[0.18], xmin=0, xmax=100, linestyle='dotted')
        plt.title('CAM')
        plt.legend()

        plt.subplot(221)
        plt.plot(LLSA_generated_bins_mean_var[:, 0], LLSA_generated_bins_mean_var[:, 2], label='variance')
        plt.hlines(y=[0.08], xmin=0, xmax=100, linestyle='dotted')
        plt.title('LLSA')
        plt.legend()

        plt.subplot(222)
        plt.plot(LLSB_generated_bins_mean_var[:, 0], LLSB_generated_bins_mean_var[:, 2], label='variance')
        plt.hlines(y=[0.07], xmin=0, xmax=100, linestyle='dotted')
        plt.title('LLSB')
        plt.legend()

        plt.tight_layout()
        plt.show()

        means = [-0.95, 0.31, -0.26, -0.09]
        variances = [0.05, 0.18, 0.08, 0.07]
        for i in range(len(LT_generated_bins_mean_var[:, 0])):
            delta_mean = LT_generated_bins_mean_var - means[0]

    if return_data:
        return total_error





def tow_visualizer(tows: list[pd.DataFrame], y_intended: list, name: str, ideal: bool):
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
    
    #set figure size
    #plt.figure(figsize=(15, 2))
    
    for i in range(len(y_intended)):
        CAM_centerline = tows[i]["center_CAM"] #take the centerline from CAM
        LT_y = tows[i]["y"] #take the y-position from LT
        intended_centerline = y_intended[i] #take the programmed y-value for a straight line
        centerline = CAM_centerline + LT_y + intended_centerline #calculate centerline in space by combining datatypes
        width = tows[i]["width_LLS_B"] #take the width from LLS B
        x = tows[i]["x"]  #take the x-position from LT
        
        
        #make the plots
        if i == 0:
            plt.plot(x, centerline, label="actual centerline", linestyle='dashed', color='grey') #plots the centerline
            plt.plot(x, centerline + 0.5 * width, label="actual tow", linestyle='solid', color='black') #plots the top edge
            plt.plot(x, centerline - 0.5 * width, linestyle='solid', color='black') #plots the bottom edge
        
        else: #do not assign a label to all other tows as this makes the legend unreadable
            plt.plot(x, centerline, linestyle='dashed', color='grey') #plots the centerline
            plt.plot(x, centerline + 0.5 * width, linestyle='solid', color='black') #plots the top edge
            plt.plot(x, centerline - 0.5 * width, linestyle='solid', color='black') #plots the bottom edge

        #plots the start end endlines of the tow
        plt.plot([x.iloc[0], x.iloc[0]], [centerline.iloc[0] - 0.5 * width.iloc[0], centerline.iloc[0] + 0.5 * width.iloc[0]], linestyle='solid', color='black')
        plt.plot([x.iloc[-1], x.iloc[-1]], [centerline.iloc[-1] - 0.5 * width.iloc[-1], centerline.iloc[-1] + 0.5 * width.iloc[-1]],linestyle='solid', color='black')
    
    if ideal == True:
        #plot the ideal tow (just a rectangle)
        plt.plot([0,1000], [tow_width_specified * 0.5, tow_width_specified * 0.5], color='green', label='ideal tow')
        plt.plot([0,1000], [-tow_width_specified * 0.5, -tow_width_specified * 0.5], color='green')
        plt.plot([0,0], [tow_width_specified * 0.5, -tow_width_specified * 0.5], color='green')
        plt.plot([1000,1000], [tow_width_specified * 0.5, -tow_width_specified * 0.5], color='green')
        plt.plot([0,1000], [0,0], color='green', linestyle='dashed', label='ideal centerline')


    # calculate the dimensions of the plots
    x_min = min(min(tow["x"].min() for tow in tows) - 50, -50)
    x_max = max(max(tow["x"].max() for tow in tows) + 50, 1050)
    y_min = min(min(tow["y"].min() for tow in tows) - 100, -50)
    y_max = max(max(tow["y"].max() for tow in tows) + 50, 1050)
    
    #plot info
    plt.xlabel("x-position [mm]")
    plt.ylabel("y-position [mm]")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid()
    plt.title(name)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()

def main():
    data = run_model()

    print("Hello world")

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else