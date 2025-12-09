'''This file is meant to run a large number of simulations at a time.'''

##############################################################################################################

# External imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import random
from dataclasses import dataclass
import seaborn as sns

# Internal imports
from constants import tow_width_specified, font_label, font_axis_ticks, figure_width, min_figure_height, color_exp, color_RS, color_RW, font_TNR, tick_length, tick_width, graph_box_thickness, font_legend
from Handling_ALL_Functions import get_synced_data
from D04_Model.Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path, generate_starting_error
from Data_ALL_statistics import main as real_hist, plot_histograms_separated, best_fit_distribution
from Model_ALL_RandomWalk import fit_random_walk, generate_random_walk, generate_RW_multitow, get_data
from D04_Model.Model_ALL_Simulation import generate_multitow_layout
from Model_ALL_RandomSampling import generate_RS_multitow

##############################################################################################################
"""Functions"""

def plot_RW_vs_exp_histograms(RW_tows: int=100, save_PDF: bool=True):
    """This function creates the plot of Random Walk vs Exponential Data for each individual sensor.
    This is equivalent to plot 2 in the paper atm."""

    # getting the experimental data per sensor
    LT_exp = get_data("LT", tows = list(np.arange(2, 32, 1)), format = "merged")[0]
    CAM_exp = get_data("CAM", tows = list(np.arange(2, 32, 1)), format = "merged")[0]
    LLSA_exp = get_data("LLS_A", tows = list(np.arange(2, 32, 1)), format = "merged")[0]
    LLSB_exp = get_data("LLS_B", tows = list(np.arange(2, 32, 1)), format = "merged")[0]

    # getting RW data per sensor
    LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params = fit_random_walk("LT", bins=100)
    CAM_steps, CAM_proposal_std, CAM_target_dist, CAM_dist, CAM_params = fit_random_walk("CAM", bins=250)
    LLS_A_steps, LLS_A_proposal_std, LLS_A_target_dist, LLS_A_dist, LLS_A_params = fit_random_walk("LLS_A", bins=100)
    LLS_B_steps, LLS_B_proposal_std, LLS_B_target_dist, LLS_B_dist, LLS_B_params = fit_random_walk("LLS_B", bins=100)

    LT_walk_data, CAM_walk_data, LLSA_walk_data, LLSB_walk_data = [], [], [], []
    for tow in range(RW_tows):
        LT_walk_data += generate_random_walk("LT", LT_steps, LT_proposal_std, LT_target_dist, LT_dist, LT_params,
                                            proposal_type="RWM")
        CAM_walk_data += generate_random_walk("CAM", CAM_steps, CAM_proposal_std, CAM_target_dist, CAM_dist, CAM_params,
                                             proposal_type="RWM")
        LLSA_walk_data += generate_random_walk("LLS_A", LLS_A_steps, LLS_A_proposal_std, LLS_A_target_dist, LLS_A_dist,
                                              LLS_A_params, proposal_type="RWM")
        LLSB_walk_data += generate_random_walk("LLS_B", LLS_B_steps, LLS_B_proposal_std, LLS_B_target_dist, LLS_B_dist,
                                              LLS_B_params, proposal_type="RWM")
    print(LT_exp, len(LT_exp))


    distribution_label_names = {
        'norm': 'Normal Distribution',
        'logistic': 'Logistic Distribution',
        'skewnorm': 'Skew-Normal Distribution',
        'genextreme': 'Generalized Extreme Value'}

    # getting the correct names for the pdf's for labels in the plot
    dist_data = [LT_dist, CAM_dist, LLS_A_dist, LLS_B_dist]
    dist_labels, y_pdfs = [], []
    for sensor_dist in dist_data:
        dist = sensor_dist
        dist_labels.append(distribution_label_names.get(dist.name, dist.name))

    #These actually have to be the distributions of the experimental data
    x_pdf_LT = np.linspace(-1.2, -0.6, 75)
    x_pdf_CAM = np.linspace(-0.6, 0.9, 188)
    x_pdf_LLS_A = np.linspace(-0.6, 0, 75)
    x_pdf_LLS_B = np.linspace(-0.3, 0.3, 75)

    #LT
    best_LT = best_fit_distribution(LT_exp, bins=100)
    best_LT_dist = best_LT['dist']
    params_LT = best_LT['params']
    mse_LT = best_LT['mse']
    shapes_LT = params_LT[:-2]    
    loc_LT = params_LT[-2]
    scale_LT = params_LT[-1]
    y_pdf_LT = best_LT_dist.pdf(x_pdf_LT, *shapes_LT, loc=loc_LT, scale=scale_LT)

    #CAM
    shrink_scale_factor_CAM = 0.9 # This factor is used to artifically stretch the skewnorm distribution to visually better fit the histogram
    best_CAM = best_fit_distribution(CAM_exp, bins=250, use_all_dist=True, shrink_scale_factor=shrink_scale_factor_CAM)
    best_CAM_dist = best_CAM['dist']
    params_CAM = best_CAM['params']
    mse_CAM = best_CAM['mse']
    shapes_CAM = params_CAM[:-2] 
    loc_CAM = params_CAM[-2]
    scale_CAM = params_CAM[-1]
    print(f'Scale_CAM: {scale_CAM}')
    y_pdf_CAM = best_CAM_dist.pdf(x_pdf_CAM, *shapes_CAM, loc=loc_CAM, scale=scale_CAM)

    #LSS A
    best_LLSA = best_fit_distribution(LLSA_exp, bins=100)
    best_LLSA_dist = best_LLSA['dist']
    params_LLSA = best_LLSA['params']
    mse_LLSA = best_LLSA['mse']
    shapes_LLSA = params_LLSA[:-2]    
    loc_LLSA = params_LLSA[-2]
    scale_LLSA = params_LLSA[-1]
    y_pdf_LLS_A = best_LLSA_dist.pdf(x_pdf_LLS_A, *shapes_LLSA, loc=loc_LLSA, scale=scale_LLSA)

    #LLS B
    best_LLSB = best_fit_distribution(LLSB_exp, bins=100)
    best_LLSB_dist = best_LLSB['dist']
    params_LLSB = best_LLSB['params']
    mse_LLSB = best_LLSB['mse']
    shapes_LLSB = params_LLSB[:-2]    
    loc_LLSB = params_LLSB[-2]
    scale_LLSB = params_LLSB[-1]
    y_pdf_LLS_B = best_LLSB_dist.pdf(x_pdf_LLS_B, *shapes_LLSB, loc=loc_LLSB, scale=scale_LLSB)

    # setting up the target distribution plots for later
    #x_pdf = np.linspace(-1.2, 1.2, 300)
    #y_pdf_LT = LT_target_dist(x_pdf)
    #y_pdf_CAM = CAM_target_dist(x_pdf)
    #y_pdf_LLS_A = LLS_A_target_dist(x_pdf)
    #y_pdf_LLS_B = LLS_B_target_dist(x_pdf)

    def annotate_mean_std(ax, data, stripe_height=0.08, lw=1.5, show_debug=False):
        """
        Annotate `ax` with:
        - Hollow red circle at mean (y=0 in data coordinates)
        - Horizontal red line from mean-std to mean+std at y=0
        - Two vertical red stripes at mean±std, centered at y=0

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        data : array-like
        stripe_height : float
            Total stripe height measured in AXES fraction (0..1). Default 0.08.
        lw : float
            Line/edge width for stripes and marker edge width.
        show_debug : bool
            If True prints internal coords for debugging.
        """

        mean = float(np.mean(data))
        std = float(np.std(data))

        # Lower spines so annotations are on top
        for spine in ax.spines.values():
            spine.set_zorder(0)
        ax.set_axisbelow(False)

        # Ensure y=0 is in the visible range
        ymin, ymax = ax.get_ylim()
        if 0 < ymin:
            ymin = 0
        if 0 > ymax:
            ymax = 0
        ax.set_ylim(ymin, ymax)

        # 1) Hollow circle at (mean, 0)
        ax.plot(mean, 0,
                marker='o', markersize=7,
                markerfacecolor='none', markeredgecolor='red',
                markeredgewidth=lw, zorder=1000, clip_on=False)

        # 2) Horizontal ±1σ line at y=0
        ax.hlines(0, mean - std, mean + std,
                colors='red', linewidth=2, zorder=900, clip_on=False)

        # 3) Compute y=0 in axes fraction
        disp_x0, disp_y0 = ax.transData.transform((mean, 0))
        _, y0_axes = ax.transAxes.inverted().transform((disp_x0, disp_y0))

        if show_debug:
            print(f"mean={mean:.4g}, std={std:.4g}, y0_axes={y0_axes:.4g}")

        # 4) Compute stripe top/bottom in axes fraction around y0_axes
        half = stripe_height / 2.0
        y_low = y0_axes - half
        y_high = y0_axes + half

        # 5) Draw vertical stripes with blended transform
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for x in (mean - std, mean + std):
            ax.plot([x, x], [y_low, y_high],
                    transform=trans,
                    color='red', linewidth=lw,
                    zorder=950, clip_on=False)

    # --------PLotting----------
    plt.rc('font', family=font_TNR)
    im0 = image.imread('Figures/robotinacc.jpg')
    im1 = image.imread('Figures/tapelatmvmt.jpg')
    im2 = image.imread('Figures/tape width.jpg')
    im3 = image.imread('Figures/tapecompaction.jpg')

    fig, axs = plt.subplots(4, 1, figsize=(figure_width, 4*min_figure_height), sharex=True)

    # LT plot
    axs[0].imshow(im0, aspect='auto', extent=(0.873, 0.967, 0.525, 0.925), transform=axs[0].transAxes)
    axs[0].hist(LT_exp, color=color_exp, bins=100, density=True, alpha=0.6, label="Experimental data")
    axs[0].hist(LT_walk_data, color=color_RW, bins=100, density=True, alpha=0.6, label="RWM simulation data")
    axs[0].plot(x_pdf_LT, y_pdf_LT, color='yellow', label="Distribution fits")
    annotate_mean_std(axs[0], LT_exp)
    axs[0].set_xlabel("Error, robot position", size=font_label)
    axs[0].set_ylabel("Density", size=font_label)
    axs[0].set_xticks(np.linspace(-1.2, 1.2, 9))
    axs[0].xaxis.set_tick_params(labelbottom=True)
    #ax = plt.gca()

    # CAM plot
    axs[1].imshow(im1, aspect='auto', extent=(0.873, 0.967, 0.525, 0.925), transform=axs[1].transAxes)
    axs[1].hist(CAM_exp, color=color_exp, bins=250, density=True, alpha=0.6)
    axs[1].hist(CAM_walk_data, color=color_RW, bins=250, density=True, alpha=0.6)
    axs[1].plot(x_pdf_CAM, y_pdf_CAM, color='yellow')
    annotate_mean_std(axs[1], CAM_exp)
    axs[1].set_xlabel("Error, tape lateral movement", size=font_label)
    axs[1].set_ylabel("Density", size=font_label)
    axs[1].set_xticks(np.linspace(-1.2, 1.2, 9))
    axs[1].xaxis.set_tick_params(labelbottom=True)

    # LLS_A 
    axs[2].imshow(im2, aspect='auto', extent=(0.873, 0.967, 0.525, 0.925), transform=axs[2].transAxes)
    axs[2].hist(LLSA_exp, color=color_exp, bins=100, density=True, alpha=0.6)
    axs[2].hist(LLSA_walk_data, color=color_RW, bins=100, density=True, alpha=0.6)
    axs[2].plot(x_pdf_LLS_A, y_pdf_LLS_A, color='yellow')
    annotate_mean_std(axs[2], LLSA_exp)
    axs[2].set_xlabel("Error, tape width before compaction", size=font_label)
    axs[2].set_ylabel("Density", size=font_label)
    axs[2].set_xticks(np.linspace(-1.2, 1.2, 9))
    axs[2].xaxis.set_tick_params(labelbottom=True)

    # LLS_B plot
    axs[3].imshow(im3, aspect='auto', extent=(0.873, 0.967, 0.525, 0.925), transform=axs[3].transAxes)
    axs[3].hist(LLSB_exp, color=color_exp, bins=100, density=True, alpha=0.6)
    axs[3].hist(LLSB_walk_data, color=color_RW, bins=100, density=True, alpha=0.6)
    axs[3].plot(x_pdf_LLS_B, y_pdf_LLS_B, color='yellow')
    annotate_mean_std(axs[3], LLSB_exp)
    axs[3].set_xlabel("Error, tape width after compaction", size=font_label)
    axs[3].set_ylabel("Density", size=font_label)
    axs[3].set_xticks(np.linspace(-1.2, 1.2, 9))
    axs[3].xaxis.set_tick_params(labelbottom=True)

    for i, ax in enumerate(axs):
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(top=True, bottom=True, left=True, right=True, direction='in', length=tick_length, width=tick_width)
        for spine in ax.spines.values():
            spine.set_linewidth(graph_box_thickness)
            spine.set_edgecolor('black')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(font_TNR)
            label.set_fontsize(font_axis_ticks)

    # Get custom order for legend entries
    handles, labels = axs[0].get_legend_handles_labels()
    desired_order = [0, 2, 1]   # <--- change index order here
    handles = [handles[i] for i in desired_order]
    labels = [labels[i] for i in desired_order]
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=1, fontsize=font_legend, frameon=True)

    if save_PDF == True:
        plt.savefig("source wise validation_310 tows.pdf", format="pdf", bbox_inches="tight")

    plt.show()

def plot_histograms(real_data: pd.DataFrame, sim_data: list, RW_data: list, title: str, bin_widths: list[float] = None):
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
        real_data[0],
        real_data[1],
        real_data[2],
        real_data[3]]

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

        ax.hist(clean, bins=bins, alpha=0.4, density=True, label='Experimental')
        ax.hist(sim_data[i], bins=bins, alpha=0.4, density=True, label='D04-Model')
        ax.hist(RW_data[i], bins=bins, alpha=0.4, density=True, label='Random Walk', color='lightgreen')
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
        RW_mean = np.array(RW_data[i]).mean()
        RW_std = np.array(RW_data[i]).std()

        print(f'{titles[i]} Experimental mean/std = {mean_val}/{std_val}')
        print(f'{titles[i]} Model mean/std = {sim_mean}/{sim_std}')
        print(f'{titles[i]} RW mean/std = {RW_mean}/{RW_std}')

        ax.axvline(mean_val, color='purple', linestyle='-', label='Experimental Mean', linewidth=1)
        ax.axvline(sim_mean, color='red', linestyle='-', label='Model Mean', linewidth=1)     # + '\n' + rf'$\sigma$ = {sim_std:.2f}'
        ax.axvline(RW_mean, color='darkgreen', linestyle='-', label='Random Walk Mean', linewidth=1)
        ax.axvline(0, color='black', linestyle='dashed')

        #ax[row, col].set_title(titles[i])
        ax.set_xlabel(titles[i], fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter('{x:0<3.1f}')

        plt.xticks(np.linspace(-1.2, 1.2, 9))
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

def run_model(generate_varying_bin_plots: bool=False, return_data: bool=True):
    '''This function executes all steps needed to plot the
        histograms for each sensor, including experimental and real.
        This involves first getting the data and then plotting.'''

    real_LT_data = pd.concat((get_synced_data(t, "LT") for t in range(3, 32, 2)), ignore_index=True)
    real_CAM_data = pd.concat((get_synced_data(t, "CAM") for t in range(3, 32, 2)), ignore_index=True)
    real_LLSA_data = pd.concat((get_synced_data(t, "LLS_A") for t in range(3, 32, 2)), ignore_index=True)
    real_LSSB_data = pd.concat((get_synced_data(t, "LLS_B") for t in range(3, 32, 2)), ignore_index=True)

    real_LT_data = real_LT_data["error_LT"]
    real_CAM_data = real_CAM_data["error_CAM"]
    real_LLSA_data = real_LLSA_data["error_LLS_A"]
    real_LSSB_data = real_LSSB_data["error_LLS_B"]
    print('experimental data has been collected.')

    #if use_saved:
    #    _save_path = "Script\\"
    #    LT_short_name = 'LT_Dist_Data'
    #    CAM_short_name = 'CAM_Dist_Data'
    #    LLSA_short_name = 'LLSA_Dist_Data'
    #    LLSB_short_name = 'LLSB_Dist_Data'
#
    #    if save_data:
    #        save_distribution_data(_save_path, LT_short_name, CAM_short_name, LLSA_short_name, LLSB_short_name)
    #    LT_dist = pd.read_pickle(_save_path + LT_short_name + ".pkl")
    #    CAM_dist = pd.read_pickle(_save_path + CAM_short_name + ".pkl")
    #    LLSA_dist = pd.read_pickle(_save_path + LLSA_short_name + ".pkl")
    #    LLSB_dist = pd.read_pickle(_save_path + LLSB_short_name + ".pkl")

    # -----code for creating the data with the D04-model---------
    LT_generated_bins_mean_var = []
    CAM_generated_bins_mean_var = []
    LLSA_generated_bins_mean_var = []
    LLSB_generated_bins_mean_var = []
    generated_bins_mean_var = [LT_generated_bins_mean_var, CAM_generated_bins_mean_var, LLSA_generated_bins_mean_var, LLSB_generated_bins_mean_var]
    for num_bins in range(130, 131, 5):
        #num_bins = 30
        rs = 42
        LT_dist = consecutive_error('LT', used_tows=list(range(2, 32, 2)), num_bins=num_bins)
        CAM_dist = consecutive_error('CAM', used_tows=list(range(2, 32, 2)), num_bins=num_bins)
        LLSA_dist = consecutive_error('LLS_A', used_tows=list(range(2, 32, 2)), num_bins=num_bins)
        LLSB_dist = consecutive_error('LLS_B', used_tows=list(range(2, 32, 2)), num_bins=num_bins)

        # ------ This section generates the simulated data used for the comparison ------
        n_runs = 50     # TODO: increase once finalising figures
        total_data = []
        n_steps = 320
        total_D04_error = [[], [], [], []]
        for run in range(n_runs):
            # starting position data
            start_cam = generate_starting_error("CAM")
            start_lt = generate_starting_error("LT")
            start_llsa = generate_starting_error("LLS_A")
            start_llsb = generate_starting_error("LLS_B")

            # generating data
            LT_error_list = generate_error_path(start_lt, n_steps, LT_dist[0], LT_dist[1], LT_dist[2], LT_dist[-3], LT_dist[-2],
                                                LT_dist[-1])
            CAM_error_list = generate_error_path(start_cam, n_steps, CAM_dist[0], CAM_dist[1], CAM_dist[2], CAM_dist[-3], CAM_dist[-2],
                                                 CAM_dist[-1])
            LLSA_error_list = generate_error_path(start_llsa, n_steps,LLSA_dist[0], LLSA_dist[1], LLSA_dist[2], LLSA_dist[-3],
                                                  LLSA_dist[-2], LLSA_dist[-1])
            LLSB_error_list = generate_error_path(start_llsb, n_steps, LLSB_dist[0], LLSB_dist[1], LLSB_dist[2], LLSB_dist[-3],
                                                  LLSB_dist[-2], LLSB_dist[-1])


            total_D04_error[0] = (total_D04_error[0] + list(LT_error_list))
            total_D04_error[1] = (total_D04_error[1] + list(CAM_error_list))
            total_D04_error[2] = (total_D04_error[2] + list(LLSA_error_list))
            total_D04_error[3] = (total_D04_error[3] + list(LLSB_error_list))


            #generated_data = []
            #x = 0
            #for i in range(len(LT_error_list)):
            #    centerline_error = LT_error_list[i] + CAM_error_list[i]
            #    width_error = LLSB_error_list[i]
            #    x +=dx
            #    generated_data.append([x, centerline_error, width_error])
            #
            #generated_data = pd.DataFrame(generated_data, columns = ['x', 'error'])
        print('DO4-model data has been generated.')

        # ---------code for generating the random walk data----------
        n_runs = 50     # TODO: increase once finalising figures
        total_RW_error = [[], [], [], []]
        for run in range(n_runs):
            LT_RW_data = generate_random_walk("LT", 'RWM')
            CAM_RW_data = generate_random_walk("CAM", 'RWM')
            LLSA_RW_data = generate_random_walk("LLS_A", 'RWM')
            LLSB_RW_data = generate_random_walk("LLS_B", 'RWM')

            total_RW_error[0] = (total_RW_error[0] + list(LT_RW_data))
            total_RW_error[1] = (total_RW_error[1] + list(CAM_RW_data))
            total_RW_error[2] = (total_RW_error[2] + list(LLSA_RW_data))
            total_RW_error[3] = (total_RW_error[3] + list(LLSB_RW_data))
        print('Random walk data has been generated.')


        plot_histograms(
            [real_LLSA_data, real_LSSB_data, real_LT_data, real_CAM_data],
            [total_D04_error[2], total_D04_error[3], total_D04_error[0], total_D04_error[1]],
            [total_RW_error[2], total_RW_error[3], total_RW_error[0], total_RW_error[1]],
            title="Sensor Error Histograms (ALL TOWS BUT NOT SPACE SYNCED), num_bins=" + str(num_bins),
            bin_widths=[0.01, 0.01, 0.005, 0.03]
        )

        for i in range(4):
            mean = float(np.array(total_D04_error[i]).mean())
            variance = float(np.array(total_D04_error[i]).std())
            generated_bins_mean_var[i].append([num_bins, mean, variance])

    print(f'LT errors: {LT_generated_bins_mean_var}')
    print(f'CAM errors: {CAM_generated_bins_mean_var}')
    print(f'LLSA errors: {LLSA_generated_bins_mean_var}')
    print(f'LLSB errors: {LLSB_generated_bins_mean_var}')

    print('total number of data points = ', len(total_D04_error[0]))
    #plt.subplot(223)
    #plt.hist(total_error[0], bins=50)
    #plt.title('LT')
    #plt.subplot(224)
    #plt.hist(total_error[1], bins=50)
    #plt.title('CAM')
    #plt.subplot(221)
    #plt.hist(total_error[2], bins=50)
    #plt.title('LLSA')
    #plt.subplot(222)
    #plt.hist(total_error[3], bins=50)
    #plt.title('LLSB')
#
    #plt.tight_layout()
    #plt.show()
#
    #real_hist()

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
        return total_D04_error

def Gap_Histogram(tows_simulated: int, plot: bool=False):
    # ------getting experimental data---------
    real_gap_data = []
    for i in range(1, 31):
        added_gap_data = list(get_synced_data(i, 'Traverse')['Gap_gap'])
        real_gap_data = real_gap_data + added_gap_data
    #print(real_gap_data)
    #print('real printed', len(real_gap_data))
    experimental_mean = np.mean(real_gap_data)
    experimental_std = np.std(real_gap_data)
    # real_gap_data = filter(lambda x: 4 >= x >= 8, real_gap_data)
    experimental_90th_percentile = np.percentile(real_gap_data, 90)
    experimental_99th_percentile = np.percentile(real_gap_data, 99)

    # -------generating D04-model data--------
    gap_overlap_df, _, _, _, _ = generate_multitow_layout(num_tows=tows_simulated, tow_spacing_mm=12.5)
    gap_overlap_df = np.array(gap_overlap_df)
    D04_gap_data = []
    for i in range (tows_simulated-1):
        D04_gap_data = D04_gap_data + list(gap_overlap_df[:, i])
    #print(D04_gap_data)
    #print('D04 printed', len(D04_gap_data))
    D04_mean = np.mean(gap_overlap_df)
    D04_std = np.std(gap_overlap_df)
    D04_90th_percentile = np.percentile(gap_overlap_df, 90)
    D04_99th_percentile = np.percentile(gap_overlap_df, 99)

    # -------generating Random Walk data--------
    RW_gap_df, _, _, _, _, _ = generate_RW_multitow(num_tows=tows_simulated, tow_spacing_mm=12.5)
    RW_gap_df = np.array(RW_gap_df)
    RW_gap_data = []
    for i in range (tows_simulated-1):
        RW_gap_data = RW_gap_data + list(RW_gap_df[:, i])
        #print(RW_gap_data, i)
    #print(RW_gap_data)
    #print('RW printed', len(RW_gap_data))
    RW_mean = np.mean(RW_gap_data)
    RW_std = np.std(RW_gap_data)
    RW_90th_percentile = np.percentile(RW_gap_data, 90)
    RW_99th_percentile = np.percentile(RW_gap_data, 99)

    # -------generating Random Sampling data--------
    RS_gap_df, _ = generate_RS_multitow(num_tows=tows_simulated, tow_spacing_mm=12.5)
    RS_gap_df = np.array(RS_gap_df)
    RS_gap_data = []
    for i in range (tows_simulated-1):
        RS_gap_data = RS_gap_data + list(RS_gap_df[:, i])
        #print(RS_gap_data, i)
    #print(RS_gap_data)
    #print('RS printed', len(RS_gap_data))
    RS_mean = np.mean(RS_gap_data)
    RS_std = np.std(RS_gap_data)
    RS_90th_percentile = np.percentile(RS_gap_data, 90)
    RS_99th_percentile = np.percentile(RS_gap_data, 99)

    print(f'Experimental mean/std/90th/99th = {experimental_mean}/{experimental_std}/{experimental_90th_percentile}/{experimental_99th_percentile}')
    print(f'D04 mean/std/90th/99th = {D04_mean}/{D04_std}/{D04_90th_percentile}/{D04_99th_percentile}')
    print(f'RW mean/std/90th/99th = {RW_mean}/{RW_std}/{RW_90th_percentile}/{RW_99th_percentile}')
    print(f'RS mean/std/90th/99th = {RS_mean}/{RS_std}/{RS_90th_percentile}/{RS_99th_percentile}')

    gap_center = 12.5-6.35
    bins = [0]+list(np.linspace(gap_center-1.2, gap_center+1.2, 100+1))+[10]

    if plot:
        # plots
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.hist(real_gap_data, label='Experimental', bins=bins, alpha=0.55, density=True)     # bins=[0]+list(np.linspace(6.15-1.2, 6.15+1.2, 80+1))+[10]
        ax.hist(D04_gap_data, label='Model', bins=bins, alpha=0.55, density=True)
        ax.hist(RW_gap_data, label='Random Walk', bins=bins, alpha=0.55, density=True, color='green')
        ax.axvline(experimental_mean, color='purple', linestyle='-', label='Experimental Mean')
        ax.axvline(D04_mean, color='red', linestyle='-', label='D04-Model Mean')
        ax.axvline(RW_mean, color='darkgreen', linestyle='-', label='RW Mean')
        ax.set_xlabel("Gap (mm)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.axvline(gap_center, color='black', linestyle='dashed', label='Ideal Gap')
        # plt.title(f"Gaps")
        ax.set_xlim(gap_center-1.2, gap_center+1.2)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.legend(fontsize=10)

        plt.xticks(np.linspace(gap_center-1.2, gap_center+1.2, 9))
        #plt.grid(True)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()
    
    return real_gap_data, D04_gap_data, RW_gap_data, RS_gap_data, experimental_mean, D04_mean, RW_mean, RS_mean, gap_center, bins

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
        
        else: #do not assign a label to all other tows as this would make the legend unreadable
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

def KDE_curves(tows_simulated: int):
    """Function to plot probability density functions using KDE plotting.
        Author: ChatGPT"""
    # Obtain data
    real_gap_data, D04_gap_data, RW_gap_data, RS_gap_data, real_mean, D04_mean, RW_mean, RS_mean, ideal_gap_center, bins = Gap_Histogram(tows_simulated)
    
    # In case we want to average over multiple laminates, not finished yet
    # 
    #RW_gap_data_list = []
    #RS_gap_data_list = []
    #RW_mean_list = []
    #RS_mean_list = []
    #for i in range(n_laminates):
    #    _, _, RW_gap_data, RS_gap_data, _, _, RW_mean, RS_mean, _, _ = Gap_Histogram(tows_simulated)
    #    RW_gap_data_list.append(RW_gap_data)
    #    RS_gap_data_list.append(RS_gap_data)
    #    RW_mean_list.append(RW_mean)
    #    RS_mean_list.append(RS_mean)
    
    
    plt.figure(figsize=(10,6))

    # Plot histograms
    plt.hist(real_gap_data, bins=bins, density=True, alpha=0.6, color="blue", label="Experimental")
    #plt.hist(D04_gap_data, bins=bins, density=True, alpha=0.2, color="orange", label="D04", hatch='o')
    plt.hist(RW_gap_data, bins=bins, density=True, alpha=0.6, color="green", label="Random Walk")
    plt.hist(RS_gap_data, bins=bins, density=True, alpha=0.6, color="orange", label="Random Sampling")
    
    # Plot smooth KDE curves
    #sns.kdeplot(real_gap_data, label="Experimental", color="blue", linewidth=2)
    #sns.kdeplot(D04_gap_data, label="D04", color="orange", linewidth=2)
    #sns.kdeplot(RW_gap_data, label="Random Walk", color="green", linewidth=2)
    #sns.kdeplot(RS_gap_data, label="Random Sampling", color="orange", linewidth=2)

    # Plot vertical lines for means and ideal gap
    #plt.axvline(real_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
    #plt.axvline(D04_mean, color="orange", linestyle="--", linewidth=1, label="D04 Mean")
    #plt.axvline(RW_mean, color="green", linestyle="--", linewidth=1, label="RW Mean")
    #plt.axvline(RS_mean, color="orange", linestyle="--", linewidth=1, label="RS Mean")
    plt.axvline(ideal_gap_center, color="black", linestyle=":", linewidth=1, label="Ideal Gap")

    # Labels and layout
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Gap (mm)", fontsize=12, fontname='Times New Roman')
    plt.ylabel("Probability Density", fontsize=12, fontname='Times New Roman')
    plt.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncols=4)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_edgecolor('black')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(top=True, bottom=True, left=True, right=True, direction='in', length=8, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(10)
    plt.tight_layout()
    plt.show()

def model_distribution_figures(tows_simulated: int, plottype: str, save_PDF: bool=False):
    """Generate 1 figure with 3 plots. 1: Experimental vs. D04. 2: Experimental vs. RW. 3: Experimental vs. Random Sampling.
       Plottype can be "single", "single no D04, "separate" or "separate no D04".
       Author: Martijn van der Voort"""
    
    print(f"DEBUG: plottype={repr(plottype)}")
    # Check for plottype
    if plottype != "single" and plottype != "single no D04" and plottype != "separate" and plottype != "separate no D04":
        raise ValueError(f'The provided plottype does not exist. Choose "single", "single no D04", "separate" or "separate no D04"')

    # ------getting experimental data---------
    experimental_gap_data = []
    for i in range(1, 31):
        added_gap_data = list(get_synced_data(i, 'Traverse')['Gap_gap'])
        experimental_gap_data = experimental_gap_data + added_gap_data
    experimental_mean = np.mean(experimental_gap_data)
    experimental_std = np.std(experimental_gap_data)
    experimental_90th_percentile = np.percentile(experimental_gap_data, 90)
    experimental_99th_percentile = np.percentile(experimental_gap_data, 99)

    # -------generating D04-model data--------
    gap_overlap_df, _, _, _, _ = generate_multitow_layout(num_tows=tows_simulated, tow_spacing_mm=12.5)
    gap_overlap_df = np.array(gap_overlap_df)
    D04_gap_data = []
    for i in range (tows_simulated-1):
        D04_gap_data = D04_gap_data + list(gap_overlap_df[:, i])
    D04_mean = np.mean(D04_gap_data)
    D04_std = np.std(D04_gap_data)
    D04_90th_percentile = np.percentile(D04_gap_data, 90)
    D04_99th_percentile = np.percentile(D04_gap_data, 99)

    # -------generating Random Walk data--------
    RW_gap_df, _, _, _, _, _ = generate_RW_multitow(num_tows=tows_simulated, tow_spacing_mm=12.5)
    RW_gap_df = np.array(RW_gap_df)
    RW_gap_data = []
    for i in range (tows_simulated-1):
        RW_gap_data = RW_gap_data + list(RW_gap_df[:, i])
    RW_mean = np.mean(RW_gap_data)
    RW_std = np.std(RW_gap_data)
    RW_90th_percentile = np.percentile(RW_gap_data, 90)
    RW_99th_percentile = np.percentile(RW_gap_data, 99)

    #-------generating Random Sampling data--------
    RS_gap_df, _, _, _ = generate_RS_multitow(num_tows=tows_simulated, tow_spacing_mm=12.5)
    RS_gap_df = np.array(RS_gap_df)
    RS_gap_data = []
    for i in range (tows_simulated-1):
        RS_gap_data = RS_gap_data + list(RS_gap_df[:, i])
    RS_mean = np.mean(RS_gap_data)
    RS_std = np.std(RS_gap_data)
    RS_90th_percentile = np.percentile(RS_gap_data, 90)
    RS_99th_percentile = np.percentile(RW_gap_data, 99)

    #-----print statements
    print(f'Experimental mean/std/90th/99th = {experimental_mean}/{experimental_std}/{experimental_90th_percentile}/{experimental_99th_percentile}')
    print(f'D04 mean/std/90th/99th = {D04_mean}/{D04_std}/{D04_90th_percentile}/{D04_99th_percentile}')
    print(f'RW mean/std/90th/99th = {RW_mean}/{RW_std}/{RW_90th_percentile}/{RW_99th_percentile}')
    print(f'RS mean/std/90th/99th = {RS_mean}/{RS_std}/{RS_90th_percentile}/{RS_99th_percentile}')

    #-------calculating plot parameters---------
    ideal_gap_center = 12.5-6.35
    bins = [0]+list(np.linspace(ideal_gap_center-1.2, ideal_gap_center+1.2, 100+1))+[10]

    #--------generating plot--------
    if plottype == "single":
        plt.figure(figsize=(12, 8))

        plt.subplot(311)
        plt.hist(experimental_gap_data, bins=bins, density=True, alpha=0.2, color="blue", label="Experimental")
        plt.hist(D04_gap_data, bins=bins, density=True, alpha=0.2, color="orange", label="D04")
        sns.kdeplot(experimental_gap_data, label="Experimental", color="blue", linewidth=2)
        sns.kdeplot(D04_gap_data, label="D04", color="orange", linewidth=2)
        plt.axvline(experimental_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
        plt.axvline(D04_mean, color="orange", linestyle="--", linewidth=1, label="D04 Mean")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
        plt.ylabel("Probability Density", fontsize=font_label)
        plt.legend(fontsize=font_legend)
        
        plt.subplot(312)
        plt.hist(experimental_gap_data, bins=bins, density=True, alpha=0.2, color="blue", label="Experimental")
        plt.hist(RW_gap_data, bins=bins, density=True, alpha=0.2, color="red", label="RW")
        sns.kdeplot(experimental_gap_data, label="Experimental", color="blue", linewidth=2)
        sns.kdeplot(RW_gap_data, label="RW", color="red", linewidth=2)
        #plt.axvline(experimental_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
        #plt.axvline(RW_mean, color="red", linestyle="--", linewidth=1, label="RW Mean")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
        plt.ylabel("Probability Density", fontsize=font_label)
        plt.legend(fontsize=font_legend)
        
        plt.subplot(313)
        plt.hist(experimental_gap_data, bins=bins, density=True, alpha=0.2, color="blue", label="Experimental")
        plt.hist(RS_gap_data, bins=bins, density=True, alpha=0.2, color="green", label="RS")
        sns.kdeplot(experimental_gap_data, label="Experimental", color="blue", linewidth=2)
        sns.kdeplot(RS_gap_data, label="RS", color="green", linewidth=2)
        #plt.axvline(experimental_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
        #plt.axvline(RS_mean, color="green", linestyle="--", linewidth=1, label="RS Mean")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
        plt.legend(fontsize=font_legend)
        plt.xlabel("Gap (mm)", fontsize=font_label)
        
        plt.ylabel("Probability Density", fontsize=font_label)
        plt.show()

    if plottype == "single no D04":
        plt.figure(figsize=(figure_width, 3*min_figure_height))

        # Shift data so that ideal gap is at x = 0
        bins = np.linspace(-1.3, 1.3, 101)
        exp_shift = np.array(experimental_gap_data) - ideal_gap_center
        rw_shift  = np.array(RW_gap_data) - ideal_gap_center
        rs_shift  = np.array(RS_gap_data) - ideal_gap_center
        
        plt.subplot(211)
        plt.hist(exp_shift, bins=bins, density=True, alpha=0.6, color=color_exp, label="Experimental")
        plt.hist(rw_shift, bins=bins, density=True, alpha=0.6, color=color_RW, label="MCMC simulation")
        #sns.kdeplot(experimental_gap_data, label="Experimental", color="blue", linewidth=2)
        #sns.kdeplot(RW_gap_data, label="RW", color="green", linewidth=2)
        #plt.axvline(experimental_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
        #plt.axvline(RW_mean, color="green", linestyle="--", linewidth=1, label="RW Mean")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)

        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = [font_TNR]
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['xtick.labelsize'] = font_axis_ticks
        mpl.rcParams['ytick.labelsize'] = font_axis_ticks
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Gap (mm)", fontsize=font_label, fontname=font_TNR)
        plt.ylabel("Density", fontsize=font_label, fontname=font_TNR)
        plt.legend(fontsize=font_legend, loc='upper right', frameon=False, ncols=1)
        x_min, x_max = -1.2, 1.2   

        ax = plt.gca()
        ax.set_xlim(x_min - 0.1, x_max + 0.1)              
        ax.set_xticks(np.linspace(x_min, x_max, 9))
        for spine in ax.spines.values():
            spine.set_linewidth(graph_box_thickness)
            spine.set_edgecolor('black')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(top=True, bottom=True, left=True, right=True,
                    direction='in', length=tick_length, width=tick_width)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(font_TNR)
            label.set_fontsize(font_label)
        ax.axvline(0, color='black', linestyle='-.', linewidth=1)
        ax.text(-0.02, ax.get_ylim()[1] * 0.95, "Ideal Gap", ha='right', va='top',
                fontsize=font_label, fontname=font_TNR)
        #plt.tight_layout()
        #plt.show()

        #plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
        #plt.ylabel("Probability Density", fontsize=font_medium)
        #plt.grid(alpha=0.5, linestyle="-")
        #plt.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -0.35))
        
        plt.subplot(212)
        plt.hist(exp_shift, bins=bins, density=True, alpha=0.6, color=color_exp, label="Experimental")
        plt.hist(rs_shift, bins=bins, density=True, alpha=0.6, color=color_RS, label="MC simulation")
        #sns.kdeplot(experimental_gap_data, label="Experimental", color="blue", linewidth=2)
        #sns.kdeplot(RS_gap_data, label="RS", color="orange", linewidth=2)
        #plt.axvline(experimental_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
        #plt.axvline(RS_mean, color="orange", linestyle="--", linewidth=1, label="RS Mean")
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)

        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = [font_TNR]
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['xtick.labelsize'] = font_axis_ticks
        mpl.rcParams['ytick.labelsize'] = font_axis_ticks
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Gap (mm)", fontsize=font_label, fontname=font_TNR)
        plt.ylabel("Density", fontsize=font_label, fontname=font_TNR)
        plt.legend(fontsize=font_legend, loc='upper right', frameon=False, ncols=1)
        x_min, x_max = -1.2, 1.2

        ax = plt.gca()
        ax.set_xlim(x_min - 0.1, x_max + 0.1)
        ax.set_xticks(np.linspace(x_min, x_max, 9))
        for spine in ax.spines.values():
            spine.set_linewidth(graph_box_thickness)
            spine.set_edgecolor('black')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(top=True, bottom=True, left=True, right=True,
                    direction='in', length=tick_length, width=tick_width)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(font_TNR)
            label.set_fontsize(font_label)
        ax.axvline(0, color='black', linestyle='-.', linewidth=1)
        ax.text(-0.02, ax.get_ylim()[1] * 0.95, "Ideal Gap", ha='right', va='top',
                fontsize=font_label, fontname=font_TNR)
        plt.subplots_adjust(hspace=0.3) 
        
        if save_PDF == True:
            plt.savefig("KDE histograms of 2 algorithms.pdf", format="pdf",bbox_inches='tight')
        plt.show()

        #plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
        #plt.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -0.35))
        #plt.xlabel("Gap (mm)", fontsize=font_medium)
        #plt.grid(alpha=0.5, linestyle="-")
        
        #plt.ylabel("Probability Density", fontsize=font_medium)
        #plt.show()

    if plottype == "separate":
        
        #D04
        plt.figure(figsize=(10,6))
        plt.hist(experimental_gap_data, bins=bins, density=True, alpha=0.2, color="blue", label="Experimental")
        plt.hist(D04_gap_data, bins=bins, density=True, alpha=0.2, color="orange", label="D04")
        sns.kdeplot(experimental_gap_data, label="Experimental", color="blue", linewidth=2)
        sns.kdeplot(D04_gap_data, label="D04", color="orange", linewidth=2)
        plt.axvline(experimental_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
        plt.axvline(D04_mean, color="orange", linestyle="--", linewidth=1, label="D04 Mean")
        plt.axvline(ideal_gap_center, color="black", linestyle=":", linewidth=1, label="Ideal Gap")
        plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Gap (mm)", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

        #RW
        plt.figure(figsize=(10,6))
        plt.hist(experimental_gap_data, bins=bins, density=True, alpha=0.2, color="blue", label="Experimental")
        plt.hist(RW_gap_data, bins=bins, density=True, alpha=0.2, color="red", label="RW")
        sns.kdeplot(experimental_gap_data, label="Experimental", color="blue", linewidth=2)
        sns.kdeplot(RW_gap_data, label="RW", color="red", linewidth=2)
        plt.axvline(experimental_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
        plt.axvline(RW_mean, color="red", linestyle="--", linewidth=1, label="RW Mean")
        plt.axvline(ideal_gap_center, color="black", linestyle=":", linewidth=1, label="Ideal Gap")
        plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Gap (mm)", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

        #RS
        plt.figure(figsize=(10,6))
        plt.hist(experimental_gap_data, bins=bins, density=True, alpha=0.2, color="blue", label="Experimental")
        plt.hist(RS_gap_data, bins=bins, density=True, alpha=0.2, color="green", label="RS")
        sns.kdeplot(experimental_gap_data, label="Experimental", color="blue", linewidth=2)
        sns.kdeplot(RS_gap_data, label="RS", color="green", linewidth=2)
        plt.axvline(experimental_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
        plt.axvline(RS_mean, color="green", linestyle="--", linewidth=1, label="RS Mean")
        plt.axvline(ideal_gap_center, color="black", linestyle=":", linewidth=1, label="Ideal Gap")
        plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Gap (mm)", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    if plottype == "separate no D04":

        #RW
        plt.figure(figsize=(10,6))
        plt.hist(experimental_gap_data, bins=bins, density=True, alpha=0.2, color="blue", label="Experimental")
        plt.hist(RW_gap_data, bins=bins, density=True, alpha=0.2, color="green", label="RW")
        #sns.kdeplot(experimental_gap_data, label="Experimental", color="blue", linewidth=2)
        #sns.kdeplot(RW_gap_data, label="RW", color="green", linewidth=2)
        plt.axvline(experimental_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
        plt.axvline(RW_mean, color="green", linestyle="--", linewidth=1, label="RW Mean")
        plt.axvline(ideal_gap_center, color="black", linestyle=":", linewidth=1, label="Ideal Gap")
        plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Gap (mm)", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

        #RS
        plt.figure(figsize=(10,6))
        plt.hist(experimental_gap_data, bins=bins, density=True, alpha=0.2, color="blue", label="Experimental")
        plt.hist(RS_gap_data, bins=bins, density=True, alpha=0.2, color="orange", label="RS")
        #sns.kdeplot(experimental_gap_data, label="Experimental", color="blue", linewidth=2)
        #sns.kdeplot(RS_gap_data, label="RS", color="orange", linewidth=2)
        plt.axvline(experimental_mean, color="blue", linestyle="--", linewidth=1, label="Exp Mean")
        plt.axvline(RS_mean, color="orange", linestyle="--", linewidth=1, label="RS Mean")
        plt.axvline(ideal_gap_center, color="black", linestyle=":", linewidth=1, label="Ideal Gap")
        plt.xlim(ideal_gap_center-1.2, ideal_gap_center+1.2)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Gap (mm)", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

##############################################################################################################
"""Run this file"""

def main():
    #data = run_model()
    #Gap_Histogram(30)
    #KDE_curves(29)
    model_distribution_figures(29, plottype="single no D04", save_PDF=True)
    #plot_RW_vs_exp_histograms(RW_tows=310, save_PDF=True)

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
