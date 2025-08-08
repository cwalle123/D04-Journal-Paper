"""This file handles the generation of the 4 statistical models.
   Written by: Manuel Cruz & Diogo Ying"""

##############################################################################################################

# External imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.stats import norm, logistic, gamma, beta, expon, lognorm, skewnorm, gumbel_r, gumbel_l, genextreme

# Internal imports
from Scripts.Handling_ALL_Functions import get_synced_data
import Scripts.constants as constants

##############################################################################################################
"""Functions"""

def statistical_values(data: pd.DataFrame):
    '''Get relevant statistical values for each error dataset'''

    errors = [
        data['error_LLS_A'],
        data['error_LLS_B'],
        data['error_LT'],
        data['error_CAM']]
    

    stats = {'mean': [], 'median': [], 'std': [], 'min': [], 'max': []}

    for e in errors:
        stats['mean'].append(round(e.mean(), 4))
        stats['median'].append(round(e.median(), 4))
        stats['std'].append(round(e.std(), 4))
        stats['min'].append(round(e.min(), 4))
        stats['max'].append(round(e.max(), 4))
    return stats

def plot_histograms(data: pd.DataFrame, title: str, bin_widths: list[float] = None, run = bool):
    '''Plots all histograms in the same Figure. 
    The x-axis is manually set for each plot for better visualization.'''

    if run == True:
        distribution_labels = {
            'norm':     'Normal Distribution',
            'logistic': 'Logistic Distribution',
            'skewnorm': 'Skew-Normal Distribution',
            'genextreme':'Generalized Extreme Value'}


        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(title)
        errors = [
            data['width error_LLS_A'],
            data['width error_LLS_B'],
            data['error_LT'],
            data['center_CAM']]
        
        names = ['Error LLS A', 'Error LLS B', 'Error Laser Tracker', 'Error Camera']

        titles = [
            'Tape width before compaction.',
            'Tape width after compaction.',
            'Robot position.',
            'Tape lateral movement.']
        

        if bin_widths is None:
            bin_widths = [None] * 4

        for i, vals in enumerate(errors):
            row, col = divmod(i, 2)
            clean = vals.dropna().to_numpy()
            mn, mx = clean.min(), clean.max()
            bw = bin_widths[i]
            bins = 40 if bw is None else np.arange(mn, mx + bw, bw)

            ax[row, col].hist(clean, bins=bins, alpha=0.6, density=True)
            best = best_fit_distribution(clean, bins=len(bins) - 1)
            dist, params = best['dist'], best['params']
            friendly = distribution_labels.get(dist.name, dist.name)

            print(f"{names[i]} best fit: {friendly}")

            x = np.linspace(mn, mx, 200)
            pdf = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
            
            ax[row, col].plot(x, pdf, '-', lw=2, label=friendly)
            ax[row, col].text(0.02, 0.95, friendly,transform=ax[row, col].transAxes,
                            va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            '''# Fix limits for individual plots for better visualization
            if i == 1:
                ax[row, col].set_xlim(-0.4, 0.2)
            elif i == 2:
                ax[row, col].set_xlim(-1.2, -0.75)
            elif i == 3:
                ax[row, col].set_xlim(-0.5, 1)'''


            mean_val = clean.mean()
            std_val  = clean.std()

            ax[row, col].axvline(mean_val, color='magenta', linestyle='-',
                                label=rf'Mean = {mean_val:.2f}' + '\n' + rf'$\sigma$ = {std_val:.2f}')
            ax[row, col].axvline(0.0, color='black', linestyle='dashed')

            ax[row, col].set_xlim(-1.2, 1.2)
            ax[row, col].set_title(titles[i])
            ax[row, col].set_xlabel(names[i])
            ax[row, col].set_ylabel('Density')
            ax[row, col].legend()
            ax[row, col].xticks(np.linspace(-1.2, 1.2, 9))

        plt.tight_layout()
        plt.show()

def plot_histograms_separated(data: pd.DataFrame, bin_widths: list[float] = None, run = bool):
    '''Plots all histograms in different Figures. 
    The x-axis has the same range for all figures.'''

    if run == True:
        distribution_labels = {
            'norm':     'Normal Distribution',
            'logistic': 'Logistic Distribution',
            'skewnorm': 'Skew-Normal Distribution',
            'genextreme':'Generalized Extreme Value'}

        errors = [
            data['width error_LLS_A'],
            data['width error_LLS_B'],
            data['error_LT'],
            data['center_CAM']]

        names = [
            'width error_LLS_A',
            'width error_LLS_B',
            'error_LT',
            'error_CAM']

        titles = [
            'Tape Width Before Compaction',
            'Tape Width After Compaction',
            'Robot Position',
            'Tape Lateral Movement']
        

        if bin_widths is None:
            bin_widths = [None] * 4

        for i, vals in enumerate(errors):
            # Clean up the data: drop NaNs and convert to a NumPy array
            clean = vals.dropna().to_numpy()
        
            # Find the data range for bin width calculation
            mn, mx = clean.min(), clean.max()
        
            # Determine bin width for this series (None -> default 40 bins)
            bw = bin_widths[i]
            bins = 40 if bw is None else np.arange(mn, mx + bw, bw)
        
            # Create a new figure for this individual histogram
            fig, ax = plt.subplots(figsize=(8, 2))
            #fig.suptitle(f"{titles[i]}")
        
            # Plot the histogram of the cleaned data
            ax.hist(clean, bins=bins, alpha=0.6, density=True)
        
            # Fit the best probability distribution to the data using best_fit_distribution()
            best = best_fit_distribution(clean, bins=len(bins) - 1)
            dist, params = best['dist'], best['params']
        
        
            friendly = distribution_labels.get(dist.name, dist.name)
        
            # Prepare x‐values for plotting the fitted PDF
            x = np.linspace(mn, mx, 200)
            # Compute the PDF using the fitted parameters
            pdf = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
        
            # Plot the fitted PDF on the histogram
            ax.plot(x, pdf, 'r-', lw=2)
        
            # Compute summary statistics for this dataset
            # Can be changed if some other statistic is interesting showing
            mean_val = clean.mean()
            std_val  = clean.std()



            ax.axvline(mean_val, color='magenta', linestyle='-')
            ax.axvline(0.0, color='black', linestyle='dashed')


            # All distributions are shown with this x-axis range
            ax.set_xlim(-1.2, 1.2)
            ax.set_title(titles[i], fontsize= constants.font_large)
            ax.set_xlabel('Error (mm)',fontsize=constants.font_medium)
            ax.set_ylabel('Density',fontsize=constants.font_medium)

            ticks = np.linspace(-1.2, 1.2, 9)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.1f}" for t in ticks])

        plt.tight_layout()
        plt.show()

def plot_LLSA_vs_LLSB(data: pd.DataFrame, title:str, bin_widths: list[float] =None, run = bool):

    if run == True:

        distribution_labels = {
            'norm':     'Normal Distribution',
            'logistic': 'Logistic Distribution',
            'skewnorm': 'Skew-Normal Distribution',
            'genextreme':'Generalized Extreme Value'}
        
        clean_A = data['width error_LLS_A'].dropna().to_numpy()
        clean_B = data['width error_LLS_B'].dropna().to_numpy()

    # Common binning based on combined data
        combined = np.concatenate((clean_A, clean_B))
        mn, mx = combined.min(), combined.max()
        bw = bin_widths[0] or (mx - mn) / 40
        bins = np.arange(mn, mx + bw, bw)

    # Plot both histograms in the same figure
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.suptitle("LLS_A vs. LLS_B")

    # Plot histograms
        ax.hist(clean_A, bins=bins, alpha=0.5, density=True, label='LLS_A')
        ax.hist(clean_B, bins=bins, alpha=0.5, density=True, label='LLS_B')

    # Fit and plot distributions
        for clean, label in [(clean_A, 'LLS_A'), (clean_B, 'LLS_B')]:
            best = best_fit_distribution(clean, bins=len(bins) - 1)
            dist, params = best['dist'], best['params']
            friendly = distribution_labels.get(dist.name, dist.name)
            x = np.linspace(mn, mx, 200)
            pdf = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
            ax.plot(x, pdf, lw=2, label=f'{label} Fit: {friendly}')

    # Styling
        ax.axvline(0.0, color='black', linestyle='dashed')
        ax.set_xlim(-1.2, 1.2)
        ax.set_xlabel("Error (mm)")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        plt.show()

def best_fit_distribution(data, bins=40, distributions=None):
    '''This function fits the best distribution to the four error types automatically'''

    # Compute the histogram of the data
    y, bin_edges = np.histogram(data, bins=bins, density=True)

    # x_mid is the center of each histogram bin, used for PDF evaluation
    x_mid = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # If no distribution list given, use a broad default set
    if distributions is None:
        distributions = [
            norm, logistic, gamma, beta, expon, lognorm, skewnorm,
            gumbel_r, gumbel_l, genextreme]


    best = {'dist': None, 'params': None, 'sse': np.inf}

    # Iterate over each candidate distribution
    for dist in distributions:
        # Skip distributions that require non-negative data if data has negatives
        if data.min() < 0 and dist in (gamma, beta, expon, lognorm, skewnorm, gumbel_r, genextreme):
            continue


        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                # Fit the best distribution to the data
                params = dist.fit(data)
                # Evaluate its PDF at the bin centers
                pdf = dist.pdf(x_mid, *params[:-2], loc=params[-2], scale=params[-1])
                # Compute sum of squared errors between histogram and PDF to check accuracy
                sse = np.sum((y - pdf) ** 2)

                # If this fit is better (lower SSE), use it
                if sse < best['sse']:
                    best.update(dist=dist, params=params, sse=sse)
            except Exception:
                # If fitting fails for any reason, skip to the next distribution
                continue

    # Return the distribution with the lowest error (SSE)
    return best

##############################################################################################################
"""Run this file"""

def main():
    df = pd.concat((get_synced_data(t, spacesynced=True) for t in range(2,32)), ignore_index=True)

    # To make the plots appear, change run=False to run=True

    # All tows are shown
    plot_histograms(
        df,
        title="Sensor Error Histograms ",
        bin_widths=[0.008, 0.008, 0.008, 0.008], 
        run = False)
    
    plot_histograms_separated(
        df,
        bin_widths=[0.005, 0.005, 0.005, 0.008],
        run = True)

    plot_LLSA_vs_LLSB(df,
        title="Error LLS A vs. Error LLS B (ALL TOWS)",
        bin_widths=[0.005, 0.005],
        run = False)

if __name__ == "__main__":
    main()
