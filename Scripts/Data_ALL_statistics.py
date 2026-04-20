"""This file handles the generation of the 4 statistical models.
   Written by: Manuel Cruz & Diogo Ying"""

##############################################################################################################

# External imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.stats import norm, logistic, gamma, beta, expon, lognorm, skewnorm, gumbel_r, gumbel_l, genextreme, pareto, weibull_min, weibull_max, cauchy, t, poisson, laplace, genlogistic
import sklearn.metrics as sklearn

# Internal imports
from Handling_ALL_Functions import get_synced_data, get_data
import constants

##############################################################################################################
"""Functions"""

# Functions for plotting:
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
            data['error_LLS_A'],
            data['error_LLS_B'],
            data['error_LT'],
            data['error_CAM']]
        
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
            ticks = np.linspace(-1.2, 1.2, 9)
            ax[row, col].set_xticks(ticks)

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

        series = [
            ('error_LLS_A', 'w_LLS_A', 'Tape Width Before Compaction'),
            ('error_LLS_B', 'w_LLS_B', 'Tape Width After Compaction'),
            ('error_LT',    'w_LT',    'Robot Position'),
            ('error_CAM',   'w_CAM',   'Tape Lateral Movement'),
        ]


        errors = [
            data['error_LLS_A'],
            data['error_LLS_B'],
            data['error_LT'],
            data['error_CAM']]

        names = [
            'error_LLS_A',
            'error_LLS_B',
            'error_LT',
            'error_CAM']

        titles = [
            'Tape Width Before Compaction',
            'Tape Width After Compaction',
            'Robot Position',
            'Tape Lateral Movement']
        

        if bin_widths is None:
            bin_widths = [None] * 4

        for (col, wcol, title), bw in zip(series, bin_widths):
            s = data[col]
            if wcol in data.columns:
                w = data[wcol]
                mask = s.notna() & w.notna() & (w > 0)
                clean = s[mask].to_numpy()
                w_clean = w[mask].to_numpy()
            else:
                clean = s.dropna().to_numpy()
                w_clean = None

            if clean.size == 0:
                continue
        
            # Find the data range for bin width calculation
            mn, mx = clean.min(), clean.max()
        
            # Determine bin width for this series (None -> default 40 bins)
            bins = 40 if bw is None else np.arange(mn, mx + bw, bw)
        
            # Create a new figure for this individual histogram
            fig, ax = plt.subplots(figsize=(8, 2))
            #fig.suptitle(f"{titles[i]}")
        
            # Plot the histogram of the cleaned data
            ax.hist(clean, bins=bins, weights=w_clean, alpha=0.6, density=True)
        
            # Fit the best probability distribution to the data using best_fit_distribution()
            best = best_fit_distribution(clean, bins=len(bins) - 1, weights=w_clean)
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
            mean_val, std_val = weighted_mean_std(clean, w_clean)

            ax.axvline(mean_val, color='magenta', linestyle='-')
            ax.axvline(0.0, color='black', linestyle='dashed')


            # All distributions are shown with this x-axis range
            ax.set_xlim(-1.2, 1.2)
            ax.set_title(title, fontsize=constants.font_large)
            ax.set_xlabel('Error (mm)',fontsize=constants.font_medium)
            ax.set_ylabel('Density',fontsize=constants.font_medium)

            ticks = np.linspace(-1.2, 1.2, 9)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.1f}" for t in ticks])

        plt.tight_layout()
        plt.show()

def plot_LLSA_vs_LLSB(data: pd.DataFrame, title:str, bin_widths: list[float] =None, run = bool):
    '''Plots Tape width before vs after compaction to see the overlap. Not used in this paper.'''
    if run == True:

        distribution_labels = {
            'norm':     'Normal Distribution',
            'logistic': 'Logistic Distribution',
            'skewnorm': 'Skew-Normal Distribution',
            'genextreme':'Generalized Extreme Value'}
        
        clean_A = data['error_LLS_A'].dropna().to_numpy()
        clean_B = data['error_LLS_B'].dropna().to_numpy()

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

# Functions for fitting and collecting data:
def best_fit_distribution(data, bins=40, distributions=None, weights=None, use_all_dist=False, plot=False, print_statement=False):
    '''This function fits the best probability distribution to the four error types automatically, for the weighted data''' 
    # Compute the histogram of the data 
    y, bin_edges = np.histogram(data, bins=bins, density=True, weights=weights) 
    # x_mid is the center of each histogram bin, used for PDF evaluation 
    x_mid = (bin_edges[:-1] + bin_edges[1:]) / 2.0 
    
    bw = np.diff(bin_edges) 
    
    # If no distribution list given, use a broad default set 
    if distributions is None and use_all_dist is True: 
        distributions = [skewnorm]
    elif distributions is None and use_all_dist is False: 
        distributions = [logistic, norm]
    
    best = {'dist': None, 'params': None, 'mse': np.inf} 
    # Iterate over each candidate distribution
    for dist in distributions:
        # Skip distributions that require non-negative data if data has negatives 
        #if data.min() < 0 and dist in (gamma, beta, expon, lognorm, skewnorm, gumbel_r, genextreme): 
        # continue 
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                # Fit the best distribution to the data
                params = dist.fit(data)
                # params = list(params)         # These 3 lines are old code to improve CAM fit which we are NOT using
                # params[-1] = params[-1] * 1
                # params = tuple(params)

                # Evaluate its PDF at the bin centers
                pdf = dist.pdf(x_mid, *params[:-2], loc=params[-2], scale=params[-1])
                # Compute sum of squared errors between histogram and PDF to check accuracy
                mse = sklearn.mean_squared_error(y, pdf)
                # If this fit is better (lower MSE), use it
                if mse < best['mse']:
                    best.update(dist=dist, params=params, mse=mse)

            except Exception:
                # If fitting fails for any reason, skip to the next distribution
                continue
        
    best_dist = best['dist']
    params = best['params']
    mse = best['mse']

    # shape parameters (can be empty)
    shapes = params[:-2]
    loc = params[-2]
    scale = params[-1]

    params = (*shapes, loc, scale)

    if print_statement == True:
        print("Best:", best_dist.name)
        print("Shape parameters:", shapes)
        print("loc:", loc, "scale:", scale)
        print("MSE:", mse)

    # Return the distribution with the lowest error (MSE)

    if plot:
        x = np.linspace(best_dist.ppf(0.001, *shapes, loc=loc, scale=scale), best_dist.ppf(0.999, *shapes, loc=loc, scale=scale), 200)
        plt.hist(data, bins=bins, density=True, alpha=0.5)
        plt.plot(x, best_dist.pdf(x, *shapes, loc=loc, scale=scale), linewidth=2)
        plt.title(f"Best fit: {best_dist.name}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.show()

    return best

def build_all_sensors_df(tow_range=range(2, 31)):
    """Returns a dataframe with columns:
      error_LLS_A, w_LLS_A,
      error_LLS_B, w_LLS_B,
      error_LT,    w_LT,
      error_CAM,   w_CAM,
      tow

    Notes
    -----
    This function assumes the synced/distilled sensor files are already aligned
    sample-by-sample within each tow. It concatenates the four sensor streams
    row-wise for each tow after resetting the index.
    """
    
    def ensure_col(df: pd.DataFrame, desired: str, fallbacks: list[str]) -> pd.DataFrame:
        """Ensure df has `desired`; if missing, rename from first available fallback."""
        df = df.copy()
        for fb in fallbacks:
            if desired not in df.columns and fb in df.columns:
                df = df.rename(columns={fb: desired})
                break
        if desired not in df.columns:
            df[desired] = np.nan
        return df

    def extract_weight(df: pd.DataFrame, new_name: str) -> pd.DataFrame:
        """Extract a single weight column, even if duplicate 'Weights' columns exist."""
        df = df.copy()
        df.columns = df.columns.astype(str).str.strip()
        weight_cols = [c for c in df.columns if c.strip().lower() == "weights"]

        if not weight_cols:
            return pd.Series(name=new_name, dtype=float).to_frame()

        wdf = df[weight_cols]
        if isinstance(wdf, pd.Series):
            s = pd.to_numeric(wdf, errors="coerce")
        else:
            wdf_num = wdf.apply(pd.to_numeric, errors="coerce")
            s = wdf_num.bfill(axis=1).iloc[:, 0]

        return s.reset_index(drop=True).rename(new_name).to_frame()

    dfs = []

    for t in tow_range:
        lt   = get_synced_data(t, sensor_type="LT").copy()
        llsa = get_synced_data(t, sensor_type="LLS_A").copy()
        llsb = get_synced_data(t, sensor_type="LLS_B").copy()
        cam  = get_synced_data(t, sensor_type="CAM").copy()

        for f in (lt, llsa, llsb, cam):
            f.columns = f.columns.astype(str).str.strip()

        lt   = ensure_col(lt,   "error_LT",    fallbacks=["error"])
        llsa = ensure_col(llsa, "error_LLS_A", fallbacks=["error"])
        llsb = ensure_col(llsb, "error_LLS_B", fallbacks=["error"])
        cam  = ensure_col(cam,  "error_CAM",   fallbacks=["center_CAM", "center", "error"])

        df_t = pd.concat(
            [
                lt[["error_LT"]].reset_index(drop=True),
                extract_weight(lt, "w_LT"),

                llsa[["error_LLS_A"]].reset_index(drop=True),
                extract_weight(llsa, "w_LLS_A"),

                llsb[["error_LLS_B"]].reset_index(drop=True),
                extract_weight(llsb, "w_LLS_B"),

                cam[["error_CAM"]].reset_index(drop=True),
                extract_weight(cam, "w_CAM"),
            ],
            axis=1
        )

        df_t["tow"] = t
        dfs.append(df_t)

    df = pd.concat(dfs, ignore_index=True)
    return df

def weighted_mean_std(x, w=None):
    '''Get the statistical data (mean and std) for the four sensors with weights on the data'''

    x = np.asarray(x, dtype=float)
    if w is None:
        return x.mean(), x.std(ddof=0)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if w.sum() == 0 or x.size == 0:
        return np.nan, np.nan
    w = w / w.sum()
    mu = np.sum(w * x)
    var = np.sum(w * (x - mu)**2)
    return mu, np.sqrt(var)

def print_weighted_stats_table(df: pd.DataFrame):
    '''function for printing the statistics fo the weighted data'''
    rows = []
    mapping = [('error_LLS_A', 'w_LLS_A'),
        ('error_LLS_B', 'w_LLS_B'),
        ('error_LT',    'w_LT'),
        ('error_CAM',  'w_CAM'),]

    for col, wcol in mapping:
        s = df[col]
        w = df[wcol] if wcol in df.columns else None
        m = s.notna() if w is None else (s.notna() & w.notna() & (w > 0))
        x = s[m].to_numpy()
        ww = None if w is None else w[m].to_numpy()
        mu, sd = weighted_mean_std(x, ww)
        rows.append((col, len(x), mu, sd, 'weighted' if ww is not None else 'unweighted'))
    print("Stats:")
    for col, n, mu, sd, kind in rows:
        print(f"- {col:<12} n={n:5d}  mean={mu: .4f}  std={sd: .4f}  ({kind})")



def plot_sensor_correlation_scatter_only(
    tow_range=range(2, 32),
    scatter_figsize=(12, 8),
    matrix_figsize=(7, 6),
    alpha=0.20,
    s=8,
    add_regression_line=True,
    show_pearson_matrix_plot=True,
    show_spearman_matrix_plot=True,
    print_table=True,
    print_matrix=True
):
    """
    Plot:
    1) Lower-triangular scatter/correlation matrix
    2) Pearson correlation heatmap
    3) Spearman correlation heatmap

    Uses the four variation sources:
        - error_LLS_A : width before compaction
        - error_LLS_B : width after compaction
        - error_LT    : robot position
        - error_CAM   : tape lateral movement

    Returns
    -------
    corr_df : pd.DataFrame
        Pairwise correlation table.
    pearson_matrix : pd.DataFrame
        Pearson correlation matrix.
    spearman_matrix : pd.DataFrame
        Spearman correlation matrix.
    """

    from scipy.stats import pearsonr, spearmanr

    # --------------------------------------------------
    # Build dataframe and ensure camera column is correct
    # --------------------------------------------------
    df = build_all_sensors_df(tow_range=tow_range).copy()

    if "error_CAM" not in df.columns:
        if "center_CAM" in df.columns:
            df = df.rename(columns={"center_CAM": "error_CAM"})
        else:
            raise KeyError("Camera error column not found. Expected 'error_CAM'.")

    cols = ["error_LLS_A", "error_LLS_B", "error_LT", "error_CAM"]
    labels = {
        "error_LLS_A": "Width before compaction",
        "error_LLS_B": "Width after compaction",
        "error_LT": "Robot position",
        "error_CAM": "Tape lateral movement"
    }

    # --------------------------------------------------
    # 1) Scatter correlation matrix (lower triangle only)
    # --------------------------------------------------
    n = len(cols)
    fig, axes = plt.subplots(n - 1, n - 1, figsize=scatter_figsize)
    fig.suptitle("Cross-correlations between the four variation sources", fontsize=18)

    correlation_rows = []

    for i in range(1, n):
        for j in range(n - 1):
            ax = axes[i - 1, j]

            if j >= i:
                ax.axis("off")
                continue

            xcol = cols[j]
            ycol = cols[i]

            pair_df = df[[xcol, ycol]].dropna()
            x = pair_df[xcol].to_numpy()
            y = pair_df[ycol].to_numpy()

            if len(x) < 2:
                ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
                ax.set_xlabel(labels[xcol])
                ax.set_ylabel(labels[ycol])
                continue

            r, _ = pearsonr(x, y)
            rho, _ = spearmanr(x, y)

            correlation_rows.append({
                "Variable 1": labels[xcol],
                "Variable 2": labels[ycol],
                "Pearson r": r,
                "Spearman rho": rho,
                "N": len(pair_df)
            })

            ax.scatter(x, y, alpha=alpha, s=s)

            if add_regression_line:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(np.min(x), np.max(x), 200)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, linewidth=2, color='red')

            ax.text(
                0.03, 0.95,
                f"r = {r:.3f}\nρ = {rho:.3f}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
            )

            ax.set_xlabel(labels[xcol])
            ax.set_ylabel(labels[ycol])

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 2) Correlation matrices
    # --------------------------------------------------
    corr_df = pd.DataFrame(correlation_rows)

    pearson_matrix = df[cols].corr(method="pearson")
    spearman_matrix = df[cols].corr(method="spearman")

    pearson_matrix = pearson_matrix.rename(index=labels, columns=labels)
    spearman_matrix = spearman_matrix.rename(index=labels, columns=labels)

    def _plot_corr_matrix(matrix, title, cmap="Greens"):
        fig, ax = plt.subplots(figsize=matrix_figsize)
        im = ax.imshow(matrix.values, cmap=cmap, vmin=-1, vmax=1)

        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
        ax.set_yticklabels(matrix.index)

        ax.set_title(title)

        # write values inside cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iloc[i, j]
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="black"
                )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation coefficient")

        plt.tight_layout()
        plt.show()

    if show_pearson_matrix_plot:
        _plot_corr_matrix(
            pearson_matrix,
            title="Pearson correlation matrix",
            cmap="Greens"
        )

    if show_spearman_matrix_plot:
        _plot_corr_matrix(
            spearman_matrix,
            title="Spearman correlation matrix",
            cmap="Greens"
        )

    # --------------------------------------------------
    # 3) Print tables
    # --------------------------------------------------
    if print_table:
        print("\nPairwise correlations between the four variation sources:\n")
        print(corr_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if print_matrix:
        print("\nPearson correlation matrix:\n")
        print(pearson_matrix.to_string(float_format=lambda x: f"{x:.3f}"))

        print("\nSpearman correlation matrix:\n")
        print(spearman_matrix.to_string(float_format=lambda x: f"{x:.3f}"))

    return corr_df, pearson_matrix, spearman_matrix

##############################################################################################################
"""Run this file"""

def main():
    
    #'''Creates dataframe with sensor data + weights from each sensor'''
    #df = build_all_sensors_df(tow_range=range(2, 32))
    #print(df.columns.tolist())   # just to ensure the dataframe has the correct data
    #print_weighted_stats_table(df)  # prints the statistical values

    # To make the plots appear, change run=False to run=True

    # All tows are shown
    #plot_histograms(
    #    df,
    #    title="Sensor Error Histograms ",
    #    bin_widths=[0.008, 0.008, 0.008, 0.008], 
    #    run = True)
    
    #'''This is the good one'''
    #plot_histograms_separated(
    #    df,
    #    bin_widths=[0.005, 0.005, 0.005, 0.008],
    #    run = False)

    #plot_LLSA_vs_LLSB(df,
    #    title="Error LLS A vs. Error LLS B (ALL TOWS)",
    #    bin_widths=[0.005, 0.005],
    #    run = False)
    
    '''sensor = "LLS_B"
    
    # Get best fit distributions
    if sensor == "CAM":
        use_all_dist = True
        bins = 250
    else:
        use_all_dist = False
        bins = 100
    data, weights = np.array(get_data(sensor, format="merged"))
    print(f"Sensor: {sensor}")
    best = best_fit_distribution(data=data, bins=bins, distributions=None, weights=weights, use_all_dist=use_all_dist, plot=True, print_statement=True)
    print(best)'''


    corr_df, pearson_matrix, spearman_matrix = plot_sensor_correlation_scatter_only(
        tow_range=range(2, 32),
        scatter_figsize=(12, 8),
        matrix_figsize=(7, 6),
        alpha=0.20,
        s=8,
        add_regression_line=True,
        show_pearson_matrix_plot=True,
        show_spearman_matrix_plot=True,
        print_table=True,
        print_matrix=True
    )

    # Get raw statistics
    #data, weights = get_data(sensor=sensor)
    #mu, std = weighted_mean_std(data, weights)
    #print(f'The chosen sensor is {sensor}')
    #print(f'The mean is {mu}')
    #print(f'The standard deviation is {std}')

if __name__ == "__main__":
    main()
