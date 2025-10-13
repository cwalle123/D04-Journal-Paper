"""A genera data plotter in order to see relations between consecutive steps"""

"""This file is currently not being used for anything except plotting"""
"""Written by: """

##############################################################################################################

# External imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D, art3d

#Internal imports
from Handling_ALL_Functions import get_synced_data
from constants import Consecutive_Error_Bins

##############################################################################################################
"""Functions"""

def plot_LT_error(tow: int):
    """Plot histogram of LT error for a given tow."""
    df = get_synced_data(tow, "LT")  # returns DataFrame
    errors = df["error_LT"]

    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'LT Error Distribution (Tow {tow})')
    plt.xlabel('Error (LT)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_LLS_A_error(tow: int):
    """Plot histogram of LLS A error for a given tow."""
    df = get_synced_data(tow, "LLS_A")
    errors = df["error_LLS_A"]

    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'LLS A Error Distribution (Tow {tow})')
    plt.xlabel('Error (LLS A)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_LLS_B_error(tow: int):
    """Plot histogram of LLS B error for a given tow."""
    df = get_synced_data(tow, "LLS_B")
    errors = df["error_LLS_B"]

    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'LLS B Error Distribution (Tow {tow})')
    plt.xlabel('Error (LLS B)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_CAM_error(tow: int):
    """Plot histogram of CAM error for a given tow."""
    df = get_synced_data(tow, "CAM")
    errors = df["error_CAM"]

    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'CAM Error Distribution (Tow {tow})')
    plt.xlabel('Error (CAM)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_consecutive_scatter(errors, sensor_name, tow):
    """Generic function to plot error[n] vs error[n+1] scatter plot for a sensor."""
    x = errors[:-1]
    y = errors[1:]

    plt.scatter(x, y, alpha=0.5, edgecolor="k", s=20)
    plt.title(f"{sensor_name} Consecutive Error Scatter (Tow {tow})")
    plt.xlabel(f"{sensor_name} Error (n)")
    plt.ylabel(f"{sensor_name} Error (n+1)")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_LT_consecutive_error(tow: int):
    """Plot LT error vs next LT error for a given tow."""
    df = get_synced_data(tow, "LT")
    plot_consecutive_scatter(df["error_LT"], "LT", tow)

def plot_LLS_A_consecutive_error(tow: int):
    """Plot LLS A error vs next LLS A error for a given tow."""
    df = get_synced_data(tow, "LLS_A")
    plot_consecutive_scatter(df["error_LLS_A"], "LLS A", tow)

def plot_LLS_B_consecutive_error(tow: int):
    """Plot LLS B error vs next LLS B error for a given tow."""
    df = get_synced_data(tow, "LLS_B")
    plot_consecutive_scatter(df["error_LLS_B"], "LLS B", tow)

def plot_CAM_consecutive_error(tow: int):
    """Plot CAM error vs next CAM error for a given tow."""
    df = get_synced_data(tow, "CAM")
    plot_consecutive_scatter(df["error_CAM"], "CAM", tow)

def plot_all_tows_consecutive_scatter(sensor: str, n_tows: int = 31):
    """
    Plot consecutive error scatter for all tows combined for a given sensor.
    
    Args:
        sensor (str): One of "LT", "LLS_A", "LLS_B", "CAM".
        n_tows (int): Number of tows to loop through (default=31).
    """
    all_x, all_y = [], []

    for tow in range(2, n_tows + 1):
        df = get_synced_data(tow, sensor)
        errors = df[f"error_{sensor}"] if sensor != "LT" else df["error_LT"]

        if len(errors) > 1:  # only if at least 2 points
            x = errors[:-1]
            y = errors[1:]
            all_x.extend(x)
            all_y.extend(y)

    plt.figure(figsize=(8, 6))
    plt.scatter(all_x, all_y, alpha=0.3, edgecolor="none", s=10)
    plt.title(f"{sensor} Consecutive Error Scatter (All {n_tows} Tows)")
    plt.xlabel(f"{sensor} Error (n)")
    plt.ylabel(f"{sensor} Error (n+1)")
    plt.grid(True, alpha=0.3)
    plt.show()

def model_consecutive_errors(sensor: str, n_tows: int = 31, n_bins: int = 40, quantile_clip: float = 0.01, min_bin_count: int = 10):
    """
    Model consecutive error data for all tows of a given sensor, ignoring outliers.

    Steps:
      1. Collect all error[n] -> error[n+1] pairs across tows
      2. Remove outliers by quantile clipping (default: drop lowest & highest 1%)
      3. Bin x (error[n]) values
      4. Compute bin means (ignore bins with too few samples)
      5. Fit regression line through bin means
      6. Fit normal distribution to residuals

    Args:
        sensor: One of "LT", "LLS_A", "LLS_B", "CAM"
        n_tows: Number of tows (default=31)
        n_bins: Number of bins for current error
        quantile_clip: Fraction to cut from each tail (default=0.01 = 1%)
        min_bin_count: Minimum points required in a bin to keep it
    """
    # Gather all pairs
    all_x, all_y = [], []
    for tow in range(2, n_tows + 1):
        df = get_synced_data(tow, sensor)
        errors = df[f"error_{sensor}"] if sensor != "LT" else df["error_LT"]
        if len(errors) > 1:
            x = errors[:-1]
            y = errors[1:]
            all_x.extend(x)
            all_y.extend(y)

    all_x, all_y = np.array(all_x), np.array(all_y)

    # Clip outliers by quantiles
    lower_x, upper_x = np.quantile(all_x, [quantile_clip, 1 - quantile_clip])
    lower_y, upper_y = np.quantile(all_y, [quantile_clip, 1 - quantile_clip])

    mask = (all_x >= lower_x) & (all_x <= upper_x) & (all_y >= lower_y) & (all_y <= upper_y)
    all_x, all_y = all_x[mask], all_y[mask]

    # Bin data
    bins = np.linspace(all_x.min(), all_x.max(), n_bins + 1)
    bin_indices = np.digitize(all_x, bins) - 1  # bin index for each point

    bin_means_x, bin_means_y, bin_counts = [], [], []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) >= min_bin_count:  # keep only bins with enough points
            bin_means_x.append(all_x[mask].mean())
            bin_means_y.append(all_y[mask].mean())
            bin_counts.append(np.sum(mask))

    bin_means_x = np.array(bin_means_x).reshape(-1, 1)
    bin_means_y = np.array(bin_means_y)
    
    # Regression line (fit on bin means)
    reg = LinearRegression().fit(bin_means_x, bin_means_y)
    y_pred = reg.predict(bin_means_x)

    # Residuals for all data points
    y_pred_all = reg.predict(all_x.reshape(-1, 1))
    residuals = all_y - y_pred_all

    # Standard deviation per bin
    bin_stds = []
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.sum(mask) >= min_bin_count:
            local_residuals = all_y[mask] - y_pred_all[mask]
            bin_stds.append(np.std(local_residuals))
    bin_stds = np.array(bin_stds)

    # Fit normal distribution to residuals
    mu, sigma = norm.fit(residuals)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(all_x, all_y, alpha=0.2, s=10, label="Data", color="blue")
    plt.scatter(bin_means_x, bin_means_y, marker='s',  s=40, color="red", label="Bin means")
    plt.plot(bin_means_x, y_pred, color="red", linewidth=2, label="Regression line")
    plt.title(f"{sensor} Consecutive Error Model (All {n_tows} Tows, Outliers Removed)")
    plt.xlabel(f"{sensor} Error (n)")
    plt.ylabel(f"{sensor} Error (n+1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot residual distribution
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50, density=True, alpha=0.6, color="blue")
    x_vals = np.linspace(residuals.min(), residuals.max(), 200)
    plt.plot(x_vals, norm.pdf(x_vals, mu, sigma), "r-", lw=2,
             label=f"Normal fit (μ={mu:.2f}, σ={sigma:.2f})")
    plt.title(f"{sensor} Residual Distribution (Outliers Removed)")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3D visualization
    # Smooth regression line across full X range
    x_line = np.linspace(all_x.min(), all_x.max(), 200).reshape(-1, 1)
    y_line = reg.predict(x_line)

    # Create figure
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter raw data
    ax.scatter(all_x, all_y, np.zeros_like(all_x), alpha=0.2, s=10, color="blue", label="Data")

    # Scatter bin means
    ax.scatter(bin_means_x, bin_means_y, marker='s', s=40, color="red", label="Bin means")

    # Smooth regression line in XY plane
    ax.plot(x_line.flatten(), y_line, zs=0, zdir='z', color='red', linewidth=2, label="Regression line")

    # Plot per-bin residual distributions
    for xi, yi, si in zip(bin_means_x, bin_means_y, bin_stds):
        y_vals = yi + np.linspace(-3*si, 3*si, 100)
        pdf = norm.pdf(y_vals, yi, si)
        z_vals = pdf / np.max(pdf) * si * 3  # scale for visibility
        ax.plot(np.full_like(y_vals, xi), y_vals, z_vals, color='green', lw=1.5)
        verts = [list(zip(np.full_like(y_vals, xi), y_vals, z_vals))]
        poly = art3d.Poly3DCollection(verts, alpha=0.25, facecolor='mediumspringgreen')
        ax.add_collection3d(poly)

    # Labels and title
    ax.set_xlabel(f"{sensor} Error (n)")
    ax.set_ylabel(f"{sensor} Error (n+1)")
    ax.set_zlabel("Residual PDF")
    ax.set_title(f"{sensor} Consecutive Error Model with Per-Bin Residual Distributions (3D View)")

    # Legend and layout
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return {
        "regression_coef": reg.coef_[0],
        "regression_intercept": reg.intercept_,
        "residual_mean": mu,
        "residual_std": sigma,
        "bins": bins,
        "bin_counts": bin_counts
    }

def model_consecutive_errors_all_sensors(n_tows: int = 31, n_bins: int = 40, quantile_clip: float = 0.01, min_bin_count: int = 10):
    """
    Make a 2x2 figure with all sensors' consecutive error models.
    CAM (top-left), LT (top-right), LLS_B (bottom-left), LLS_A (bottom-right).
    """

    # 🔧 Tweak these limits manually inside the function
    limits = {
        "CAM":   {"x": (-0.6, 0.9), "y": (-0.6, 0.9)},
        "LT":    {"x": (-1.2, -0.6), "y": (-1.2, -0.6)},
        "LLS_B": {"x": (-0.6, 0.3), "y": (-0.6, 0.3)},
        "LLS_A": {"x": (-0.6, 0.3), "y": (-0.6, 0.3)},
    }

    sensors = [("CAM", (0, 0)), ("LT", (0, 1)), ("LLS_B", (1, 0)), ("LLS_A", (1, 1))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    results = {}

    for sensor, (row, col) in sensors:
        ax = axes[row, col]

        # Gather all pairs
        all_x_full, all_y_full = [], []
        for tow in range(2, n_tows + 1):
            df = get_synced_data(tow, sensor)
            errors = df[f"error_{sensor}"] if sensor != "LT" else df["error_LT"]
            if len(errors) > 1:
                x = errors[:-1]
                y = errors[1:]
                all_x_full.extend(x)
                all_y_full.extend(y)

        all_x_full, all_y_full = np.array(all_x_full), np.array(all_y_full)

        # Clip outliers for regression only
        lower_x, upper_x = np.quantile(all_x_full, [quantile_clip, 1 - quantile_clip])
        lower_y, upper_y = np.quantile(all_y_full, [quantile_clip, 1 - quantile_clip])
        mask = (all_x_full >= lower_x) & (all_x_full <= upper_x) & (all_y_full >= lower_y) & (all_y_full <= upper_y)

        all_x = all_x_full[mask]
        all_y = all_y_full[mask]

        # Bin data
        bins = np.linspace(all_x.min(), all_x.max(), n_bins + 1)
        bin_indices = np.digitize(all_x, bins) - 1
        bin_means_x, bin_means_y, bin_counts = [], [], []
        for i in range(n_bins):
            mask_bin = bin_indices == i
            if np.sum(mask_bin) >= min_bin_count:
                bin_means_x.append(all_x[mask_bin].mean())
                bin_means_y.append(all_y[mask_bin].mean())
                bin_counts.append(np.sum(mask_bin))

        bin_means_x = np.array(bin_means_x).reshape(-1, 1)
        bin_means_y = np.array(bin_means_y)

        # Regression
        reg = LinearRegression().fit(bin_means_x, bin_means_y)
        y_pred = reg.predict(bin_means_x)

        # Residuals
        y_pred_all = reg.predict(all_x.reshape(-1, 1))
        residuals = all_y - y_pred_all
        mu, sigma = norm.fit(residuals)

        # Plot: full data in light color, bin means and regression in red
        ax.scatter(all_x_full, all_y_full, alpha=0.2, s=10, color="blue", label="All data")
        ax.scatter(bin_means_x, bin_means_y, color="red", marker="s", s=15, label="Bin means")
        ax.plot(bin_means_x, y_pred, color="red", linewidth=2, label="Regression line")
        ax.set_title(sensor)
        ax.set_xlabel("Error (n)")
        ax.set_ylabel("Error (n+1)")
        ax.grid(True, alpha=0.3)

        # Apply manual axis limits
        if sensor in limits:
            if "x" in limits[sensor]:
                ax.set_xlim(limits[sensor]["x"])
            if "y" in limits[sensor]:
                ax.set_ylim(limits[sensor]["y"])

        if sensor == "CAM":  # only first subplot has legend
            ax.legend()

        # Store results
        results[sensor] = {
            "regression_coef": reg.coef_[0],
            "regression_intercept": reg.intercept_,
            "residual_mean": mu,
            "residual_std": sigma,
            "bins": bins,
            "bin_counts": bin_counts
        }

    fig.suptitle(f"Consecutive Error Models (All {n_tows} Tows, Outliers Removed for Regression)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return results

####################################################################################################
"""Run this file"""

def main():
    tow = 3
    # plot_LT_consecutive_error(tow)
    # plot_all_tows_consecutive_scatter("LT")
    model_consecutive_errors("LT")
    # model_consecutive_errors_all_sensors()

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
