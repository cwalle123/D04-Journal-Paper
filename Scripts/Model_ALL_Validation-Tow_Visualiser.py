"""This file is used to generate a figure of a simulated tow vs an experimental tow.
   This file is currently not being used for anything except plotting"""

##############################################################################################################

# External imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from scipy.fft import fft, fftfreq
import os

#Internal imports
from Data_ALL_importer import LLS_B_excel_to_array, CAM_excel_to_array, LT_x_excel_to_array, LT_y_normalized_excel_to_array
from Handling_ALL_Functions import get_synced_data
from Data_ALL_traverse import traverse_tow_constructor
from Model_ALL_ConsecutiveErrorTheo import consecutive_error, generate_error_path
from Model_ALL_Simulation import generate_multitow_layout
from constants import number_of_steps, Consecutive_Error_Bins

##############################################################################################################
"""Functions"""

def plot_real_tow(tow: int, tow_length_mm=1000, plot: bool = True):
    """
    Plot a real tow profile using Traverse interpolated data
    """

    real_df = traverse_tow_constructor(tow)
    x_right = real_df["x_right"].to_numpy()
    y_right = real_df["y_right"].to_numpy()
    x_left = real_df["x_left"].to_numpy()
    y_left = real_df["y_left"].to_numpy()

    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(x_right, y_right, "-", linewidth=2.0, label="Right edge")
        plt.plot(x_left, y_left, "-", linewidth=2.0, label="Left edge")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.title(f"Real tow {tow} from traverse interpolated data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return real_df

def plot_Layup_vs_Traverse_tow(tow: int, tow_length_mm=1000):
    # Check for validity of tow
    if tow not in range(2, 31):
        print(f'Tow 1 or 31 can not be recreated from traverse data.')
        print(f'Provide a tow number between 2 and 30 inclusive')
        return None

    else:   
        # Load layup data
        base_path = r"Cached Data"
        file_name = f"LAYUP_{tow}.csv"
        file_path = os.path.join(base_path, file_name)
        Layup_CAM = "center_CAM"
        Layup_LT_x = "x"
        Layup_LT_y = "y"
        Layup_LLS_B = "width_LLS_B"
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        Layup_data = df[[Layup_CAM, Layup_LT_x, Layup_LT_y, Layup_LLS_B]].to_numpy()

        # Load traverse data
        Traverse_df = traverse_tow_constructor(tow)
        Traverse_x_right = Traverse_df["x_right"].to_numpy()
        Traverse_y_right = Traverse_df["y_right"].to_numpy()
        Traverse_x_left = Traverse_df["x_left"].to_numpy()
        Traverse_y_left = Traverse_df["y_left"].to_numpy()

        # Calculate lay-up tows
        Layup_centerline = Layup_data[:, 0] + Layup_data[:, 2] + 5.5 # Plus 5.5 to place traverse and layup on top of each other
        Layup_width = Layup_data[:, 3]
        Layup_x_right = Layup_data[:, 1]
        Layup_y_right = Layup_centerline - 0.5 * Layup_width
        Layup_x_left = Layup_data[:, 1]
        Layup_y_left = Layup_centerline + 0.5 * Layup_width

        # Make plot
        plt.figure(figsize=(10, 3))
        plt.plot(Traverse_x_right, Traverse_y_right, "-", color = "r", linewidth=2.0, label="Traverse right edge")
        plt.plot(Traverse_x_left, Traverse_y_left, "-", color = "b", linewidth=2.0, label="Traverse left edge")
        plt.plot(Layup_x_right, Layup_y_right, "-", color = "g", linewidth=2.0, label="Layup right edge")
        plt.plot(Layup_x_left, Layup_y_left, "-", color = "y", linewidth=2.0, label="Layup left edge")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.title(f"Real tow {tow} from traverse interpolated data and layup data")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return

def plot_simulated_vs_real_tow(tow: int, tow_length_mm=1000, scaled: bool = False, plot: bool = True):
    """
    Overlay a simulated tow on a real tow.
    Real tow is built from Traverse Data
    Simulated tow is generated with generate_multitow_layout.

    Parameters
    ----------
    tow : int
        Tow index.
    tow_length_mm : int, optional
        Tow length in mm (default 1000).
    scaled : bool, optional
        If True, plot is shown with 1:1 scale (equal aspect ratio).
    """

    # --- Get real tow ---
    real_df = plot_real_tow(tow, tow_length_mm=tow_length_mm, plot=False)

    # Extract edges
    real_x = real_df["x_right"].to_numpy()
    real_y_right = real_df["y_right"].to_numpy()
    real_y_left = real_df["y_left"].to_numpy()

    # Compute centerline
    real_centerline = 0.5 * (real_y_left + real_y_right)

    # Normalize so centerline starts at 0
    real_offset = real_centerline[0]
    real_centerline = real_centerline - real_offset
    real_y_left = real_y_left - real_offset
    real_y_right = real_y_right - real_offset

    real_data = pd.DataFrame({
        "x_mm": real_x,
        "centerline": real_centerline,
        "left_edge": real_y_left,
        "right_edge": real_y_right,
        "width": real_y_left - real_y_right})

    # --- Generate simulated tow ---
    gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent = generate_multitow_layout(
        num_tows=1, tow_length_mm=tow_length_mm, plot=False)

    num_bins = Consecutive_Error_Bins
    n_steps = int(tow_length_mm * 340 / 1000)
    n_steps = number_of_steps

    # Load error model fits
    bin_stats_cam, slope_cam, intercept_cam, _, _, _, x_sorted_cam, bin_edges_cam, devs_cam = consecutive_error(
        "CAM", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_lt, slope_lt, intercept_lt, _, _, _, x_sorted_lt, bin_edges_lt, devs_lt = consecutive_error(
        "LT", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))
    bin_stats_llsb, slope_llsb, intercept_llsb, _, _, _, x_sorted_llsb, bin_edges_llsb, devs_llsb = consecutive_error(
        "LLS_B", test_ratio=0.5, num_bins=num_bins, bins_show=False, plot_fit=False,
        random_state=random.randint(0, 10000))

    # Generate simulated paths
    start_cam = random.uniform(-0.75, 0.75)
    start_lt = random.uniform(-0.9, -0.7)
    start_llsb = random.uniform(-0.21, -0.02)

    cam_path = generate_error_path(start_cam, n_steps, slope_cam, intercept_cam,
                                   x_sorted_cam, bin_edges_cam, devs_cam)
    lt_path = generate_error_path(start_lt, n_steps, slope_lt, intercept_lt,
                                  x_sorted_lt, bin_edges_lt, devs_lt)
    tow_centerline_sim = cam_path + lt_path

    width_error = generate_error_path(start_llsb, n_steps, slope_llsb, intercept_llsb,
                                      x_sorted_llsb, bin_edges_llsb, devs_llsb)
    tow_widths_sim = 6.35 + width_error  # nominal width + error

    sim_top = tow_centerline_sim + 0.5 * tow_widths_sim
    sim_bottom = tow_centerline_sim - 0.5 * tow_widths_sim
    sim_x = np.linspace(0, tow_length_mm, len(tow_centerline_sim))

    # Normalize so sim starts at y=0
    sim_offset = tow_centerline_sim[0]
    tow_centerline_sim = tow_centerline_sim - sim_offset
    sim_top = sim_top - sim_offset
    sim_bottom = sim_bottom - sim_offset

    sim_data = pd.DataFrame({
        "x_mm": sim_x,
        "centerline": tow_centerline_sim,
        "top_edge": sim_top,
        "bottom_edge": sim_bottom,
        "width": tow_widths_sim})

    # --- Plot both ---
    if plot:
        plt.figure(figsize=(10, 6))

        # Real tow
        plt.plot(real_data["x_mm"], real_data["centerline"], "--", color="blue", label="Real centerline")
        plt.plot(real_data["x_mm"], real_data["left_edge"], "-", color="blue", alpha=0.6, label="Real edges")
        plt.plot(real_data["x_mm"], real_data["right_edge"], "-", color="blue", alpha=0.6)

        # Simulated tow
        plt.plot(sim_data["x_mm"], sim_data["centerline"], "--", color="red", label="Sim centerline")
        plt.plot(sim_data["x_mm"], sim_data["top_edge"], "-", color="red", alpha=0.6, label="Sim edges")
        plt.plot(sim_data["x_mm"], sim_data["bottom_edge"], "-", color="red", alpha=0.6)

        plt.xlabel("Tow length (mm)", fontsize=14)
        plt.ylabel("Position (mm)", fontsize=14)
        plt.title(f"Tow {tow}: Real vs Simulated (aligned to y=0)", fontsize=16)
        plt.legend()
        plt.grid(True)

        if scaled:
            plt.axis("equal")

        plt.tight_layout()
        plt.show()

    return real_data, sim_data

def compare_simulated_vs_real_tow(tow: int, tow_length_mm=1000, plot: bool = True):
    """
    Compare simulated and real tow edges by calculating average lateral error.

    Parameters
    ----------
    tow : int
        Tow index.
    tow_length_mm : int, optional
        Tow length in mm (default 1000).

    Returns
    -------
    errors : dict
        Dictionary with average errors between real and simulated tow edges.
        Includes mean absolute error (MAE) and root mean square error (RMSE).
    """

    # --- Get real tow ---
    real_df = plot_real_tow(tow, tow_length_mm=tow_length_mm)

    # --- Generate one simulated tow (no plot, just data) ---
    _, sim_df = plot_simulated_vs_real_tow(tow, tow_length_mm=tow_length_mm, scaled=False, plot=plot)

    # --- Interpolate real edges to simulated x positions ---
    real_left_interp = np.interp(sim_df["x_mm"], real_df["x_mm"], real_df["left_edge"])
    real_right_interp = np.interp(sim_df["x_mm"], real_df["x_mm"], real_df["right_edge"])

    # --- Compute edge errors (corrected mapping) ---
    err_top = sim_df["top_edge"] - real_left_interp      # Sim top vs Real left
    err_bottom = sim_df["bottom_edge"] - real_right_interp  # Sim bottom vs Real right

    # --- Error metrics ---
    mae_top = float(np.mean(np.abs(err_top)))
    mae_bottom = float(np.mean(np.abs(err_bottom)))
    rmse_top = float(np.sqrt(np.mean(err_top**2)))
    rmse_bottom = float(np.sqrt(np.mean(err_bottom**2)))

    errors = {
        "MAE_top_edge (SimTop vs RealLeft)": mae_top,
        "MAE_bottom_edge (SimBottom vs RealRight)": mae_bottom,
        "RMSE_top_edge (SimTop vs RealLeft)": rmse_top,
        "RMSE_bottom_edge (SimBottom vs RealRight)": rmse_bottom}

    print("MAE_top_edge (SimTop vs RealLeft)", mae_top, "mm")
    print("MAE_bottom_edge (SimBottom vs RealRight)", mae_bottom, "mm")
    print("RMSE_top_edge (SimTop vs RealLeft)", rmse_top, "mm")
    print("RMSE_bottom_edge (SimBottom vs RealRight)", rmse_bottom, "mm")

    return errors

def compare_multiple_simulations(tow: int, n_simulations: int = 50, tow_length_mm: int = 1000):
    """
    Run multiple simulated vs real tow comparisons and return error statistics,
    plus plot histograms with Gaussian fits of the error distributions.

    Parameters
    ----------
    tow : int
        Tow index.
    n_simulations : int, optional
        Number of simulation runs (default = 50).
    tow_length_mm : int, optional
        Tow length in mm (default 1000).

    Returns
    -------
    stats : dict
        Dictionary with mean and std deviation of the errors across runs.
    """

    results = []

    for i in range(n_simulations):
        errors = compare_simulated_vs_real_tow(tow, tow_length_mm=tow_length_mm, plot=False)
        results.append(errors)

    # Convert to DataFrame for easier aggregation
    df = pd.DataFrame(results)

    stats = {}
    for col in df.columns:
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std()
        }

    print("\n=== Error Statistics over", n_simulations, "simulations ===")
    for col in df.columns:
        print(f"{col}: mean = {stats[col]['mean']:.3f} mm, std = {stats[col]['std']:.3f} mm")

    # --- Plot histograms with Gaussian fit ---
    num_metrics = len(df.columns)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 4), constrained_layout=True)

    if num_metrics == 1:
        axes = [axes]

    for ax, col in zip(axes, df.columns):
        data = df[col].values
        mu, sigma = stats[col]["mean"], stats[col]["std"]

        # Histogram
        count, bins, _ = ax.hist(data, bins=50, density=True, alpha=0.6,
                                 color="steelblue", edgecolor="black")

        # Gaussian curve
        x = np.linspace(min(bins), max(bins), 200)
        pdf = norm.pdf(x, mu, sigma)
        ax.plot(x, pdf, "r--", linewidth=2, label=f"N({mu:.2f}, {sigma:.2f}²)")

        ax.set_title(col, fontsize=12)
        ax.set_xlabel("Error (mm)")
        ax.set_ylabel("Density")
        ax.legend()

    plt.show()

    return stats

##############################################################################################################
"""TEST functions to check FFT""" # Use Model_ALL_Validation-FFT.py instead of this function!

def compare_fft_real_vs_sim(real_df, sim_df, tow_length_mm=1000, plot=True):
    """
    Compare FFT spectra of real vs simulated tow centerlines.
    """
    # --- Extract centerlines ---
    real = real_df["centerline"].to_numpy()
    sim = sim_df["centerline"].to_numpy()

    # --- Sampling steps (different lengths -> different dx) ---
    dx_real = tow_length_mm / len(real)
    dx_sim = tow_length_mm / len(sim)

    # --- FFT for real ---
    freqs_real = fftfreq(len(real), d=dx_real)[: len(real) // 2]
    fft_real = np.abs(fft(real)[: len(real) // 2])
    fft_real /= np.max(fft_real)

    # --- FFT for sim ---
    freqs_sim = fftfreq(len(sim), d=dx_sim)[: len(sim) // 2]
    fft_sim = np.abs(fft(sim)[: len(sim) // 2])
    fft_sim /= np.max(fft_sim)

    # --- Plot ---
    if plot == True:
        plt.figure(figsize=(10, 6))
        plt.plot(freqs_real, fft_real, label="Real Tow FFT", color="blue")
        plt.plot(freqs_sim, fft_sim, label="Simulated Tow FFT", color="red", alpha=0.7)
        plt.xlabel("Spatial frequency (1/mm)", fontsize=14)
        plt.ylabel("Normalized magnitude", fontsize=14)
        plt.title("FFT Comparison: Real vs Simulated Tow", fontsize=16)
        plt.xlim(0, 0.75)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return freqs_real, fft_real, freqs_sim, fft_sim

def optimize_fft_match(tow: int,
                       tow_length_mm=1000,
                       steps_range=(200, 600),      # search range for number_of_steps
                       bins_range=(5, 20),          # search range for num_bins
                       n_trials=20,
                       plot_best=True):
    """
    Optimize number_of_steps and Consecutive_Error_Bins so simulated FFT
    matches the real FFT as closely as possible.
    """

    best_error = np.inf
    best_params = None
    best_result = None

    # Try combinations
    for _ in range(n_trials):
        # Random pick in ranges
        n_steps = random.randint(*steps_range)
        n_bins = random.randint(*bins_range)

        # Update globals (since generate_multitow_layout depends on them)
        global number_of_steps, Consecutive_Error_Bins
        number_of_steps = n_steps
        Consecutive_Error_Bins = n_bins

        # Run sim vs real tow
        real_df, sim_df = plot_simulated_vs_real_tow(tow, tow_length_mm, plot=False)
        freqs_real, fft_real, freqs_sim, fft_sim = compare_fft_real_vs_sim(real_df, sim_df, tow_length_mm, plot=False)

        # Interpolate both FFTs onto a common frequency axis
        f_common = np.linspace(0, min(freqs_real.max(), freqs_sim.max()), 500)
        fft_real_interp = np.interp(f_common, freqs_real, fft_real)
        fft_sim_interp = np.interp(f_common, freqs_sim, fft_sim)

        # Compute error (MSE)
        error = np.mean((fft_real_interp - fft_sim_interp) ** 2)

        if error < best_error:
            best_error = error
            best_params = (n_steps, n_bins)
            best_result = (f_common, fft_real_interp, fft_sim_interp)

    # Plot best result
    if plot_best and best_result is not None:
        f_common, fft_real_best, fft_sim_best = best_result
        plt.figure(figsize=(10, 6))
        plt.plot(f_common, fft_real_best, label="Real Tow FFT", color="blue")
        plt.plot(f_common, fft_sim_best, label="Simulated Tow FFT (best fit)", color="red", alpha=0.7)
        plt.xlabel("Spatial frequency (1/mm)", fontsize=14)
        plt.ylabel("Normalized magnitude", fontsize=14)
        plt.title(f"Best FFT Match: steps={best_params[0]}, bins={best_params[1]}", fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"Best parameters: steps={best_params[0]}, bins={best_params[1]} (error={best_error:.6f})")
    return best_params, best_error

##############################################################################################################
"""Run this file"""

def main():
    #plot_real_tow(15)
    # for tow in range(1,32):
    #    print(tow)
    #    plot_Layup_vs_Traverse_tow(tow)

    # plot_Layup_vs_Traverse_tow(2)

    real_df, sim_df = plot_simulated_vs_real_tow(8, plot = True)
    compare_fft_real_vs_sim(real_df, sim_df)

    # best_params, best_error = optimize_fft_match(tow=8, steps_range=(600, 1000), bins_range=(60, 100), n_trials=50)

    # compare_simulated_vs_real_tow(8)
    # compare_multiple_simulations(8, 50)

if __name__ == "__main__":
    main()