"""This file deals with traverse data handling and plotting
   Written by: Martijn van der Voort, Clifton-John Walle and Manuel Cruz."""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pareto
import os
from tqdm import tqdm

# Internal imports
from constants import NOMINAL_LLS_A, NOMINAL_CAM, NOMINAL_LLS_B, NOMINAL_LT_Y, y_offset_traverse, y_increment_traverse, frame_width_traverse, tow_width_specified, number_of_steps
from Handling_ALL_Functions import get_synced_data
from Data_ALL_importer import Traverse_LT_excel_to_array, Traverse_Gap_excel_to_array

##############################################################################################################
"""Functions"""

def traverse_LT_viewer(tow: int):
    """
    Plot LT y-data along x for a single tow, using Z-synced traverse data.
    Shows left and right edges of the tow.
    """
    # --- Load trimmed traverse data ---
    bottom_tow_data = get_synced_data(tow - 1, "Traverse", overwrite=True) 
    top_tow_data = get_synced_data(tow, "Traverse", overwrite=True)

    # --- Extract data for right edge ---
    x_bottom = bottom_tow_data["LT_x"].to_numpy()
    LT_y_bottom = bottom_tow_data["LT_y"].to_numpy()
    bottom_edge = bottom_tow_data["Gap_leftedge"].to_numpy()

    # --- Extract data for left edge ---
    x_top = top_tow_data["LT_x"].to_numpy()
    LT_y_top = top_tow_data["LT_y"].to_numpy()
    top_edge = top_tow_data["Gap_rightedge"].to_numpy()

    # --- Calculate y positions of edges ---
    y_bottom = LT_y_bottom + 0.5 * frame_width_traverse - bottom_edge
    y_top = LT_y_top + 0.5 * frame_width_traverse - top_edge

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(x_bottom, LT_y_bottom, "--", color="orange", label="Raw LT_y right")
    plt.plot(x_top, LT_y_top, "--", color="cyan", label="Raw LT_y left")
    plt.plot(x_bottom, y_bottom, "-", color="red", linewidth=2, label="Edge right")
    plt.plot(x_top, y_top, "-", color="blue", linewidth=2, label="Edge left")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title(f"Traverse LT_y and edges for Tow {tow}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def traverse_tow_constructor(tow: int, normalize: bool = False):
    """Construct edge lines of a tow from traverse data (outliers already removed in get_synced_data)."""
    
    if tow not in range(2, 31):
        print("Tow 1 or 31 cannot be recreated from traverse data.")
        print("Provide a tow number between 2 and 30 inclusive.")
        return None

    # --- Load synced & trimmed data for adjacent gaps ---
    bottom_tow_data = get_synced_data(tow - 1, "Traverse") 
    top_tow_data = get_synced_data(tow, "Traverse")

    # --- Extract relevant data ---
    x_bottom = bottom_tow_data["LT_x"].to_numpy()
    y_bottom = bottom_tow_data["LT_y"].to_numpy()
    #bottom_edge = bottom_tow_data["Gap_rightedge"].to_numpy()
    bottom_edge = bottom_tow_data["Gap_leftedge"].to_numpy()

    x_top = top_tow_data["LT_x"].to_numpy()
    y_top = top_tow_data["LT_y"].to_numpy()
    #top_edge = top_tow_data["Gap_leftedge"].to_numpy()
    top_edge = top_tow_data["Gap_rightedge"].to_numpy()

    # --- Truncate all arrays to the shortest length to ensure alignment ---
    min_len = min(len(x_bottom), len(y_bottom), len(bottom_edge), len(x_top), len(y_top), len(top_edge))
    x_bottom = x_bottom[:min_len]
    y_bottom = y_bottom[:min_len]
    bottom_edge = bottom_edge[:min_len]
    x_top = x_top[:min_len]
    y_top = y_top[:min_len]
    top_edge = top_edge[:min_len]

    # --- Calculate y edges and centerline ---
    y_bottom_edge = y_bottom - bottom_edge
    y_top_edge = y_top - top_edge
    y_centerline = (y_bottom_edge + y_top_edge)/2

    # --- Translate the tow down to y = 0 ---
    if normalize == True:
        real_offset = 112 + (tow - 1)*y_increment_traverse
        y_centerline = y_centerline - real_offset
        y_top_edge = y_top_edge - real_offset
        y_bottom_edge = y_bottom_edge - real_offset

    # --- Construct final dataframe ---
    traverse_tow = pd.DataFrame({
        "x_right": x_bottom,
        "y_right": y_bottom_edge,
        "x_left": x_top,
        "y_left": y_top_edge,
        "x_centerline": x_bottom,
        "y_centerline": y_centerline})

    return traverse_tow

def traverse_tow_gaps_and_overlaps(plot=True, tow_spacing=None, print_statement=True):
    """
    Collect normalized traverse tow data (tows 2–30).
    Apply +6.35 mm offset per tow index after tow 2.
    Compute gap/overlap percentages between adjacent tows.
    """

    top_edge_paths, bottom_edge_paths = [], []
    x_vals_list = []

    if tow_spacing == None:
        tow_spacing = tow_width_specified

    # --- Collect traverse tow edges with offsets ---
    for tow in range(2, 31):  # Tow 2..30
        traverse_tow = traverse_tow_constructor(tow, normalize=True)
        if traverse_tow is None:
            continue

        # Offset in y direction
        offset_mm = (tow - 2) * tow_spacing

        x_vals_list.append(traverse_tow["x_centerline"].to_numpy())
        top_edge_paths.append(traverse_tow["y_left"].to_numpy() + offset_mm)
        bottom_edge_paths.append(traverse_tow["y_right"].to_numpy() + offset_mm)

    # --- Truncate all arrays to the global minimum length ---
    min_len = min(len(arr) for arr in x_vals_list)
    x_vals = x_vals_list[0][:min_len]  # use first tow's x-values, cut to min length
    top_edge_paths = [arr[:min_len] for arr in top_edge_paths]
    bottom_edge_paths = [arr[:min_len] for arr in bottom_edge_paths]

    # --- Gap/Overlap analysis ---
    gap_overlap_dict = {
        f"Gap/overlap_Tow{tow_idx+2}_Tow{tow_idx+3}": 
            bottom_edge_paths[tow_idx+1] - top_edge_paths[tow_idx]
        for tow_idx in range(len(top_edge_paths) - 1)
    }
    gap_overlap_df = pd.DataFrame(gap_overlap_dict, index=x_vals)

    gap_df = gap_overlap_df.where(gap_overlap_df > 0)
    overlap_df = gap_overlap_df.where(gap_overlap_df < 0)

    # --- Area calculations ---
    highest_tow_edge = top_edge_paths[-1]
    lowest_tow_edge = bottom_edge_paths[0]
    total_layout_area = np.trapezoid(highest_tow_edge - lowest_tow_edge, x_vals)

    total_gap_area = sum(np.trapezoid(np.clip(values, 0, None), x_vals) for values in gap_overlap_df.values.T)
    total_overlap_area = sum(np.trapezoid(np.clip(-values, 0, None), x_vals) for values in gap_overlap_df.values.T)

    gap_percent = (total_gap_area / total_layout_area) * 100 if total_layout_area > 0 else 0
    overlap_percent = (total_overlap_area / total_layout_area) * 100 if total_layout_area > 0 else 0

    # --- Print summary ---
    if print_statement == True:
        print(f"\nTotal layout area: {total_layout_area:.2f} mm²")
        print(f"Gap area: {total_gap_area:.2f} mm² ({gap_percent:.2f}%)")
        print(f"Overlap area: {total_overlap_area:.2f} mm² ({overlap_percent:.2f}%)")

    # --- Plotting ---
    if plot:
        plt.figure(figsize=(10, 6))
        for i, (top, bottom) in enumerate(zip(top_edge_paths, bottom_edge_paths)):
            color = plt.get_cmap("tab10")(i % 10)
            tow_number = i + 2
            plt.plot(x_vals, (top + bottom) / 2, "--", color=color, label=f"Tow {tow_number} centerline")
            plt.plot(x_vals, top, "-", color=color)
            plt.plot(x_vals, bottom, "-", color=color)
        plt.xlabel("Tow length (mm)")
        plt.ylabel("Tow position (mm)")
        plt.title("Traverse Tow Layout with 12.5 mm Offsets")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent

def traverse_tow_gaps_and_overlaps_lengths(plot=True, histogram_bins=30, force_steps=False):
    """
    Compute lengths of gaps and overlaps between normalized traverse tows (2–30),
    applying +6.35 mm offset per tow index after tow 2.
    Optionally plot histograms and fit Pareto distributions.
    
    Args:
        plot (bool): Whether to plot histograms of gap and overlap lengths.
        histogram_bins (int): Number of bins for histograms.
        force_steps (bool): If True, resample all tows to a fixed number of steps.
        number_of_steps (int): Target number of samples per tow if force_steps=True.
    """

    top_edge_paths, bottom_edge_paths = [], []
    x_vals_list = []

    # --- Collect traverse tow edges with offsets ---
    for tow in range(2, 31):  # Tow 2..30
        traverse_tow = traverse_tow_constructor(tow, normalize=True)
        if traverse_tow is None:
            continue

        # --- Apply force_steps logic here ---
        if force_steps:
            target_points = number_of_steps
            n_points = len(traverse_tow["x_centerline"])
            if n_points > target_points:
                indices = np.linspace(0, n_points - 1, target_points, dtype=int)
                for col in ["x_right", "y_right", "x_left", "y_left", "x_centerline", "y_centerline"]:
                    if col in traverse_tow.columns:
                        traverse_tow[col] = traverse_tow[col].iloc[indices].reset_index(drop=True)
                traverse_tow = traverse_tow.reset_index(drop=True)

        offset_mm = (tow - 2) * tow_width_specified, #Op 6.272
        x_vals_list.append(traverse_tow["x_centerline"].to_numpy())
        top_edge_paths.append(traverse_tow["y_left"].to_numpy() + offset_mm)
        bottom_edge_paths.append(traverse_tow["y_right"].to_numpy() + offset_mm)

    # --- Truncate all arrays to the global minimum length ---
    min_len = min(len(arr) for arr in x_vals_list)
    x_vals = x_vals_list[0][:min_len]
    top_edge_paths = [arr[:min_len] for arr in top_edge_paths]
    bottom_edge_paths = [arr[:min_len] for arr in bottom_edge_paths]

    # --- Compute gap/overlap arrays between adjacent tows ---
    gap_overlap_list = [
        bottom_edge_paths[i + 1] - top_edge_paths[i] for i in range(len(top_edge_paths) - 1)
    ]

    # --- Helper to extract continuous segment lengths ---
    def extract_lengths(values, positive=True):
        mask = values > 0 if positive else values < 0
        lengths = []
        run_length = 0
        for i in range(len(mask)):
            if mask[i]:
                run_length += 1
            elif run_length > 0:
                lengths.append(run_length)
                run_length = 0
        if run_length > 0:
            lengths.append(run_length)
        dx = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 1
        return np.array(lengths) * dx

    # --- Extract lengths for all tow pairs ---
    gap_lengths, overlap_lengths = [], []
    for gap_overlap in gap_overlap_list:
        gap_lengths.extend(extract_lengths(gap_overlap, positive=True))
        overlap_lengths.extend(extract_lengths(gap_overlap, positive=False))

    gap_lengths = np.array(gap_lengths)
    overlap_lengths = np.array(overlap_lengths)

    # --- Pareto fit helper ---
    def fit_pareto(data):
        if len(data) == 0:
            return {"shape": 0, "loc": 0, "scale": 0, "mean": 0, "std": 0}
        shape, loc, scale = pareto.fit(data, floc=0)
        mean = pareto.mean(shape, loc=loc, scale=scale)
        std = pareto.std(shape, loc=loc, scale=scale)
        return {"shape": shape, "loc": loc, "scale": scale, "mean": mean, "std": std}

    gap_fit = fit_pareto(gap_lengths)
    overlap_fit = fit_pareto(overlap_lengths)

    # --- Plot histograms ---
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for i, (data, title, fit) in enumerate([
            (gap_lengths, "Gap Lengths (Pareto Fit)", gap_fit),
            (overlap_lengths, "Overlap Lengths (Pareto Fit)", overlap_fit)
        ]):
            if len(data):
                counts, bins, _ = ax[i].hist(
                    data,
                    bins=histogram_bins,
                    density=False,
                    alpha=0.7,
                    edgecolor="black",
                    label="Empirical counts"
                )
                x = np.linspace(min(data), max(data), 400)
                pdf = pareto.pdf(x, fit["shape"], loc=fit["loc"], scale=fit["scale"])
                bin_width = bins[1] - bins[0]
                pdf_scaled = pdf * len(data) * bin_width
                ax[i].plot(x, pdf_scaled, "r-", linewidth=2, label=f"Pareto α={fit['shape']:.2f}")
                ax[i].axvline(fit["mean"], color="blue", linestyle="--", linewidth=1.5,
                              label=f"Mean={fit['mean']:.2f} mm")
                ax[i].set_xlabel("Length (mm)")
                ax[i].set_ylabel("Count")
                ax[i].set_title(title)
                ax[i].legend(fontsize=9)
                ax[i].grid(True, linestyle=":")

        plt.tight_layout()
        plt.show()

    return gap_lengths, overlap_lengths, gap_fit, overlap_fit

def LT_velocity_check(tow: int):
    # --- Load data ---
    LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
    t_data = LT_arr[:, 0]   # time
    x_data = LT_arr[:, 1]   # x
    z_data = LT_arr[:, 3]   # z (not used here)

    # --- Find first continuous segment where 0 <= x <= 1000 ---
    mask = (x_data >= 0) & (x_data <= 1000)
    if not np.any(mask):
        raise ValueError("No x values between 0 and 1000 mm in this dataset.")

    start_idx = np.argmax(mask)  # first True
    end_idx = start_idx
    while end_idx < len(mask) and mask[end_idx]:
        end_idx += 1

    # --- Trim data ---
    t_trim = t_data[start_idx:end_idx]
    x_trim = x_data[start_idx:end_idx]
    z_trim = z_data[start_idx:end_idx]

    # --- Compute velocities ---
    v_inst = np.gradient(x_trim, t_trim)  # instantaneous velocity
    v_const = (x_trim[-1] - x_trim[0]) / (t_trim[-1] - t_trim[0])  # constant
    v_const_line = np.full_like(t_trim, v_const)

    # --- Plot ---
    plt.figure(figsize=(10, 6))

    # Position
    plt.subplot(2, 1, 1)
    plt.plot(t_trim, x_trim, label="x(t)", color="steelblue")
    plt.axhline(0, color="black", linestyle="--", alpha=0.6, label="0 mm cutoff")
    plt.axhline(1000, color="red", linestyle="--", alpha=0.6, label="1000 mm cutoff")
    plt.xlabel("Time (s)")
    plt.ylabel("X Position (mm)")
    plt.title(f"Tow {tow} Position and Velocity (trimmed 0–1000 mm)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Velocity
    plt.subplot(2, 1, 2)
    plt.plot(t_trim, v_inst, label="Instantaneous Velocity", color="darkorange")
    plt.plot(t_trim, v_const_line, "--", label=f"Constant Velocity = {v_const:.3f}", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mm/s)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    return t_trim, x_trim, z_trim, v_inst, v_const

def GAP_velocity_check(tow: int):
    # --- Load data ---
    Gap_arr, Gap_cols = Traverse_Gap_excel_to_array(tow)
    t_data = Gap_arr[:, 0]       # seconds (already elapsed time)
    gap_data = Gap_arr[:, 3]     # actual gap values, but we’ll track along-tow position

    # --- Build virtual x-axis for 1000 mm tow ---
    # Sampling rate = 4 ms
    dt = 0.004
    n_points = len(t_data)
    x_data = np.linspace(0, 1000, n_points)   # assume evenly spaced along tow length

    # --- Instantaneous velocity (mm/s) ---
    v_inst = np.gradient(x_data, t_data)

    # --- Average velocity ---
    v_avg = 1000 / (t_data[-1] - t_data[0])
    v_avg_line = np.full_like(t_data, v_avg)

    # --- Plot ---
    plt.figure(figsize=(10, 6))

    # Gap vs position
    plt.subplot(2, 1, 1)
    plt.plot(x_data, gap_data, label="Gap", color="steelblue")
    plt.xlabel("Tow Position (mm)")
    plt.ylabel("Gap (mm)")
    plt.title(f"Tow {tow} Gap and Velocity (0–1000 mm)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Velocity comparison
    plt.subplot(2, 1, 2)
    plt.plot(t_data, v_inst, label="Instantaneous Velocity", color="darkorange")
    plt.plot(t_data, v_avg_line, "--", label=f"Average Velocity = {v_avg:.3f}", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mm/s)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    return t_data, x_data, gap_data, v_inst, v_avg

def LT_z_check(tow: int):
    # --- Load data ---
    LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
    t_data = LT_arr[:, 0]   # time
    x_data = LT_arr[:, 1]   # x
    z_data = LT_arr[:, 3]   # z

    # --- Find first continuous segment where 0 <= x <= 1000 ---
    mask = (x_data >= -1000) & (x_data <= 2000)
    if not np.any(mask):
        raise ValueError("No x values between 0 and 1000 mm in this dataset.")

    start_idx = np.argmax(mask)  # first True
    end_idx = start_idx
    while end_idx < len(mask) and mask[end_idx]:
        end_idx += 1

    # --- Trim data ---
    t_trim = t_data[start_idx:end_idx]
    x_trim = x_data[start_idx:end_idx]
    z_trim = z_data[start_idx:end_idx]

    # --- Plot ---
    plt.figure(figsize=(10, 8))

    # z over x
    plt.subplot(2, 1, 1)
    plt.plot(x_trim, z_trim, label="z(x)", color="purple")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Z Position (mm)")
    plt.title(f"Tow {tow} Z over X (trimmed 0–1000 mm)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # z over time
    plt.subplot(2, 1, 2)
    plt.plot(t_trim, z_trim, label="z(t)", color="purple")
    plt.xlabel("Time (s)")
    plt.ylabel("Z Position (mm)")
    plt.title(f"Tow {tow} Z over Time (trimmed 0–1000 mm)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return t_trim, x_trim, z_trim

def plot_all_tows_trimmed():
    tow_numbers = range(1, 32)
    all_data = []

    for tow in tow_numbers:
        LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
        t_data = LT_arr[:, 0]
        x_data = LT_arr[:, 1]
        z_data = LT_arr[:, 3]

        # --- Mask where x is in [0, 1000] ---
        mask = (x_data >= 0) & (x_data <= 1000)
        if not np.any(mask):
            raise ValueError(f"Tow {tow} has no x values between 0 and 1000 mm")

        # First index where condition is True
        start_idx = np.argmax(mask)

        # Last continuous index before x leaves [0, 1000]
        end_idx = start_idx
        while end_idx < len(mask) and mask[end_idx]:
            end_idx += 1

        # --- Trim data ---
        t_trim = t_data[start_idx:end_idx]
        x_trim = x_data[start_idx:end_idx]
        z_trim = z_data[start_idx:end_idx]

        all_data.append((t_trim, x_trim, z_trim))

    # --- Plot ---
    plt.figure(figsize=(12, 10))

    # z vs x
    plt.subplot(2, 1, 1)
    for tow, (t_trim, x_trim, z_trim) in zip(tow_numbers, all_data):
        plt.plot(x_trim, z_trim, label=f"Tow {tow}")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Z Position (mm)")
    plt.title("Z vs X for all tows (individually trimmed to first 0–1000 mm region)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # z vs time
    plt.subplot(2, 1, 2)
    for tow, (t_trim, x_trim, z_trim) in zip(tow_numbers, all_data):
        plt.plot(t_trim, z_trim, label=f"Tow {tow}")
    plt.xlabel("Time (s)")
    plt.ylabel("Z Position (mm)")
    plt.title("Z vs Time for all tows (individually trimmed to first 0–1000 mm region)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return all_data

def plot_lt_y_error_histogram(tow: int, bins: int = 50):
    """
    Plots histogram of LT_y error (measured - nominal position) for a given tow.
    Overlays mean and standard deviation as vertical lines.
    """
    # --- Load traverse data ---
    df = get_synced_data(tow, "Traverse")

    if "LT_y" not in df.columns:
        raise KeyError("Traverse data does not contain 'LT_y' column")

    # --- Nominal tow position ---
    nominal_y = 125 + (tow - 1) * 12.5

    # --- Compute error ---
    lt_y_error = df["LT_y"].dropna().values - nominal_y

    if len(lt_y_error) == 0:
        raise ValueError(f"No LT_y data available for tow {tow}")

    # --- Compute statistics ---
    mean_val = np.mean(lt_y_error)
    std_val = np.std(lt_y_error)

    # --- Plot histogram ---
    plt.figure(figsize=(8, 5))
    counts, bins_edges, _ = plt.hist(lt_y_error, bins=bins, density=True, alpha=0.6, color="orange", edgecolor="black")

    # Overlay Gaussian approximation
    x = np.linspace(min(lt_y_error), max(lt_y_error), 500)
    y = (1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_val) / std_val) ** 2)
    plt.plot(x, y, "r--", linewidth=2, label="Gaussian approx.")

    # Overlay mean and std lines
    plt.axvline(mean_val, color="red", linestyle="-", linewidth=2, label=f"Mean = {mean_val:.3f}")
    plt.axvline(mean_val - std_val, color="green", linestyle="--", linewidth=1.5, label=f"± Std = {std_val:.3f}")
    plt.axvline(mean_val + std_val, color="green", linestyle="--", linewidth=1.5)

    plt.title(f"LT_y Error Distribution (Tow {tow})")
    plt.xlabel("LT_y Error [mm]")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def analyze_real_tow_spacing_effect(spacing_values_mm: list = None, print_progress: bool = True, existing_data: pd.DataFrame | str | None = None):
    """
    Analyze the effect of tow spacing on gap and overlap percentage using *real traverse tows*.

    This version behaves like `analyze_tow_spacing_effect()`, but instead of running synthetic 
    simulations with `generate_RW_multitow()`, it uses the real traverse data via 
    `traverse_tow_gaps_and_overlaps()` for each tested spacing value.

    Parameters
    ----------
    spacing_values_mm : list of float, optional
        Tow spacing values (mm) to analyze. If None, uses np.linspace(5.0, 7.5, 9).
    tow_width_mm : float, optional
        Nominal tow width (default 6.35 mm).
    tow_length_mm : float, optional
        Tow length in mm (default 1000).
    print_progress : bool, optional
        Print progress updates (default True).
    existing_data : pd.DataFrame | str | None
        If provided, plots data from a CSV file or DataFrame instead of re-running analysis.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with average gap and overlap percentages vs. tow spacing,
        including intersection spacing and value columns.
    """

    # --- CASE 1: Data provided (plot only) ---
    if existing_data is not None:
        if isinstance(existing_data, str):
            results_df = pd.read_csv(existing_data)
        elif isinstance(existing_data, pd.DataFrame):
            results_df = existing_data.copy()
        else:
            raise ValueError("`existing_data` must be a pandas DataFrame or a CSV file path.")

    # --- CASE 2: Run real data analysis ---
    else:
        if spacing_values_mm is None:
            spacing_values_mm = np.linspace(5.0, 7.5, 9)

        avg_gap_percentages = []
        avg_overlap_percentages = []

        for spacing in tqdm(spacing_values_mm, desc="Analyzing real tow spacing"):
            if print_progress:
                print(f"\n--- Analyzing for tow spacing = {spacing:.2f} mm ---")

            try:
                _, _, _, gap_percent, overlap_percent = traverse_tow_gaps_and_overlaps(
                    plot=False, tow_spacing=spacing, print_statement=False)

                avg_gap_percentages.append(gap_percent)
                avg_overlap_percentages.append(overlap_percent)

                if print_progress:
                    print(f"  → Gap: {gap_percent:.3f}% | Overlap: {overlap_percent:.3f}%")

            except Exception as e:
                print(f"⚠️ Skipped spacing {spacing:.2f} mm due to error: {e}")
                avg_gap_percentages.append(np.nan)
                avg_overlap_percentages.append(np.nan)

        # Create DataFrame
        results_df = pd.DataFrame({
            "Tow Spacing (mm)": spacing_values_mm,
            "Average Gap (%)": avg_gap_percentages,
            "Average Overlap (%)": avg_overlap_percentages})

    # --- Find intersection (where gap = overlap) ---
    gap_arr = np.array(results_df["Average Gap (%)"])
    overlap_arr = np.array(results_df["Average Overlap (%)"])
    spacing_values_mm = np.array(results_df["Tow Spacing (mm)"])
    diff = gap_arr - overlap_arr

    intersection_spacing = None
    intersection_gap_value = None

    for i in range(len(diff) - 1):
        if np.isnan(diff[i]) or np.isnan(diff[i + 1]):
            continue
        if diff[i] * diff[i + 1] < 0:
            x1, x2 = spacing_values_mm[i], spacing_values_mm[i + 1]
            y1, y2 = diff[i], diff[i + 1]
            intersection_spacing = x1 - y1 * (x2 - x1) / (y2 - y1)

            g1, g2 = gap_arr[i], gap_arr[i + 1]
            intersection_gap_value = g1 + (g2 - g1) * ((intersection_spacing - x1) / (x2 - x1))
            break

    # --- Add intersection info ---
    results_df["Intersection Spacing (mm)"] = intersection_spacing
    results_df["Intersection Gap/Overlap (%)"] = intersection_gap_value

    # --- Save CSV ---
    if existing_data is None:
        os.makedirs("Cached Data", exist_ok=True)
        csv_path = os.path.join(
            "Cached Data",
            f"Tow_spacing_effect_Traverse.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n✅ Results (including intersection columns) saved to: {csv_path}")

    # --- Plot ---
    plt.figure(figsize=(9.25, 2.90))
    ax = plt.gca()

    plt.plot(spacing_values_mm, gap_arr, color="blue", label="Gap", linewidth=1)
    plt.plot(spacing_values_mm, overlap_arr, color="red", label="Overlap", linewidth=1)

    plt.xlabel("Programmed shift (mm)", fontname="Times New Roman", fontsize=15)
    plt.ylabel("Defect area (%)", fontname="Times New Roman", fontsize=15)

    plt.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("black")

    plt.xticks(fontname="Times New Roman", fontsize=15)
    plt.yticks(fontname="Times New Roman", fontsize=15)
    plt.legend(prop={"family": "Times New Roman", "size": 15})
    plt.tight_layout()
    
    ax.tick_params(
        top=True, bottom=True, left=True, right=True,
        direction='in', length=6, width=1)

    xmin, xmax = 5, 7.5
    ymin, ymax = 0, 10
    ax.set_xlim(xmin - 0.02*(xmax - xmin), xmax + 0.02*(xmax - xmin))
    ax.set_ylim(ymin - 0.1*(ymax - ymin), ymax + 0.1*(ymax - ymin))

    plt.show()

    # --- Print intersection info ---
    if intersection_spacing is not None:
        print(f"\n🔴 Intersection point at {intersection_spacing:.3f} mm "
              f"(Gap ≈ Overlap = {intersection_gap_value:.3f}%)")
    else:
        print("\n⚠️ No intersection found between gap and overlap curves.")

    return results_df

##############################################################################################################
"""Run this file"""

def main():
    #traverse_LT_viewer(10)
    #traverse_tow_gaps_and_overlaps_lengths(plot=True, histogram_bins=30, force_steps=False)
    # z_check(5)
    # LT_velocity_check(5)
    # GAP_velocity_check(5)
    #plot_all_tows_trimmed()
    # print(traverse_tow_constructor(5))
    # gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent=traverse_tow_gaps_and_overlaps()
    # print(gap_overlap_df)
    #plot_lt_y_error_histogram(10)
    analyze_real_tow_spacing_effect(existing_data="Cached Data/Tow_spacing_effect_Traverse.csv")
    

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else