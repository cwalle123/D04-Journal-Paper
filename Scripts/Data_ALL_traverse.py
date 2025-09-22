"""This file deals with traverse data handling and plotting"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Internal imports
from constants import NOMINAL_LLS_A, NOMINAL_CAM, NOMINAL_LLS_B, NOMINAL_LT_Y, y_offset_traverse, y_increment_traverse, frame_width_traverse
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

def traverse_tow_constructor(tow: int):
    """Construct edge lines of a tow from traverse data (outliers already removed in get_synced_data)."""
    
    if tow not in range(2, 31):
        print("Tow 1 or 31 cannot be recreated from traverse data.")
        print("Provide a tow number between 2 and 30 inclusive.")
        return None

    # --- Load synced & trimmed data for adjacent gaps ---
    bottom_tow_data = get_synced_data(tow - 1, "Traverse", overwrite=True) 
    top_tow_data = get_synced_data(tow, "Traverse", overwrite=True)

    # --- Extract relevant data ---
    x_bottom = bottom_tow_data["LT_x"].to_numpy()
    y_bottom = bottom_tow_data["LT_y"].to_numpy()
    bottom_edge = bottom_tow_data["Gap_rightedge"].to_numpy()

    x_top = top_tow_data["LT_x"].to_numpy()
    y_top = top_tow_data["LT_y"].to_numpy()
    top_edge = top_tow_data["Gap_leftedge"].to_numpy()

    # --- Truncate all arrays to the shortest length to ensure alignment ---
    min_len = min(len(x_bottom), len(y_bottom), len(bottom_edge), len(x_top), len(y_top), len(top_edge))
    x_bottom = x_bottom[:min_len]
    y_bottom = y_bottom[:min_len]
    bottom_edge = bottom_edge[:min_len]
    x_top = x_top[:min_len]
    y_top = y_top[:min_len]
    top_edge = top_edge[:min_len]

    # --- Calculate y positions using frame width ---
    # y_offset = 0.5 * frame_width_traverse
    # y_bottom = y_bottom + y_offset - bottom_edge
    # y_top = y_top + y_offset - top_edge

    y_bottom_edge = y_bottom + bottom_edge
    y_top_edge = y_top + top_edge

    # --- Construct final dataframe ---
    traverse_tow = pd.DataFrame({
        "x_right": x_bottom,
        "y_right": y_bottom_edge,
        "x_left": x_top,
        "y_left": y_top_edge})

    return traverse_tow

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

##############################################################################################################
"""Run this file"""

def main():
    # traverse_LT_viewer(2)
    # z_check(5)
    # LT_velocity_check(5)
    # GAP_velocity_check(5)
    # plot_all_tows_trimmed()
    print(traverse_tow_constructor(5))

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else