"""This file deals with traverse data handling and plotting"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Internal imports
from constants import NOMINAL_LLS_A, NOMINAL_CAM, NOMINAL_LLS_B, NOMINAL_LT_Y, y_offset_traverse, y_increment_traverse, frame_width_traverse
from Handling_ALL_Functions import get_synced_data
from Data_ALL_importer import Traverse_LT_excel_to_array

##############################################################################################################
"""Functions"""

def traverse_tow_constructor(tow: int):
    """Function to construct edge lines of tows from traverse data.
        Edge lines can be plotted and used in FFT validation.
        
        Arguments:
            tow (int): tow number
        
        Returns:
            traverse_tow (DataFrame): Dataframe consisting of coordinates for the left and right edges of the tow
            
        Note: Currently assumes 2 things:
                1) Frame width (frame_width_traverse) of LLS B, resulting in incorrect gap size, but correct shape of tow edges
                2) Frame center (y_offset_traverse, y_increment_traverse) of LLS B, resulting in incorrect
                    absolute y-coordinates of tow edges, but correct shape of tow edges
    """
    # Check for valid tow number
    if tow not in range(2, 31):
        print(f'Tow 1 or 31 can not be recreated from traverse data.')
        print(f'Provide a tow number between 2 and 30 inclusive')
        return None
    
    elif tow in range(2,31): 
        # Load synced and interpolated data of both gaps adjacent to tow, set rows outside range of tow (0-1000 mm) equal to NaN
        gap_right_data_full = get_synced_data((tow-1), "Traverse_Interpolated")
        gap_left_data_full = get_synced_data(tow, "Traverse_Interpolated")
        gap_right_data_full = get_synced_data((tow-1), "Traverse_Interpolated_Z_Sync") # Temporarily overwrites the previous calls of get_synced_data to see the Z synced data
        gap_left_data_full = get_synced_data(tow, "Traverse_Interpolated_Z_Sync") # Temporarily overwrites the previous calls of get_synced_data to see the Z synced data
        gap_right_data = gap_right_data_full.where((gap_right_data_full["LT_x"] > 0) & (gap_right_data_full["LT_x"] < 1000))
        gap_left_data = gap_left_data_full.where((gap_left_data_full["LT_x"] > 0) & (gap_left_data_full["LT_x"] < 1000))

        # Extract relevant data for right edge (lower side)
        x_right = gap_right_data["LT_x"].to_numpy()
        LT_y_right = gap_right_data["LT_y"].to_numpy()
        edge_right = gap_right_data["Gap_leftedge"].to_numpy() #Note: Gap_leftedge is intentionally selected
        
        # Extract relevant data for left edge (upper side)
        x_left = gap_left_data["LT_x"].to_numpy()
        LT_y_left = gap_left_data["LT_y"].to_numpy()
        edge_left = gap_left_data["Gap_rightedge"].to_numpy() #Note: Gap_rightedge is intentionally selected

        # Calculate y-positions of the right edge (lower side)
        y_right = LT_y_right + 0.5 * frame_width_traverse - edge_right

        # Calculate y-positions of the left edge (upper side)
        y_left = LT_y_left + 0.5 * frame_width_traverse - edge_left

        # Pad shortest columns with NaNs
        arrays = [x_right, y_right, x_left, y_left]
        max_len = max(map(len, arrays))
        x_right, y_right, x_left, y_left = [np.pad(arr, (0, max_len - len(arr)), mode="empty") for arr in arrays]

        # Construct final dataframe
        traverse_tow = pd.DataFrame({"x_right": x_right, "y_right": y_right, "x_left": x_left, "y_left": y_left})

    return traverse_tow

def raw_vs_interpolated_comparison(tow: int):
    """Plot raw data from Siddharth vs interpolated data from get_synced_data to compare.
    Visualize whether interpolation happens correctly"""
    #raw_data = 
    return NotImplementedError

def velocity_check(tow: int):
    # --- Load data ---
    LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
    t_data = LT_arr[:, 0]   # time
    x_data = LT_arr[:, 1]   # x
    z_data = LT_arr[:, 3]   # z (not used here)

    # --- Find first continuous segment where 0 <= x <= 1000 ---
    mask = (x_data >= 0) & (x_data <= 999.5)
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

def z_check(tow: int):
    # --- Load data ---
    LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
    t_data = LT_arr[:, 0]   # time
    x_data = LT_arr[:, 1]   # x
    z_data = LT_arr[:, 3]   # z

    # --- Find first continuous segment where 0 <= x <= 1000 ---
    mask = (x_data >= 0) & (x_data <= 999.5)
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
    last_indices = []

    # --- Load all tows and find last index where z < -0.03 ---
    for tow in tow_numbers:
        LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
        t_data = LT_arr[:, 0]
        x_data = LT_arr[:, 1]
        z_data = LT_arr[:, 3]

        # Find all indices where z < -0.03
        mask = z_data < -0.03
        if not np.any(mask):
            raise ValueError(f"Tow {tow} has no z < -0.03")

        last_idx = np.max(np.where(mask)[0])  # last index where condition is True
        last_indices.append(last_idx)
        all_data.append((t_data, x_data, z_data))

    # --- Determine shortest length to trim all tows ---
    trim_index = min(last_indices)

    plt.figure(figsize=(12, 10))

    # z vs x
    plt.subplot(2, 1, 1)
    for tow, (t_data, x_data, z_data) in zip(tow_numbers, all_data):
        plt.plot(x_data[:trim_index+1], z_data[:trim_index+1], label=f"Tow {tow}")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Z Position (mm)")
    plt.title(f"Z vs X for all tows trimmed to shortest last z<-0.03")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # z vs time
    plt.subplot(2, 1, 2)
    for tow, (t_data, x_data, z_data) in zip(tow_numbers, all_data):
        plt.plot(t_data[:trim_index+1], z_data[:trim_index+1], label=f"Tow {tow}")
    plt.xlabel("Time (s)")
    plt.ylabel("Z Position (mm)")
    plt.title(f"Z vs Time for all tows trimmed to shortest last z<-0.03")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return all_data, trim_index

##############################################################################################################
"""Run this file"""

def main():
    z_check(5)
    # plot_all_tows_trimmed()
    print(traverse_tow_constructor(2))

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else