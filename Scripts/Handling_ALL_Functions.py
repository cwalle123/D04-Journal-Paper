"""This file deals with all of the important functions that are used all throughout the project.
   Written by Martijn van der Voort, Clifton-John Walle and Sam Rotteveel"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Internal imports
from Data_ALL_importer import LLS_A_excel_to_array, LLS_B_excel_to_array, CAM_excel_to_array, LT_x_excel_to_array, LT_y_normalized_excel_to_array, Traverse_Gap_excel_to_array, Traverse_LT_excel_to_array
from constants import NOMINAL_LLS_A, NOMINAL_CAM, NOMINAL_LLS_B, NOMINAL_LT_Y, y_offset_traverse, y_increment_traverse, frame_width_traverse, TCP_LLS_B
CACHE_FOLDER = "Cached Data"

##############################################################################################################
"""Functions"""

# Functions for saving, loading, and purging data:
def save_cached_data(name: str, array: np.ndarray, columns: list[str]):
    """
    Save array to 'Cached Data' with given column names in first row.
    """
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    path = os.path.join(CACHE_FOLDER, f"{name}.csv")
    header = ",".join(columns)
    np.savetxt(path, array, delimiter=",", header=header, comments='')
    print(f"[CACHE] Saved '{name}' to {path}")

def load_cached_data(name: str) -> tuple[np.ndarray, list[str]]:
    """
    Load cached array and return (array, column_names).
    """
    path = os.path.join(CACHE_FOLDER, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached file found for '{name}'")

    with open(path, 'r') as f:
        first_line = f.readline().strip()
    col_names = first_line.split(",")

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data, col_names

def purge_cached_data():
    """
    Delete all cached CSVs.
    """
    if os.path.exists(CACHE_FOLDER):
        for file in os.listdir(CACHE_FOLDER):
            os.remove(os.path.join(CACHE_FOLDER, file))
        print("[CACHE] All cached data purged.")

# Functions for calling data:
def get_synced_data(tow: int, sensor_type: str, overwrite=False, helper=False, print_statement=False) -> pd.DataFrame:
    """
    Loads processed data for a given tow & sensor, with caching.
    Combines multiple arrays horizontally if needed.
    Adds error column based on nominal value for the sensor type.
    Returns a Pandas DataFrame instead of NumPy array.
    """
    if sensor_type not in ["LT", "LLS_A", "LLS_B", "CAM", "TRAVERSE_GAP", "TRAVERSE_LT", "Traverse"]:
        raise KeyError(f"The key '{sensor_type}' is invalid")
    if tow not in range(1, 32):
        raise IndexError(f"Tow ID {tow} is out of range")

    name = f"{sensor_type}_{tow}"

    # --- Try cache ---
    if not helper and not overwrite:
        try:
            cached_array, cached_cols = load_cached_data(name)
            if print_statement: print(f"[CACHE] Loaded '{name}' from cache")
            return pd.DataFrame(cached_array, columns=cached_cols)
        except FileNotFoundError:
            print(f"[CACHE] No cache found for '{name}'. Processing new data...")

    arrays = []
    col_names = []

    if sensor_type == "LT":
        arr1, cols1 = LT_x_excel_to_array(tow)  # X data
        arr2, cols2 = LT_y_normalized_excel_to_array(tow)  # Y normalized data
        arrays.append(arr1)
        col_names.extend(cols1)
        arrays.append(arr2)
        col_names.extend(cols2)

        # Error from normalized Y column (last one in LT data)
        error_col = arr2[:, 0] - NOMINAL_LT_Y
        arrays.append(error_col[:, None])
        col_names.append("error_LT")

    elif sensor_type == "CAM":
        arr, cols = CAM_excel_to_array(tow)
        arrays.append(arr)
        col_names.extend(cols)

        # Error from first column
        error_col = arr[:, 0] - NOMINAL_CAM
        arrays.append(error_col[:, None])
        col_names.append("error_CAM")

    elif sensor_type == "LLS_A":
        arr, cols = LLS_A_excel_to_array(tow)
        arrays.append(arr)
        col_names.extend(cols)

        # Error from first column
        error_col = arr[:, 0] - NOMINAL_LLS_A
        arrays.append(error_col[:, None])
        col_names.append("error_LLS_A")

    elif sensor_type == "LLS_B":
        arr, cols = LLS_B_excel_to_array(tow)
        arrays.append(arr)
        col_names.extend(cols)

        # Error from first column
        error_col = arr[:, 0] - NOMINAL_LLS_B
        arrays.append(error_col[:, None])
        col_names.append("error_LLS_B")

    elif sensor_type == "Traverse":
        """
        Load one tow, trim LT to first continuous region where 107 <= x <= 1107,
        reset LT time and x (start at 0),
        compute Gap x-position from instantaneous Gap velocity,
        interpolate Gap data onto LT_x using linear splines,
        horizontally stack, remove rows with outliers,
        and print info when outliers are removed.
        """

        # --- Load LT and Gap data ---
        LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
        Gap_arr, Gap_cols = Traverse_Gap_excel_to_array(tow) if tow < 31 else (None, None)

        if LT_arr.size == 0:
            raise ValueError(f"Tow {tow}: LT array empty.")

        # --- Trim LT to first continuous 107 <= x <= 1107 ---
        x_col = LT_arr[:, 1]
        mask = (x_col >= TCP_LLS_B) & (x_col <= (1000 + TCP_LLS_B))  # TCP_LLS_B from constants.py
        if not np.any(mask):
            raise ValueError(f"Tow {tow}: no x values between 107 and 1107.")

        start_idx = np.argmax(mask)
        end_idx = start_idx
        while end_idx < len(mask) and mask[end_idx]:
            end_idx += 1
        LT_arr = LT_arr[start_idx:end_idx, :]

        # --- Reset LT time and x to start at 0 ---
        LT_arr[:, 0] -= LT_arr[0, 0]
        LT_arr[:, 1] -= LT_arr[0, 1]
        LT_x = LT_arr[:, 1]

        # --- Process Gap data ---
        if Gap_arr is not None:
            # Gap time relative to start
            gap_time = Gap_arr[:, 0] - Gap_arr[0, 0]

            # --- Compute instantaneous gap velocity (mm/s) ---
            # Assume tow length = 1000 mm spread across Gap samples
            n_points = len(gap_time)
            gap_pos = np.linspace(0, 1000, n_points)  # this essentially applies the sampling rate of 4ms
            gap_v_inst = np.gradient(gap_pos, gap_time)

            # --- Compute gap_x from integrating instantaneous velocity ---
            gap_x = np.cumsum(gap_v_inst * np.gradient(gap_time))
            gap_x -= gap_x[0]  # start at 0

            # --- Interpolate Gap columns onto LT_x axis ---
            Gap_interp = np.zeros((LT_x.shape[0], 3))
            interp_kind = "linear"

            for i in range(1, 4):  # skip Gap time column
                f = interp1d(gap_x, Gap_arr[:, i], kind=interp_kind, bounds_error=False, fill_value=np.nan)
                Gap_interp[:, i - 1] = f(LT_x)

            # --- Combine LT and Gap data ---
            synced = np.column_stack([
                LT_arr[:, 0],       # time
                Gap_interp,         # Gap_leftedge, Gap_rightedge, Gap_gap
                LT_x,               # LT_x
                LT_arr[:, 2:4]])    # LT_y, LT_z
            
            synced_cols = ["time", "Gap_leftedge", "Gap_rightedge", "Gap_gap", "LT_x", "LT_y", "LT_z"]

            # --- Remove spike rows but keep NaNs ---
            spike_threshold = 0.21  # mm
            left_diff = np.diff(synced[:, 1], prepend=synced[0, 1])
            right_diff = np.diff(synced[:, 2], prepend=synced[0, 2])
            keep_mask = np.ones(synced.shape[0], dtype=bool)

            for idx in range(synced.shape[0]):
                if not np.isnan(left_diff[idx]) and np.abs(left_diff[idx]) > spike_threshold:
                    keep_mask[idx] = False
                    if print_statement: print(f"Tow {tow}: removed spike at x = {synced[idx, 4]:.2f} mm, Gap_left jump = {left_diff[idx]:.2f} mm")
                if not np.isnan(right_diff[idx]) and np.abs(right_diff[idx]) > spike_threshold:
                    keep_mask[idx] = False
                    if print_statement: print(f"Tow {tow}: removed spike at x = {synced[idx, 4]:.2f} mm, Gap_right jump = {right_diff[idx]:.2f} mm")

            synced = synced[keep_mask, :]

        else:
            synced = LT_arr
            synced_cols = LT_cols

        arrays.append(synced)
        col_names.extend(synced_cols)
    
    processed_data = arrays[0] if len(arrays) == 1 else np.hstack(arrays)
    #drop_cols = ["time", "leftedge", "rightedge", "gap"]  # adjust as needed
    #keep_cols = [c for c in col_names if c not in drop_cols]
    #keep_indices = [col_names.index(c) for c in keep_cols]
    #processed_data = processed_data[:, keep_indices]
    #col_names = keep_cols

    # Save to cache unless helper
    if not helper:
        save_cached_data(name, processed_data, col_names)

    # Return as DataFrame
    return pd.DataFrame(processed_data, columns=col_names)

def get_data(sensor: str, tows: list = list(np.arange(2, 32, 1)), format: str = "merged"):
    """
    This function gets the required data for the specified sensor.
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

        if format == "merged":
            # put data into correct format (pairs)
            for i in range(len(tow_data[:, 0])):
                data.append(float(tow_data[i, 0]))
                weights.append(float(tow_data[i, 1]))

        elif format == "separated":
            data.append(tow_data[:, 0])
            weights.append(tow_data[:, 1])

        elif format == "paired":
            """This option puts the data into pairs with their weights: 
            format = np.array([[1st_data_point, 2nd, weight], [2nd, 3rd, weight], ...])"""
            pairs = []
            for i in range(len(tow_data[:-1, 0])):
                point_0 = tow_data[i, 0]  # x_values in plot
                point_1 = tow_data[i + 1, 0]  # y_values in plot
                weight = 0.5 * tow_data[i, 1] + 0.5 * tow_data[i + 1, 1]
                pairs.append([point_0, point_1, weight])
            return np.array(pairs)

        else: print('Invalid format. Possible values are "merged" and "separated".')
    return data, weights   

##############################################################################################################
"""Run this file"""

def main():
    # x = get_synced_data(5, "Traverse", overwrite=False)

    # Just to check if the new data with weights is correct (it is)
    # for tow in range(1,32):
    #     x = get_synced_data(tow, "Traverse", overwrite=True)
    #     print(np.shape(x))

    #print("Columns:", x.columns.tolist())
    data, weights = get_data("CAM", format="separated")
    print(weights)
    print(get_synced_data(2, "LLS_B"))
    
if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
