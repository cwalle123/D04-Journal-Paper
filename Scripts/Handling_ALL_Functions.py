"""This file deals with all of the important functions that are used all throughout the project"""

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

##############################################################################################################
"""Functions for saving, loading, and purging data"""

CACHE_FOLDER = "Cached Data"

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

##############################################################################################################
"""Functions for calling data"""

def get_synced_data(tow: int, sensor_type: str, overwrite=False, helper=False) -> pd.DataFrame:
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
            print(f"[CACHE] Loaded '{name}' from cache")
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
        reset LT time (start at 0),
        compute Gap x-position from time (velocity = 50 mm/s),
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
        mask = (x_col >= TCP_LLS_B) & (x_col <= (1000 + TCP_LLS_B)) # TCP_LLS_B from constants.py
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
            # Compute Gap x from tow average velocity
            gap_time = Gap_arr[:, 0] - Gap_arr[0, 0]
            gap_velocity = 1000 / gap_time[-1] # average velocity for respective tow
            gap_x = gap_time * gap_velocity  # mm/s

            # Interpolate Gap columns onto LT_x axis
            Gap_interp = np.zeros((LT_x.shape[0], 3))
            interp_kind = "linear"

            for i in range(1, 4):  # skip Gap time column
                f = interp1d(gap_x, Gap_arr[:, i], kind=interp_kind, bounds_error=False, fill_value=np.nan)
                Gap_interp[:, i - 1] = f(LT_x)

            # --- Combine LT and Gap data ---
            synced = np.column_stack([
                LT_arr[:, 0],   # time
                Gap_interp,     # Gap_leftedge, Gap_rightedge, Gap_gap
                LT_x,           # LT_x
                LT_arr[:, 2:4]  # LT_y, LT_z
            ])
            synced_cols = ["time", "Gap_leftedge", "Gap_rightedge", "Gap_gap", "LT_x", "LT_y", "LT_z"]

            # --- Remove spike rows but keep NaNs ---
            spike_threshold = 0.3  # mm
            left_diff = np.diff(synced[:, 1], prepend=synced[0, 1])
            right_diff = np.diff(synced[:, 2], prepend=synced[0, 2])
            keep_mask = np.ones(synced.shape[0], dtype=bool)

            for idx in range(synced.shape[0]):
                if not np.isnan(left_diff[idx]) and np.abs(left_diff[idx]) > spike_threshold:
                    keep_mask[idx] = False
                    print(f"Tow {tow}: removed spike at x = {synced[idx, 4]:.2f} mm, Gap_left jump = {left_diff[idx]:.2f} mm")
                if not np.isnan(right_diff[idx]) and np.abs(right_diff[idx]) > spike_threshold:
                    keep_mask[idx] = False
                    print(f"Tow {tow}: removed spike at x = {synced[idx, 4]:.2f} mm, Gap_right jump = {right_diff[idx]:.2f} mm")

            synced = synced[keep_mask, :]

            # --- Remove rows with NaN ---
            # valid_rows = ~np.isnan(synced).any(axis=1)
            # synced = synced[valid_rows, :]

        else:
            synced = LT_arr
            synced_cols = LT_cols

        arrays.append(synced)
        col_names.extend(synced_cols)
    
    """elif sensor_type == "TRAVERSE_GAP":
        
        if tow == 1:
            arr, cols = Traverse_Gap_excel_to_array(tow)
            arr = np.array(arr, dtype=float)
            arrays.append(arr)
            col_names.extend(cols)

            #Calculate x-values from velocity and time and add in column
            x_col = (1.0 / arr[-1, 0]) * arr[:, 0]
            arrays.append(x_col[:, None])
            col_names.append("x")

            #Take left edge to equal outer edge of LLS B scan window
            y_col_LE = np.full_like(x_col, 125)
            arrays.append(y_col_LE[:, None])
            col_names.append("y_LE")

            #Calculate right edge from data
            y_col_RE = (y_offset_traverse + tow * y_increment_traverse + 0.5 * frame_width_traverse) - arr[:, 2]
            arrays.append(y_col_RE[:, None])
            col_names.append("y_RE")

        elif tow == 31:
            arr, cols = Traverse_Gap_excel_to_array((tow-1))
            arr = np.array(arr, dtype=float)
            arrays.append(arr)
            col_names.extend(cols)

            #Calculate x-values from velocity and time and add in column
            x_col = (1.0 / arr[-1, 0]) * arr[:, 0]
            arrays.append(x_col[:, None])
            col_names.append("x")

            #Calculate left edge from data
            y_col_LE = (y_offset_traverse + (tow - 1) * y_increment_traverse + 0.5 * frame_width_traverse) - arr[:, 1]
            arrays.append(y_col_LE[:, None])
            col_names.append("y_LE")

            #Take right edge to equal outer edge of LLS B scan window
            y_col_RE = np.full_like(x_col, 500)
            arrays.append(y_col_RE[:, None])
            col_names.append("y_RE")

        else:
            #Take data for left edge
            arr_LE, cols_LE = Traverse_Gap_excel_to_array((tow-1))
            arr_LE = np.array(arr_LE, dtype=float)
            arrays.append(arr_LE)
            col_names.extend(cols_LE)

            #Calculate the x-positions of the measurements for the left edge
            x_col_LE = (1.0 / arr_LE[-1, 0]) * arr_LE[:, 0]
            arrays.append(x_col_LE[:, None])
            col_names.append("x_LE")

                #Calculate the y-positions of the measurements for the left edge
            y_col_LE = (y_offset_traverse + (tow - 1) * y_increment_traverse + 0.5 * frame_width_traverse) - arr_LE[:, 1]
            arrays.append(y_col_LE[:, None])
            col_names.append("y_LE")

            #Take data for right edge
            arr_RE, cols_RE = Traverse_Gap_excel_to_array(tow)
            arr_RE = np.array(arr_RE, dtype=float)
            arrays.append(arr_RE)
            col_names.extend(cols_RE)

            #Calculate the x-positions of the measurements for the right edge
            x_col_RE = (1.0 / arr_RE[-1, 0]) * arr_RE[:, 0]
            arrays.append(x_col_RE[:, None])
            col_names.append("x_RE")

            #Calculate the y-positions of the measurements for the right edge
            y_col_RE = (y_offset_traverse + tow * y_increment_traverse + 0.5 * frame_width_traverse) - arr_RE[:, 2]
            arrays.append(y_col_RE[:, None])
            col_names.append("y_RE")
    
    elif sensor_type == "TRAVERSE_LT":
        arr_trav, cols_trav = Traverse_LT_excel_to_array(tow)
        arrays.append(arr_trav)
        col_names.extend(cols_trav)"""
    
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

def traverse_vs_layup_data(tow: int):

    path_traverse = r'C:\Users\manue\OneDrive\Documents\GitHub\D04-Journal-Paper\Synced data from Siddharth\ExportedCSVs\Traverse\Traverse tracker data\TrackerData_7_Traverse.csv'
    path_layup = r'C:\Users\manue\OneDrive\Documents\GitHub\D04-Journal-Paper\Synced data from Siddharth\ExportedCSVs\Layup data\Data\Data_Run07_Tracker.csv'

    #path_traverse = f"Cached Data/TRAVERSE_GAP_{tow}.csv"
    #path_layup = f"Cached Data/LAYUP_{tow}.csv"

    traverse = pd.read_csv(path_traverse)
    layup = pd.read_csv(path_layup)

    # Plot traverse dataframe
    plt.plot(traverse.iloc[:, 2], traverse.iloc[:, 3], label="Traverse", linestyle='-',linewidth=1.5, color='b')
    #plt.plot((traverse.iloc[:, 2] * 500 + 500 * traverse.iloc[:, 0]), (0.5 * traverse.iloc[:, 3] + 0.5 * traverse.iloc[:, 1]), label="Traverse", linestyle='-',linewidth=1.5, color='b')

    # Plot layup dataframe
    plt.plot(layup.iloc[:, 2], layup.iloc[:, 3], label="Layup", linestyle='-',linewidth=1.5, color='r')
    #plt.plot(layup.iloc[:, 7], (layup.iloc[:, 8] + layup.iloc[:, 3]), label="Layup", linestyle='-',linewidth=1.5, color='r')

    # Labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Traverse vs Layup")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

##############################################################################################################
"""Run this file"""

def main():
    x = get_synced_data(9, "Traverse", overwrite=True)

    # Just to check if the new data with weights is correct (it is)
    #for tow in range(1,32):
    #    x = get_synced_data(tow, "Traverse", overwrite=True)
    #    print(np.shape(x))

    #print("Columns:", x.columns.tolist())

    #print(get_synced_data(10, "TRAVERSE_LT"))

    #get_synced_data(5, "Traverse", overwrite=True)

    #traverse_vs_layup_data(10)
    
if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
