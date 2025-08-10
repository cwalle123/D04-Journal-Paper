"""This file deals with all of the important functions that are used all throughout the project"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import os

# Internal imports
from Data_ALL_importer import LLS_A_excel_to_array, LLS_B_excel_to_array, CAM_excel_to_array, LT_x_excel_to_array, LT_y_normalized_excel_to_array, GAP_excel_to_array
from constants import NOMINAL_LLS_A, NOMINAL_CAM, NOMINAL_LLS_B, NOMINAL_LT_Y

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
    if sensor_type not in ["LT", "LLS_A", "LLS_B", "CAM"]:
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

    # Combine horizontally
    processed_data = arrays[0] if len(arrays) == 1 else np.hstack(arrays)

    # Save to cache unless helper
    if not helper:
        save_cached_data(name, processed_data, col_names)

    # Return as DataFrame
    return pd.DataFrame(processed_data, columns=col_names)

##############################################################################################################
"""Run this file"""

def main():
    for tow in range(1,32):
        x = get_synced_data(tow, "LLS_B")
    print(np.shape(x))

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
