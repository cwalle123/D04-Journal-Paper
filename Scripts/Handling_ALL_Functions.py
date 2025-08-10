"""This file deals with all of the important functions that are used all throughout the project"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import os

# Internal imports
from Data_ALL_importer import LLS_A_excel_to_array, LLS_B_excel_to_array, CAM_excel_to_array, LT_x_excel_to_array, LT_y_normalized_excel_to_array, GAP_excel_to_array

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

def get_synced_data(tow: int, sensor_type: str, overwrite=False, helper=False) -> np.ndarray:
    """
    Loads processed data for a given tow & sensor, with caching.
    Combines multiple arrays horizontally if needed.
    Uses actual column names from excel_to_array functions.

    Args:
        tow (int): Tow number (1 to 31).
        sensor_type (str): One of ["LT", "LLS_A", "LLS_B", "CAM"].
        overwrite (bool): If True, ignore cache and overwrite cache from files.
        helper (bool): Internal use flag to skip caching.

    Returns:
        np.ndarray: Combined data array for given tow and sensor type.
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
            return cached_array
        except FileNotFoundError:
            print(f"[CACHE] No cache found for '{name}'. Processing new data...")

    arrays = []
    col_names = []

    if sensor_type == "LT":
        arr1, cols1 = LT_x_excel_to_array(tow)
        arr2, cols2 = LT_y_normalized_excel_to_array(tow)
        arrays.append(arr1)
        col_names.extend(cols1)
        arrays.append(arr2)
        col_names.extend(cols2)

    elif sensor_type == "CAM":
        arr, cols = CAM_excel_to_array(tow)
        arrays.append(arr)
        col_names.extend(cols)

    elif sensor_type == "LLS_A":
        arr, cols = LLS_A_excel_to_array(tow)
        arrays.append(arr)
        col_names.extend(cols)

    elif sensor_type == "LLS_B":
        arr, cols = LLS_B_excel_to_array(tow)
        arrays.append(arr)
        col_names.extend(cols)

    # Combine horizontally if more than one array, else just the single array
    processed_data = arrays[0] if len(arrays) == 1 else np.hstack(arrays)

    # Save to cache unless helper
    if not helper:
        save_cached_data(name, processed_data, col_names)

    return processed_data

def get_joined_data(tow: int, overwrite=False, helper=False) -> np.ndarray:
    """
    Load joined data for a given tow directly from the corresponding CSV file.
    Returns the data as a numpy array and caches it using column names.

    Args:
        tow (int): Tow number (1 to 31).
        overwrite (bool): If True, ignore cache and reload from file.
        helper (bool): Internal use flag to skip caching.

    Returns:
        np.ndarray: Data array loaded from joined CSV.
    """
    if tow not in range(1, 32):
        raise IndexError(f"Tow ID {tow} is out of range")

    name = f"ALL_{tow}"
    if not helper and not overwrite:
        try:
            cached_array, _ = load_cached_data(name)
            print(f"[CACHE] Loaded '{name}' from cache")
            return cached_array
        except FileNotFoundError:
            print(f"[CACHE] No cache found for '{name}'. Reading from file...")

    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\DataStreamsJoined"
    tow_str = str(tow).zfill(2)
    file_name = f"DataStreamsJoined_Run{tow_str}_Run.csv"
    file_path = os.path.join(base_path, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Joined data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Convert datetime column (assumed first column) to elapsed seconds relative to first timestamp
    # Replace first column with elapsed seconds
    datetime_col = df.columns[0]
    df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True)  # dayfirst=True for format like 26.07.2023

    # Calculate elapsed seconds from first timestamp
    start_time = df[datetime_col].iloc[0]
    elapsed_seconds = (df[datetime_col] - start_time).dt.total_seconds()
    df[datetime_col] = elapsed_seconds

    # Now all columns should be numeric or convertable to numeric, so keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    data_array = numeric_df.to_numpy()  # float array with NaNs allowed
    columns = list(numeric_df.columns)

    if not helper:
        save_cached_data(name, data_array, columns)

    return data_array

##############################################################################################################
"""Run this file"""

def main():
    tow = 2
    x = get_synced_data(tow, "LT")
    print(np.shape(x))

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
