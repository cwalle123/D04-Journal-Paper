"""This file deals with all of the important functions that are used all throughout the project"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import glob
import os
import shutil

# Internal imports
from Data_ALL_importer import LLS_A_excel_to_array, LLS_B_excel_to_array, CAM_excel_to_array, LT_x_excel_to_array, LT_y_normalized_excel_to_array, GAP_excel_to_array

###################################################################################################################################################################################################
"""Functions for saving, loading, and purging data"""

CACHE_FOLDER = "Cached Data"

def save_cached_data(name: str, array: np.ndarray):
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    path = os.path.join(CACHE_FOLDER, f"{name}.csv")
    np.savetxt(path, array, delimiter=",")
    print(f"[CACHE] Saved '{name}' to {path}")

def load_cached_data(name: str) -> np.ndarray:
    path = os.path.join(CACHE_FOLDER, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached file found for '{name}'")
    return np.loadtxt(path, delimiter=",")

def purge_cached_data():
    if os.path.exists(CACHE_FOLDER):
        for file in os.listdir(CACHE_FOLDER):
            os.remove(os.path.join(CACHE_FOLDER, file))
        print("[CACHE] All cached data purged.")

################################################################################################################
"""Functions for calling data"""

def get_synced_data(tow: int, sensor_type: str, overwrite=False, helper=False) -> np.ndarray:
    '''
    Loads processed data for a given tow & sensor, with caching.
    Combines multiple arrays horizontally if needed.
    '''
    if sensor_type not in ["LT", "LLS_A", "LLS_B", "CAM"]:
        raise KeyError(f"The key '{sensor_type}' is invalid")
    if tow not in range(1, 32):
        raise IndexError(f"Tow ID {tow} is out of range")

    name = f"{sensor_type}_{tow}"

    # --- Try cache ---
    if not helper and not overwrite:
        try:
            cached_array = load_cached_data(name)
            print(f"[CACHE] Loaded '{name}' from cache")
            return cached_array
        except FileNotFoundError:
            print(f"[CACHE] No cache found for '{name}'. Processing new data...")

    # --- Generate list of arrays ---
    arrays = []
    match sensor_type:
        case "LT":
            arrays.append(LT_x_excel_to_array()[:, (tow-1)*2:(tow-1)*2 + 2])
            arrays.append(LT_y_normalized_excel_to_array()[:, (tow-1)*2:(tow-1)*2 + 2])
        case "CAM":
            arrays.append(CAM_excel_to_array()[:, (tow-1)*2:(tow-1)*2 + 2])
        case "LLS_A":
            arrays.append(LLS_A_excel_to_array()[:, (tow-1)*2:(tow-1)*2 + 2])
        case "LLS_B":
            arrays.append(LLS_B_excel_to_array()[:, (tow-1)*2:(tow-1)*2 + 2])

    # --- Combine arrays dynamically ---
    processed_data = arrays[0] if len(arrays) == 1 else np.hstack(arrays)

    # --- Save to cache ---
    if not helper:
        save_cached_data(name, processed_data)

    return processed_data

##############################################################################################################
"""Run this file"""

def main():
    get_synced_data(2, "LT", True)

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
