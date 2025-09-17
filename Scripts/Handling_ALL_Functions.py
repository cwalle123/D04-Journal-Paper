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
from constants import NOMINAL_LLS_A, NOMINAL_CAM, NOMINAL_LLS_B, NOMINAL_LT_Y, y_offset_traverse, y_increment_traverse, frame_width_traverse

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
    if sensor_type not in ["LT", "LLS_A", "LLS_B", "CAM", "TRAVERSE_GAP", "TRAVERSE_LT", "Traverse", "Traverse_Interpolated"]:
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
        #This branch synchronizes the data of the LT and LLS B by connecting nearest in space points of the LT to the LLS B data.
        #It moves datapoints around, which is incorrect and can have problematic results.
        #Synchronization with interpolation might be a better idea, see next branch.
        if tow in range(1,31):
        # Load the LT and Gap data arrays
            LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
            Gap_arr, Gap_cols = Traverse_Gap_excel_to_array(tow)

            # Guard against empty arrays
            if LT_arr.shape[0] == 0 or Gap_arr.shape[0] == 0:
                synced_cols = Gap_cols + ["Gap_x"] + [c for c in LT_cols if c != "LT_x"]
                arrays.append(np.empty((0, len(synced_cols))))
                col_names.extend(synced_cols)
                print("USED GUARD AGAINST EMPTY ARRAYS")
            else:
                # Calculate Gap_x (normalized using last Gap time)
                if Gap_arr[-1, 0] == 0:
                    raise ValueError("Last Gap time is zero, cannot compute Gap_x.")
                Gap_x_col = (1.0 / Gap_arr[-1, 0]) * Gap_arr[:, 0]

                # Extract LT_x
                LT_x = LT_arr[:, LT_cols.index("LT_x")]

                # Vectorized nearest-neighbor lookup
                nearest_idx = np.searchsorted(LT_x, Gap_x_col)
                nearest_idx = np.clip(nearest_idx, 1, len(LT_x)-1)
                left_is_better = (np.abs(Gap_x_col - LT_x[nearest_idx-1]) < np.abs(Gap_x_col - LT_x[nearest_idx]))
                chosen_idx = np.where(left_is_better, nearest_idx-1, nearest_idx)

                # Select matched LT rows and drop LT_x
                LT_matched = LT_arr[chosen_idx]
                lt_keep_mask = [c != "LT_x" for c in LT_cols]
                LT_matched = LT_matched[:, lt_keep_mask]
                LT_keep_cols = [c for c in LT_cols if c != "LT_x"]

                # Stack Gap + Gap_x + matched LT rows
                synced = np.hstack([Gap_arr, Gap_x_col[:, None], LT_matched])
                synced_cols = Gap_cols + ["Gap_x"] + LT_keep_cols

                # Append to arrays for later processing
                arrays.append(synced)
                col_names.extend(synced_cols)
        
        elif tow == 31:
            LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
            arrays.append(LT_arr)
            col_names.extend(LT_cols)

    elif sensor_type == "Traverse_Interpolated":
        if tow in range(1, 31):
            # Step 1: Load LT and Gap arrays from Excel
            LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
            Gap_arr, Gap_cols = Traverse_Gap_excel_to_array(tow)

            # Step 2: Guard against empty arrays
            if LT_arr.size == 0 or Gap_arr.size == 0:
                # If one dataset is missing, return an empty placeholder
                synced_cols = Gap_cols + LT_cols
                arrays.append(np.empty((0, len(synced_cols))))
                col_names.extend(synced_cols)
                print(f"Tow {tow}: one of the arrays was empty, skipping.")
            
            else:
                # Step 3: Extract raw time axes
                gap_time = Gap_arr[:, 0]                   
                lt_time  = LT_arr[:, 0]

                
                # Step 4: Build common time axis
                # Use the union of all unique time points from Gap and LT
                common_time = np.union1d(gap_time, lt_time)

                # Step 5: Interpolate Gap data onto common_time
                gap_interp_values = []
                for col_index in range(Gap_arr.shape[1]):
                    # Build a linear interpolator for each column in Gap
                    f_gap = interp1d(gap_time, Gap_arr[:, col_index], kind="linear", fill_value="extrapolate")
                    # Evaluate on the common time grid
                    gap_interp_values.append(f_gap(common_time))
                gap_interp = np.column_stack(gap_interp_values)

                # Step 6: Interpolate LT data onto common_time
                lt_interp_values = []
                LT_keep_cols = [c for c in LT_cols if c != "LT_time"]  # ignore raw time col
                for col_name in LT_keep_cols:
                    col_index = LT_cols.index(col_name)
                    f_lt = interp1d(lt_time, LT_arr[:, col_index], kind="linear", fill_value="extrapolate")
                    lt_interp_values.append(f_lt(common_time))
                lt_interp = np.column_stack(lt_interp_values)

                # Step 7: Combine everything into one array
                # Format: [time, Gap columns..., LT columns...]
                synced = np.column_stack([common_time, gap_interp[:, 1:], lt_interp])
                synced_cols = ["time"] + Gap_cols[1:] + LT_keep_cols

                arrays.append(synced)
                col_names.extend(synced_cols)

        elif tow == 31:
            # Special case: just keep LT data as-is
            LT_arr, LT_cols = Traverse_LT_excel_to_array(tow)
            arrays.append(LT_arr)
            col_names.extend(LT_cols)
    
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
    # x = get_synced_data(2, "Traverse_Interpolated", overwrite=True)

    # Just to check if the new data with weights is correct (it is)
    for tow in range(1,32):
        x = get_synced_data(tow, "Traverse_Interpolated", overwrite=True)
        print(np.shape(x))

    #print("Columns:", x.columns.tolist())

    #print(get_synced_data(10, "TRAVERSE_LT"))

    #get_synced_data(5, "Traverse", overwrite=True)

    #traverse_vs_layup_data(10)
    
if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
