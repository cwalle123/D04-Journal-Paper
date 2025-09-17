'''This file imports all the data from the excel files and groups them into one array per sensor
   Written by: Martijn van der Voort, Clifton-John Walle and Manuel Cruz"""'''

##############################################################################################################

# External imports
import pandas as pd
import numpy  as np
import os
from pathlib import Path

##############################################################################################################
"""Functions"""

def LLS_A_excel_to_array(tow_num):
    """
    Read the LLS A Excel/CSV file for a single specified tow.
    It first scans all 31 tows to find the shortest valid data length (excluding NaNs),
    then loads and trims the specified tow's data to that length.

    Args:
        tow_num (int): The tow number (1 to 31) to load.

    Returns:
        tuple: (numpy.ndarray, list)
            - Data array with shape (shortest_length, 2) containing columns [TapeWidth1, Weights].
            - List of column names: ['TapeWidth1', 'Weights'].
    """
    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\Distilled\lls a"
    LLS_A_width = "TapeWidth_Arc1"
    LLS_A_weights = "Weights"

    smallest_file_length = None

    # Find shortest valid length by scanning all tows
    for tow in range(1, 32):
        file_name = f"LLSA_distilled_Run{tow:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

        if LLS_A_width not in df.columns or LLS_A_weights not in df.columns:
            raise ValueError(f"Missing columns in file: {file_path}")

        data_sub = df[[LLS_A_width, LLS_A_weights]]
        nan_indices = np.where(data_sub.isna().any(axis=1))[0]
        valid_length = nan_indices[0] if len(nan_indices) > 0 else len(data_sub)

        if smallest_file_length is None or valid_length < smallest_file_length:
            smallest_file_length = valid_length

    # Load the requested tow data and trim
    file_name = f"LLSA_distilled_Run{tow_num:02d}_Run.csv"
    file_path = os.path.join(base_path, file_name)
    df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

    tow_data = df[[LLS_A_width, LLS_A_weights]].iloc[:smallest_file_length].to_numpy()

    return tow_data, ['width_LLS_A', LLS_A_weights]

def LLS_B_excel_to_array(tow_num):
    """
    Read the LLS B Excel/CSV file for a single specified tow.
    Finds shortest valid data length across all 31 tows and trims the data.

    Args:
        tow_num (int): Tow number (1 to 31).

    Returns:
        tuple: (numpy.ndarray, list)
            - Data array with shape (shortest_length, 2) containing columns [TapeWidthAfterCompaction, Weights].
            - List of column names: ['TapeWidthAfterCompaction', 'Weights'].
    """
    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\Distilled\lls b"
    LLS_B_width = "TapeWidthAfterCompaction_Rotated"
    LLS_B_weights = "Weights"

    smallest_file_length = None

    for tow in range(1, 32):
        file_name = f"LLSB_distilled_Run{tow:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

        if LLS_B_width not in df.columns or LLS_B_weights not in df.columns:
            raise ValueError(f"Missing columns in file: {file_path}")

        data_sub = df[[LLS_B_width, LLS_B_weights]]
        nan_indices = np.where(data_sub.isna().any(axis=1))[0]
        valid_length = nan_indices[0] if len(nan_indices) > 0 else len(data_sub)

        if smallest_file_length is None or valid_length < smallest_file_length:
            smallest_file_length = valid_length

    file_name = f"LLSB_distilled_Run{tow_num:02d}_Run.csv"
    file_path = os.path.join(base_path, file_name)
    df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

    tow_data = df[[LLS_B_width, LLS_B_weights]].iloc[:smallest_file_length].to_numpy()

    return tow_data, ['width_LLS_B', LLS_B_weights]

def CAM_excel_to_array(tow_num):
    """
    Read the Camera Excel/CSV file for a single specified tow.
    Finds shortest valid data length across all 31 tows and trims the data.

    Args:
        tow_num (int): Tow number (1 to 31).

    Returns:
        tuple: (numpy.ndarray, list)
            - Data array with shape (shortest_length, 2) containing columns [TapeCenterLine, Weights].
            - List of column names: ['TapeCenterLine', 'Weights'].
    """
    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\Distilled\camera"
    CAM_centerline = "TapeCenterLine"
    CAM_weights = "Weights"

    smallest_file_length = None

    for tow in range(1, 32):
        file_name = f"Camera_distilled_Run{tow:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

        if CAM_centerline not in df.columns or CAM_weights not in df.columns:
            raise ValueError(f"Missing columns in file: {file_path}")

        data_sub = df[[CAM_centerline, CAM_weights]]
        nan_indices = np.where(data_sub.isna().any(axis=1))[0]
        valid_length = nan_indices[0] if len(nan_indices) > 0 else len(data_sub)

        if smallest_file_length is None or valid_length < smallest_file_length:
            smallest_file_length = valid_length

    file_name = f"Camera_distilled_Run{tow_num:02d}_Run.csv"
    file_path = os.path.join(base_path, file_name)
    df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

    tow_data = df[[CAM_centerline, CAM_weights]].iloc[:smallest_file_length].to_numpy()

    return tow_data, ['center_CAM', CAM_weights]

def LT_x_excel_to_array(tow_num):
    """
    Read the LT tracker x-position Excel/CSV file for a single specified tow.
    Finds shortest valid data length across all 31 tows and trims the data.

    Args:
        tow_num (int): Tow number (1 to 31).

    Returns:
        tuple: (numpy.ndarray, list)
            - Data array with shape (shortest_length, 2) containing columns [X_mm, Weights].
            - List of column names: ['X_mm', 'Weights'].
    """
    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\Distilled\tracker"
    LT_x = "X_mm"
    LT_weights = "Weights"

    smallest_file_length = None

    for tow in range(1, 32):
        file_name = f"Tracker_distilled_Run{tow:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

        if LT_x not in df.columns or LT_weights not in df.columns:
            raise ValueError(f"Missing columns in file: {file_path}")

        data_sub = df[[LT_x, LT_weights]]
        nan_indices = np.where(data_sub.isna().any(axis=1))[0]
        valid_length = nan_indices[0] if len(nan_indices) > 0 else len(data_sub)

        if smallest_file_length is None or valid_length < smallest_file_length:
            smallest_file_length = valid_length

    file_name = f"Tracker_distilled_Run{tow_num:02d}_Run.csv"
    file_path = os.path.join(base_path, file_name)
    df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

    tow_data = df[[LT_x, LT_weights]].iloc[:smallest_file_length].to_numpy()

    return tow_data, ['x', LT_weights]

def LT_y_normalized_excel_to_array(tow_num):
    """
    Read the LT tracker y-normalized Excel/CSV file for a single specified tow.
    Finds shortest valid data length across all 31 tows and trims the data.

    Args:
        tow_num (int): Tow number (1 to 31).

    Returns:
        tuple: (numpy.ndarray, list)
            - Data array with shape (shortest_length, 2) containing columns [Y_normalised, Weights].
            - List of column names: ['Y_normalised', 'Weights'].
    """
    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\Distilled\tracker"
    LT_y_normalized = "Y_normalised"
    LT_weights = "Weights"

    smallest_file_length = None

    for tow in range(1, 32):
        file_name = f"Tracker_distilled_Run{tow:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

        if LT_y_normalized not in df.columns or LT_weights not in df.columns:
            raise ValueError(f"Missing columns in file: {file_path}")

        data_sub = df[[LT_y_normalized, LT_weights]]
        nan_indices = np.where(data_sub.isna().any(axis=1))[0]
        valid_length = nan_indices[0] if len(nan_indices) > 0 else len(data_sub)

        if smallest_file_length is None or valid_length < smallest_file_length:
            smallest_file_length = valid_length

    file_name = f"Tracker_distilled_Run{tow_num:02d}_Run.csv"
    file_path = os.path.join(base_path, file_name)
    df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

    tow_data = df[[LT_y_normalized, LT_weights]].iloc[:smallest_file_length].to_numpy()

    return tow_data, ['y', LT_weights]

def GAP_excel_to_array(gap_num):
    """
    Reads the traverse gap CSV data for a specified tow number.
    Finds the shortest valid length across all 30 tows by checking for NaNs,
    then trims the data of the requested tow to that shortest length.

    Args:
        gap_num (int): gap number (1 to 30 inclusive).

    Returns:
        tuple: (numpy.ndarray, list)
            - 2D numpy array of gap values trimmed to shortest valid length.
            - List containing the column name ['Gap'].
    """
    base_path = r"Synced data from Siddharth\ExportedCSVs\Traverse\Traverse Gap Data from LLS"
    gap_col = "Gap"
    
    if gap_num < 1 or gap_num > 30:
        raise ValueError("tow_num for GAP must be between 1 and 30 inclusive.")
    
    smallest_length = None
    for tow in range(1, 31):
        start_tow = tow
        end_tow = tow + 1
        file_name = f"TraverseData_Gap_{start_tow}_{end_tow}.csv"
        file_path = os.path.join(base_path, file_name)
        df = pd.read_csv(file_path)
        
        if gap_col not in df.columns:
            raise ValueError(f"Missing column '{gap_col}' in file: {file_path}")
        
        gap_values = df[gap_col].to_numpy()
        nan_indices = np.where(np.isnan(gap_values))[0]
        first_nan = nan_indices[0] if len(nan_indices) > 0 else len(gap_values)
        
        if smallest_length is None or first_nan < smallest_length:
            smallest_length = first_nan
    
    start_tow = gap_num
    end_tow = gap_num + 1
    file_name = f"TraverseData_Gap_{start_tow}_{end_tow}.csv"
    file_path = os.path.join(base_path, file_name)
    df = pd.read_csv(file_path)
    
    df_trim = df[gap_col].iloc[:smallest_length].dropna()
    data_array = df_trim.to_numpy().reshape(-1, 1)  # Keep 2D shape for consistency

    return data_array, [gap_col]

def Traverse_LT_excel_to_array(tow_nr):
    """
    Reads the traverse tracker data for a specified tow number.
    
    Args:
        tow_nr (int): tow number (1 to 31 inclusive)
        
    Returns:
        tuple: (numpy.ndarray, list)
            - Data array with shape (shortest_length, 2) containing columns [time, x, y, z].
            - List of column names: ['time', 'x', 'y', 'z'].
    """

    path_base = r"Synced data from Siddharth\ExportedCSVs\Traverse\Traverse tracker data"
    time_col = "TimeStamp"
    x_col = "X_mm_"
    y_col = "Y_mm_"
    z_col = "Z_mm_"

    smallest_file_length = None

    if tow_nr < 1 or tow_nr > 31:
        raise ValueError("tow_nr must be between 1 and 31 inclusive.")
    
    for tow in range(1,32):
        file_name = f"TrackerData_{tow}_Traverse.csv"
        file_path = os.path.join(path_base, file_name)
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
    
        if time_col not in df.columns or x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
            raise ValueError(f"Missing columns in file: {file_path}")
        
        data_sub = df[[time_col, x_col, y_col, z_col]]
        nan_indices = np.where(data_sub.isna().any(axis=1))[0]
        valid_length = nan_indices[0] if len(nan_indices) > 0 else len(data_sub)

        if smallest_file_length is None or valid_length < smallest_file_length:
            smallest_file_length = valid_length
    
    file_name = f"TrackerData_{tow_nr}_Traverse.csv"
    file_path = os.path.join(path_base, file_name)
    df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)

    #Calculate seconds passed since beginning of measurement
    df[time_col] = pd.to_datetime(df[time_col], errors="raise", format="%d.%m.%Y %H:%M:%S.%f")
    df[time_col] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()

    tow_traverse_data = df[[time_col, x_col, y_col, z_col]].iloc[:smallest_file_length].to_numpy()

    return tow_traverse_data, ['LT_time', 'LT_x', 'LT_y', 'LT_z']

def _first_existing(base: Path, tow_a: int, tow_b: int) -> Path:
    """Return the first existing path among padded and unpadded variants. Partially created with AI"""
    candidates = [
        base / f"TraverseData_Gap_{tow_a:02d}_{tow_b:02d}.csv",
        base / f"TraverseData_Gap_{tow_a}_{tow_b}.csv",
        base / f"TraverseData_Gap_{tow_a:02d}_{tow_b:02d}.xlsx",
        base / f"TraverseData_Gap_{tow_a}_{tow_b}.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Just return the first candidate (padded csv) for error messaging
    return candidates[0]

def Traverse_Gap_excel_to_array(tows_nrs: int):
    """
    Reads traverse-gap data between tow n and n+1,
    
    Returns:
      np.ndarray [time, leftedge, rightedge, gap], columns names list.
    """

    if not (1 <= tows_nrs <= 30):
        raise ValueError("tows_nrs must be between 1 and 30 inclusive.")

    path_base = Path("Synced data from Siddharth/ExportedCSVs/Traverse/Traverse Gap Data from LLS")

    time_col = "Time"
    left_col = "TapeEdgeLeft"
    right_col = "TapeEdgeRight"
    gap_col = "Gap"
    wanted_cols = [time_col, left_col, right_col, gap_col]

    # Find global smallest valid length across all files that exist
    smallest_file_length = None
    for tow in range(1, 31):
        tow2 = tow + 1
        path = _first_existing(path_base, tow, tow2)

        df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)

        data_sub = df[wanted_cols]
        nan_rows = np.where(data_sub.isna().any(axis=1))[0]
        valid_len = int(nan_rows[0]) if len(nan_rows) else len(data_sub)

        if smallest_file_length is None or valid_len < smallest_file_length:
            smallest_file_length = valid_len

    # Load the specific pair by `tows_nrs`
    tow2 = tows_nrs + 1
    target_path = _first_existing(path_base, tows_nrs, tow2)
    if not target_path.exists():
        raise FileNotFoundError(f"File does not exist: {target_path}")

    df = pd.read_excel(target_path) if target_path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(target_path)

    # Convert to elapsed seconds from start of measurement
    df[time_col] = pd.to_datetime(df[time_col], errors="raise")
    df[time_col] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()

    # Construct the dataframe
    arr = df[[time_col, left_col, right_col, gap_col]].iloc[:smallest_file_length].to_numpy()

    return arr, ['Gap_time', 'Gap_leftedge', 'Gap_rightedge', 'Gap_gap']

##############################################################################################################
"""Run this file"""

def main():
    #tow = 3
    #x, names = LLS_A_excel_to_array(tow)
    #print(names)
    #print(np.shape(x))


    arr, data = Traverse_Gap_excel_to_array(4)
    print(arr)
    print(data)
    

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
