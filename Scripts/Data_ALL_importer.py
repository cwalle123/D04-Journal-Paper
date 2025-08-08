'''This file imports all the data from the excel files and groups them into one array'''

##############################################################################################################

# External imports
import pandas as pd
import numpy  as np
import os

##############################################################################################################
"""Functions"""

def LLS_A_excel_to_array():
    """Read the LLS A Excel files for all 31 tows and output into a combined NumPy array.
       The format of the array's columns is [tow_1_width, tow_1_weight, tow_2_width, tow_2_weight, ...]"""

    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\Distilled\lls a"
    LLS_A_width = "TapeWidth1"
    LLS_A_weights = "Weights"
    
    tow_arrays = []
    smallest_file_lenght = None
    
    # First pass — find the shortest number of rows
    for tow_num in range(1, 32):
        file_name = f"LLSA_distilled_Run{tow_num:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        
        if LLS_A_width not in df.columns or LLS_A_weights not in df.columns:
            raise ValueError(f"Missing correct column names in file: {file_path}")
        
        if smallest_file_lenght is None or len(df) < smallest_file_lenght:
            smallest_file_lenght = len(df)
    
    # Second pass — actually load data and trim
    for tow_num in range(1, 32):
        file_name = f"LLSA_distilled_Run{tow_num:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        
        tow_data = df[[LLS_A_width, LLS_A_weights]].iloc[:smallest_file_lenght].to_numpy()
        tow_arrays.append(tow_data)
    
    # Stack side-by-side
    final_array = np.hstack(tow_arrays)
    return final_array

def LLS_B_excel_to_array():
    """Read the LLS B Excel files for all 31 tows and output into a combined NumPy array.
       The format of the array's columns is [tow_1_width, tow_1_weight, tow_2_width, tow_2_weight, ...]"""

    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\Distilled\lls b"
    LLS_B_width = "TapeWidthAfterCompaction"
    LLS_B_weights = "Weights"
    
    tow_arrays = []
    smallest_file_lenght = None
    
    # First pass — find the shortest number of rows
    for tow_num in range(1, 32):
        file_name = f"LLSB_distilled_Run{tow_num:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        
        if LLS_B_width not in df.columns or LLS_B_weights not in df.columns:
            raise ValueError(f"Missing correct column names in file: {file_path}")
        
        if smallest_file_lenght is None or len(df) < smallest_file_lenght:
            smallest_file_lenght = len(df)
    
    # Second pass — actually load data and trim
    for tow_num in range(1, 32):
        file_name = f"LLSB_distilled_Run{tow_num:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        
        tow_data = df[[LLS_B_width, LLS_B_weights]].iloc[:smallest_file_lenght].to_numpy()
        tow_arrays.append(tow_data)
    
    # Stack side-by-side
    final_array = np.hstack(tow_arrays)
    return final_array

def CAM_excel_to_array():
    """Read the Camera Excel files for all 31 tows and output into a combined NumPy array.
       The format of the array's columns is [tow_1_centerline, tow_1_weight, tow_2_centerline, tow_2_weight, ...]"""

    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\Distilled\camera"
    CAM_centerline = "TapeCenterLine"
    CAM_weights = "Weights"
    
    tow_arrays = []
    smallest_file_lenght = None
    
    # First pass — find the shortest number of rows
    for tow_num in range(1, 32):
        file_name = f"Camera_distilled_Run{tow_num:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        
        if CAM_centerline not in df.columns or CAM_weights not in df.columns:
            raise ValueError(f"Missing correct column names in file: {file_path}")
        
        if smallest_file_lenght is None or len(df) < smallest_file_lenght:
            smallest_file_lenght = len(df)
    
    # Second pass — actually load data and trim
    for tow_num in range(1, 32):
        file_name = f"Camera_distilled_Run{tow_num:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        
        tow_data = df[[CAM_centerline, CAM_weights]].iloc[:smallest_file_lenght].to_numpy()
        tow_arrays.append(tow_data)
    
    # Stack side-by-side
    final_array = np.hstack(tow_arrays)
    return final_array

def LT_x_excel_to_array():
    """Read the LT Excel files for the x data for all 31 tows and output into a combined NumPy array.
       The format of the array's columns is [tow_1_x, tow_1_weight, tow_2_x, tow_2_weight, ...]"""

    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\Distilled\tracker"
    LT_x = "X_mm"
    LT_weights = "Weights"
    
    tow_arrays = []
    smallest_file_lenght = None
    
    # First pass — find the shortest number of rows
    for tow_num in range(1, 32):
        file_name = f"Tracker_distilled_Run{tow_num:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        
        if LT_x not in df.columns or LT_weights not in df.columns:
            raise ValueError(f"Missing correct column names in file: {file_path}")
        
        if smallest_file_lenght is None or len(df) < smallest_file_lenght:
            smallest_file_lenght = len(df)
    
    # Second pass — actually load data and trim
    for tow_num in range(1, 32):
        file_name = f"Tracker_distilled_Run{tow_num:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        
        tow_data = df[[LT_x, LT_weights]].iloc[:smallest_file_lenght].to_numpy()
        tow_arrays.append(tow_data)
    
    # Stack side-by-side
    final_array = np.hstack(tow_arrays)
    return final_array

def LT_y_normalized_excel_to_array():
    """Read the LT Excel files for the y normalized data for all 31 tows and output into a combined NumPy array.
       The format of the array's columns is [tow_1_y_normalized, tow_1_weight, tow_2_y_normalized, tow_2_weight, ...]"""

    base_path = r"Synced data from Siddharth\ExportedCSVs\Layup data\Distilled\tracker"
    LT_y_normalized = "Y_normalised"
    LT_weights = "Weights"
    
    tow_arrays = []
    smallest_file_lenght = None
    
    # First pass — find the shortest number of rows
    for tow_num in range(1, 32):
        file_name = f"Tracker_distilled_Run{tow_num:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        
        if LT_y_normalized not in df.columns or LT_weights not in df.columns:
            raise ValueError(f"Missing correct column names in file: {file_path}")
        
        if smallest_file_lenght is None or len(df) < smallest_file_lenght:
            smallest_file_lenght = len(df)
    
    # Second pass — actually load data and trim
    for tow_num in range(1, 32):
        file_name = f"Tracker_distilled_Run{tow_num:02d}_Run.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_excel(file_path) if file_path.lower().endswith('.xlsx') else pd.read_csv(file_path)
        
        tow_data = df[[LT_y_normalized, LT_weights]].iloc[:smallest_file_lenght].to_numpy()
        tow_arrays.append(tow_data)
    
    # Stack side-by-side
    final_array = np.hstack(tow_arrays)
    return final_array

def GAP_excel_to_array():
    """Read the Traverse files for all consecutive tow pairs and output into a combined NumPy array.
       Stops at the first NaN in any file, ensuring all columns have valid values.
       The format of the array's columns is [gap_between_tow_1_and_2, gap_between_tow_2_and_3, ...]"""

    base_path = r"Synced data from Siddharth\ExportedCSVs\Traverse\Traverse Gap Data from LLS"
    GAP = "Gap"
    
    tow_arrays = []
    smallest_valid_length = None  # shortest row length with no NaNs
    
    # First pass — find earliest NaN across all files
    for start_tow in range(1, 31):  # 1 to 30 inclusive
        end_tow = start_tow + 1
        file_name = f"TraverseData_Gap_{start_tow}_{end_tow}.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_csv(file_path)
        
        if GAP not in df.columns:
            raise ValueError(f"Missing correct column name in file: {file_path}")
        
        gap_values = df[GAP].to_numpy()
        
        # Find first NaN index if it exists
        nan_index = np.where(np.isnan(gap_values))[0]
        if len(nan_index) > 0:
            first_nan = nan_index[0]
        else:
            first_nan = len(gap_values)
        
        # Track minimum valid length
        if smallest_valid_length is None or first_nan < smallest_valid_length:
            smallest_valid_length = first_nan
    
    # Second pass — load & trim to smallest_valid_length
    for start_tow in range(1, 31):
        end_tow = start_tow + 1
        file_name = f"TraverseData_Gap_{start_tow}_{end_tow}.csv"
        file_path = os.path.join(base_path, file_name)
        
        df = pd.read_csv(file_path)
        
        gap_data = df[GAP].iloc[:smallest_valid_length].to_numpy().reshape(-1, 1)
        tow_arrays.append(gap_data)
    
    # Combine into one NumPy array
    final_array = np.hstack(tow_arrays)
    return final_array

##############################################################################################################
"""Run this file"""

def main():
    x = GAP_excel_to_array()
    print(x)
    print(np.shape(x))

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
