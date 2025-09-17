"""This file deals with traverse data handling and plotting"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Internal imports
from constants import NOMINAL_LLS_A, NOMINAL_CAM, NOMINAL_LLS_B, NOMINAL_LT_Y, y_offset_traverse, y_increment_traverse, frame_width_traverse
from Handling_ALL_Functions import get_synced_data

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
                    absolute y-coordinates of tow edges, but correct shape (correct relative y-coordinates) of tow edges
    """
    # Check for valid tow number
    if tow == 1 or tow == 31:
        print(f'Tow 1 or 31 can not be recreated from traverse data.')
        print(f'Provide a tow number between 2 and 30 inclusive')
        return None
    
    elif tow in range(2,31): 
        # Load synced and interpolated data of both gaps adjacent to tow, set rows outside range of tow (0-1000 mm) equal to NaN
        gap_right_data_full = get_synced_data((tow-1), "Traverse_Interpolated")
        gap_left_data_full = get_synced_data(tow, "Traverse_Interpolated")
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

        # Pad shortest columns with zeroes
        arrays = [x_right, y_right, x_left, y_left]
        max_len = max(map(len, arrays))
        x_right, y_right, x_left, y_left = [np.pad(arr, (0, max_len - len(arr)), mode="constant") for arr in arrays]

        # Construct final dataframe
        traverse_tow = pd.DataFrame({"x_right": x_right, "y_right": y_right, "x_left": x_left, "y_left": y_left})

    return traverse_tow

##############################################################################################################
"""Run this file"""

def main():
    print(traverse_tow_constructor(2))

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else