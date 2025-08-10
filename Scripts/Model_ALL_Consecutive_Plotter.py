"""A genera data plotter in order to see relations between consecutive steps"""

"""This file is currently not being used for anything except plotting"""

##############################################################################################################

# External imports
import matplotlib.pyplot as plt
import numpy as np

#Internal imports
from Handling_ALL_Functions import get_synced_data
from constants import Consecutive_Error_Bins, NOMINAL_LLS_A, NOMINAL_LLS_B, NOMINAL_CAM, NOMINAL_LT_Y

##############################################################################################################
"""Functions"""

def plot_LT_error(tow: int):
    """Plot histogram of LT error for a given tow."""
    data = get_synced_data(tow, "LT")  # returns np.ndarray
    value_column = 2  # index for LT y value
    
    values = data[:, value_column]
    errors = values - NOMINAL_LT_Y
    
    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'LT Error Distribution (Tow {tow})')
    plt.xlabel('Error (LT)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_LLS_A_error(tow: int):
    """Plot histogram of LLS A error for a given tow."""
    data = get_synced_data(tow, "LLS_A")  # returns np.ndarray
    value_column = 0  # index for LLS A width
    
    values = data[:, value_column]
    errors = values - NOMINAL_LLS_A
    
    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'LLS  A Error Distribution (Tow {tow})')
    plt.xlabel('Error (LLS A)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_LLS_B_error(tow: int):
    """Plot histogram of LLS B error for a given tow."""
    data = get_synced_data(tow, "LLS_B")  # returns np.ndarray
    value_column = 0  # index for LLS B width
    
    values = data[:, value_column]
    errors = values - NOMINAL_LLS_B
    
    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'LLS  B Error Distribution (Tow {tow})')
    plt.xlabel('Error (LLS B)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_CAM_error(tow: int):
    """Plot histogram of CAM error for a given tow."""
    data = get_synced_data(tow, "CAM")  # returns np.ndarray
    value_column = 0  # index for CAM value
    
    values = data[:, value_column]
    errors = values - NOMINAL_CAM
    
    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'CAM Error Distribution (Tow {tow})')
    plt.xlabel('Error (CAM)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

##############################################################################################################
"""Run this file"""

def main():
    tow = 2
    plot_CAM_error(tow)

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
