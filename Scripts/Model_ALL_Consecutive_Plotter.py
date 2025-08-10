"""A genera data plotter in order to see relations between consecutive steps"""

"""This file is currently not being used for anything except plotting"""

##############################################################################################################

# External imports
import matplotlib.pyplot as plt

#Internal imports
from Handling_ALL_Functions import get_synced_data
from constants import Consecutive_Error_Bins

##############################################################################################################
"""Functions"""

def plot_LT_error(tow: int):
    """Plot histogram of LT error for a given tow."""
    df = get_synced_data(tow, "LT")  # returns DataFrame
    errors = df["error_LT"]

    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'LT Error Distribution (Tow {tow})')
    plt.xlabel('Error (LT)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_LLS_A_error(tow: int):
    """Plot histogram of LLS A error for a given tow."""
    df = get_synced_data(tow, "LLS_A")
    errors = df["error_LLS_A"]

    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'LLS A Error Distribution (Tow {tow})')
    plt.xlabel('Error (LLS A)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_LLS_B_error(tow: int):
    """Plot histogram of LLS B error for a given tow."""
    df = get_synced_data(tow, "LLS_B")
    errors = df["error_LLS_B"]

    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'LLS B Error Distribution (Tow {tow})')
    plt.xlabel('Error (LLS B)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_CAM_error(tow: int):
    """Plot histogram of CAM error for a given tow."""
    df = get_synced_data(tow, "CAM")
    errors = df["error_CAM"]

    plt.hist(errors, bins=Consecutive_Error_Bins, density=True, alpha=0.7, edgecolor='black')
    plt.title(f'CAM Error Distribution (Tow {tow})')
    plt.xlabel('Error (CAM)')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.show()

####################################################################################################
"""Run this file"""

def main():
    tow = 2
    plot_CAM_error(tow)

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
