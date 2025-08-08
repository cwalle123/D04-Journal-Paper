"""A genera data plotter in order to see relations between consecutive steps"""

"""This file is currently not being used for anything"""

##############################################################################################################

# External imports
import matplotlib.pyplot as plt
import pandas as pd

#Internal imports
from Handling_ALL_Functions import get_synced_data

##############################################################################################################
"""Functions"""

def plot_LT(data: pd.DataFrame, name: str):
    '''plots the LT data'''
    time = data["time"]
    x = data["x"]
    E = data["error_LT"]
    print(f"time = {len(time)}, x = {len(x)}")


    v_x, v_E = [0], [0]
    for i in range(len(time)-1):
        dt = 0.01  # time[i+1] - time[i]

        dx = x[i+1] - x[i]
        x_velocity = dx/dt
        dE = E[i+1] - E[i]
        E_velocity = dE/dt

        v_x.append(x_velocity)
        v_E.append(E_velocity)

    '''plots for the data itself'''
    plt.subplot(221)
    plt.plot(time, x)
    plt.title('LT, x-coordinates')
    plt.xlabel("Time")

    plt.subplot(222)
    plt.plot(time, E)
    plt.title('LT, err-coordinates')
    plt.xlabel("Time")

    plt.subplot(223)
    plt.plot(time, v_x)
    plt.title('LT, x-velocities')
    plt.xlabel("Time")

    plt.subplot(224)
    plt.plot(time, v_E)
    plt.title('LT, err-velocities')
    plt.xlabel("Time")

    plt.tight_layout()
    plt.show()

    err_range = max(E) - min(E)
    minimum = min(E)
    maximum = max(E)
    split = 0.2

    split_1, split_2, split_3, split_4, split_5 = [], [], [], [], []
    V_1, V_2, V_3, V_4, V_5 = [], [], [], [], []
    for i in range(len(time)-1):
        err = E[i]

        if minimum <= err <= minimum + split * err_range:
            split_1.append(err)
            V_1.append(v_E[i])
        elif minimum + split * err_range <= err <= minimum + 2 * split * err_range:
            split_2.append(err)
            V_2.append(v_E[i])
        elif minimum + 2 * split * err_range <= err <= minimum + 3 * split * err_range:
            split_3.append(err)
            V_3.append(v_E[i])
        elif minimum + 3 * split * err_range <= err <= minimum + 4 * split * err_range:
            split_4.append(err)
            V_4.append(v_E[i])
        elif minimum + 4 * split * err_range <= err <= minimum + 5 * split * err_range:
            split_5.append(err)
            V_5.append(v_E[i])
        else:
            print("error: data point outside of splits")

    '''plots for subsets of the data so I can see how relations in error-change change based on what the error is.'''
    plt.subplot(231)
    plt.hist(v_E, bins=100)
    plt.title('LT, err')
    plt.xlabel("V")

    plt.subplot(232)
    bins_needed = int(len(split_1)/5)
    plt.hist(V_1, bins=bins_needed)
    plt.title('LT, split 1')
    plt.xlabel("V")

    plt.subplot(233)
    bins_needed = int(len(split_2)/2)
    plt.hist(V_2, bins=bins_needed)
    plt.title('LT, split 2')
    plt.xlabel("V")

    plt.subplot(234)
    bins_needed = int(len(split_3)/2)
    plt.hist(V_3, bins=bins_needed)
    plt.title('LT, split 3')
    plt.xlabel("V")

    plt.subplot(235)
    bins_needed = int(len(split_4)/2)
    plt.hist(V_4, bins=bins_needed)
    plt.title('LT, split 4')
    plt.xlabel("V")

    plt.subplot(236)
    bins_needed = int(len(split_5)/2)
    plt.hist(V_5, bins=bins_needed)
    plt.title('LT, split 5')
    plt.xlabel("V")

    plt.tight_layout()
    plt.show()

def plot_LLS(data: pd.DataFrame, name: str):
    time = data["time"]
    width = data["width"]
    E = data["width error"]

    v_width, v_E = [0], [0]
    for i in range(len(time)-1):
        dt = time[i + 1] - time[i]

        dw = width[i + 1] - width[i]
        x_velocity = dw / dt
        dE = E[i + 1] - E[i]
        E_velocity = dE / dt

        v_width.append(x_velocity)
        v_E.append(E_velocity)

    '''plots for the data itself'''
    plt.subplot(121)
    plt.plot(time, width, label='width', color='red')
    plt.plot(time, E, label='center', color='blue')
    plt.legend(loc='upper right')
    plt.title('LLS, coordinates')
    plt.xlabel("Time")
    plt.ylabel("location")

    plt.subplot(122)
    plt.plot(time, v_width, label='v_width', color='red')
    plt.plot(time, v_E, label='v_center', color='blue')
    plt.legend(loc='upper right')
    plt.title('LLS, velocities')
    plt.xlabel("Time")
    plt.ylabel("velocity")

    plt.tight_layout()
    plt.show()

    err_range = max(E) - min(E)
    minimum = min(E)
    maximum = max(E)
    split = 0.2

    split_1, split_2, split_3, split_4, split_5 = [], [], [], [], []
    V_1, V_2, V_3, V_4, V_5 = [], [], [], [], []
    for i in range(len(time) - 1):
        err = E[i]

        if minimum <= err <= minimum + split * err_range:
            split_1.append(err)
            V_1.append(v_E[i])
        elif minimum + split * err_range <= err <= minimum + 2 * split * err_range:
            split_2.append(err)
            V_2.append(v_E[i])
        elif minimum + 2 * split * err_range <= err <= minimum + 3 * split * err_range:
            split_3.append(err)
            V_3.append(v_E[i])
        elif minimum + 3 * split * err_range <= err <= minimum + 4 * split * err_range:
            split_4.append(err)
            V_4.append(v_E[i])
        elif minimum + 4 * split * err_range <= err <= minimum + 5 * split * err_range:
            split_5.append(err)
            V_5.append(v_E[i])
        else:
            print("error: data point outside of splits")

    '''plots for subsets of the data so I can see how relations in error-change change based on what the error is.'''
    plt.subplot(231)
    plt.hist(v_E, bins=30)
    plt.title('LLS, err')
    plt.xlabel("V")

    plt.subplot(232)
    bins_needed = int(len(split_1) / 2)
    plt.hist(V_1, bins=bins_needed)
    plt.title('LLS, split 1')
    plt.xlabel("V")

    plt.subplot(233)
    bins_needed = int(len(split_2) / 5)
    plt.hist(V_2, bins=bins_needed)
    plt.title('LLS, split 2')
    plt.xlabel("V")

    plt.subplot(234)
    bins_needed = int(len(split_3) / 10)
    plt.hist(V_3, bins=bins_needed)
    plt.title('LLS, split 3')
    plt.xlabel("V")

    plt.subplot(235)
    bins_needed = int(len(split_4) / 5)
    plt.hist(V_4, bins=bins_needed)
    plt.title('LLS, split 4')
    plt.xlabel("V")

    plt.subplot(236)
    bins_needed = int(len(split_5) / 2)
    plt.hist(V_5, bins=bins_needed)
    plt.title('LLS, split 5')
    plt.xlabel("V")

    plt.tight_layout()
    plt.show()

def plot_camera(data: pd.DataFrame, name: str):
    time = data["time"]
    center = data["center"]
    E = data["error"]

    v_E, v_center = [0], [0]
    for i in range(len(time - 1)):
        dt = time[i + 1] - time[i]

        dE = E[i + 1] - E[i]
        E_velocity = dE / dt
        dc = center[i + 1] - center[i]
        y_velocity = dc / dt

        v_E.append(E_velocity)
        v_center.append(y_velocity)

    '''plots for the data itself'''
    plt.subplot(121)
    plt.plot(time, E, label='width', color='red')
    plt.plot(time, center, label='center', color='blue')
    plt.legend(loc='upper_right')
    plt.title('CAM, coordinates')
    plt.xlabel("Time")
    plt.ylabel("location")

    plt.subplot(122)
    plt.plot(time, v_E, label='v_width', color='red')
    plt.plot(time, v_center, label='v_center', color='blue')
    plt.legend(loc='upper_right')
    plt.title('CAM, velocities')
    plt.xlabel("Time")
    plt.ylabel("velocity")

    plt.tight_layout()
    plt.show()

    err_range = max(E) - min(E)
    min = min(E)
    max = max(E)
    split = 0.2 * err_range

    split_1, split_2, split_3, split_4, split_5 = [], [], [], [], []
    V_1, V_2, V_3, V_4, V_5 = [], [], [], [], []
    for i in range(len(time)):
        err = E[i]

        if min <= err <= min + split * err_range:
            split_1.append(err)
            V_1.append(v_E[i])
        elif min + split * err_range <= err <= min + 2 * split * err_range:
            split_2.append(err)
            V_2.append(v_E[i])
        elif min + 2 * split * err_range <= err <= min + 3 * split * err_range:
            split_3.append(err)
            V_3.append(v_E[i])
        elif min + 3 * split * err_range <= err <= min + 4 * split * err_range:
            split_4.append(err)
            V_4.append(v_E[i])
        elif min + 4 * split * err_range <= err <= min + 5 * split * err_range:
            split_5.append(err)
            V_5.append(v_E[i])
        else:
            print("error: data point outside of splits")

    '''plots for subsets of the data so I can see how relations in error-change change based on what the error is.'''
    plt.subplot(231)
    plt.hist(v_E)
    plt.title('CAM, err')
    plt.xlabel("V")

    plt.subplot(232)
    plt.hist(V_1)
    plt.title('CAM, split 1')
    plt.xlabel("V")

    plt.subplot(233)
    plt.hist(V_2)
    plt.title('CAM, split 2')
    plt.xlabel("V")

    plt.subplot(234)
    plt.hist(V_3)
    plt.title('CAM, split 3')
    plt.xlabel("V")

    plt.subplot(235)
    plt.hist(V_4)
    plt.title('CAM, split 4')
    plt.xlabel("V")

    plt.subplot(236)
    plt.hist(V_5)
    plt.title('CAM, split 5')
    plt.xlabel("V")

    plt.tight_layout()
    plt.show()

##############################################################################################################
"""Run this file"""

def main():
    "hi"

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
