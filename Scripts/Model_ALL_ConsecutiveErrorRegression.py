'''
This code makes regression models for mean and variation of the data bins that were extracted in the ConsecutiveErrorRegressions.
'''

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


def fit_linear(data: pd.DataFrame):   #bin_error: np.array, bin_mean: np.array, bin_variance: np.array
    '''a piece of code to fit and plot a linear regression to the data'''
    bin_error = np.array(data["y_mean"])    # TODO: should be x_mean
    bin_mean = np.array(data["deviation_mean"])
    bin_variance = np.array(data["deviation_variance"])
    # mean
    mu_a, mu_b = get_regression_linear(bin_error, bin_mean)
    plot_function_linear(mu_a, mu_b, bin_error, bin_mean, 'mean')

    # variance
    var_a, var_b = get_regression_linear(bin_error, bin_variance)
    plot_function_linear(var_a, var_b, bin_error, bin_variance, 'variance')

def fit_cubic(data: pd.DataFrame):   #bin_error: np.array, bin_mean: np.array, bin_variance: np.array
    '''a piece of code to fit and plot a cubic regression to the data'''
    bin_error = np.array(data["y_mean"])    # TODO: x_mean
    bin_mean = np.array(data["deviation_mean"])
    bin_variance = np.array(data["deviation_variance"])
    # mean
    mu_a, mu_b, mu_c, mu_d = get_regression_cubic(bin_error, bin_mean)
    plot_function(mu_a, mu_b, mu_c, mu_d, bin_error, bin_mean, 'mean')

    # variance
    var_a, var_b, var_c, var_d = get_regression_cubic(bin_error, bin_variance)
    plot_function(var_a, var_b, var_c, var_d, bin_error, bin_variance, 'variance')

def get_regression_linear(x_cords: np.array, y_cords: np.array):
    '''This code fits a linear regression and returns the factors of the polynomial.'''

    X, Y = [], []
    for i in range(len(x_cords)):
        X.append([1, x_cords[i]])
        Y.append(y_cords[i])

        #X = np.array(((1, x_cords[0], x_cords[0]**2), (1, x_cords[1], x_cords[1]**2), (1, x_cords[2], x_cords[2]**2)))
        #Y = np.array((y_cords[0], y_cords[1], y_cords[2]))

    print(X, Y)
    X_T = np.transpose(X)
    inverse_factor = np.linalg.inv(np.matmul(X_T, X))

    Beta = np.matmul(inverse_factor, X_T).dot(Y)

    return Beta[0], Beta[1]    # c, b, a

def get_regression_cubic(x_cords: np.array, y_cords: np.array):
    '''This code fits a cubic regression and returns the factors of the polynomial.'''
    X, Y = [], []
    for i in range(len(x_cords)):
        X.append([1, x_cords[i], x_cords[i]**2, x_cords[i]**3])
        Y.append(y_cords[i])

        #X = np.array(((1, x_cords[0], x_cords[0]**2), (1, x_cords[1], x_cords[1]**2), (1, x_cords[2], x_cords[2]**2)))
        #Y = np.array((y_cords[0], y_cords[1], y_cords[2]))

    print(X, Y)
    X_T = np.transpose(X)
    inverse_factor = np.linalg.inv(np.matmul(X_T, X))

    Beta = np.matmul(inverse_factor, X_T).dot(Y)

    return Beta[0], Beta[1], Beta[2], Beta[3]    # d, c, b, a

def plot_function_linear(a: float, b: float, bin_error: list, bin_mean: list, title: str):     # y = a + bx + cx^2
    '''This code plots the linear regression against the data points.'''
    points = 101
    min, max = np.min(bin_error), np.max(bin_error)
    start = min - 0.05*(max-min)
    end = max + 0.05*(max-min)
    step = (end - start)/(points-1)

    x_list, y_list = [], []
    for x in np.arange(start, end, step):
        y = a + b*x

        x_list.append(x)
        y_list.append(y)

    plt.plot(x_list, y_list)
    plt.scatter(bin_error, bin_mean)
    plt.title(title)
    # plt.ylim((min(y_list), max(y_list)))
    plt.show()

def plot_function(a: float, b: float, c: float, d: float, bin_error: list, bin_mean: list, title: str):     # y = a + bx + cx^2
    '''This code plots the cubic regression against the data points.'''
    points = 101
    min, max = np.min(bin_error), np.max(bin_error)
    start = min - 0.05*(max-min)
    end = max + 0.05*(max-min)
    step = (end - start)/(points-1)

    x_list, y_list = [], []
    for x in np.arange(start, end, step):
        y = a + b*x + c*(x**2) + d*(x**3)

        x_list.append(x)
        y_list.append(y)

    plt.plot(x_list, y_list)
    plt.scatter(bin_error, bin_mean)
    plt.title(title)
    # plt.ylim((min(y_list), max(y_list)))
    plt.show()

def test():
    '''Just some test data to check the code'''
    bin_error = np.array([-2.2, -0.9, 0, 1, 2.1])
    bin_mean = np.array([-1, -0.5, 0, 0.7, 1.1])
    bin_variance = np.array([5, 1.2, 0, 2, 3.8])

import Model_ALL_ConsecutiveErrorTheo

stats = Model_ALL_ConsecutiveErrorTheo.consecutive_error('LLS_A', '', 0.2)

x = fit_linear(stats)    # Model_ALL_functionsErrorPredict.bin_stats_df
# x = test()
