import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def obtain_tbxt_data(file_path):
    """
    Obtain TBXT data from files
    """
    df = pd.read_csv(file_path)
    # get tbxt distance
    tbxt_dist = df["distance"].to_numpy()[:-1]
    tbxt_dist = tbxt_dist / np.max(tbxt_dist)
    # get tbxt intensity
    tbxt_intensity = df["mean"].to_numpy()[:-1]
    tbxt_sd = df["sd"].to_numpy()[:-1]
    return tbxt_dist, tbxt_intensity, tbxt_sd

def fit_simulation(tbxt_distance, tbxt_intensity):
    """
    Fit TBXT intensity to simulation dataframe
    """
    simulation_file_path = "../processed_data/rd_simulations/simulation_intensity_circle.csv"
    simulation_data = pd.read_csv(simulation_file_path)
    simulation_distance = np.linspace(0, 1, 10)
    
    gamma = np.arange(5, 101, 1)
    errors = np.zeros(len(gamma))
    for i in range(len(gamma)):
        # flip the array from edge to centre
        activator_int = np.flip(simulation_data[f"gamma{gamma[i]}"].to_numpy())
        activator_int = np.interp(tbxt_distance, simulation_distance, activator_int)
        errors[i] = metrics.mean_squared_error(activator_int, tbxt_intensity)

    best_fit_gamma = gamma[np.argmin(errors)]
    best_fit_intensity = np.flip(simulation_data[f"gamma{best_fit_gamma}"].to_numpy())

    return simulation_distance, best_fit_intensity, best_fit_gamma, gamma, errors