import os
import numpy as np
import pandas as pd

## full processing a folder:
# from all raw dataframes of same organoid parameter
# get maximum distance and make a discretised distance array
# pass that to process all dataframes individually - normalise by dapi and cluster
# with processed dataframes, divide by maximum mean intensity
# compare this across all organoid data

def process_neuruloids_in_folder(folder_path, 
                                 num_discretise,
                                 channels=[2,3], 
                                 subtract_bg=[False, True]):
    """
    Process all data in a folder and save in a dictionary
    Args:
        folder_path (str): path containing dataframes output from segmentation
        num_discretise (int): number of discretisation of space (edge to centre)
        channels (list): list of channel number to be saved
        subtract_bg (list of bool): list of len(channels). True if normalise by max and min value, False if max value only
    """
    dataframe_names = os.listdir(folder_path)
    max_dist = get_max_distance_from_dataframes(folder_path)
    discrete_distance_array = np.linspace(0, max_dist, num_discretise)

    # create dataframes to store data
    dataframe_dict = {}
    for j in range(len(channels)):
        dataframe_dict[f"channel_{channels[j]}"] = pd.DataFrame({"distance":discrete_distance_array})

    for i in range(len(dataframe_names)):
        data_name = dataframe_names[i]
        df = pd.read_csv(os.path.join(folder_path, data_name))
        normalised_df = process_raw_to_norm_dataframe(df,
                                                      discrete_distance_array,
                                                      channels,
                                                      subtract_bg)
        
        for j in range(len(channels)):
            dataframe_dict[f"channel_{channels[j]}"][data_name] = normalised_df[f"channel_{channels[j]}"]
    return dataframe_dict

def dataframe_to_mean_sd(dataframe, return_full=False, save_path=None):
    """Dataframe containing signals from multiple organoid to mean and sd"""
    dataframe_columns = dataframe.columns.to_list()[1:]
    signal = np.array([dataframe[dataframe_columns[i]] for i in range(len(dataframe_columns))])
    mean_signal, sd_signal = np.nanmean(signal, axis=0), np.nanstd(signal, axis=0)

    if return_full:
        # if return full add into a dataframe and save into save_path 
        df_dict = {"distance":dataframe["distance"], "mean":mean_signal, "sd":sd_signal}
        mean_dataframe = pd.DataFrame(df_dict)
        mean_dataframe.to_csv(save_path)

    return mean_signal, sd_signal, len(signal)

def process_raw_to_norm_dataframe(dataframe,
                                  discrete_distance_array,
                                  channels=[2,3],
                                  subtract_bg=[False, True]):
    processed_df = process_raw_dataframe(dataframe, 
                                         discrete_distance_array, 
                                         channels)
    normalised_max_df = normalising_intensities_to_max(processed_df, channels, subtract_bg)
    return normalised_max_df

def normalising_intensities_to_max(processed_df,
                                   channels=[2,3],
                                   subtract_bg=[False, True]):
    """
    From processed dataframe with mean intensities, 
    divide by maximum to get intensities between 0-1
    """
    normalised_max_df = pd.DataFrame(columns=["discrete_distance"])
    normalised_max_df["discrete_distance"] = processed_df["discrete_distance"]

    for i in range(len(channels)):
        processed_intensity = processed_df[f"channel_{channels[i]}_mean"].to_numpy()
        
        if subtract_bg[i]:
            norm_to_max = (processed_intensity - np.nanmin(processed_intensity))/ (np.nanmax(processed_intensity) - np.nanmin(processed_intensity))

        else:
            norm_to_max = processed_intensity / np.nanmax(processed_intensity)

        normalised_max_df[f"channel_{channels[i]}"] = norm_to_max
    return normalised_max_df

def process_raw_dataframe(dataframe, 
                          discrete_distance_array,
                          channels=[2,3]):
    """
    From single raw dataframe to processed dataframe by
    - normalising with dapi
    - mean intensity for each distance range
    """
    # normalise dataframe by dapi channel
    normalised_df = normalise_channels_by_dapi(dataframe, 
                                               channels)
    # get mean for discrete distance
    processed_df = mean_intensities(normalised_df, discrete_distance_array, channels)
    return processed_df

def mean_intensities(intensity_dataframe,
                     discrete_distance_array,
                     channels=[2,3]):
    """
    From dataframe, return mean intensities for 
    discretised distance for all listed channels
    Returns:
        dataframe with columns:
        ["discrete_distance", "channel_1_mean", "channel_1_sd",.... "cell_number"]
    """
    distances = intensity_dataframe["distance"]
    dataframe = pd.DataFrame(columns=["discrete_distance", "cell_number"])
    dataframe["discrete_distance"] = discrete_distance_array[1:]

    for i in range(len(channels)):
        intensities = intensity_dataframe[f"channel_{channels[i]}"]
        mean_intensity, sd_intensity, cell_number = mean_intensity_signal(discrete_distance_array, 
                                                                          distances, 
                                                                          intensities)
        dataframe[f"channel_{channels[i]}_mean"] = mean_intensity
        dataframe[f"channel_{channels[i]}_sd"] = sd_intensity

    dataframe["cell_number"] = cell_number
    return dataframe

def mean_intensity_signal(discrete_distance, 
                          distances, 
                          intensities):
    """
    Computes the mean and standard deviation of intensity values within 
    specified distance intervals.
    Args:
        discrete_distance (array-like): Sorted boundary values for distance intervals.
        distances (array-like): Distance values corresponding to intensity measurements.
        intensities (array-like): Intensity values at respective distances.
    Returns:
        tuple: (mean intensities, standard deviations) for each interval.
    """
    mean_intensity = np.empty(len(discrete_distance) - 1)
    sd_intensity = np.empty(len(discrete_distance) - 1)
    cell_number = np.empty(len(discrete_distance) - 1)
    
    for i, (d1, d2) in enumerate(zip(discrete_distance[:-1], discrete_distance[1:])):
        mask = (distances > d1) & (distances <= d2)
        intensity_in_range = intensities[mask]
        cell_number[i] = intensity_in_range.size

        if intensity_in_range.size > 0:
            mean_intensity[i], sd_intensity[i] = np.mean(intensity_in_range), np.std(intensity_in_range)
        else:
            mean_intensity[i], sd_intensity[i] = np.nan, np.nan

    return mean_intensity, sd_intensity, cell_number

def normalise_channels_by_dapi(dataframe, channels=[2,3]):
    """
    Normalise channel intensities in dataframe with dapi
    Args:
        dataframe of segmentation where column "channel_1" is Dapi (pd.DataFrame)
        mean_dapi (float): factor to multiply the signals by
        channels (list): list of channel to normalise
        subtract_bg (list): list of channels to subtract background
    Returns:
        new dataframe without column channel_1 (pd.DataFrame)
    """
    dapi_signal = dataframe["channel_1"].to_numpy()
    mean_dapi = np.mean(dapi_signal)

    normalised_dataframe = pd.DataFrame(columns=["cell_id", "distance"])
    normalised_dataframe["cell_id"] = dataframe["cell_id"]
    normalised_dataframe["distance"] = dataframe["distance"]

    for i in range(len(channels)):
        intensity = dataframe[f"channel_{channels[i]}"].to_numpy()
        norm_intensity = intensity * mean_dapi / dapi_signal
        min_intensity = np.min(norm_intensity)
        norm_intensity = norm_intensity - min_intensity # subtract by minimum for all signals
        normalised_dataframe[f"channel_{channels[i]}"] = norm_intensity

    return normalised_dataframe

def get_max_distance_from_dataframes(folder_path):
    dataframe_names = os.listdir(folder_path)
    max_dists = np.zeros(len(dataframe_names))
    for i in range(len(dataframe_names)):
        df = pd.read_csv(os.path.join(folder_path, dataframe_names[i]))
        max_dists[i] = np.max(df["distance"].to_numpy())
    return np.max(max_dists)