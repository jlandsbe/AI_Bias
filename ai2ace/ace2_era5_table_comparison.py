###Imports, data loading, saving functions
import xarray as xr
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cdsapi
from datetime import datetime
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from ace_ERA5_pdf_comp_experiments import exp_dictionary

def load_file(filepath):
    """
    Loads a NetCDF file into an xarray.Dataset.

    Parameters:
        filepath (str): The path to the NetCDF file.

    Returns:
        xarray.Dataset: The loaded dataset.
    """
    try:
        dataset = xr.open_dataset(filepath)
        print(f"Successfully loaded NetCDF file: {filepath}")
        return dataset
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

def save_to_netcdf(dataset, output_path):
    """
    Save an xarray dataset to a NetCDF file.

    Parameters:
    - dataset (xarray.Dataset): The dataset to save.
    - output_path (str): The path where the NetCDF file will be saved.
    """
    try:
        dataset.to_netcdf(output_path)
        print(f"Dataset saved to {output_path}")
    except Exception as e:
        print(f"Error saving dataset: {e}")


def filter_dataset_by_lat_lon(dataset, bounds):
    """
    Filters an xarray dataset to only include values within the specified latitude and longitude bounds.

    Parameters:
        dataset (xarray.Dataset): The input dataset with 'lat' and 'lon' dimensions.
        bounds (dict): A dictionary with 'lat' and 'lon' keys specifying the bounds.
                       Example: {'lat': [-80, 80], 'lon': [-180, 180]}

    Returns:
        xarray.Dataset: The filtered dataset.
    """
    lat_min, lat_max = bounds['lat']
    lon_min, lon_max = bounds['lon']

    # Filter the dataset based on the latitude and longitude bounds
    filtered_dataset = dataset.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )

    return filtered_dataset

import xarray as xr

def create_land_ocean_masks(filepath ='/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/forcing/forcing_1940.nc', maskout_type = "ocean"):
    """
    Loads a NetCDF file, computes the mean land fraction over time, 
    and creates land and ocean masks based on a threshold.

    Parameters:
        filepath (str): The path to the NetCDF file that contains land fraction data.

    Returns:
        tuple: A tuple containing two xarray.DataArrays:
               - land_maskout: The land mask (land_frac < 0.5).
               - ocean_maskout: The ocean mask (land_frac >= 0.5).
    """
    # Load the file
    dataset = xr.open_dataset(filepath)
    
    # Compute the mean land fraction over time
    land_frac = dataset.mean(dim='time').land_fraction
    
    # Create land and ocean masks
    land_maskout = land_frac.where(land_frac < 0.5).values
    ocean_maskout = land_frac.where(land_frac >= 0.5).values
    no_mask = land_frac.where(land_frac >= -1).values
    
    if maskout_type == "land":
        return land_maskout
    elif maskout_type == "ocean":
        return ocean_maskout
    else:
        return no_mask



import xarray as xr
import numpy as np

import numpy as np

def split_surface_temp_by_decade(surface_temp_vals, start_year, time_chunk=10, spatial_mean=1, drop_excess=1):
    """
    Splits surface temperature values into chunks based on decades, using real xarray time coordinates,
    and returns NumPy arrays like in the original version.

    Parameters:
        surface_temp_vals (xarray.DataArray): Surface temperature values.
        start_year (int): Starting year for first chunk.
        time_chunk (int): Number of years in each chunk (default 10).
        spatial_mean (bool): Whether to spatially average (default True).
        drop_excess (bool): Whether to drop leftover data that doesn't fit a full chunk (default True).

    Returns:
        dict: Dictionary of {decade_name: numpy.ndarray}.
    """
    if spatial_mean:
        weights = np.cos(np.deg2rad(surface_temp_vals.lat))
        weighted_mean = surface_temp_vals.weighted(weights).mean(dim=("lat", "lon"), skipna=True)
        surface_temp_vals = weighted_mean

    # Make sure 'time' is sorted
    surface_temp_vals = surface_temp_vals.sortby('time')

    # Get years from time
    years = surface_temp_vals['time'].dt.year

    # Initialize
    dictionary = {}

    current_start = start_year

    while current_start <= years.max():
        current_end = current_start + time_chunk

        # Select time slice
        time_slice = surface_temp_vals.sel(time=slice(f"{current_start}-01-01", f"{current_end-1}-12-31"))

        # Check if enough years are present
        unique_years = np.unique(time_slice['time'].dt.year)

        # If we are dropping excess and there aren't enough years, stop
        if drop_excess and len(unique_years) < time_chunk:
            break

        # Skip if empty
        if len(time_slice['time']) == 0:
            break

        # Name key
        if time_chunk == 10:
            key = f"{current_start}s"
        else:
            key = f"{current_start}-{current_end-1}"

        # Convert to squeezed numpy array
        dictionary[key] = np.squeeze(time_slice.values)

        # Move to next chunk
        current_start += time_chunk

    return dictionary


import numpy as np

def subset_dict_by_percentile(data_dict, cutoff):
    """
    Subsets values in each array in a dictionary based on a percentile cutoff.
    
    Parameters:
    - data_dict: dict of 1D numpy arrays
    - cutoff: float or list/tuple of two floats
        - If float:
            - If 0: returns original arrays
            - If > 0: returns values greater than the given percentile
            - If < 0: returns values less than the given absolute percentile
        - If list/tuple of two numbers: returns values between the two percentiles
    
    Returns:
    - A new dictionary with filtered arrays
    """
    subsetted_dict = {}

    for key, arr in data_dict.items():
        if isinstance(cutoff, (list, tuple)):
            low, high = sorted(cutoff)  # Ensure low <= high
            low_thresh = np.percentile(arr, low)
            high_thresh = np.percentile(arr, high)
            subsetted_dict[key] = arr[(arr >= low_thresh) & (arr <= high_thresh)]
        else:
            if cutoff == 0:
                subsetted_dict[key] = arr
            elif cutoff > 0:
                threshold = np.percentile(arr, cutoff)
                subsetted_dict[key] = arr[arr > threshold]
            elif cutoff < 0:
                threshold = np.percentile(arr, abs(cutoff))
                subsetted_dict[key] = arr[arr < threshold]

    return subsetted_dict



def compute_dist_table(dict1, dict2, dict1_name, dict2_name, percentile_cutoff=0, metric="emd", save_fig_title="exp", mean_only=0):   
    """
    Computes Earth Mover's Distance (Wasserstein distance) between each pair of arrays in two dictionaries.
    Returns a DataFrame and a heatmap.
    """
    dict1 = subset_dict_by_percentile(dict1, percentile_cutoff)
    dict2 = subset_dict_by_percentile(dict2, percentile_cutoff)

    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    
    # Initialize distance matrix
    distance_matrix = np.zeros((len(keys1), len(keys2)))
    mean_distance_matrix = np.zeros((len(keys1), len(keys2)))
    # Compute pairwise distances
    for i, k1 in enumerate(keys1):
        for j, k2 in enumerate(keys2):
            # Unravel multidimensional data if necessary
            data1 = dict1[k1].ravel() if dict1[k1].ndim > 1 else dict1[k1]
            data2 = dict2[k2].ravel() if dict2[k2].ndim > 1 else dict2[k2]
            
            if metric == "emd":
                distance_matrix[i, j] = wasserstein_distance(data1, data2)
                mean_distance_matrix[i, j] = np.mean(data1) - np.mean(data2)
            elif metric == "js":
                distance_matrix[i, j] = jensenshannon(data1, data2)
                mean_distance_matrix[i, j] = np.abs(np.mean(data1) - np.mean(data2))
            else:
                raise ValueError("Unsupported metric.")

    # Create DataFrame
    df = pd.DataFrame(distance_matrix, index=keys1, columns=keys2)
    if mean_only:
        df = pd.DataFrame(mean_distance_matrix, index=keys1, columns=keys2)

    # Prepare bold annotation labels for min values
    annotations = df.copy().astype(str)
    for i in range(df.shape[0]):
        min_idx = df.iloc[i].argmin()
        annotations.iloc[i, min_idx] = f"**{df.iloc[i, min_idx]:.2f}**"

    # Decide color map and center based on mean_only
    if mean_only:
        cmap = "coolwarm"
        center = 0
    else:
        cmap = "Reds"
        center = None

    # Create 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: No normalization
    sns.heatmap(
        df, 
        annot=df.applymap("{:.2f}".format), 
        fmt="", 
        cmap=cmap, 
        cbar=True, 
        center=center,
        xticklabels=True, 
        yticklabels=True, 
        ax=axes[0],
        linewidths=1,
        linecolor="lightgrey"
    )
    for i in range(len(keys1)):
        axes[0].add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=3))
    axes[0].set_title("No Normalization")
    axes[0].set_ylabel(dict1_name)
    axes[0].set_xlabel(dict2_name)

    # Plot 2: Row normalization
    df_row_norm = df.sub(df.min(axis=1), axis=0).div(df.max(axis=1) - df.min(axis=1), axis=0)
    sns.heatmap(
        df_row_norm, 
        annot=df.applymap("{:.2f}".format), 
        fmt="", 
        cmap=cmap, 
        cbar=True, 
        center=center,
        xticklabels=True, 
        yticklabels=True, 
        ax=axes[1],
        linewidths=1,
        linecolor="lightgrey"
    )
    for i in range(len(keys1)):
        axes[1].add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=3))
    axes[1].set_title("Row Normalization")
    axes[1].set_ylabel(dict1_name)
    axes[1].set_xlabel(dict2_name)

    # Plot 3: Column normalization
    df_col_norm = df.sub(df.min(axis=0), axis=1).div(df.max(axis=0) - df.min(axis=0), axis=1)
    sns.heatmap(
        df_col_norm, 
        annot=df.applymap("{:.2f}".format), 
        fmt="", 
        cmap=cmap, 
        cbar=True, 
        center=center,
        xticklabels=True, 
        yticklabels=True, 
        ax=axes[2],
        linewidths=1,
        linecolor="lightgrey"
    )
    for i in range(len(keys1)):
        axes[2].add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=3))
    axes[2].set_title("Column Normalization")
    axes[2].set_ylabel(dict1_name)
    axes[2].set_xlabel(dict2_name)


    plt.tight_layout()
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/{save_fig_title}_{percentile_cutoff}_percentile_{metric}_table.png")
    plt.show()

    return df

def filter_by_months(ds1, ds2, months=[1,2,3,4,5,6,7,8,9,10,11,12]):

    
    # Check if ds1 and ds2 have different lengths
    if len(ds1['time']) != len(ds2['time']):
        diff = len(ds2['time']) - len(ds1['time'])
        if diff > 0:
            # Trim the difference from the beginning of ds2
            ds2 = ds2.isel(time=slice(diff, None))
        else:
            # Trim the difference from the beginning of ds1
            ds1 = ds1.isel(time=slice(-diff, None))
    mask = ds1['time'].dt.month.isin(months)
    ds1_filtered = ds1.sel(time=mask)
    ds2_filtered = ds2.sel(time=mask)
    print(len(ds1_filtered['time']), len(ds2_filtered['time']))
    return ds1_filtered, ds2_filtered


def plot_empirical_pdfs_for_key(dict_ace, dict_era5, key, bins=100, percentile_cutoff=0, exp_name=""):
    """
    Plots empirical PDFs (histograms) for the values corresponding to a single key in two dictionaries.
    
    Parameters:
    - dict1, dict2: dictionaries of 1D numpy arrays
    - key: the key for which to plot the PDFs in both dictionaries
    - bins: number of bins for the histogram (default = 100)
    """

    dict1 = subset_dict_by_percentile(dict_ace, percentile_cutoff)
    dict2 = subset_dict_by_percentile(dict_era5, percentile_cutoff)
    # Retrieve data for the specified key in both dictionaries
    data1 = dict1.get(key)
    data2 = dict2.get(key)

    if percentile_cutoff != 0:
        bins = int(bins/2)
   
    if data1 is None or data2 is None:
        raise KeyError(f"The key '{key}' does not exist in both dictionaries.")

    # Compute the normalized histogram (empirical PDF) for dict1
    counts1, bin_edges1 = np.histogram(data1, bins=bins, density = 1)
    bin_centers1 = 0.5 * (bin_edges1[1:] + bin_edges1[:-1])  # Get the bin centers

    # Compute the normalized histogram (empirical PDF) for dict2
    counts2, bin_edges2 = np.histogram(data2, bins=bins, density = 1)
    bin_centers2 = 0.5 * (bin_edges2[1:] + bin_edges2[:-1])  # Get the bin centers

    # Plot empirical PDF (line plot) for dict1
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers1, counts1, label=f'{key} (ACE2)', color='blue', lw=2)
    
    # Plot empirical PDF (line plot) for dict2
    plt.plot(bin_centers2, counts2, label=f'{key} (ERA5)', color='red', lw=2, linestyle='--')
    if percentile_cutoff < 0:
        print(np.sum(counts1), np.sum(counts2))
    plt.title(f'Empirical PDF Comparison for: {key}')
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/empirical_pdf_{key}_{exp_name}.png")

def preprocess_and_run_experiment(fp_ACE, fp_ERA5, percentile = 0, mask_type=0, months = [1,2,3,4,5,6,7,8,9,10,11,12], lat_bounds=None, lon_bounds=None, start_year=1940, time_chunk=10,exp_save_name="", mean_only=0):
    """
    Preprocesses ACE2 and ERA5 data, applies filtering, and splits data into time chunks for analysis.

    Parameters:
        fp_ACE (str): Filepath to the ACE2 data.
        fp_ERA5 (str): Filepath to the ERA5 data.
        mask_type (str): Specify "land" or "ocean" for filtering. Default is "ocean".
        lat_bounds (list): Latitude bounds as [min_lat, max_lat]. Default is None (no filtering).
        lon_bounds (list): Longitude bounds as [min_lon, max_lon]. Default is None (no filtering).
        start_year (int): The starting year for splitting data into time chunks. Default is 1940.
        time_chunk (int): The number of years in each time chunk. Default is 10.

    Returns:
        pd.DataFrame: A DataFrame containing distance metrics between ACE2 and ERA5 data.
    """
    # Step 1: Load ACE2 and ERA5 data
    ACE2_data = load_file(fp_ACE)
    ERA5_data = load_file(fp_ERA5)
    print('Data loaded')
    ERA5_data, ACE2_data = filter_by_months(ERA5_data, ACE2_data, months=months)

    print(ERA5_data)
    print(ACE2_data)



    if mask_type == "ocean" or mask_type == "land":
        # Step 2: Create land or ocean mask
        mask = create_land_ocean_masks(maskout_type=mask_type)
        print(f'{mask_type.capitalize()} mask created')

        # Step 3: Apply mask to ACE2 and ERA5 data
        ACE2_data_filtered = ACE2_data * mask
        ERA5_data_filtered = ERA5_data * mask
        print('Mask applied to data')
    else:
        ACE2_data_filtered = ACE2_data
        ERA5_data_filtered = ERA5_data
        print('No mask applied to data')

    print(ACE2_data_filtered)
    print(ERA5_data_filtered)
    # Step 4: Filter by latitude and longitude bounds if specified
    if lat_bounds and lon_bounds:
        ACE2_data_filtered = filter_dataset_by_lat_lon(ACE2_data_filtered, {'lat': lat_bounds, 'lon': lon_bounds})
        ERA5_data_filtered = filter_dataset_by_lat_lon(ERA5_data_filtered, {'lat': lat_bounds, 'lon': lon_bounds})
        print(f'Data filtered by lat/lon bounds: lat={lat_bounds}, lon={lon_bounds}')

    print(ACE2_data_filtered)
    print(ERA5_data_filtered)
    ace_split_dict = split_surface_temp_by_decade(ACE2_data_filtered['surface_temperature'], start_year=start_year, time_chunk=time_chunk)
    era_split_dict = split_surface_temp_by_decade(ERA5_data_filtered['surface_temperature'], start_year=start_year, time_chunk=time_chunk)
    print('Data split into time chunks')
    print(ace_split_dict)
    print(era_split_dict)

    # Step 6: Compute distance metrics between ACE2 and ERA5 data
    df = compute_dist_table(ace_split_dict, era_split_dict, "ACE2 Temperature", "ERA5 Temperature", percentile_cutoff=percentile, metric="emd", save_fig_title=exp_save_name, mean_only=mean_only)
    print('Distance metrics computed')

    return ace_split_dict,era_split_dict

# Filepaths for ACE2 and ERA5 data
fp_ACE = '/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ACE2_processed.nc'
fp_ERA5 = '/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ERA5_processed.nc'
Experiment_list =  list(exp_dictionary.keys())
#for experiment in Experiment_list:
for experiment in Experiment_list:
    exp_name = exp_dictionary[experiment]["exp_name"]
    exp_name = exp_name+"_15year"
    mask_type = exp_dictionary[experiment]["mask_type"]
    lat_bounds = exp_dictionary[experiment]["lat_bounds"]
    lon_bounds = exp_dictionary[experiment]["lon_bounds"]
    start_year = exp_dictionary[experiment]["start_year"]
    start_year = 1945
    time_chunk = exp_dictionary[experiment]["time_chunk"]
    time_chunk = 15
    percentile = exp_dictionary[experiment]["percentile"]
    months = exp_dictionary[experiment]["months"]
    #months = [1,2,3,4,5,6,7,8,9,10,11,12]
    mean_only = exp_dictionary[experiment]["mean_only"]
    print(f"Running experiment: {exp_name}")
    ace_split_dict, era_split_dict = preprocess_and_run_experiment(fp_ACE, fp_ERA5, percentile, mask_type, months, lat_bounds, lon_bounds, start_year, time_chunk, exp_name, mean_only)

    if mean_only or 1:

        ace_split_dict_filtered = subset_dict_by_percentile(ace_split_dict, percentile)
        era_split_dict_filtered = subset_dict_by_percentile(era_split_dict, percentile)
        # Compute yearly averages for ACE2 and ERA5
        ace_yearly_avg = {key: np.mean(values) for key, values in ace_split_dict_filtered.items()}
        era_yearly_avg = {key: np.mean(values) for key, values in era_split_dict_filtered.items()}

        # Plot the time series of average yearly temperatures
        plt.figure(figsize=(10, 6))
        plt.plot(list(ace_yearly_avg.keys()), list(ace_yearly_avg.values()), label="ACE2 Yearly Avg Temp")
        plt.plot(list(era_yearly_avg.keys()), list(era_yearly_avg.values()), label="ERA5 Yearly Avg Temp")
        plt.title(f"Time Series of Average Yearly Temperatures ({exp_name})")
        plt.xlabel("Time Period")
        plt.ylabel("Average Temperature")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/yearly_avg_temp_{exp_name}.png")
        plt.show()

