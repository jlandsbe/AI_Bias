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
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from matplotlib import font_manager as fm
import matplotlib as mpl

# Path to your font files
font_path_regular = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-VariableFont_wght.ttf"
font_path_italic = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-Italic-VariableFont_wght.ttf"

# Register fonts
fm.fontManager.addfont(font_path_regular)
fm.fontManager.addfont(font_path_italic)

# Set global font family to Red Hat Display
mpl.rcParams['font.family'] = 'Red Hat Display'

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
import numpy as np

import xarray as xr
import numpy as np

def subset_xarray_by_percentile_per_year(data, cutoff, spatial_mean=True, subtract_mean=False):
    """
    Subsets an xarray.DataArray based on a percentile cutoff,
    after optionally spatial averaging (over lat/lon only) but preserving other dims (e.g., 'sample').

    Parameters:
    - data: xarray.DataArray with 'time' and optionally 'lat', 'lon', and 'sample' dimensions
    - cutoff: float or tuple/list of two floats
    - spatial_mean: bool, if True, average over lat/lon before processing
    - subtract_mean: bool, if True, remove mean before thresholding

    Returns:
    - Masked xarray.DataArray (time + other non-spatial dims like sample)
    """

    if spatial_mean:
        weights = np.cos(np.deg2rad(data.lat))
        data = data.weighted(weights).mean(dim=("lat", "lon"), skipna=True)


    # Group by 'time.year'
    grouped = data.groupby('time.year')

    def filter_group(g):
        if isinstance(cutoff, (list, tuple)):
            low, high = sorted(cutoff)
            low_thresh = g.quantile(low / 100, dim='time')
            high_thresh = g.quantile(high / 100, dim='time')
            mask = (g >= low_thresh) & (g <= high_thresh)
        else:
            if cutoff >= 0:
                thresh = g.quantile(cutoff / 100, dim='time')
                mask = g >= thresh
            else:
                thresh = g.quantile(abs(cutoff) / 100, dim='time')
                mask = g <= thresh

        if subtract_mean:
            print(g)
            g = g - g.mean(dim='time')
            print(g)

        return g.where(mask)

    # Apply per year
    filtered = grouped.map(filter_group)

    return filtered





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
        return xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/low_res_poles_mask.nc").__xarray_dataarray_variable__.values
    else:
        return no_mask

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
    return ds1_filtered, ds2_filtered

def chunker(surface_temp_vals, start_year, time_chunk=10, spatial_mean=1, drop_excess=1):
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


    # Make sure 'time' is sorted
    surface_temp_vals = surface_temp_vals.sortby('time')

    # Get years from time
    years = surface_temp_vals['time'].dt.year
    dictionary = {}

    current_start = start_year

    while current_start <= years.max().item():
        current_end = current_start + time_chunk
        # Select time slice
        time_slice = surface_temp_vals.sel(time=slice(f"{current_start}-01-01", f"{current_end-1}-12-31"))

        # Check if enough years are present
        unique_years = np.unique(time_slice['time'].dt.year)

        # Skip if empty
        if len(time_slice['time']) == 0:
            break

        # If we are dropping excess and there aren't enough years, stop
        if drop_excess and len(unique_years) < time_chunk:
            break

        # Name key
        if time_chunk == 10:
            #key = f"{current_start}s"
            key = f"{current_start}-{current_end-1}"
        elif time_chunk == 1:
            key = f"{current_start}"
        else:
            key = f"{current_start}-{current_end-1}"
        # Convert to squeezed numpy array
        dictionary[key] = np.squeeze(time_slice.values)

        # Move to next chunk
        current_start += time_chunk

    return dictionary



import matplotlib.pyplot as plt
import numpy as np
import re

def plot_single_time_series(data, start_year, time_chunk = 10, title = "Mean Temperature Differnce (ACE- ERA5)", exp_name=""):
    chunk_keys = list(data.keys())
    x_vals = np.arange(len(chunk_keys))  # Numeric x-values for plotting
    vals_to_plot = np.array([np.nanmean(data[k]) for k in chunk_keys])
    ylabel = "Mean Temperature Difference (°C)"
    filename = f"{exp_name}_average_temp_all_land.png"
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    old_color = '#136F63'  # Dark cyan color
    ax.plot(x_vals, vals_to_plot, '-o', color='#102e4a', label='Mean Temperature Difference', linewidth=3, markersize=8)


    ax.axhline(y=0, color='#FFBC42', linestyle='solid', linewidth=2, label='Zero Temperature Difference', alpha=0.8)
    ax.fill_between(
        x_vals, vals_to_plot, 0,
        where=(vals_to_plot <= 0),
        color='cornflowerblue', alpha=0.5, interpolate=True, label='ACE2 < ERA5'
    )

    ax.fill_between(
        x_vals, vals_to_plot, 0,
        where=(vals_to_plot > 0),
        color='tomato', alpha=0.5, interpolate=True, label='ACE2 > ERA5'
    )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time Period", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # X-ticks as chunk_keys (e.g., '1940-1950', '1950-1960', etc.)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(chunk_keys, rotation=45, ha='right')

    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/{filename}")
    plt.close()


def plot_timechunk_means(ACE2_mean, ERA5_mean, start_year, time_chunk, title, exp_name="", plot_derivative=False, center_at_zero=False):
    chunk_keys = list(ERA5_mean.keys())
    x_vals = np.arange(len(chunk_keys))  # Numeric x-values for plotting

    # ERA5: mean everything
    ERA5_vals = np.array([np.nanmean(ERA5_mean[k]) for k in chunk_keys])

    # ACE2: mean only across time (axis = time axis)
    ACE2_vals = []
    for k in chunk_keys:
        ace_vals = ACE2_mean[k]
        if ace_vals.ndim == 1:
            ACE2_vals.append(np.nanmean(ace_vals))
        else:
            ACE2_vals.append(np.nanmean(ace_vals, axis=-1))  # Mean across time, keep samples

    ACE2_vals = np.stack(ACE2_vals, axis=1)  # samples × timechunks
    if center_at_zero:
        ACE2_vals = ACE2_vals - np.mean(ACE2_vals, axis=1, keepdims=True)
        ERA5_vals = ERA5_vals - np.mean(ERA5_vals)

    ACE2_sample_mean = np.mean(ACE2_vals, axis=0)

    if plot_derivative:
        # Compute empirical derivative
        ERA5_vals = np.gradient(ERA5_vals)
        ACE2_vals = np.gradient(ACE2_vals, axis=1)
        ACE2_sample_mean = np.mean(ACE2_vals, axis=0)
        year_label = f"{time_chunk} year" if int(time_chunk) == 1 else f"{time_chunk} years"
        ylabel = f"Average Temperature Trend (°C per {year_label})"
        filename = f"{exp_name}_temp_trend_all_land.png"
    else:
        ylabel = "Average Temperature (°C)"
        filename = f"{exp_name}_average_temp_all_land.png"

    # Plot
    plt.figure(figsize=(10, 6))

    if plot_derivative:
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Mean Trend')

    # n_samples = ACE2_vals.shape[0]
    # for s in range(n_samples):
    #     plt.plot(x_vals, ACE2_vals[s, :], color="lightskyblue", linestyle='--', alpha=0.5,
    #              label='ACE2 Ensemble Member' if s == 0 else None)

    # Plot the lines
    plt.plot(x_vals, ERA5_vals, color='gold', label='ERA5 Mean', linewidth=2)
    plt.axhline(y=np.mean(ERA5_vals), color='darkgoldenrod', linestyle='--', linewidth=1.5, label='ERA5 mean')

    plt.plot(x_vals, ACE2_sample_mean, color='darkcyan', label='ACE2 Mean', linewidth=2)
    plt.axhline(y=np.mean(ACE2_sample_mean), color='darkslategrey', linestyle='--', linewidth=1.5, label='ACE2 mean')


    # # === ADD SHADED BOXES BASED ON YEAR RANGES ===
    # highlight_ranges = [(1999, 2011), (2020, 2022)]
    # for i, chunk in enumerate(chunk_keys):
    #     start, end = map(int, chunk.split('-'))
    #     for h_start, h_end in highlight_ranges:
    #         if end >= h_start and start <= h_end:
    #             plt.axvspan(i - 0.5, i + 0.5, color='grey', alpha=0.9, zorder=0)
    # # Add vertical dashed lines at specific years
    # highlight_years = [1979, 2015]
    # for year in highlight_years:
    #     for i, chunk in enumerate(chunk_keys):
    #         start, end = map(int, chunk.split('-'))
    #         if start <= year <= end:
    #             plt.axvline(i, color='red', linestyle='--', linewidth=1.5, label=f'{year}' if year == highlight_years[0] else None)
    #             break

    plt.title(title)
    plt.xlabel("Time Chunk")
    plt.ylabel(ylabel)

    # X-ticks as chunk_keys
    plt.xticks(ticks=x_vals, labels=chunk_keys, rotation=45, ha='right')

    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/{filename}")
    plt.close()




def get_time_split_data(fp_ACE, fp_ERA5,start_year, end_year, mask_type="ocean", months=[1,2,12], global_mean= True):
    """
    Splits ACE2 and ERA5 data into time chunks based on the specified start and end years,
    and filters by specified months.

    Parameters:
        ffp_ACE (str): Filepath to the ACE2 data.
        fp_ERA5 (str): Filepath to the ERA5 data.
        start_year (int): The starting year for splitting data into time chunks.
        end_year (int): The ending year for splitting data into time chunks.
        months (list): List of months to filter by. Default is [1, 2, 12].

    Returns:
        tuple: A tuple containing two dictionaries with time chunked data for ACE2 and ERA5.
    """
    ACE2_data = load_file(fp_ACE)
    ERA5_data = load_file(fp_ERA5)

    ACE2_data, ERA5_data = filter_by_months(ACE2_data, ERA5_data, months=months)
    if mask_type == "ocean" or mask_type == "land":
        # Step 2: Create land or ocean mask
        mask = create_land_ocean_masks(maskout_type=mask_type)
        print(f'{mask_type.capitalize()} mask created')

        # Step 3: Apply mask to ACE2 and ERA5 data
        ACE2_data_filtered = ACE2_data * mask
        print(ACE2_data)
        print(ACE2_data_filtered)
        print(mask)
        ERA5_data_filtered = ERA5_data * mask
        print('Mask applied to data')
    else:
        ACE2_data_filtered = ACE2_data
        ERA5_data_filtered = ERA5_data
        print('No mask applied to data')
    if global_mean:
        # Compute global mean if specified
        ACE2_data_filtered = ACE2_data_filtered.mean(dim=['lat', 'lon'], skipna=True)
        ERA5_data_filtered = ERA5_data_filtered.mean(dim=['lat', 'lon'], skipna=True)
    ACE2_data_filtered = ACE2_data_filtered.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    ERA5_data_filtered = ERA5_data_filtered.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    #take just 1st sample of Ace
    ACE2_data_filtered = ACE2_data_filtered.isel(sample=0)
    ACE2_data_filtered_vals = ACE2_data_filtered['surface_temperature'].values
    ERA5_data_filtered_vals = ERA5_data_filtered['surface_temperature'].values
    return ACE2_data_filtered_vals.flatten(), ERA5_data_filtered_vals.flatten()





import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
def plot_dual_list_pdfs(list1, list2, colors, labels, title, num_points=1000, bandwidth_adjust=1):

    """
    Plot Gaussian KDE PDFs of two lists of 3 numpy arrays each using Matplotlib.
    
    Parameters:
    - list1: List of 3 numpy arrays (dashed lines, lower alpha)
    - list2: List of 3 numpy arrays (solid lines)
    - colors: List of 3 colors (color for each array pair)
    - labels: List of 6 labels (list1[0], list2[0], list1[1], list2[1], ...)
    - title: Title of the plot
    - num_points: Number of x-points in the PDF curve
    - bandwidth_adjust: Factor to adjust KDE smoothness (default = 1)
    """
    plt.figure(figsize=(10, 6))

    # Determine a common x-axis range across all data
    all_data = np.concatenate(list1 + list2)
    x_min, x_max = all_data.min(), all_data.max()
    x = np.linspace(x_min, x_max, num_points)
    
    for i in range(3):
        # KDE for list1[i] (dashed)
        kde1 = gaussian_kde(list1[i], bw_method='scott')
        kde1.set_bandwidth(kde1.factor * bandwidth_adjust)
        plt.plot(x, kde1(x), linestyle='--', linewidth=3, color=colors[i], alpha=0.6, label=labels[2*i])
        if i<3:
        # KDE for list2[i] (solid)
            kde2 = gaussian_kde(list2[i], bw_method='scott')
            kde2.set_bandwidth(kde2.factor * bandwidth_adjust)
            plt.plot(x, kde2(x), linestyle='-', linewidth=3, color=colors[i], alpha=1.0, label=labels[2*i+1])
    
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/{title.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()





def preprocess_and_run_experiment(fp_ACE, fp_ERA5, percentile = 0, mask_type=0, months = [1,2,3,4,5,6,7,8,9,10,11,12], lat_bounds=None, lon_bounds=None, start_year=1940, time_chunk=10,exp_save_name="", mean_only=1, cutoff=0, plot=1, subtract_mean=0, trend = 0, z_center=0):
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

    if mask_type == "ocean" or mask_type == "land":
        # Step 2: Create land or ocean mask
        mask = create_land_ocean_masks(maskout_type=mask_type)
        print(f'{mask_type.capitalize()} mask created')

        # Step 3: Apply mask to ACE2 and ERA5 data
        ACE2_data_filtered = ACE2_data * mask
        print(ACE2_data)
        print(ACE2_data_filtered)
        print(mask)
        ERA5_data_filtered = ERA5_data * mask
        print('Mask applied to data')
    else:
        ACE2_data_filtered = ACE2_data
        ERA5_data_filtered = ERA5_data
        print('No mask applied to data')

    # Step 4: Filter by latitude and longitude bounds if specified
    if lat_bounds and lon_bounds:
        ACE2_data_filtered = filter_dataset_by_lat_lon(ACE2_data_filtered, {'lat': lat_bounds, 'lon': lon_bounds})
        ERA5_data_filtered = filter_dataset_by_lat_lon(ERA5_data_filtered, {'lat': lat_bounds, 'lon': lon_bounds})
        print(f'Data filtered by lat/lon bounds: lat={lat_bounds}, lon={lon_bounds}')
        # shape of sample x time x lat x lon or time x lat x lon
    #optionally filter by percentile
    print(f"subtract mean: {subtract_mean}")
    ACE2_data_filtered_perc = subset_xarray_by_percentile_per_year(ACE2_data_filtered['surface_temperature'], cutoff=percentile, subtract_mean=subtract_mean)
    ERA5_data_filtered_perc = subset_xarray_by_percentile_per_year(ERA5_data_filtered['surface_temperature'], cutoff=percentile, subtract_mean=subtract_mean)





    ACE2_time_chunk_mean = chunker(ACE2_data_filtered_perc, start_year = start_year, time_chunk = time_chunk)
    ERA5_time_chunk_mean = chunker(ERA5_data_filtered_perc,start_year = start_year, time_chunk = time_chunk)
    ttl = f'Winter Global Land {exp_save_name} Mean'
    if plot:
        plot_timechunk_means(ACE2_time_chunk_mean, ERA5_time_chunk_mean, start_year, time_chunk,ttl , exp_save_name, plot_derivative=trend, center_at_zero=z_center)
    return ACE2_time_chunk_mean, ERA5_time_chunk_mean


# Filepaths
fp_ACE = '/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ACE2_processed.nc'
fp_ERA5 = '/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ERA5_processed.nc'

early_ace, early_era5 = get_time_split_data(fp_ACE, fp_ERA5, start_year=1940, end_year=1980, mask_type="ocean", months=[12,1,2])

late_ace, late_era5 = get_time_split_data(fp_ACE, fp_ERA5, start_year=1981, end_year=2020, mask_type="ocean", months=[12,1,2])

training_ace1, training_era5_1 = get_time_split_data(fp_ACE, fp_ERA5, start_year=1940, end_year=1995, mask_type="ocean", months=[12,1,2])
training_ace2, training_era5_2 = get_time_split_data(fp_ACE, fp_ERA5, start_year=2011, end_year=2019, mask_type="ocean", months=[12,1,2])

training_ace = np.concatenate([training_ace1, training_ace2])
training_era5 = np.concatenate([training_era5_1, training_era5_2])

ace_list = [early_ace, late_ace, training_ace]
era5_list = [early_era5, late_era5, training_era5]

plot_dual_list_pdfs(era5_list, ace_list, colors=['cornflowerblue', 'tomato','#102e4a'],
                    labels=['ERA5 Early', 'ACE2 Early', 'ERA5 Late', 'ACE2 Late', 'ERA5 Training', 'ACE2 Training'],
                    title='PDF of ACE2 and ERA5 Winter Land Temperature',)




# # 1, 5, 10, 15 year means
# year1_means_ace, year1_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=None, lon_bounds=None,
#     start_year=1940, time_chunk=1, exp_save_name="1_year", mean_only=1, months=[12,1,2]
# )


# year10_means_ace, year10_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=None, lon_bounds=None,
#     start_year=1940, time_chunk=10, exp_save_name="10_year", mean_only=1, months=[12,1,2]
# )

# # 1, 5, 10, 15 year means
# year1_means_ace, year1_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=None, lon_bounds=None,
#     start_year=1940, time_chunk=1, exp_save_name="1_year_no_mean", mean_only=1, months=[12,1,2], subtract_mean=1
# )


# year10_means_ace, year10_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=None, lon_bounds=None,
#     start_year=1940, time_chunk=10, exp_save_name="10_year_no_mean", mean_only=1, months=[12,1,2], subtract_mean=1)

# Setup for 90th and 10th percentile US regions
dict1_run = exp_dictionary['High_lats_Temperature_10th']
dict2_run = exp_dictionary['N_US_Temperature_10th']
dict4_run = exp_dictionary['E_Russia_Temperature_10th']


# # 1, 5, 10, 15 year means
# year1_means_ace, year1_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict1_run["lat_bounds"], lon_bounds=dict1_run["lon_bounds"],
#     start_year=1940, time_chunk=1, exp_save_name="us_all_1_year", mean_only=1, months=[12,1,2]
# )
# # 10-year means
# year10_means_ace, year10_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict1_run["lat_bounds"], lon_bounds=dict1_run["lon_bounds"],
#     start_year=1940, time_chunk=10, exp_save_name="us_all_10_year", mean_only=1, months=[12,1,2]
# )

# # 1-year means with 90th percentile cutoff
# year1_90th_means_ace, year1_90th_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict1_run["lat_bounds"], lon_bounds=dict1_run["lon_bounds"],
#     start_year=1940, time_chunk=1, exp_save_name="us_90th_1_year", mean_only=1, months=[12,1,2], percentile=90
# )

# # 10-year means with 90th percentile cutoff
# year10_90th_means_ace, year10_90th_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict1_run["lat_bounds"], lon_bounds=dict1_run["lon_bounds"],
#     start_year=1940, time_chunk=10, exp_save_name="us_90th_10_year", mean_only=1, months=[12,1,2], percentile=90
# )

# # 1-year means with -10th percentile cutoff
# year1_neg10th_means_ace, year1_neg10th_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict2_run["lat_bounds"], lon_bounds=dict2_run["lon_bounds"],
#     start_year=1940, time_chunk=1, exp_save_name="us_10th_1_year", mean_only=1, months=[12,1,2], percentile=-10
# )

# # 10-year means with -10th percentile cutoff
# year10_neg10th_means_ace, year10_neg10th_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict2_run["lat_bounds"], lon_bounds=dict2_run["lon_bounds"],
#     start_year=1940, time_chunk=10, exp_save_name="us_10th_10_year", mean_only=1, months=[12,1,2], percentile=-10
# )

# # 1-year means with 90th percentile cutoff and subtract_mean=1
# year1_90th_sub_means_ace, year1_90th_sub_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict1_run["lat_bounds"], lon_bounds=dict1_run["lon_bounds"],
#     start_year=1940, time_chunk=1, exp_save_name="us_90th_1_year_relative", mean_only=1, months=[12,1,2], percentile=90, subtract_mean=1
# )

# # 10-year means with 90th percentile cutoff and subtract_mean=1
# year10_90th_sub_means_ace, year10_90th_sub_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict1_run["lat_bounds"], lon_bounds=dict1_run["lon_bounds"],
#     start_year=1940, time_chunk=10, exp_save_name="us_90th_10_year_relative", mean_only=1, months=[12,1,2], percentile=90, subtract_mean=1
# )

# 1-year means with -10th percentile cutoff and subtract_mean=1

print("running here!")

# year1_neg10th_sub_means_ace, year1_neg10th_sub_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict2_run["lat_bounds"], lon_bounds=dict2_run["lon_bounds"],
#     start_year=1980, time_chunk=5, exp_save_name="us_10th_5_year_relative_trend", mean_only=1, months=[12,1,2], percentile=-10, subtract_mean=1, trend=1 
# )

# # 10-year means with -10th percentile cutoff and subtract_mean=1
# year10_neg10th_sub_means_ace, year10_neg10th_sub_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict2_run["lat_bounds"], lon_bounds=dict2_run["lon_bounds"],
#     start_year=1980, time_chunk=10, exp_save_name="us_10th_10_year_relative_trend", mean_only=1, months=[12,1,2], percentile=-10, subtract_mean=1, trend=1
# )
# # 1-year means with -10th percentile cutoff and subtract_mean=1
# year1_neg10th_sub_means_ace, year1_neg10th_sub_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict1_run["lat_bounds"], lon_bounds=dict1_run["lon_bounds"],
#     start_year=1980, time_chunk=5, exp_save_name="arct_10th_5_year_relative_trend", mean_only=1, months=[12,1,2], percentile=-10, subtract_mean=1, trend=1 
# )

# # 10-year means with -10th percentile cutoff and subtract_mean=1
# year10_neg10th_sub_means_ace, year10_neg10th_sub_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict1_run["lat_bounds"], lon_bounds=dict1_run["lon_bounds"],
#     start_year=1980, time_chunk=10, exp_save_name="arct_10th_10_year_relative_trend", mean_only=1, months=[12,1,2], percentile=-10, subtract_mean=1, trend=1
# )

# # 1-year means with -10th percentile cutoff and subtract_mean=1
# year1_neg10th_sub_means_ace, year1_neg10th_sub_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict4_run["lat_bounds"], lon_bounds=dict4_run["lon_bounds"],
#     start_year=1980, time_chunk=5, exp_save_name="russ_10th_5_year_relative_trend", mean_only=1, months=[12,1,2], percentile=-10, subtract_mean=1, trend=1 
# )

# # 10-year means with -10th percentile cutoff and subtract_mean=1
# year10_neg10th_sub_means_ace, year10_neg10th_sub_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict4_run["lat_bounds"], lon_bounds=dict4_run["lon_bounds"],
#     start_year=1980, time_chunk=10, exp_save_name="russ_10th_10_year_relative_trend", mean_only=1, months=[12,1,2], percentile=-10, subtract_mean=1, trend=1
# )


##from 1940

# # 1-year means with -10th percentile cutoff and subtract_mean=1
# russ_10th_ace, russ_10th_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict4_run["lat_bounds"], lon_bounds=dict4_run["lon_bounds"],
#     start_year=1940, time_chunk=10, exp_save_name="full_range_russ_10th_10_year_temps", mean_only=1, months=[12,1,2], percentile=-10, subtract_mean=0, trend=0
# )

# us_10th_ace, us_10th_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict2_run["lat_bounds"], lon_bounds=dict2_run["lon_bounds"],
#     start_year=1940, time_chunk=10, exp_save_name="full_range_us_10th_10_year_temps", mean_only=1, months=[12,1,2], percentile=-10, subtract_mean=0, trend=0
# )


# # 1-year means with -10th percentile cutoff and subtract_mean=1
# arct_10th_ace, arct_10th_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=dict1_run["lat_bounds"], lon_bounds=dict1_run["lon_bounds"],
#     start_year=1940, time_chunk=10, exp_save_name="full_range_arct_10th_10_year_temps", mean_only=1, months=[12,1,2], percentile=-10, subtract_mean=0, trend=0
# )

###means

# 1-year means with -10th percentile cutoff and subtract_mean=1

# year10_means_ace, year10_means_era = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=None, lon_bounds=None,
#     start_year=1940, time_chunk=10, exp_save_name="full_range_global_all_10_year", mean_only=1, months=[12,1,2], percentile=0
# )

# year10_means_ace_10th, year10_means_era_10th = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=None, lon_bounds=None,
#     start_year=1940, time_chunk=10, exp_save_name="full_range_global_10th_10_year", mean_only=1, months=[12,1,2], percentile=-10
# )

# year10_means_ace_90th, year10_means_era_90th = preprocess_and_run_experiment(
#     fp_ACE, fp_ERA5, mask_type="ocean", lat_bounds=None, lon_bounds=None,
#     start_year=1940, time_chunk=10, exp_save_name="full_range_global_90th_10_year", mean_only=1, months=[12,1,2], percentile=90
# )

# # russ_ace_10th_to_mean_diff = {key: np.nanmean(russ_10th_ace[key],axis=-1, keepdims=True) - np.nanmean(russ_mean_ace[key],axis=-1, keepdims=True) for key in russ_10th_ace.keys()}
# # russ_era_10th_to_mean_diff = {key: np.nanmean(russ_10th_era[key], keepdims=True) - np.nanmean(russ_mean_era[key], keepdims=True) for key in russ_10th_era.keys()}
# # us_ace_10th_to_mean_diff = {key: np.nanmean(us_10th_ace[key],axis=-1, keepdims=True) - np.nanmean(us_mean_ace[key],axis=-1, keepdims=True) for key in us_10th_ace.keys()}
# # us_era_10th_to_mean_diff = {key: np.nanmean(us_10th_era[key], keepdims=True) - np.nanmean(us_mean_era[key], keepdims=True) for key in us_10th_era.keys()}
# # arct_ace_10th_to_mean_diff = {key: np.nanmean(arct_10th_ace[key],axis=-1, keepdims=True) - np.nanmean(arct_mean_ace[key],axis=-1, keepdims=True) for key in arct_10th_ace.keys()}
# # arct_era_10th_to_mean_diff = {key: np.nanmean(arct_10th_era[key], keepdims=True) - np.nanmean(arct_mean_era[key], keepdims=True) for key in arct_10th_era.keys()}


# global_mean_diff = {key: np.nanmean(year10_means_ace[key],axis=-1, keepdims=True) - np.nanmean(year10_means_era[key],axis=-1, keepdims=True) for key in year10_means_ace.keys()}
# global_10th_diff = {key: np.nanmean(year10_means_ace_10th[key],axis=-1, keepdims=True) - np.nanmean(year10_means_era_10th[key],axis=-1, keepdims=True) for key in year10_means_era_10th.keys()}
# global_90th_diff = {key: np.nanmean(year10_means_ace_90th[key],axis=-1, keepdims=True) - np.nanmean(year10_means_era_90th[key],axis=-1, keepdims=True) for key in year10_means_era_90th.keys()}

# plot_single_time_series(global_mean_diff, start_year=1940, time_chunk=10, title="Global Mean Temperature Difference (ACE- ERA5)", exp_name="global_mean_temp_diff")
# plot_single_time_series(global_10th_diff, start_year=1940, time_chunk=10, title="Global 10th Percentile Temperature Difference (ACE- ERA5)", exp_name="global_10th_temp_diff")
# print("global_90th_diff", global_90th_diff)
# plot_single_time_series(global_90th_diff, start_year=1940, time_chunk=10, title="Global 90th Percentile Temperature Difference (ACE- ERA5)", exp_name="global_90th_temp_diff")

# # plot_timechunk_means(russ_ace_10th_to_mean_diff, russ_era_10th_to_mean_diff, 1940, 10,"Russia 10th - Mean" , "full_range_russ_10th_vs_mean", plot_derivative=0, center_at_zero=0)
# # plot_timechunk_means(us_ace_10th_to_mean_diff, us_era_10th_to_mean_diff, 1940, 10,"US 10th - Mean" , "full_range_us_10th_vs_mean", plot_derivative=0, center_at_zero=0)
# # plot_timechunk_means(arct_ace_10th_to_mean_diff, arct_era_10th_to_mean_diff, 1940, 10,"Arctic 10th - Mean" , "full_range_arct_10th_vs_mean", plot_derivative=0, center_at_zero=0)


