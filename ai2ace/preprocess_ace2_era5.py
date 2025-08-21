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
import os
import xesmf as xe

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
        dataset.to_netcdf(output_path, mode='w')
        print(f"Dataset saved to {output_path}")
    except Exception as e:
        print(f"Error saving dataset: {e}")

#data processing functions
def coarsen_time_average(filepath, time_chunk=100, coarsen_step=4, boundary='trim'):
    """
    Open an xarray dataset with chunking, coarsen it along the time dimension,
    and return the averaged dataset.

    Parameters:
    - filepath (str): Path to the NetCDF file.
    - time_chunk (int): Size of Dask chunk along time.
    - coarsen_step (int): Number of time steps to average over.
    - boundary (str): Coarsen boundary behavior ('trim', 'pad', or 'exact').

    Returns:
    - xarray.Dataset: Coarsened and averaged dataset.
    """
    ds = xr.open_dataset(filepath, chunks={'time': time_chunk})
    ds_avg = ds.coarsen(time=coarsen_step, boundary=boundary).mean()
    return ds_avg.load() 
import xarray as xr
import xesmf as xe
import os

# Step 1: Coarsen ACE2 and save
# Filepaths for individual prediction files
filepaths = [
    "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/output/1940_01_01_autoregressive_predictions.nc",
    "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/output/1940_01_02_autoregressive_predictions.nc",
    "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/output/1940_01_03_autoregressive_predictions.nc",
    "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/output/1940_01_04_autoregressive_predictions.nc",
    "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/output/1940_01_05_autoregressive_predictions.nc"
]

# Coarsen each dataset individually
coarsened_datasets = []
for filepath in filepaths:
    coarsened_ds = coarsen_time_average(filepath, time_chunk=100, coarsen_step=4, boundary='trim')
    coarsened_datasets.append(coarsened_ds)

# Combine coarsened datasets along the 'sample' dimension
combined_predictions = xr.concat(coarsened_datasets, dim='sample')

# Save the combined dataset to a single NetCDF file
combined_predictions.to_netcdf('/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/output/combined_predictions.nc')

# Set the filepath for the combined dataset
fp_ace = '/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/output/combined_predictions.nc'
ACE2_data = load_file(fp_ace)

# Process ERA5
folder_path = "/barnes-engr-scratch1/DATA/ERA5/daily_temp/skin_temperature"
start_year = 1940
end_year = 2022

files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith("temperature.nc")])
print(len(files), "files found")

# Combine along the time dimension
ERA5_data = xr.open_mfdataset(files, combine='by_coords',)

#Rename valid_time and skt
ERA5_data = ERA5_data.rename({'valid_time': 'time', 'skt': 'surface_temperature'})


ERA5_data = ERA5_data.sel(time=slice(f"{start_year}-01-02", f"{end_year}-12-31"))
print(len(ERA5_data.time), "time steps found")

ACE2_data = ACE2_data.assign_coords(time=ERA5_data.time)
vars_to_drop = ['valid_time', 'number', 'init_time']
ACE2_data = ACE2_data.drop_vars([v for v in vars_to_drop if v in ACE2_data])

#ACE2_data = ACE2_data.assign_coords(time = ERA5_data.time)






# Regrid
# Define the target grid using ACE2's lat/lon
target_grid = xr.Dataset({'lat': ACE2_data.lat, 'lon': ACE2_data.lon})

# Create the regridder with 'bilinear' method
regridder = xe.Regridder(ERA5_data, target_grid, method='bilinear', periodic=True)

# Regrid the ERA5 data
ERA5_regridded = regridder(ERA5_data)  # FIXED: previously used `ds1`, now correctly uses `ERA5_data`

ERA5_processed, ACE2_processed = xr.align(ERA5_regridded, ACE2_data, join='inner')
# Save the regridded ERA5 data
save_to_netcdf(ERA5_processed, '/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ERA5_processed.nc')

save_to_netcdf(ACE2_processed, '/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ACE2_processed.nc')



