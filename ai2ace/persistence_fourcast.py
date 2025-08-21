####Comparing ERA5 to FourCastNet v2.0

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib import font_manager as fm
import matplotlib as mpl
import os
import re
from datetime import datetime
import os
import re
from datetime import datetime
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb

import os
import re
from datetime import datetime
import xarray as xr
import os
import re
import xarray as xr
from datetime import datetime
import pandas as pd






# Path to your font files
font_path_regular = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-VariableFont_wght.ttf"
font_path_italic = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-Italic-VariableFont_wght.ttf"

# Register fonts
fm.fontManager.addfont(font_path_regular)
fm.fontManager.addfont(font_path_italic)

# Set global font family to Red Hat Display
mpl.rcParams['font.family'] = 'Red Hat Display'


fourcast_9day = xr.open_dataset("/barnes-engr-scratch1/DATA/Fourcastv2/Fourcast_V2_winter_2020_2025.nc")
fourcast_2day = xr.open_dataset("/barnes-engr-scratch1/DATA/Fourcastv2/Fourcast_V2_winter_2020_2025_2day.nc")
era5_2015_2020_climatology = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2015-2020.nc").rename({'valid_time': 'time'})
era5_2020_2025_persistence = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/6-hourly_0.25deg_data/winter2020-2025/persistence_dates_sorted.nc")
era5_2020_2025_truth = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/6-hourly_0.25deg_data/winter2020-2025/combined_file.nc")

def get_climatology(ds, variable_name, start_year=1979, end_year=2025):
    """
    Calculate a 366-day climatology (including Feb 29 interpolated).
    
    Parameters:
        ds (xarray.Dataset): Dataset with 'time' coordinate.
        variable_name (str): Variable to compute climatology for.
        start_year (int): Start year.
        end_year (int): End year.
    
    Returns:
        xarray.DataArray: Climatology with dimension 'dayofyear' (1–366).
    """
    # Subset the data
    ds_subset = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # Remove Feb 29 to normalize calendar for mean computation
    is_feb29 = (ds_subset['time.month'] == 2) & (ds_subset['time.day'] == 29)
    ds_no_feb29 = ds_subset.sel(time=~is_feb29)

    # Compute climatology for dayofyear 1–365
    clim_365 = ds_no_feb29[variable_name].groupby('time.dayofyear').mean(dim='time')

    # Interpolate Feb 29 as average of Feb 28 (59) and Mar 1 (60)
    feb_28 = clim_365.sel(dayofyear=59)
    feb_29 = feb_28
    feb_29 = feb_29.assign_coords(dayofyear=60)

    # Insert Feb 29 back
    clim_366 = xr.concat([
        clim_365.sel(dayofyear=slice(1, 59)),
        feb_29.expand_dims("dayofyear"),
        clim_365.sel(dayofyear=slice(60, 365))
    ], dim='dayofyear')

    return clim_366

import xarray as xr

def apply_climatology_to_times(ds_target, clim_366):
    """
    Expand a day-of-year climatology to match the time dimension of a target dataset.

    Parameters:
        ds_target (xarray.Dataset or DataArray): Target with 'time' coordinate.
        clim_366 (xarray.DataArray): Climatology with 'dayofyear' from 1 to 366
                                     (Feb 29 interpolated at dayofyear=60).

    Returns:
        xarray.DataArray: Climatology expanded to match ds_target.time.
    """
    # 1. Identify Feb 29 in the target
    is_feb29 = (ds_target.time.dt.month == 2) & (ds_target.time.dt.day == 29)
    ds_target_no_feb29 = ds_target.sel(time=~is_feb29)
    
    # 2. Get dayofyear for dates excluding Feb 29 (1–365 only)
    doy = ds_target_no_feb29.time.dt.dayofyear
        # Check what's actually in the climatology
    print("Climatology dayofyear values:", clim_366.dayofyear.values)

    # Get target dayofyear values
    doy = ds_target.time.dt.dayofyear

    # Check which dayofyear values aren't present in the climatology
    invalid_doys = np.setdiff1d(doy.values, clim_366.dayofyear.values)

    print("Invalid dayofyear values (not in climatology):", invalid_doys)
    # 3. Select matching climatology
    clim_matched = clim_366.sel(dayofyear=doy)
    clim_matched = clim_matched.assign_coords(time=ds_target_no_feb29.time)
    
    # 4. Interpolate Feb 29 as average of Feb 28 (59) and Mar 1 (60)
    feb_28 = clim_366.sel(dayofyear=59)
    mar_1 = clim_366.sel(dayofyear=60)
    feb_29_interp = 0.5 * (feb_28 + mar_1)

    # 5. Broadcast interpolated Feb 29 across all Feb 29 dates
    feb29_times = ds_target.time.where(is_feb29, drop=True)
    feb29_clim = feb_29_interp.expand_dims(time=feb29_times)
    feb29_clim = feb29_clim.assign_coords(time=feb29_times)
    
    # 6. Combine both parts and sort by time
    full_climatology = xr.concat([clim_matched, feb29_clim], dim='time')
    full_climatology = full_climatology.sortby('time')
    
    return full_climatology





def shift_djf_seasons_forward(data, variable_name, shift_amount = 9):
    """
    Shifts each DJF season (Nov to Feb) forward by 1 day.
    November 30 becomes December 1, Feb 27/28 becomes Feb 28/29.
    The last day of February is dropped from each season.
    
    Parameters:
        data (xarray.DataArray or xarray.Dataset): Input data with time coordinate from Nov to Feb across years.

    Returns:
        xarray.DataArray or xarray.Dataset: Shifted data with seasons aligned forward by 1 day.
    """
    # Ensure datetime index
    data = data[variable_name]
    time = data['time'].to_index()
    season_years = time.year + (time.month == 12).astype(int)

    # Store shifted data chunks
    shifted_chunks = []

    for year in sorted(set(season_years)):
        # Define seasonal bounds
        start = pd.Timestamp(f"{year - 1}-11-01")
        end = pd.Timestamp(f"{year}-03-01")  # exclusive March 1
        
        # Select chunk
        chunk = data.sel(time=slice(start, end - pd.Timedelta(days=1)))
        
        if chunk.time.size == 0:
            continue  # skip if no data for this season

        # Shift forward in time by 1 day
        chunk_shifted = chunk.shift(time=shift_amount)

        chunk_shifted = chunk_shifted.sel(time=chunk_shifted['time.month'].isin([12, 1, 2]))

        shifted_chunks.append(chunk_shifted)

    # Concatenate all chunks back
    result = xr.concat(shifted_chunks, dim="time")
    return result

def is_year_leap(year):
    """
    Check if a year is a leap year.
    
    Parameters:
        year (int): Year to check.
    
    Returns:
        bool: True if leap year, False otherwise.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def expand_clim(dataset, years=range(2020,2026), array_shape = (451,721,1440)):
    array_out = np.full(array_shape, np.nan)
    day_tracker = 0
    for idx, year in enumerate(years):
        if idx==0:
            #first year, just copy climatology from december
            filtered = dataset.sel(dayofyear=slice(335, 365))
            filtered_vals = filtered.values
            num_days = len(filtered_vals)
            array_out[day_tracker:day_tracker+num_days, :, :] = filtered_vals
            day_tracker += num_days
            print(day_tracker)
        elif idx == len(years) - 1:
            print(year)
            print(day_tracker)
            #last year, just copy climatology from january and february
            filtered = dataset.where(dataset['dayofyear'] <= 60, drop=True)
            if not is_year_leap(year):
                # remove Feb 29 if not leap year
                filtered_vals = filtered.where(filtered['dayofyear'] != 60, drop=True).values
            else:
                filtered_vals = filtered.values
            num_days = len(filtered_vals)
            print(num_days)
            print(day_tracker)
            array_out[day_tracker:day_tracker+num_days, :, :] = filtered_vals 
            day_tracker += num_days              
        else:
            if is_year_leap(year):
                print("in leap year")
                print(day_tracker)
                #leap year, copy climatology from december and february
                filtered_vals = dataset.values
                num_days = len(filtered_vals)
                print(num_days)
                array_out[day_tracker:day_tracker+num_days, :, :] = filtered_vals
                day_tracker += num_days
            else:
                print(day_tracker)
                print(dataset.dayofyear.values)
                filtered = dataset.where(dataset['dayofyear'] != 60, drop=True)
                print(filtered.dayofyear.values)
                filtered_vals = filtered.values
                num_days = len(filtered_vals)
                print(num_days)
                array_out[day_tracker:day_tracker+num_days, :, :] = filtered_vals
                day_tracker += num_days
    return array_out

clim366 = get_climatology(era5_2015_2020_climatology, 't2m',)
print(clim366)
print(clim366.dayofyear.values)
persistence_forecast_test = shift_djf_seasons_forward(era5_2020_2025_persistence, 't2m', shift_amount=1)
print(persistence_forecast_test)

#extract numpy arrays
P = persistence_forecast_test.values
C = expand_clim(clim366, years=range(2020,2026), array_shape=P.shape)
T = era5_2020_2025_truth.t2m.values

##save P, C, T to disk using pickle
# import pickle
# with open("/home/jlandsbe/ai_weather_to_climate_ats780A8/persistence_forecast_test.pkl", "wb") as f:
#     pickle.dump(P, f)
# with open("/home/jlandsbe/ai_weather_to_climate_ats780A8/clim366.pkl", "wb") as f:
#     pickle.dump(C, f)
# with open("/home/jlandsbe/ai_weather_to_climate_ats780A8/era5_2020_2025_truth.pkl", "wb") as f:
#     pickle.dump(T, f)

T_minus_C = T - C
P_minus_C = P - C


##fit
# Compute numerator and denominator
numerator = np.sum(T_minus_C * P_minus_C, axis=0)      # shape (x, y)
denominator = np.sum(P_minus_C * P_minus_C, axis=0)        # shape (x, y)

# Avoid division by zero
denominator = np.where(denominator == 0, np.nan, denominator)

# Fit coefficient a (shape x, y)
a = numerator / denominator

predictions = a * (P - C) + C
print(a)
era5_2020_2025_truth['predictions'] = (('time', 'latitude', 'longitude'), predictions)
era5_2020_2025_truth.to_netcdf("/home/jlandsbe/ai_weather_to_climate_ats780A8/era5_2020_2025_predictions.nc")
# climatology = apply_climatology_to_times(persistence_forecast, clim366)
# print(climatology)
##save both of these
#persistence_forecast_test.to_netcdf("/home/jlandsbe/ERA5_2020_2025_persistence.nc")
#climatology.to_netcdf("/home/jlandsbe/ERA5_2015_2020_climatology.nc")


