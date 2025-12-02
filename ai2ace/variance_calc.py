##variance calculations
#pvalues for significance
import xarray as xr
import numpy as np
import os

import numpy as np
import xarray as xr

from scipy.stats import norm

#for every season/year in ERA5, subtract the mean at every grid point
def assign_fast_season_year(data):
    months = data['time'].dt.month
    years  = data['time'].dt.year

    season_offset = np.zeros(len(data['time']), dtype=int)
    season_offset[np.isin(months.data, [12, 1, 2])] = 10000   # DJF
    season_offset[np.isin(months.data, [3, 4, 5])]   = 20000   # MAM
    season_offset[np.isin(months.data, [6, 7, 8])]   = 30000   # JJA
    season_offset[np.isin(months.data, [9, 10, 11])] = 40000   # SON

    season_year = years.values + (months == 12).values

    season_year_fast = np.array([y + s for y, s in zip(season_year, season_offset)], dtype=int)

    # Assign as a coordinate along the time dimension
    return data.assign_coords(season_year=("time", season_year_fast))



def mean_subtract_by_season_year(data):
    """
    Subtract the mean of each season_year group from each timestep.
    Returns anomalies with the same shape as input.
    """
    data = assign_fast_season_year(data)
    grouped = data.groupby("season_year")
    return grouped - grouped.mean(dim="time")

def thresholded_mean(ds, var_name, percentile=0, months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], start_year = 1940, end_year = 2025):
    """
    Calculate the yearly mean of a variable with optional thresholding by percentile and filtering by months.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with dimensions including time, lat, lon (and optional sample).
    var_name : str
        Name of the variable in the dataset to process.
    percentile : float
        Percentile threshold for filtering.
        - 0 = no thresholding
        - positive = keep values above percentile
        - negative = keep values below abs(percentile)
    months : list of int
        List of months to include in the calculation (default is all months).
        
    Returns
    -------
    xarray.DataArray
        Resulting DataArray of shape (year, lat, lon)
    """
    # Extract the variable
    da = ds[var_name]
    
    # Filter by specified months and years
    da = da.where(da['time'].dt.month.isin(months), drop=True)
    years = np.arange(start_year, end_year)
    da = da.where(da['time'].dt.year.isin(years), drop=True)
    if percentile == 0:
    # No thresholding: simple mean over time
        data_subsetted = da
    else:
        # Compute threshold
        thresh = da.quantile(abs(percentile) / 100, dim='time')
        
        if percentile > 0:
            mask = da >= thresh
        else:
            mask = da <= thresh

        # Mask and compute mean over time
        masked = da.where(mask)
        data_subsetted = masked
    return data_subsetted


def percentile_filtering_by_season(data, percentile=-10):
    """
    Apply percentile filtering within each DJF season (Dec–Feb) separately.

    Parameters:
    -----------
    data : xr.DataArray
        Must have a 'time' dimension with datetime64.
    percentile : int
        Percentile threshold to apply. Negative = bottom X%, Positive = top X%.

    Returns:
    --------
    xr.DataArray
        Same shape as input, with values not in the specified percentile masked as NaN,
        but thresholding is done **within each DJF season separately**.
    """
    # Build a new season_year coordinate: DJF of 2022–2023 → season_year = 2023
    time = data['time'].to_index()
    season_years = time.year + (time.month == 12).astype(int)
    data = data.assign_coords(season_year=('time', season_years))

    # Group by season year
    def _season_thresh_filter(season_data):
        q = np.abs(percentile) / 100.0
        threshold = season_data.quantile(q=q, dim='time', skipna=True)

        if percentile == 0:
            return season_data
        elif percentile < 0:
            return xr.where(season_data < threshold, season_data, np.nan)
        else:
            return xr.where(season_data > threshold, season_data, np.nan)

    return data.groupby('season_year').map(_season_thresh_filter)




def compute_variance_ace(da, months = [1,2,3,4,5,6,7,8,9,10,11,12], start_year = 1940, end_year = 2025, percentile=0, var_name = 'surface_temperature', model_type=""):
    """
    Compute the variance of a DataArray after optional thresholding and filtering by months.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray with dimensions including time, lat, lon (and optional sample).
    months : list of int
        List of months to include in the calculation (default is all months).
    start_year : int
        Start year for filtering (default is 1940).
    end_year : int
        End year for filtering (default is 2025).
    percentile : float
        Percentile threshold for filtering.
        - 0 = no thresholding
        - positive = keep values above percentile
        - negative = keep values below abs(percentile)
    var_name : str
        Name of the variable in the dataset to process (default is 'surface_temperature').
    Returns
    -------
    xarray.DataArray
        Variance of the processed DataArray with dimensions (lat, lon).
    """
    if months == [12,1,2]:
        month_name = 'DJF'
    elif months == [3,4,5]:
        month_name = 'MAM'
    elif months == [6,7,8]:
        month_name = 'JJA'
    elif months == [9,10,11]:
        month_name = 'SON'
    elif months is None:
        month_name = 'Annual'
    else:
        month_name = 'Unknown'
    # Apply thresholded mean
    data_subsetted = thresholded_mean(da, var_name, percentile, months, start_year, end_year)
    era5_ace_deseasoned = mean_subtract_by_season_year(data_subsetted)
    # Compute variance over time
    variance = era5_ace_deseasoned.var(dim='time', skipna=True)
    #save variance dataarray
    save_dir = '/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data'
    np.save(os.path.join(save_dir,f"{model_type}_variance_{month_name}_{np.abs(percentile)}_percentile.npy"), variance.values)

    return variance



era5_aligned_ace = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ERA5_processed.nc")
era5_2020_2025 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/6-hourly_0.25deg_data/winter2020-2025/combined_file.nc")
era5_2015_2020 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2015-2020.nc").rename({'valid_time': 'time'})
era5_2010_2015 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2010-2015.nc").rename({'valid_time': 'time'})
era5_2005_2010 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2005-2010.nc").rename({'valid_time': 'time'})
era5_2000_2005 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2000-2005.nc").rename({'valid_time': 'time'})
era5_1995_2000 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1995-2000.nc").rename({'valid_time': 'time'})
era5_1990_1995 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1990-1995.nc").rename({'valid_time': 'time'})
era5_1985_1990 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1985-1990.nc").rename({'valid_time': 'time'})
era5_1980_1985 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1980-1985.nc").rename({'valid_time': 'time'})

#combine all era5 data
era5_list = [era5_1980_1985, era5_1985_1990, era5_1990_1995, era5_1995_2000, era5_2000_2005, era5_2005_2010, era5_2010_2015, era5_2015_2020, era5_2020_2025]
era5_combined_wx = xr.concat(era5_list, dim='time')


#do this for ace winter, summer, winter -10 percentile, and winter 90 percentile
compute_variance_ace(era5_aligned_ace, months = [12, 1, 2], percentile=0, var_name = 'surface_temperature', model_type="ACE")
compute_variance_ace(era5_aligned_ace, months = [6, 7, 8], percentile=0, var_name = 'surface_temperature', model_type="ACE")
compute_variance_ace(era5_aligned_ace, months = [12, 1, 2], percentile=-10, var_name = 'surface_temperature', model_type="ACE")
compute_variance_ace(era5_aligned_ace, months = [12, 1, 2], percentile=90, var_name = 'surface_temperature', model_type="ACE")

#do this for era5_combined_wx winter, winter -10, winter 90
compute_variance_ace(era5_combined_wx, months = [12, 1, 2], percentile=0, var_name = 't2m', model_type="ERA5")
compute_variance_ace(era5_combined_wx, months = [12, 1, 2], percentile=-10, var_name = 't2m', model_type="ERA5")
compute_variance_ace(era5_combined_wx, months = [12, 1, 2], percentile=90, var_name = 't2m', model_type="ERA5")
compute_variance_ace(era5_combined_wx, months = [12, 1, 2], percentile=-5, var_name = 't2m', model_type="ERA5")
compute_variance_ace(era5_combined_wx, months = [12, 1, 2], percentile=95, var_name = 't2m', model_type="ERA5")

compute_variance_ace(era5_combined_wx, months = [12, 1, 2], percentile=-20, var_name = 't2m', model_type="ERA5")
compute_variance_ace(era5_combined_wx, months = [12, 1, 2], percentile=80, var_name = 't2m', model_type="ERA5")





