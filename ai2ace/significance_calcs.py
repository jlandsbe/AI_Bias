#pvalues for significance
import xarray as xr
import numpy as np
import os

from matplotlib import font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, to_rgb
from scipy.stats import norm


# Path to your font files
font_path_regular = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-VariableFont_wght.ttf"
font_path_italic = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-Italic-VariableFont_wght.ttf"

# Register fonts
fm.fontManager.addfont(font_path_regular)
fm.fontManager.addfont(font_path_italic)

# Set global font family to Red Hat Display
mpl.rcParams['font.family'] = 'Red Hat Display'

###ACE2 p values###
era5_aligned = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ERA5_processed.nc")
ace2_aligned = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ACE2_processed.nc")




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

import xarray as xr
import numpy as np

def avg_gap_1d(arr, large_value=np.inf):
    """Return average index gap between True values in a 1D boolean array."""
    idx = np.where(arr)[0]

    if len(idx) < 2:
        return large_value

    return np.diff(idx).mean()

def avg_gap_3d(mask_da):
    return xr.apply_ufunc(
        avg_gap_1d,
        mask_da,
        input_core_dims=[["time"]],   # operate along time only
        output_core_dims=[[]],        # scalar output per (lat, lon)
        vectorize=True,               # apply separately to each grid cell
        dask="parallelized",          # works with or without Dask
        output_dtypes=[float],        
    )


def effective_sample_size(da, percentile=0, gap_days=7, coarsen=False):
    """
    Compute effective sample size for extreme events in da (e.g., 90th percentile),
    computing lag-1 autocorrelation within each season.

    Parameters
    ----------
    da : xr.DataArray
        Time series with dimensions (time, lat, lon)
    percentile : int
        Percentile threshold for extremes (e.g., 90)
    gap_days : int
        Threshold to define separate seasons for splitting

    Returns
    -------
    n_eff_total : xarray.DataArray
        Effective sample size for extreme events at each (lat, lon)
    """
    # 1. Identify extreme events
    extreme_mask = percentile_filtering_by_season(da, percentile=percentile)

    # 2. Identify season breaks based on time gaps
    dt = (da.time.diff('time') / np.timedelta64(1, 'D')).values
    breaks = np.where(dt >= gap_days)[0] + 1
    season_starts = np.r_[0, breaks]
    season_ends = np.r_[breaks, len(da.time)]

    n_eff_list = []
    print("dt array:")
    print(np.shape(da))
    # 3. Loop over seasons
    for s0, s1 in zip(season_starts, season_ends):
        sub = da.isel(time=slice(s0, s1))
        sub_mask = extreme_mask.isel(time=slice(s0, s1))
        average_gap_days = avg_gap_3d(sub_mask.notnull())
        print("sub mask shape")
        print(np.shape(sub_mask))

        # 4. Compute lag-1 correlation for this season
        anom = sub - sub.mean("time")
        r1 = xr.corr(anom, anom.shift(time=1), dim="time")
        r1 = xr.where(np.abs(r1) >= 0.999, 0.999 * np.sign(r1), r1)

        # 5. Broadcast time to match mask shape
        sub_times_3d, sub_mask_3d = xr.broadcast(sub.time, sub_mask)
        extreme_times = sub_times_3d.where(sub_mask_3d.notnull(), drop=True)

        # Skip if too few extreme events
        if extreme_times.size < 2:
            n_eff_list.append(extreme_times.notnull().sum(dim="time"))
            continue

        # 6. Compute gaps between consecutive extremes (in days)
        r_effective = r1 ** average_gap_days
        print("shape of r_effective and average gap days")
        print(np.shape(r_effective))
        print(np.shape(average_gap_days))
        # 7. Compute n_eff for this season
        n_extreme = extreme_times.notnull().sum(dim="time")
        n_eff_season = n_extreme * (1 - r_effective) / (1 + r_effective)
        n_eff_list.append(n_eff_season)

    # 8. Sum across seasons
    n_eff_total = xr.concat(n_eff_list, dim="season").sum("season")

    #if there is a sample coordinate, sum along that too
    if 'sample' in n_eff_total.dims:
        n_eff_total = n_eff_total.sum('sample')
    print("initial shape of n")
    print(np.shape(n_eff_total))

    if coarsen:
        n_eff_total = n_eff_total.coarsen(latitude=4, longitude=4, boundary="trim").sum(skipna=True)
    print("final shape of n")
    print(np.shape(n_eff_total))
    return n_eff_total
##subset the data based on what you want (season or percentile)
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
def subset_ace_data_p_vals(ace_data, era5_data, months=None, percentile=None, start_year=None, end_year=None,var_path=""):
    # Count valid points along time (exclude NaNs)
    print(ace_data)
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

    ace_subset = thresholded_mean(ace_data, 'surface_temperature', months=months, percentile=percentile, start_year=start_year, end_year=end_year)
    era5_subset = thresholded_mean(era5_data, 'surface_temperature', months=months, percentile=percentile, start_year=start_year, end_year=end_year)
    n = effective_sample_size(ace_subset, gap_days=7)

    # Mean over time
    mean_data = ace_subset.mean(dim=['time','sample']) - era5_subset.mean(dim=['time'])
    variance = np.load(var_path)
    # Standard error
    se = np.sqrt(variance / n.values)

    # z-score
    z_scores = mean_data.values / se
    print(z_scores)
    # One-sided p-values
    p_values = (1 - norm.cdf(np.abs(z_scores)))

    save_dir = '/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data'
    os.makedirs(save_dir, exist_ok=True)  # creates the folder if it doesn't exist

    np.save(os.path.join(save_dir, f'ace2_p_values_{month_name}_{np.abs(percentile)}.npy'), p_values)
#save the pvalues as numpy array
# subset_ace_data_p_vals(ace2_aligned, era5_aligned, start_year = 1996, end_year=2010, months=[12,1,2], percentile=0, var_path="/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ACE_variance_DJF_0_percentile.npy")
# subset_ace_data_p_vals(ace2_aligned, era5_aligned, start_year = 1996, end_year=2010, months=[6,7,8], percentile=0, var_path="/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ACE_variance_JJA_0_percentile.npy")
subset_ace_data_p_vals(ace2_aligned, era5_aligned, start_year = 1996, end_year=2010, months=[12,1,2], percentile=-10, var_path="/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ACE_variance_DJF_10_percentile.npy")
subset_ace_data_p_vals(ace2_aligned, era5_aligned, start_year = 1996, end_year=2010, months=[12,1,2], percentile=90, var_path="/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ACE_variance_DJF_90_percentile.npy")



# ### Wx Model p values###
fourcast_9day = xr.open_dataset("/barnes-engr-scratch1/DATA/Fourcastv2/Fourcast_V2_winter_2020_2025.nc")
fourcast_2day = xr.open_dataset("/barnes-engr-scratch1/DATA/Fourcastv2/Fourcast_V2_winter_2020_2025_2day.nc")
pangu_2day = xr.open_dataset("/barnes-engr-scratch1/DATA/Pangu/PANGU_V1_combined_2day.nc")
pangu_9day = xr.open_dataset("/barnes-engr-scratch1/DATA/Pangu/PANGU_V1_combined_9day.nc")
era5_2020_2025 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/6-hourly_0.25deg_data/winter2020-2025/combined_file.nc")

def subset_wx_data_p_vals(wx_data, era5_data, var_path="", percentile=None, model_name = '', coarsen=False, mask=True):
    keep_land_mask = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/remove_poles_mask.nc").__xarray_dataarray_variable__

    wx_subset = percentile_filtering_by_season(wx_data, percentile)
    if mask:
        wx_subset = wx_subset * keep_land_mask
    n = effective_sample_size(wx_data, gap_days=7, coarsen=coarsen)
    # Mean over time
    mean_data = wx_subset.mean(dim='time') - era5_data.mean(dim='time')
    if coarsen:
        mean_data = mean_data.coarsen(latitude=4, longitude=4, boundary='trim').mean()
    variance = np.load(var_path)
    #coarsen numpy array
    if coarsen:
        lat_trim = variance.shape[0] - (variance.shape[0] % 4)
        lon_trim = variance.shape[1] - (variance.shape[1] % 4)

        variance = variance[:lat_trim, :lon_trim]
        variance = variance.reshape((lat_trim//4, 4, lon_trim//4, 4)).mean(axis=(1,3))
    print("shape of n and variance")
    print(np.shape(n))
    print(np.shape(variance))
    # Standard error
    se = np.sqrt(variance / n.values)

    # z-score
    z_scores = mean_data.values / se
    print(z_scores)
    # Two-sided p-values
    p_values = (1 - norm.cdf(np.abs(z_scores)))   

    save_dir = '/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data'
    os.makedirs(save_dir, exist_ok=True)  # creates the folder if it doesn't exist

    np.save(os.path.join(save_dir, f'{model_name}_p_values_{np.abs(percentile)}.npy'), p_values)

subset_wx_data_p_vals(fourcast_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_0_percentile.npy", percentile=0, model_name='fourcast_9day', coarsen=True)
subset_wx_data_p_vals(pangu_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_0_percentile.npy", percentile=0, model_name='pangu_9day', coarsen=True)
subset_wx_data_p_vals(pangu_2day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_0_percentile.npy", percentile=0, model_name='pangu_2day', coarsen=True)
subset_wx_data_p_vals(fourcast_2day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_0_percentile.npy", percentile=0, model_name='fourcast_2day', coarsen=True)

subset_wx_data_p_vals(fourcast_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_90_percentile.npy", percentile=90, model_name='fourcast_9day', coarsen=True)
subset_wx_data_p_vals(fourcast_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_10_percentile.npy", percentile=-10, model_name='fourcast_9day', coarsen=True)
subset_wx_data_p_vals(fourcast_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_95_percentile.npy", percentile=95, model_name='fourcast_9day', coarsen=True)
subset_wx_data_p_vals(fourcast_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_5_percentile.npy", percentile=-5, model_name='fourcast_9day', coarsen=True)
subset_wx_data_p_vals(fourcast_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_80_percentile.npy", percentile=80, model_name='fourcast_9day', coarsen=True)
subset_wx_data_p_vals(fourcast_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_20_percentile.npy", percentile=-20, model_name='fourcast_9day', coarsen=True)
subset_wx_data_p_vals(pangu_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_90_percentile.npy", percentile=90, model_name='pangu_9day', coarsen=True)
subset_wx_data_p_vals(pangu_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_10_percentile.npy", percentile=-10, model_name='pangu_9day', coarsen=True)
subset_wx_data_p_vals(pangu_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_95_percentile.npy", percentile=95, model_name='pangu_9day', coarsen=True)
subset_wx_data_p_vals(pangu_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_5_percentile.npy", percentile=-5, model_name='pangu_9day', coarsen=True)
subset_wx_data_p_vals(pangu_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_80_percentile.npy", percentile=80, model_name='pangu_9day', coarsen=True)
subset_wx_data_p_vals(pangu_9day.t2m, era5_2020_2025.t2m, "/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ERA5_variance_DJF_20_percentile.npy", percentile=-20, model_name='pangu_9day', coarsen=True)




# #can so a simple variance calc since only have 
