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




era5_2020_2025 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/6-hourly_0.25deg_data/winter2020-2025/combined_file.nc")
era5_2015_2020 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2015-2020.nc").rename({'valid_time': 'time'})
era5_2010_2015 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2010-2015.nc").rename({'valid_time': 'time'})
era5_2005_2010 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2005-2010.nc").rename({'valid_time': 'time'})
era5_2000_2005 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2000-2005.nc").rename({'valid_time': 'time'})
era5_1995_2000 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1995-2000.nc").rename({'valid_time': 'time'})
era5_1990_1995 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1990-1995.nc").rename({'valid_time': 'time'})
era5_1985_1990 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1985-1990.nc").rename({'valid_time': 'time'})
era5_1980_1985 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1980-1985.nc").rename({'valid_time': 'time'})
keep_land_mask = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/remove_poles_mask.nc").__xarray_dataarray_variable__

mode = "pangu"

#take asubset to only include 2017 or earlier
era5_2015_2020_subsetted = era5_2015_2020.sel(time=slice("2015-01-01", "2020-12-31"))


def percentile_filtering(data, percentile=-10):
    """
    Apply a percentile filter to the data.
    """
    threshold = data.quantile(q=np.abs(percentile) / 100.0, dim='time', skipna=True)
    if percentile == 0:
        return data, threshold
    if percentile < 0:
        return xr.where(data < threshold, data, np.nan), threshold
    else:
        return xr.where(data > threshold, data, np.nan), threshold


def quantile_counter(da_list, quantiles, greater = 0, mask=None):
    array_out = np.zeros(np.shape(da_list[0].t2m.values)[1:])
    lats = da_list[0].latitude.values
    lons = da_list[0].longitude.values
    total_time = 0
    for da in da_list:
        total_time += len(da.time)
        if greater == 0:
            array_out += np.sum((da.t2m.values < quantiles.t2m.values).astype(int),axis=0)
        else:
            array_out += np.sum((da.t2m.values > quantiles.t2m.values).astype(int),axis=0)
    array_out = array_out / total_time
    if mask is not None:
        array_out = (array_out * mask.values)
    return xr.DataArray(
        array_out,
        coords={"latitude": lats, "longitude": lons},
        dims=("latitude", "longitude"),
        name="num_quantiles"
    )

if mode == "pangu":
    da_list = [
        era5_2015_2020_subsetted,
        era5_2010_2015,
        era5_2005_2010,
        era5_2000_2005,
        era5_1995_2000,
        era5_1990_1995,
        era5_1985_1990,
        era5_1980_1985]

else:
    da_list = [
        era5_2010_2015,
        era5_2005_2010,
        era5_2000_2005,
        era5_1995_2000,
        era5_1990_1995,
        era5_1985_1990,
        era5_1980_1985]
import os
import xarray as xr

percentiles = list(range(-5, -96, -5)) + list(range(5, 100, 5))
quantile_files = {
    p: f"/home/jlandsbe/ai_weather_to_climate_ats780A8/ERA5_2020_2025_{abs(p)}th_quantile.nc"
    for p in percentiles
}

quantile_datasets = {}

# Loop through each percentile
for p in percentiles:
    path = quantile_files[p]
    if not os.path.exists(path):
        print(f"{p}th percentile quantile file does not exist, calculating...")
        _, quantile = percentile_filtering(era5_2020_2025, p)
        quantile.to_netcdf(path)
        quantile_datasets[p] = quantile
    else:
        print(f"{p}th percentile quantile file already exists, skipping calculation.")
        quantile_datasets[p] = xr.open_dataset(path)

# Calculate extreme frequency counts using quantile_counter
as_extreme = {}

for p in range(-5, -96, -5):
    as_extreme[p] = quantile_counter(da_list, quantile_datasets[p], greater=0, mask=keep_land_mask)
    as_extreme[p].to_netcdf(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/ERA5_as_extreme_{abs(p)}th_below_{mode}.nc")

for p in range(5, 100, 5):
    as_extreme[p] = quantile_counter(da_list, quantile_datasets[p], greater=1, mask=keep_land_mask)
    as_extreme[p].to_netcdf(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/ERA5_as_extreme_{abs(p)}th_above_{mode}.nc")
