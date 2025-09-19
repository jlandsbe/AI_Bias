###ERA5 climate change weather models

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


import xarray as xr
import matplotlib.pyplot as plt
# Set default font size (e.g., 14)
plt.rcParams.update({'font.size': 12})


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


font_path_regular = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-VariableFont_wght.ttf"
font_path_italic = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-Italic-VariableFont_wght.ttf"

# Register fonts
fm.fontManager.addfont(font_path_regular)
fm.fontManager.addfont(font_path_italic)

# Set global font family to Red Hat Display
mpl.rcParams['font.family'] = 'Red Hat Display'



keep_land_mask = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/remove_poles_mask.nc").__xarray_dataarray_variable__

era5_2020_2025 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/6-hourly_0.25deg_data/winter2020-2025/combined_file.nc")
era5_2015_2020 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2015-2020.nc").rename({'valid_time': 'time'})
era5_2010_2015 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2010-2015.nc").rename({'valid_time': 'time'})
era5_2005_2010 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2005-2010.nc").rename({'valid_time': 'time'})
era5_2000_2005 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2000-2005.nc").rename({'valid_time': 'time'})
era5_1995_2000 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1995-2000.nc").rename({'valid_time': 'time'})
era5_1990_1995 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1990-1995.nc").rename({'valid_time': 'time'})
era5_1985_1990 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1985-1990.nc").rename({'valid_time': 'time'})
era5_1980_1985 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1980-1985.nc").rename({'valid_time': 'time'})
import xarray as xr

# Concatenate 1980–1997 inclusive
era5_1980_1997 = xr.concat(
    [
        era5_1980_1985,
        era5_1985_1990,
        era5_1990_1995,
        era5_1995_2000.sel(time=slice("1995-01-01", "1998-03-1"))
    ],
    dim="time"
)

# Concatenate 1997–2015 (exclude 1997, include 2015)
era5_1997_2015 = xr.concat(
    [
        era5_1995_2000.sel(time=slice("1998-12-01", "2001-03-01")),
        era5_2000_2005,
        era5_2005_2010,
        era5_2010_2015
    ],
    dim="time"
)

def plot_comparison(fourcast, era5, variable, title="", cmap='RdBu_r', maskout=None, vmin=None, vmax=None, tag="", highlight_box=None):
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.EqualEarth()})
    
    # Find difference between FourCastNet and ERA5
    if isinstance(fourcast, xr.Dataset):
        fourcast_minus_era5 = fourcast[variable].mean(dim='time', skipna = True) - era5[variable].mean(dim='time', skipna = True)
    elif isinstance(fourcast, xr.DataArray):
        fourcast_minus_era5 = fourcast.mean(dim='time', skipna = True) - era5.mean(dim='time', skipna = True)
    else:
        raise ValueError("Input must be an xarray.Dataset or xarray.DataArray")
    
    if type(maskout) != type(None):
        fourcast_minus_era5 = fourcast_minus_era5*maskout
    #weight by cosing of latitude and calculate mean
    weights = np.cos(np.deg2rad(fourcast_minus_era5.latitude))
    weights.name = "weights"
    difference_weighted = fourcast_minus_era5.weighted(weights)
    mean_diff = difference_weighted.mean(dim=('latitude','longitude')).values
    #plot the difference
    im = fourcast_minus_era5.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label(f"{variable} Difference from ERA5", fontsize=12)
    ax.set_title(title, fontsize=16)

    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")

        # Add annotation over Southern Ocean (centered above Antarctica)
    ax.text(0, -60, f"Mean: {mean_diff:.2f} K", transform=ccrs.PlateCarree(),
            ha='center', va='center', fontsize=14, color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    # 🔲 Draw box if specified: highlight_box = (lat_min, lat_max, lon_min, lon_max)
    if highlight_box is not None:
        if isinstance(highlight_box[0], (list, tuple)):  # multiple boxes
            boxes = highlight_box
        else:  # single box
            boxes = [highlight_box]

        for box in boxes:
            lat_min, lat_max, lon_min, lon_max = box
            box_lats = [lat_min, lat_max, lat_max, lat_min, lat_min]
            box_lons = [lon_min, lon_min, lon_max, lon_max, lon_min]
            ax.plot(box_lons, box_lats, color='#FFBC42', linewidth=2,
                    transform=ccrs.PlateCarree(), zorder=10)
    print(f"saving figure with tag: {tag}")
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/mean_fourcast_era5_comparison_{tag}.png", bbox_inches='tight', dpi=300)
    return fourcast_minus_era5, mean_diff
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

thresholded_datasets = {}

for perc in [-10,90]:
    era5_early = percentile_filtering_by_season(era5_1980_1997.t2m, percentile=perc)
    era5_late = percentile_filtering_by_season(era5_1997_2015.t2m, percentile=perc)
    tag = f"ERA5_winter_{abs(perc)}th_percentile_climate_change"
    thresholded_datasets[f'ERA5_{abs(perc)}th'], mean_diff = plot_comparison(
        era5_late, era5_early, "t2m",
        title=f"ERA5 {abs(perc)}th Percentile Climate Change (Winter)",
        maskout=keep_land_mask, vmin=-2, vmax=2, tag=tag
    )


mean_diff_2020_2025, total_diff_2020_2025 = plot_comparison(era5_1997_2015, era5_1980_1997, "t2m", title="ERA5 Winter Season Comparison", maskout=keep_land_mask, vmin=-2, vmax=2, tag="fcast_pangu_era5_climate_change_winter")
