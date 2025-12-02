####Comparing ERA5 to FourCastNet v2.0

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






# Path to your font files
font_path_regular = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-VariableFont_wght.ttf"
font_path_italic = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-Italic-VariableFont_wght.ttf"

# Register fonts
fm.fontManager.addfont(font_path_regular)
fm.fontManager.addfont(font_path_italic)

# Set global font family to Red Hat Display
mpl.rcParams['font.family'] = 'Red Hat Display'





fourcast_9day = xr.open_dataset("/barnes-engr-scratch1/DATA/Fourcastv2/Fourcast_V2_winter_2020_2025.nc")

persistence_forcast = xr.open_dataset("/home/jlandsbe/ERA5_2020_2025_persistence.nc")

damped_persistence_forcast = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/era5_2020_2025_predictions.nc").predictions

fourcast_2day = xr.open_dataset("/barnes-engr-scratch1/DATA/Fourcastv2/Fourcast_V2_winter_2020_2025_2day.nc")

pangu_2day = xr.open_dataset("/barnes-engr-scratch1/DATA/Pangu/PANGU_V1_combined_2day.nc")
pangu_9day = xr.open_dataset("/barnes-engr-scratch1/DATA/Pangu/PANGU_V1_combined_9day.nc")


era5_2020_2025 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/6-hourly_0.25deg_data/winter2020-2025/combined_file.nc")
era5_2015_2020 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2015-2020.nc").rename({'valid_time': 'time'})
era5_2010_2015 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2010-2015.nc").rename({'valid_time': 'time'})
era5_2005_2010 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2005-2010.nc").rename({'valid_time': 'time'})
era5_2000_2005 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_2000-2005.nc").rename({'valid_time': 'time'})
era5_1995_2000 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1995-2000.nc").rename({'valid_time': 'time'})
era5_1990_1995 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1990-1995.nc").rename({'valid_time': 'time'})
era5_1985_1990 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1985-1990.nc").rename({'valid_time': 'time'})
era5_1980_1985 = xr.open_dataset("/barnes-engr-scratch1/DATA/ERA5/daily_temp/winter_1979_2025/combined_era5_winter_1980-1985.nc").rename({'valid_time': 'time'})
# mask = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/ERA5_hres_land_mask.nc")
# mask = mask.mean(dim="valid_time")
# keep_ocean_mask = xr.where(mask.lsm <= 0.5, 1, np.nan)
# keep_land_mask = xr.where(mask.lsm > 0.5, 1, np.nan)
keep_land_mask = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/remove_poles_mask.nc").__xarray_dataarray_variable__

import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

def make_custom_colormap(low_color="#FE5F55", mid_color="#FCDFA6", high_color="#4CB963", n_colors=9, power=2.0):
    """
    Creates a custom colormap with more gradation at the high end.
    
    Parameters:
    - low_color, mid_color, high_color: colors at bottom, middle, and top.
    - n_colors: total number of colors in the colormap.
    - power: controls gradation skew. >1 = more emphasis at top, <1 = more at bottom.
    """
    n_colors -= 2
    n_colors = 9
    # Convert to RGB arrays
    low = np.array(to_rgb(low_color))
    mid = np.array(to_rgb(mid_color))
    high = np.array(to_rgb(high_color))

    # Number of gradient steps
    n_half = (n_colors - 1) // 2

    # Generate nonlinear spaced weights
    low_to_mid_weights = np.linspace(0, 1, n_half + 1) ** power
    mid_to_high_weights = np.linspace(0, 1, n_half + 1) ** power

    # Interpolate colors with nonlinear weights
    low_to_mid = [low + (mid - low) * w for w in low_to_mid_weights]
    mid_to_high = [mid + (high - mid) * w for w in mid_to_high_weights]

    all_colors = [low_to_mid[0], low_to_mid[0]] + low_to_mid + mid_to_high[1:]
    all_colors = low_to_mid + mid_to_high[1:]
    all_colors = np.clip(all_colors, 0, 1)

    return ListedColormap(all_colors)



def plot_emd_years(data, low_color="#FE5F55", mid_color="#FCDFA6", high_color="#4CB963",
                   n_colors=9, variable=None, title="", maskout=None, vmin=None, vmax=None,
                   tag="", time_mean=False, coarsen=True, pvals=None):


    cmap = make_custom_colormap(low_color, mid_color, high_color, n_colors)



    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.EqualEarth()})



    # Extract the variable if Dataset
    if isinstance(data, xr.Dataset):
        if variable is None:
            raise ValueError("You must specify the variable name when passing a Dataset.")
        plot_data = data[variable]
    elif isinstance(data, xr.DataArray):
        print("Using DataArray directly.")
        print(data)
        plot_data = data
    else:
        raise ValueError("Input must be an xarray.Dataset or xarray.DataArray")

    print("Min:", plot_data.min().values)
    print("Max:", plot_data.max().values)
    print("Total non-NaN values:", np.count_nonzero(~np.isnan(plot_data.values)))
    non_nan_mask = ~plot_data.isnull()
    valid_points = non_nan_mask.sum(dim=["latitude", "longitude"])
    print("Valid gridpoints (non-NaN):", valid_points.values)

    # Optionally average over time
    if time_mean:
        plot_data = plot_data.mean(dim='time', skipna=True)

    # Apply mask if provided
    if maskout is not None:
        print(np.shape((maskout)))
        print(np.shape((plot_data)))
        # if coarsen:
        #     maskout = maskout.coarsen(latitude=4, longitude=4, boundary='trim').mean()
        plot_data = plot_data * maskout
        print(np.shape(plot_data))

    print("Min:", plot_data.min().values)
    print("Max:", plot_data.max().values)
    print("Total non-NaN values:", np.count_nonzero(~np.isnan(plot_data.values)))
    non_nan_mask = ~plot_data.isnull()
    valid_points = non_nan_mask.sum(dim=["latitude", "longitude"])
    print("Valid gridpoints (non-NaN):", valid_points.values)

    # Compute weighted mean (global)
    weights = np.cos(np.deg2rad(plot_data.latitude))
    weights.name = "weights"
    weighted = plot_data.weighted(weights)
    mean_diff = weighted.mean(dim=('latitude', 'longitude')).values

    # Compute weighted mean over contiguous US
    # Define lat/lon bounds for CONUS (approx: 24-50N, 235-295E)
    lat_min_us, lat_max_us = 25, 42
    lon_min_us, lon_max_us = 265, 290

    # Draw CONUS box on the map in black
    box_lats = [lat_min_us, lat_max_us, lat_max_us, lat_min_us, lat_min_us]
    box_lons = [lon_min_us, lon_min_us, lon_max_us, lon_max_us, lon_min_us]
    ax.plot(box_lons, box_lats, color='black', linewidth=2, transform=ccrs.PlateCarree(), zorder=10)

    # Subset data for CONUS
    plot_data_us = plot_data.sel(
        latitude=slice(lat_max_us, lat_min_us),
        longitude=slice(lon_min_us, lon_max_us)
    )

    # Compute weights for CONUS
    weights_us = np.cos(np.deg2rad(plot_data_us.latitude))
    weights_us.name = "weights"
    weighted_us = plot_data_us.weighted(weights_us)
    mean_contiguous_us = weighted_us.mean(dim=('latitude', 'longitude')).values
    if coarsen:
        print("boundary")
        plot_data = plot_data.coarsen(latitude=4, longitude=4, boundary='trim').mean()
    # Plot the data
    im = plot_data.plot(
    ax=ax,
    cmap=cmap,
    vmin=0,
    vmax=n_colors,
    add_colorbar=False,
    transform=ccrs.PlateCarree()  # <--- This is crucial!
)
    if pvals is not None:
        if pvals.shape != plot_data.shape:
            raise ValueError("pvals must have same shape as data being plotted.")

        sig_mask = pvals < 0.1

        # Convert mask → coordinates
        y_idx, x_idx = np.where(sig_mask)

        # Get actual lon/lat arrays
        lats = plot_data.latitude.values
        lons = plot_data.longitude.values

        # Stipple
        ax.scatter(
            lons[x_idx],
            lats[y_idx],
            s=1,
            c='k',
            alpha=0.5,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=10
        )
    
    # Colorbar setup
    tick_positions = np.arange(n_colors) + 0.5  # centers of color patches
    tick_labels = ['1980-1985', '1985-1990', '1990-1995', '1995-2000', '2000-2005', '2005-2010', '2010-2015', '2015-2020', '2020-2025']

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05,
                        boundaries=np.arange(n_colors + 1),
                        ticks=tick_positions)

    cbar.set_label("Best Matching Time Period", fontsize=12)
    # Add mean marker to colorbar
    mean_index = mean_diff  # This should already be in the range [0, n_colors]
    cbar_ax = cbar.ax
    print(mean_index)
    # Plot a triangle marker at the mean index
    cbar_ax.plot([mean_index + 0.5], .87, marker='v', color='black', clip_on=True, transform=cbar_ax.transData)

    # Add label below the marker
    cbar_ax.text(mean_index + 0.5, 1.3, "Global Mean", ha='center', va='top', fontsize=10, fontweight=800 ,transform=cbar_ax.transData)

    mean_index_us = mean_contiguous_us
    print(mean_index_us)
    cbar_ax.plot([mean_index_us + 0.5], .87, marker='v', color='black', clip_on=True, transform=cbar_ax.transData)
    if tag == "best_matching_time_period_t2m_persist":
        cbar_ax.text(mean_index_us + 0.5, 1.31, "E U.S.", ha='center', fontsize=10, fontweight=800, transform=cbar_ax.transData)
    else: 
        cbar_ax.text(mean_index_us + 0.5, 1.3, "E U.S.", ha='center', va='top', fontsize=10, fontweight=800, transform=cbar_ax.transData)
    cbar.ax.set_xticklabels(tick_labels)
    cbar.ax.tick_params(which='both', length=0)
        # Alternate font size (e.g., large for even indices, small for odd)

    # Set all tick label font sizes to 10
    for label in cbar.ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_ha('center')  # Optional, for horizontal alignment
    plt.setp(cbar.ax.get_xticklabels(), rotation=0, ha='center')
           # Add map features
    ax.set_title(title, fontsize=16)
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")

    # Save figure
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/year_plot_{tag}.png",
                bbox_inches='tight', dpi=300)
    plt.close()






def plot_basic(data, variable, title="", cmap='RdBu_r', maskout=None, vmin=None, vmax=None, tag="", time_mean=True, emd=False, trend=False, quantile=False, highlight_box=None, pvals=None):
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.EqualEarth()})
        # Find difference between FourCastNet and ERA5 
    if isinstance(data, xr.Dataset):
        if time_mean:
            plot_data = data[variable].mean(dim='time', skipna=True)
        else:
            plot_data = data[variable]
    elif isinstance(data, xr.DataArray):
        if time_mean:
            plot_data = data.mean(dim='time', skipna=True)
        else:
            plot_data = data
    else:
        raise ValueError("Input must be an xarray.Dataset or xarray.DataArray")
    # Find difference between FourCastNet and ERA5

    if type(maskout) != type(None):
        plot_data = plot_data*maskout
    #weight by cosing of latitude and calculate mean
    weights = np.cos(np.deg2rad(plot_data.latitude))
    weights.name = "weights"
    difference_weighted = plot_data.weighted(weights)
    mean_diff = difference_weighted.mean(dim=('latitude','longitude')).values
    #plot the difference
    im = plot_data.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False,transform=ccrs.PlateCarree())
    if pvals is not None:
        if pvals.shape != plot_data.shape:
            raise ValueError("pvals must have same shape as data being plotted.")

        sig_mask = pvals < 0.1

        # Convert mask → coordinates
        y_idx, x_idx = np.where(sig_mask)

        # Get actual lon/lat arrays
        lats = plot_data.latitude.values
        lons = plot_data.longitude.values

        # Stipple
        ax.scatter(
            lons[x_idx],
            lats[y_idx],
            s=1,
            c='k',
            alpha=0.5,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=10
        )
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    ax.set_title(title, fontsize=16)
    if emd:
        cbar.set_label("EMD", fontsize=12)
        text_str = f"Mean: {mean_diff:.2f}"
    elif trend:
        cbar.set_label("Trend (K/5 years)", fontsize=12)
        text_str = f"Mean: {mean_diff:.3f} K/year"
    elif quantile:
        cbar.set_label(r"% of Data as Extreme", fontsize=12)
        text_str = f"Mean: {mean_diff:.2f} %"
    else:
        cbar.set_label(f"{variable} Difference from ERA5", fontsize=12)
        text_str = f"Mean: {mean_diff:.2f} K"



    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")


    # Add annotation over Southern Ocean (centered above Antarctica)
    ax.text(0, -60, text_str, transform=ccrs.PlateCarree(),
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
            ax.plot(box_lons, box_lats, color='black', linewidth=2,
                    transform=ccrs.PlateCarree(), zorder=10)
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/mean_{tag}.png", bbox_inches='tight', dpi=300)
    return plot_data, mean_diff

def plot_comparison(fourcast, era5, variable, title="", cmap='RdBu_r', maskout=None, vmin=None, vmax=None, tag="", highlight_box=None, coarsen=True, pvals=None):
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
    if coarsen: 
        fourcast_minus_era5_plot = fourcast_minus_era5.coarsen(latitude=4, longitude=4, boundary='trim').mean()
    im = fourcast_minus_era5_plot.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False, transform=ccrs.PlateCarree())
    if pvals is not None:
        if pvals.shape != fourcast_minus_era5_plot.shape:
            raise ValueError("pvals must have same shape as data being plotted.")

        sig_mask = pvals < 0.1

        # Convert mask → coordinates
        y_idx, x_idx = np.where(sig_mask)

        # Get actual lon/lat arrays
        lats = fourcast_minus_era5_plot.latitude.values
        lons = fourcast_minus_era5_plot.longitude.values

        # Stipple
        ax.scatter(
            lons[x_idx],
            lats[y_idx],
            s=1,
            c='k',
            alpha=0.5,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=10
        )
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
        color = '#FFBC42'
        colors = ['black', '#FFBC42']
        for c, box in enumerate(boxes):
            lat_min, lat_max, lon_min, lon_max = box
            box_lats = [lat_min, lat_max, lat_max, lat_min, lat_min]
            box_lons = [lon_min, lon_min, lon_max, lon_max, lon_min]
            ax.plot(box_lons, box_lats, color=colors[c], linewidth=2,
                    transform=ccrs.PlateCarree(), zorder=10)
    print(f"saving figure with tag: {tag}")
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/mean_fourcast_era5_comparison_{tag}.png", bbox_inches='tight', dpi=300)
    return fourcast_minus_era5, mean_diff

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


def x_y_plot(xlabels, ydata, ylabel, title, tag, plot_lab="", vmin=None, vmax=None, absolute = 0, xlab="Time", colors=None):
    """
    Create a time series plot with specified x-axis labels and y-axis data.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # Round xlabels to nearest whole number and ydata to 2 decimal points
    #xlabels_rounded = [round(float(x),1) for x in xlabels]
    ydata_rounded = [round(float(y), 2) for y in ydata]

    if not absolute:
        ax.plot(xlabels, ydata_rounded, marker='o', linestyle='-', color='#102e4a', linewidth=2, markersize=8, label=plot_lab)
        #ax.plot(xlabels, ydata_rounded, marker='o', color='#FCDFA6', markersize=8)
    else:
        # Plot the line connecting all points
        ax.plot(xlabels, [abs(y) for y in ydata_rounded], linestyle='-', color='#102e4a', linewidth=3, label=plot_lab)
        #ax.plot(xlabels, [abs(y) for y in ydata_rounded], marker='o', color='#FCDFA6', markersize=8)
    # Track if we've added a label yet
    if colors is not None:
        plotted_labels = {"Warm Tail": False, "Cold Tail": False}
        for i in range(len(xlabels)):
            if colors[i] == 'warm':
                color = 'tomato'
                label = 'Warm Tail' if not plotted_labels["Warm Tail"] else None
                plotted_labels["Warm Tail"] = True
            else:
                color = 'cornflowerblue'
                label = 'Cold Tail' if not plotted_labels["Cold Tail"] else None
                plotted_labels["Cold Tail"] = True
            ax.plot(xlabels[i], np.abs(ydata[i]), marker='o', color=color, markersize=10, label=label)
        ax.legend(loc='upper left', fontsize=12)
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    # ax.set_xticks(xlabels_rounded)
    # ax.set_xticklabels(xlabels_rounded, rotation=45)
    #set limits
    if vmin is not None and vmax is not None:
        ax.set_ylim(vmin, vmax)
    plt.tight_layout()
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/{tag}.png", bbox_inches='tight', dpi=300)

def time_series_plot(xlabels, ydata, ylabel, title, tag, vmin=None, vmax=None, abs = 0, xlab="Time", y_data2 = None):
    """
    Create a time series plot with specified x-axis labels and y-axis data.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if not abs:
        ax.plot(xlabels, ydata, marker='o', linestyle='-', color='seagreen', linewidth=2, markersize=8, label='FourCastNet Difference in Mean Temperature')
        if y_data2 is not None:
            ax.plot(xlabels, y_data2, marker='o', linestyle='--', color='seagreen', linewidth=2, markersize=8, label='Pangu Difference in Global Mean Temperature')
    else:
                # Plot the line connecting all points
        ax.plot(xlabels, np.abs(ydata), linestyle='--', color='#102e4a', linewidth=4, label='FourCastNet Absolute Difference in Mean Temperature')
        ax.plot(xlabels, np.abs(y_data2), linestyle='-', color='#FFBC42', linewidth=4, label='Pangu Absolute Difference in Mean Temperature')

    # Track if we've added a label yet
    plotted_labels = {"Warm Bias": False, "Cold Bias": False}

    for i in range(len(xlabels)):
        value = np.abs(ydata[i]) if abs else ydata[i]
        
        if ydata[i] > 0:
            color = 'tomato'
            label = 'Warm Bias' if not plotted_labels["Warm Bias"] else None
            plotted_labels["Warm Bias"] = True
        else:
            color = 'cornflowerblue'
            label = 'Cold Bias' if not plotted_labels["Cold Bias"] else None
            plotted_labels["Cold Bias"] = True

        ax.plot(xlabels[i], value, marker='o', color=color, markersize=12, label=label)
        
        # Add annotation in same color, slightly offset vertically
        ax.text(xlabels[i], value + 0.04, f"{value:.2f}", color=color, fontsize=12, ha='center', weight = 'bold')


    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(xlabels)
    ax.set_xticklabels(xlabels, rotation=45)
    
    if vmax is not None:
        ax.set_ylim(0, vmax)

    plt.tight_layout()
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/{tag}.png", bbox_inches='tight', dpi=300)

from scipy.stats import wasserstein_distance

def compute_emd_per_gridpoint(da1, da2):
    """
    Compute Earth Mover's Distance (EMD) at each lat/lon gridpoint between two xarray DataArrays.

    Parameters:
        da1, da2 (xarray.DataArray): Input DataArrays with dimensions (time, latitude, longitude).
                                     Time values do not need to match.

    Returns:
        xarray.DataArray: 2D array (latitude, longitude) of EMD values.
    """
    lats = da1.latitude
    lons = da1.longitude

    # Convert to numpy arrays for efficient indexing
    data1 = da1.values
    data2 = da2.values

    emd_values = np.full((len(lats), len(lons)), np.nan)

    for i in range(len(lats)):
        for j in range(len(lons)):
            x = data1[:, i, j]
            y = data2[:, i, j]

            # Remove NaNs before computing EMD
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]

            # Skip if either dataset has no valid values at this location
            if len(x) > 0 and len(y) > 0:
                emd_values[i, j] = wasserstein_distance(x, y)
            else:
                emd_values[i, j] = np.nan

    return xr.DataArray(
        emd_values,
        coords={"latitude": lats, "longitude": lons},
        dims=("latitude", "longitude"),
        name="emd"
    )


def time_mean(ds,era5, maskout, variable="t2m"):
        # Find difference between FourCastNet and ERA5
    if isinstance(ds, xr.Dataset):
        fourcast_minus_era5 = ds[variable].mean(dim='time', skipna = True) - era5[variable].mean(dim='time', skipna = True)
    elif isinstance(ds, xr.DataArray):
        fourcast_minus_era5 = ds.mean(dim='time', skipna = True) - era5.mean(dim='time', skipna = True)
    else:
        raise ValueError("Input must be an xarray.Dataset or xarray.DataArray")
    
    if type(maskout) != type(None):
        fourcast_minus_era5 = fourcast_minus_era5*maskout

    return fourcast_minus_era5


# # ##mean comparison
# damped_persistence_diff_2020_2025 = time_mean(damped_persistence_forcast, era5_2020_2025.t2m, keep_land_mask)
# damped_persistence_diff_2015_2020 = time_mean(damped_persistence_forcast, era5_2015_2020.t2m, keep_land_mask)
# damped_persistence_diff_2010_2015 = time_mean(damped_persistence_forcast, era5_2010_2015.t2m, keep_land_mask)
# damped_persistence_diff_2005_2010 = time_mean(damped_persistence_forcast, era5_2005_2010.t2m, keep_land_mask)
# damped_persistence_diff_2000_2005 = time_mean(damped_persistence_forcast, era5_2000_2005.t2m, keep_land_mask)
# damped_persistence_diff_1995_2000 = time_mean(damped_persistence_forcast, era5_1995_2000.t2m, keep_land_mask)
# damped_persistence_diff_1990_1995 = time_mean(damped_persistence_forcast, era5_1990_1995.t2m, keep_land_mask)
# damped_persistence_diff_1985_1990 = time_mean(damped_persistence_forcast, era5_1985_1990.t2m, keep_land_mask)
# damped_persistence_diff_1980_1985 = time_mean(damped_persistence_forcast, era5_1980_1985.t2m, keep_land_mask)


# damped_persist_list = [damped_persistence_diff_1980_1985, damped_persistence_diff_1985_1990,damped_persistence_diff_1990_1995,
#                 damped_persistence_diff_1995_2000, damped_persistence_diff_2000_2005, damped_persistence_diff_2005_2010,
#                 damped_persistence_diff_2010_2015, damped_persistence_diff_2015_2020, damped_persistence_diff_2020_2025]


# # Assume you have a list of DataArrays with the same lat/lon dims
# # Stack them into a new dimension, e.g., 'index'
# stacked = xr.concat(damped_persist_list, dim='index')
# stacked = np.abs(stacked)


# # Create mask where all values are NaN at each gridpoint
# all_nan_mask = stacked.isnull().all(dim='index')

# # Fill NaNs with +inf before calling argmin (inf will never be selected unless all are inf)
# stacked_filled = stacked.fillna(np.inf)

# # Compute argmin safely
# min_index = stacked_filled.argmin(dim='index')

# # Set index to NaN where all were NaN
# min_index = min_index.where(~all_nan_mask)

# plot_emd_years(min_index, variable="t2m", title="Best Matching ERA5 Time Period for 2m Temperature (Damped Persistence Forecast)",
#                maskout=keep_land_mask, vmin=0, vmax=8, tag="best_matching_time_period_t2m_damped_persist")










persitance_diff_2020_2025 = time_mean(persistence_forcast, era5_2020_2025, keep_land_mask)
persitance_diff_2015_2020 = time_mean(persistence_forcast, era5_2015_2020, keep_land_mask)
persitance_diff_2010_2015 = time_mean(persistence_forcast, era5_2010_2015, keep_land_mask)
persitance_diff_2005_2010 = time_mean(persistence_forcast, era5_2005_2010, keep_land_mask)
persitance_diff_2000_2005 = time_mean(persistence_forcast, era5_2000_2005, keep_land_mask)
persitance_diff_1995_2000 = time_mean(persistence_forcast, era5_1995_2000, keep_land_mask)
persitance_diff_1990_1995 = time_mean(persistence_forcast, era5_1990_1995, keep_land_mask)
persitance_diff_1985_1990 = time_mean(persistence_forcast, era5_1985_1990, keep_land_mask)
persitance_diff_1980_1985 = time_mean(persistence_forcast, era5_1980_1985, keep_land_mask)

persist_list = [persitance_diff_1980_1985, persitance_diff_1985_1990, persitance_diff_1990_1995,
                persitance_diff_1995_2000, persitance_diff_2000_2005, persitance_diff_2005_2010,
                persitance_diff_2010_2015, persitance_diff_2015_2020, persitance_diff_2020_2025]


# Assume you have a list of DataArrays with the same lat/lon dims
# Stack them into a new dimension, e.g., 'index'
stacked = xr.concat(persist_list, dim='index')
stacked = np.abs(stacked)


# Create mask where all values are NaN at each gridpoint
all_nan_mask = stacked.isnull().all(dim='index')

# Fill NaNs with +inf before calling argmin (inf will never be selected unless all are inf)
stacked_filled = stacked.fillna(np.inf)

# Compute argmin safely
min_index = stacked_filled.argmin(dim='index')

# Set index to NaN where all were NaN
min_index = min_index.where(~all_nan_mask)

plot_emd_years(min_index, variable="t2m", title="Persistence",
               maskout=keep_land_mask, vmin=0, vmax=8, tag="best_matching_time_period_t2m_persist")



land_mask_coarsened = keep_land_mask.coarsen(latitude=4, longitude=4, boundary='trim').mean()

p_vals_fourcast_0_2day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/fourcast_2day_p_values_0.npy") * land_mask_coarsened.values

p_vals_fourcast_0_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/fourcast_9day_p_values_0.npy") * land_mask_coarsened.values

p_vals_pangu_0_2day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/pangu_2day_p_values_0.npy")* land_mask_coarsened.values
p_vals_pangu_0_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/pangu_9day_p_values_0.npy")* land_mask_coarsened.values

p_vals_fourcast_10_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/fourcast_9day_p_values_10.npy")* land_mask_coarsened.values
p_vals_fourcast_90_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/fourcast_9day_p_values_90.npy")* land_mask_coarsened.values
p_vals_fourcast_80_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/fourcast_9day_p_values_80.npy")* land_mask_coarsened.values
p_vals_fourcast_20_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/fourcast_9day_p_values_20.npy")* land_mask_coarsened.values
p_vals_fourcast_95_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/fourcast_9day_p_values_95.npy")* land_mask_coarsened.values
p_vals_fourcast_5_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/fourcast_9day_p_values_5.npy")* land_mask_coarsened.values

p_vals_pangu_10_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/pangu_9day_p_values_10.npy") * land_mask_coarsened.values
p_vals_pangu_90_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/pangu_9day_p_values_90.npy") * land_mask_coarsened.values
p_vals_pangu_80_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/pangu_9day_p_values_80.npy") * land_mask_coarsened.values
p_vals_pangu_20_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/pangu_9day_p_values_20.npy") * land_mask_coarsened.values
p_vals_pangu_95_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/pangu_9day_p_values_95.npy") * land_mask_coarsened.values
p_vals_pangu_5_9day = np.load("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/pangu_9day_p_values_5.npy") * land_mask_coarsened.values

#recomment

# mean_diff_2020_2025, total_diff_2020_2025 = plot_comparison(fourcast_9day, era5_2020_2025, "t2m", title="FourCastNet 9-day", maskout = keep_land_mask,vmin=-2, vmax=2, tag="9day_2020_2025_mean", pvals = p_vals_fourcast_0_9day)
# mean_diff_2015_2020, total_diff_2015_2020 = plot_comparison(fourcast_9day, era5_2015_2020, "t2m", title="FourCastNet 9-day Mean 2mT Difference (2015-2020)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="9day_2015_2020_mean")
# mean_diff_2010_2015, total_diff_2010_2015 = plot_comparison(fourcast_9day, era5_2010_2015, "t2m", title="FourCastNet 9-day Mean 2mT Difference (2010-2015)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="9day_2010_2015_mean", pvals = p_vals_fourcast_0_9day)
# mean_diff_2005_2010, total_diff_2005_2010 = plot_comparison(fourcast_9day, era5_2005_2010, "t2m", title="FourCastNet 9-day Mean 2mT Difference (2005-2010)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="9day_2005_2010_mean", pvals = p_vals_fourcast_0_9day)
# mean_diff_2000_2005, total_diff_2000_2005 = plot_comparison(fourcast_9day, era5_2000_2005, "t2m", title="FourCastNet 9-day Mean 2mT Difference (2000-2005)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="9day_2000_2005_mean", pvals = p_vals_fourcast_0_9day)
# mean_diff_1995_2000, total_diff_1995_2000 = plot_comparison(fourcast_9day, era5_1995_2000, "t2m", title="FourCastNet 9-day Mean 2mT Difference (1995-2000)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="9day_1995_2000_mean", pvals = p_vals_fourcast_0_9day)
# mean_diff_1990_1995, total_diff_1990_1995 = plot_comparison(fourcast_9day, era5_1990_1995, "t2m", title="FourCastNet 9-day Mean 2mT Difference (1990-1995)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="9day_1990_1995_mean", pvals = p_vals_fourcast_0_9day)
# mean_diff_1985_1990, total_diff_1985_1990 = plot_comparison(fourcast_9day, era5_1985_1990, "t2m", title="FourCastNet 9-day Mean 2mT Difference (1985-1990)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="9day_1985_1990_mean", pvals = p_vals_fourcast_0_9day)
# mean_diff_1980_1985, total_diff_1980_1985 = plot_comparison(fourcast_9day, era5_1980_1985, "t2m", title="FourCastNet 9-day Mean 2mT Difference (1980-1985)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="9day_1980_1985_mean", pvals = p_vals_fourcast_0_9day)

# # Pangu 9-day lead comparisons
# pangu_mean_diff_2020_2025, pangu_total_diff_2020_2025 = plot_comparison(pangu_9day, era5_2020_2025, "t2m", title="Pangu 9-day", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_9day_2020_2025_mean", pvals = p_vals_pangu_0_9day)
# pangu_mean_diff_2015_2020, pangu_total_diff_2015_2020 = plot_comparison(pangu_9day, era5_2015_2020, "t2m", title="Pangu 9-day Mean 2mT Difference (2015-2020)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_9day_2015_2020_mean")
# pangu_mean_diff_2010_2015, pangu_total_diff_2010_2015 = plot_comparison(pangu_9day, era5_2010_2015, "t2m", title="Pangu 9-day Mean 2mT Difference (2010-2015)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_9day_2010_2015_mean")
# pangu_mean_diff_2005_2010, pangu_total_diff_2005_2010 = plot_comparison(pangu_9day, era5_2005_2010, "t2m", title="Pangu 9-day Mean 2mT Difference (2005-2010)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_9day_2005_2010_mean")
# pangu_mean_diff_2000_2005, pangu_total_diff_2000_2005 = plot_comparison(pangu_9day, era5_2000_2005, "t2m", title="Pangu 9-day Mean 2mT Difference (2000-2005)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_9day_2000_2005_mean")
# pangu_mean_diff_1995_2000, pangu_total_diff_1995_2000 = plot_comparison(pangu_9day, era5_1995_2000, "t2m", title="Pangu 9-day Mean 2mT Difference (1995-2000)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_9day_1995_2000_mean", pvals = p_vals_pangu_0_9day)
# pangu_mean_diff_1990_1995, pangu_total_diff_1990_1995 = plot_comparison(pangu_9day, era5_1990_1995, "t2m", title="Pangu 9-day Mean 2mT Difference (1990-1995)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_9day_1990_1995_mean", pvals = p_vals_pangu_0_9day)
# pangu_mean_diff_1985_1990, pangu_total_diff_1985_1990 = plot_comparison(pangu_9day, era5_1985_1990, "t2m", title="Pangu 9-day Mean 2mT Difference (1985-1990)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_9day_1985_1990_mean", pvals = p_vals_pangu_0_9day)
# pangu_mean_diff_1980_1985, pangu_total_diff_1980_1985 = plot_comparison(pangu_9day, era5_1980_1985, "t2m", title="Pangu 9-day Mean 2mT Difference (1980-1985)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_9day_1980_1985_mean", pvals = p_vals_pangu_0_9day)

# # Pangu 2-day lead comparisons
# pangu_mean_diff_2020_2025_2day, pangu_total_diff_2020_2025_2day = plot_comparison(pangu_2day, era5_2020_2025, "t2m", title="Pangu 2-day", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_2day_2020_2025_mean", pvals= p_vals_pangu_0_2day)
# pangu_mean_diff_2015_2020_2day, pangu_total_diff_2015_2020_2day = plot_comparison(pangu_2day, era5_2015_2020, "t2m", title="Pangu 2-day Mean 2mT Difference (2015-2020)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_2day_2015_2020_mean", pvals= p_vals_pangu_0_2day)
# pangu_mean_diff_2010_2015_2day, pangu_total_diff_2010_2015_2day = plot_comparison(pangu_2day, era5_2010_2015, "t2m", title="Pangu 2-day Mean 2mT Difference (2010-2015)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_2day_2010_2015_mean")
# pangu_mean_diff_2005_2010_2day, pangu_total_diff_2005_2010_2day = plot_comparison(pangu_2day, era5_2005_2010, "t2m", title="Pangu 2-day Mean 2mT Difference (2005-2010)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_2day_2005_2010_mean")
# pangu_mean_diff_2000_2005_2day, pangu_total_diff_2000_2005_2day = plot_comparison(pangu_2day, era5_2000_2005, "t2m", title="Pangu 2-day Mean 2mT Difference (2000-2005)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_2day_2000_2005_mean")
# pangu_mean_diff_1995_2000_2day, pangu_total_diff_1995_2000_2day = plot_comparison(pangu_2day, era5_1995_2000, "t2m", title="Pangu 2-day Mean 2mT Difference (1995-2000)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_2day_1995_2000_mean")
# pangu_mean_diff_1990_1995_2day, pangu_total_diff_1990_1995_2day = plot_comparison(pangu_2day, era5_1990_1995, "t2m", title="Pangu 2-day Mean 2mT Difference (1990-1995)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_2day_1990_1995_mean")
# pangu_mean_diff_1985_1990_2day, pangu_total_diff_1985_1990_2day = plot_comparison(pangu_2day, era5_1985_1990, "t2m", title="Pangu 2-day Mean 2mT Difference (1985-1990)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_2day_1985_1990_mean")
# pangu_mean_diff_1980_1985_2day, pangu_total_diff_1980_1985_2day = plot_comparison(pangu_2day, era5_1980_1985, "t2m", title="Pangu 2-day Mean 2mT Difference (1980-1985)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_2day_1980_1985_mean")




# print("forcast_diff")
# # print(mean_diff_2020_2025)
# # #save this



# # #2day
# mean_diff_2020_2025_2day, total_diff_2020_2025_2day = plot_comparison(fourcast_2day, era5_2020_2025, "t2m", title="FourCastNet 2-day", maskout = keep_land_mask,vmin=-2, vmax=2, tag="2day_2020_2025_mean", pvals= p_vals_fourcast_0_2day)
# mean_diff_2015_2020_2day, total_diff_2015_2020_2day = plot_comparison(fourcast_2day, era5_2015_2020, "t2m", title="FourCastNet Mean 2mT Difference (2015-2020)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="2day_2015_2020_mean", pvals= p_vals_fourcast_0_2day)
# mean_diff_2010_2015_2day, total_diff_2010_2015_2day = plot_comparison(fourcast_2day, era5_2010_2015, "t2m", title="FourCastNet Mean 2mT Difference (2010-2015)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="2day_2010_2015_mean", pvals= p_vals_fourcast_0_2day)
# mean_diff_2005_2010_2day, total_diff_2005_2010_2day = plot_comparison(fourcast_2day, era5_2005_2010, "t2m", title="FourCastNet Mean 2mT Difference (2005-2010)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="2day_2005_2010_mean", pvals= p_vals_fourcast_0_2day)
# mean_diff_2000_2005_2day, total_diff_2000_2005_2day = plot_comparison(fourcast_2day, era5_2000_2005, "t2m", title="FourCastNet Mean 2mT Difference (2000-2005)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="2day_2000_2005_mean", pvals= p_vals_fourcast_0_2day)
# mean_diff_1995_2000_2day, total_diff_1995_2000_2day = plot_comparison(fourcast_2day, era5_1995_2000, "t2m", title="FourCastNet Mean 2mT Difference (1995-2000)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="2day_1995_2000_mean", pvals= p_vals_fourcast_0_2day)
# mean_diff_1990_1995_2day, total_diff_1990_1995_2day = plot_comparison(fourcast_2day, era5_1990_1995, "t2m", title="FourCastNet Mean 2mT Difference (1990-1995)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="2day_1990_1995_mean", pvals= p_vals_fourcast_0_2day)
# mean_diff_1985_1990_2day, total_diff_1985_1990_2day = plot_comparison(fourcast_2day, era5_1985_1990, "t2m", title="FourCastNet 2-day Mean 2mT Difference (1985-1990)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="2day_1985_1990_mean", pvals= p_vals_fourcast_0_2day)
# mean_diff_1980_1985_2day, total_diff_1980_1985_2day = plot_comparison(fourcast_2day, era5_1980_1985, "t2m", title="FourCastNet 2-day Mean 2mT Difference (1980-1985)", maskout = keep_land_mask,vmin=-2, vmax=2, tag="2day_1980_1985_mean", pvals= p_vals_fourcast_0_2day)


# xlabels = ["1980-1985", "1985-1990", "1990-1995", "1995-2000", "2000-2005", "2005-2010", "2010-2015", "2015-2020", "2020-2025"]
# ydata_fourcast = [total_diff_1980_1985, total_diff_1985_1990, total_diff_1990_1995, total_diff_1995_2000, total_diff_2000_2005, total_diff_2005_2010, total_diff_2010_2015, total_diff_2015_2020, total_diff_2020_2025]
# ydata_pangu_9_day = [total_diff_1980_1985, total_diff_1985_1990, total_diff_1990_1995, total_diff_1995_2000, total_diff_2000_2005, total_diff_2005_2010, total_diff_2010_2015, total_diff_2015_2020, total_diff_2020_2025]
# time_series_plot(xlabels, ydata_fourcast, ylabel="Mean Difference (K)", title="2-Day Lead Absolute Global Mean Temperature Difference", tag="time_series_mean_diff_abs", abs=1, vmax=1, y_data2 = ydata_pangu_9_day)

# da_list_pangu = [pangu_mean_diff_1980_1985, pangu_mean_diff_1985_1990, pangu_mean_diff_1990_1995,
#            pangu_mean_diff_1995_2000, pangu_mean_diff_2000_2005, pangu_mean_diff_2005_2010,
#            pangu_mean_diff_2010_2015, pangu_mean_diff_2015_2020, pangu_mean_diff_2020_2025]

# da_list_2day_pangu = [pangu_mean_diff_1980_1985_2day, pangu_mean_diff_1985_1990_2day, pangu_mean_diff_1990_1995_2day,
#                pangu_mean_diff_1995_2000_2day, pangu_mean_diff_2000_2005_2day, pangu_mean_diff_2005_2010_2day,
#                pangu_mean_diff_2010_2015_2day, pangu_mean_diff_2015_2020_2day, pangu_mean_diff_2020_2025_2day]





# #Assume you have a list of DataArrays with the same lat/lon dims
# #Stack them into a new dimension, e.g., 'index'
# stacked = xr.concat(da_list_2day_pangu, dim='index')
# stacked = np.abs(stacked)


# # Create mask where all values are NaN at each gridpoint
# all_nan_mask = stacked.isnull().all(dim='index')

# # Fill NaNs with +inf before calling argmin (inf will never be selected unless all are inf)
# stacked_filled = stacked.fillna(np.inf)

# # Compute argmin safely
# min_index = stacked_filled.argmin(dim='index')

# # Set index to NaN where all were NaN
# min_index = min_index.where(~all_nan_mask)

# plot_emd_years(min_index, variable="t2m", title="Pangu 2-day",
#                maskout=keep_land_mask, vmin=0, vmax=8, tag="pangu_best_matching_time_period_t2m_2day", pvals= p_vals_pangu_0_2day)

# #Assume you have a list of DataArrays with the same lat/lon dims
# #Stack them into a new dimension, e.g., 'index'
# stacked = xr.concat(da_list_pangu, dim='index')
# stacked = np.abs(stacked)


# # Create mask where all values are NaN at each gridpoint
# all_nan_mask = stacked.isnull().all(dim='index')

# # Fill NaNs with +inf before calling argmin (inf will never be selected unless all are inf)
# stacked_filled = stacked.fillna(np.inf)

# # Compute argmin safely
# min_index = stacked_filled.argmin(dim='index')

# # Set index to NaN where all were NaN
# min_index = min_index.where(~all_nan_mask)

# plot_emd_years(min_index, variable="t2m", title="Pangu 9-day",
#                maskout=keep_land_mask, vmin=0, vmax=8, tag="pangu_best_matching_time_period_t2m_9day", pvals= p_vals_pangu_0_9day)


# da_list = [mean_diff_1980_1985, mean_diff_1985_1990, mean_diff_1990_1995,
#            mean_diff_1995_2000, mean_diff_2000_2005, mean_diff_2005_2010,
#            mean_diff_2010_2015, mean_diff_2015_2020, mean_diff_2020_2025]

# da_list_2day = [mean_diff_1980_1985_2day, mean_diff_1985_1990_2day, mean_diff_1990_1995_2day,
#                mean_diff_1995_2000_2day, mean_diff_2000_2005_2day, mean_diff_2005_2010_2day,
#                mean_diff_2010_2015_2day, mean_diff_2015_2020_2day, mean_diff_2020_2025_2day]





# #Assume you have a list of DataArrays with the same lat/lon dims
# #Stack them into a new dimension, e.g., 'index'
# stacked = xr.concat(da_list_2day, dim='index')
# stacked = np.abs(stacked)


# # Create mask where all values are NaN at each gridpoint
# all_nan_mask = stacked.isnull().all(dim='index')

# # Fill NaNs with +inf before calling argmin (inf will never be selected unless all are inf)
# stacked_filled = stacked.fillna(np.inf)

# # Compute argmin safely
# min_index = stacked_filled.argmin(dim='index')

# # Set index to NaN where all were NaN
# min_index = min_index.where(~all_nan_mask)

# plot_emd_years(min_index, variable="t2m", title="FourCastNet 2-day",
#                maskout=keep_land_mask, vmin=0, vmax=8, tag="fourcastnet_best_matching_time_period_t2m_2day", pvals= p_vals_fourcast_0_2day)

# #Assume you have a list of DataArrays with the same lat/lon dims
# #Stack them into a new dimension, e.g., 'index'
# stacked = xr.concat(da_list, dim='index')
# stacked = np.abs(stacked)


# # Create mask where all values are NaN at each gridpoint
# all_nan_mask = stacked.isnull().all(dim='index')

# # Fill NaNs with +inf before calling argmin (inf will never be selected unless all are inf)
# stacked_filled = stacked.fillna(np.inf)

# # Compute argmin safely
# min_index = stacked_filled.argmin(dim='index')

# # Set index to NaN where all were NaN
# min_index = min_index.where(~all_nan_mask)

# plot_emd_years(min_index, variable="t2m", title="FourCastNet 9-day",
#                maskout=keep_land_mask, vmin=0, vmax=8, tag="fourcastnet_best_matching_time_period_t2m_9day", pvals= p_vals_fourcast_0_9day)





##recomment


plot_comparison(fourcast_9day, era5_2020_2025, "t2m", title="FourCastNet 9-day Mean 2m Temperature Difference", maskout = keep_land_mask,vmin=-2, vmax=2, tag="fourcast_9day", pvals= p_vals_fourcast_0_9day)
plot_comparison(fourcast_2day, era5_2020_2025, "t2m", title="FourCastNet 2-day Mean 2m Temperature Difference", maskout = keep_land_mask, vmin=-2, vmax=2, tag="fourcast_2day", pvals= p_vals_fourcast_0_2day)

plot_comparison(pangu_9day, era5_2020_2025, "t2m", title="Pangu 9-day Mean 2m Temperature Difference", maskout = keep_land_mask,vmin=-2, vmax=2, tag="pangu_9day", pvals= p_vals_pangu_0_9day)
plot_comparison(pangu_2day, era5_2020_2025, "t2m", title="Pangu 2-day Mean 2m Temperature Difference", maskout = keep_land_mask, vmin=-2, vmax=2, tag="pangu_2day", pvals= p_vals_pangu_0_2day)

mean = 0
fourcast_color="#FE5F55"
era5_color="#FFBC42"
pangu_color = '#73877B'
for perc in [(5,95), (10,90), (20,80)]:
    lat_min, lat_max = -35, -22
    lon_min, lon_max = 302, 315
    # Subset to lat-lon box
    era5_box = era5_2020_2025.t2m.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    fourcast_box = fourcast_9day.t2m.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    pangu_box = pangu_9day.t2m.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    # Compute weights
    lat_weights = np.cos(np.deg2rad(era5_box.latitude))
    lat_weights.name = "lat_weights"

    if mean:
        # Weighted means
        temps_era5 = era5_box.weighted(lat_weights).mean(dim=("latitude", "longitude")).values
        temps_fourcast = fourcast_box.weighted(lat_weights).mean(dim=("latitude", "longitude")).values
        temps_pangu = pangu_box.weighted(lat_weights).mean(dim=("latitude", "longitude")).values
    else:
        # Unweighted means
        temps_era5 = era5_box.values
        temps_fourcast = fourcast_box.values
        temps_pangu = pangu_box.values


    # Flatten in case data carry an extra singleton dimension
    temps_era5       = temps_era5.ravel()
    temps_fourcast   = temps_fourcast.ravel()
    temps_pangu      = temps_pangu.ravel()

    temps_era5 = temps_era5[~np.isnan(temps_era5)]
    temps_fourcast = temps_fourcast[~np.isnan(temps_fourcast)]
    temps_pangu = temps_pangu[~np.isnan(temps_pangu)]

    # Compute common bin edges
    all_temps = np.concatenate([temps_era5, temps_fourcast, temps_pangu])
    bins = np.linspace(all_temps.min(), all_temps.max(), 40)

    # Compute histograms (normalized)
    hist_era5, _ = np.histogram(temps_era5, bins=bins, density=True)
    hist_fourcast, _ = np.histogram(temps_fourcast, bins=bins, density=True)
    hist_pangu, _ = np.histogram(temps_pangu, bins=bins, density=True)

    # Bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, hist_era5, label='ERA5 (2020–2025)', color=era5_color, linewidth=3)
    plt.plot(bin_centers, hist_fourcast, label='FourCastNet 9-day', color=fourcast_color, linewidth=3)
    plt.plot(bin_centers, hist_pangu, label='Pangu 9-day', color=pangu_color, linewidth=3)

    # Add 10th and 90th percentiles
    p10_era5 = np.percentile(temps_era5, perc[0])
    p90_era5 = np.percentile(temps_era5, perc[1])
    p10_fc = np.percentile(temps_fourcast, perc[0])
    p90_fc = np.percentile(temps_fourcast, perc[1])
    p10_pangu = np.percentile(temps_pangu, perc[0])
    p90_pangu = np.percentile(temps_pangu, perc[1])


    for val, color, label, hist in zip(
    [p10_era5, p90_era5, p10_fc, p90_fc, p10_pangu, p90_pangu],
    [era5_color, era5_color, fourcast_color, fourcast_color, pangu_color, pangu_color],
    [f"ERA5 {perc[0]}th", f"ERA5 {perc[1]}th", f"FourCast {perc[0]}th", f"FourCast {perc[1]}th", f"Pangu {perc[0]}th", f"Pangu {perc[1]}th"],
    [hist_era5, hist_era5, hist_fourcast, hist_fourcast, hist_pangu, hist_pangu],
):
       #Find the bin index closest to the percentile
        # Interpolate to find the vertical position
        ymax = np.interp(val, bin_centers, hist)

        # Plot vertical dashed line from y=0 to ymax
        plt.vlines(val, 0, ymax, linestyle='--', color=color, linewidth=2)

        # Add vertical label slightly above the line
        if val == p10_era5 or val == p10_fc:
            if p10_fc < p10_era5:
                add_bit = .2 + .5 * (-2*(label[0:4]=="Four"))
            else:
                add_bit = .2 + .5 * (-2*(label[0:4]=="ERA5"))
        else:
            add_bit = .2 + .5 * (-2*(label[0:4]=="Four"))

        plt.text(val + add_bit, 0, label, color=color, rotation=90,
                ha='left', va='bottom', fontsize=10, fontweight='bold')


    # Labels and legend
    plt.xlabel('Temperatures (K)')
    plt.ylabel('Probability Density')
    # set x limits
    # plt.xlim(287, 301)
    plt.title('SE South America Temperatures', fontsize=16)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    # Remove top and right axes (spines)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/SA_temperatures_pdf_{perc[0]}_{perc[1]}.png",
                    bbox_inches='tight', dpi=300)

    mean = 0
    lat_min, lat_max = 33, 38
    lon_min, lon_max = 265, 273
    # lat_min, lat_max = 25, 42
    # lon_min, lon_max = 265, 290
    # Subset to lat-lon box
    era5_box = era5_2020_2025.t2m.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    fourcast_box = fourcast_9day.t2m.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    pangu_box = pangu_9day.t2m.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    # Compute weights
    lat_weights = np.cos(np.deg2rad(era5_box.latitude))
    lat_weights.name = "lat_weights"

    if mean:
        # Weighted means
        temps_era5 = era5_box.weighted(lat_weights).mean(dim=("latitude", "longitude")).values
        temps_fourcast = fourcast_box.weighted(lat_weights).mean(dim=("latitude", "longitude")).values
        temps_pangu = pangu_box.weighted(lat_weights).mean(dim=("latitude", "longitude")).values
    else:
        # Unweighted means
        temps_era5 = era5_box.values
        temps_fourcast = fourcast_box.values
        temps_pangu = pangu_box.values


    # Flatten in case data carry an extra singleton dimension
    temps_era5       = temps_era5.ravel()
    temps_fourcast   = temps_fourcast.ravel()
    temps_pangu      = temps_pangu.ravel()

    temps_era5 = temps_era5[~np.isnan(temps_era5)]
    temps_fourcast = temps_fourcast[~np.isnan(temps_fourcast)]
    temps_pangu = temps_pangu[~np.isnan(temps_pangu)]

    # Compute common bin edges
    all_temps = np.concatenate([temps_era5, temps_fourcast, temps_pangu])
    bins = np.linspace(all_temps.min(), all_temps.max(), 40)

    # Compute histograms (normalized)
    hist_era5, _ = np.histogram(temps_era5, bins=bins, density=True)
    hist_fourcast, _ = np.histogram(temps_fourcast, bins=bins, density=True)
    hist_pangu, _ = np.histogram(temps_pangu, bins=bins, density=True)

    # Bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, hist_era5, label='ERA5 (2020–2025)', color=era5_color, linewidth=3)
    plt.plot(bin_centers, hist_fourcast, label='FourCastNet 9-day', color=fourcast_color, linewidth=3)
    plt.plot(bin_centers, hist_pangu, label='Pangu 9-day', color=pangu_color, linewidth=3)

    # Add 10th and 90th percentiles
    p10_era5 = np.percentile(temps_era5, perc[0])
    p90_era5 = np.percentile(temps_era5, perc[1])
    p10_fc = np.percentile(temps_fourcast, perc[0])
    p90_fc = np.percentile(temps_fourcast, perc[1])
    p10_pangu = np.percentile(temps_pangu, perc[0])
    p90_pangu = np.percentile(temps_pangu, perc[1])


    for val, color, label, hist in zip(
    [p10_era5, p90_era5, p10_fc, p90_fc, p10_pangu, p90_pangu],
    [era5_color, era5_color, fourcast_color, fourcast_color, pangu_color, pangu_color],
    [f"ERA5 {perc[0]}th", f"ERA5 {perc[1]}th", f"FourCast {perc[0]}th", f"FourCast {perc[1]}th", f"Pangu {perc[0]}th", f"Pangu {perc[1]}th"],
    [hist_era5, hist_era5, hist_fourcast, hist_fourcast, hist_pangu, hist_pangu],
):
       #Find the bin index closest to the percentile
        # Interpolate to find the vertical position
        ymax = np.interp(val, bin_centers, hist)

        # Plot vertical dashed line from y=0 to ymax
        plt.vlines(val, 0, ymax, linestyle='--', color=color, linewidth=2)

        # Add vertical label slightly above the line
        if val == p10_era5 or val == p10_fc:
            if p10_fc < p10_era5:
                add_bit = .2 + .5 * (-2*(label[0:4]=="Four"))
            else:
                add_bit = .2 + .5 * (-2*(label[0:4]=="ERA5"))
            
        else:
            add_bit = .2 + .5 * (-2*(label[0:4]=="Four"))
        if label[0:4]=="ERA5":
            if val == p10_era5:
                # Centered horizontally, offset a bit lower vertically
                plt.text(val, -0.001, label, color=color, fontsize=10, fontweight='bold', ha='center', va='top')
            else:
                plt.text(val + add_bit, 0, label, color=color, rotation=90,
                ha='left', va='bottom', fontsize=10, fontweight='bold')
        elif label[0:4]=="Pang" and val == p10_pangu:
            plt.text(val -.8, 0, label, color=color, rotation=90,
            ha='left', va='bottom', fontsize=10, fontweight='bold')

        else:
            plt.text(val + add_bit, 0, label, color=color, rotation=90,
                ha='left', va='bottom', fontsize=10, fontweight='bold')

    plt.xlim(left=268)  # lower limit only

    # Labels and legend
    plt.xlabel('Temperatures (K)')
    plt.ylabel('Probability Density')
    # set x limits
    # plt.xlim(287, 301)
    plt.title('SE U.S. Temperatures', fontsize=16)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    # Remove top and right axes (spines)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/SE_US_temperatures_pdf_{perc[0]}_{perc[1]}_wx.png",
                    bbox_inches='tight', dpi=300)






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



# # # 10th and 90th percentile comparison (original)
era5_2020_2025_t2m_10th = percentile_filtering_by_season(era5_2020_2025.t2m, percentile=-10)
fourcast_9day_t2m_10th = percentile_filtering_by_season(fourcast_9day.t2m, percentile=-10)
pangu_9day_t2m_10th = percentile_filtering_by_season(pangu_9day.t2m, percentile=-10)
fourcast_9day_t2m_90th = percentile_filtering_by_season(fourcast_9day.t2m, percentile=90)
pangu_9day_t2m_90th = percentile_filtering_by_season(pangu_9day.t2m, percentile=90)
era5_2020_2025_t2m_90th = percentile_filtering_by_season(era5_2020_2025.t2m, percentile=90)
plot_comparison(fourcast_9day_t2m_10th, era5_2020_2025_t2m_10th, "t2m", title="FourCastNet 9-day 10th Percentile", maskout = keep_land_mask, vmin=-2, vmax=2, tag="fourcast_9day_10th", pvals= p_vals_fourcast_10_9day)
plot_comparison(fourcast_9day_t2m_90th, era5_2020_2025_t2m_90th, "t2m", title="FourCastNet 9-day 90th Percentile", maskout = keep_land_mask, vmin=-2, vmax=2, tag="fourcast_9day_90th", pvals= p_vals_fourcast_90_9day)
plot_comparison(pangu_9day_t2m_10th, era5_2020_2025_t2m_10th, "t2m", title="Pangu 9-day 10th Percentile", maskout = keep_land_mask, vmin=-2, vmax=2, tag="pangu_9day_10th", pvals= p_vals_pangu_10_9day)
plot_comparison(pangu_9day_t2m_90th, era5_2020_2025_t2m_90th, "t2m", title="Pangu 9-day 90th Percentile", maskout = keep_land_mask, vmin=-2, vmax=2, tag="pangu_9day_90th", pvals= p_vals_pangu_90_9day)


# Repeat for -5 to -80 (by 5) and 50 to 95 (by 5)
# Define highlight boxes for SE US and SE SA
highlight_boxes = [
    (33, 38, 263, 273),   # SE US: lat_min, lat_max, lon_min, lon_max
]
highlight_boxes = [(25, 42, 265, 290),(33, 38, 265, 273)]  # Updated to cover larger SE US region


for i, perc in enumerate([-5, -10, -20]):
    era5_perc = percentile_filtering_by_season(era5_2020_2025.t2m, percentile=perc)
    fourcast_perc = percentile_filtering_by_season(fourcast_9day.t2m, percentile=perc)
    tag = f"FourCastNet_9day_{abs(perc)}th_below"
    pvals = [p_vals_fourcast_5_9day, p_vals_fourcast_10_9day, p_vals_fourcast_20_9day]
    _, mean_diff = plot_comparison(
        fourcast_perc, era5_perc, "t2m",
        title=f"FourCastNet 9-day {abs(perc)}th Percentile and Below",
        highlight_box=highlight_boxes,
        maskout=keep_land_mask, vmin=-2, vmax=2, tag=tag, pvals=pvals[i]
    )

for i, perc in enumerate([95,90,80]):
    era5_perc = percentile_filtering_by_season(era5_2020_2025.t2m, percentile=perc)
    fourcast_perc = percentile_filtering_by_season(fourcast_9day.t2m, percentile=perc)
    tag = f"FourCastNet_9day_{perc}th_above"
    pvals = [p_vals_fourcast_95_9day, p_vals_fourcast_90_9day, p_vals_fourcast_80_9day]
    _, mean_diff = plot_comparison(
        fourcast_perc, era5_perc, "t2m",
        title=f"FourCastNet 9-day {perc}th Percentile and Above", highlight_box=highlight_boxes,
        maskout=keep_land_mask, vmin=-2, vmax=2, tag=tag, pvals=pvals[i]
    )

#as extreme for below percentiles
# Load the datasets for extreme percentiles
for perc in [-5,-10,-20]:
    as_extreme = xr.open_dataset(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/ERA5_as_extreme_{abs(perc)}th_below.nc")
    _, mean_quant = plot_basic(as_extreme.num_quantiles*100, "t2m", cmap='Purples', 
               title=f"FourCastNet Training Data as Extreme as 2020-2025 {abs(perc)}th Percentile", 
               maskout=keep_land_mask, vmin=0, vmax=-3*perc, 
               tag=f"fourcast_era5_{abs(perc)}th_quantile_below", time_mean=False, quantile=True)
    dict_tag = f"pangu_9day_{abs(perc)}th_below"

    
for perc in [95,90,80]:
    as_extreme = xr.open_dataset(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/ERA5_as_extreme_{perc}th_above.nc")
    _, mean_quant = plot_basic(as_extreme.num_quantiles*100, "t2m", cmap='Purples', 
               title=f"FourCastNet Training Data as Extreme as 2020-2025 {perc}th Percentile", 
               maskout=keep_land_mask, vmin=0, vmax=3*(100-perc), 
               tag=f"fourcast_era5_{perc}th_quantile_above", time_mean=False, quantile=True)


# Repeat for -5 to -80 (by 5) and 50 to 95 (by 5)
for j, perc in enumerate([-5,-10,-20]):
    era5_perc = percentile_filtering_by_season(era5_2020_2025.t2m, percentile=perc)
    pangu_perc = percentile_filtering_by_season(pangu_9day.t2m, percentile=perc)
    tag = f"Pangu_9day_{abs(perc)}th_below"
    pvals = [p_vals_pangu_5_9day, p_vals_pangu_10_9day, p_vals_pangu_20_9day]
    _, mean_diff = plot_comparison(
        pangu_perc, era5_perc, "t2m",
        title=f"Pangu 9-day {abs(perc)}th Percentile and Below", highlight_box=highlight_boxes,
        maskout=keep_land_mask, vmin=-2, vmax=2, tag=tag, pvals=pvals[j]
    )

for j, perc in enumerate([95,90,80]):
    era5_perc = percentile_filtering_by_season(era5_2020_2025.t2m, percentile=perc)
    pangu_perc = percentile_filtering_by_season(pangu_9day.t2m, percentile=perc)
    tag = f"Pangu_9day_{perc}th_above"
    pvals = [p_vals_pangu_95_9day, p_vals_pangu_90_9day, p_vals_pangu_80_9day]
    print(pvals)
    _, mean_diff = plot_comparison(
        pangu_perc, era5_perc, "t2m",
        title=f"Pangu 9-day {abs(perc)}th Percentile and Above", highlight_box=highlight_boxes,
        maskout=keep_land_mask, vmin=-2, vmax=2, tag=tag, pvals=pvals[j]
    )
#as extreme for below percentiles
# Load the datasets for extreme percentiles
for perc in [-5,-10,-20]:
    as_extreme = xr.open_dataset(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/ERA5_as_extreme_{abs(perc)}th_below_pangu.nc")
    _, mean_quant = plot_basic(as_extreme.num_quantiles*100, "t2m", cmap='Purples', 
               title=f"Pangu Training Data as Extreme as 2020-2025 {abs(perc)}th Percentile", 
               maskout=keep_land_mask, vmin=0, vmax=-3*perc, 
               tag=f"pangu_era5_{abs(perc)}th_quantile_below", time_mean=False, quantile=True)
    dict_tag = f"pangu_9day_{abs(perc)}th_below"

    
for perc in [95,90,80]:
    as_extreme = xr.open_dataset(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/ERA5_as_extreme_{perc}th_above_pangu.nc")
    _, mean_quant = plot_basic(as_extreme.num_quantiles*100, "t2m", cmap='Purples', 
               title=f"Pangu Training Data as Extreme as 2020-2025 {perc}th Percentile", 
               maskout=keep_land_mask, vmin=0, vmax=3*(100-perc), 
               tag=f"pangu_era5_{perc}th_quantile_above", time_mean=False, quantile=True)







# compared_to_2020_2025 = compute_emd_per_gridpoint(era5_2020_2025.t2m, fourcast_9day.t2m)
# compared_to_2015_2020 = compute_emd_per_gridpoint(era5_2015_2020.t2m, fourcast_9day.t2m)
# compared_to_2010_2015 = compute_emd_per_gridpoint(era5_2010_2015.t2m, fourcast_9day.t2m)
# compared_to_2005_2010 = compute_emd_per_gridpoint(era5_2005_2010.t2m, fourcast_9day.t2m)
# compared_to_2000_2005 = compute_emd_per_gridpoint(era5_2000_2005.t2m, fourcast_9day.t2m)
# compared_to_1995_2000 = compute_emd_per_gridpoint(era5_1995_2000.t2m, fourcast_9day.t2m)
# compared_to_1990_1995 = compute_emd_per_gridpoint(era5_1990_1995.t2m, fourcast_9day.t2m)
# compared_to_1985_1990 = compute_emd_per_gridpoint(era5_1985_1990.t2m, fourcast_9day.t2m)
# compared_to_1980_1985 = compute_emd_per_gridpoint(era5_1980_1985.t2m, fourcast_9day.t2m)
# plot_basic(compared_to_2020_2025, "emd", cmap='hot_r', title="EMD between FourCastNet and ERA5 (2020-2025)", maskout = keep_land_mask, vmin=0, vmax=2, tag="emd_2020_2025", time_mean=False, emd=True)
# plot_basic(compared_to_2015_2020, "emd", cmap='hot_r', title="EMD between FourCastNet and ERA5 (2015-2020)", maskout = keep_land_mask, vmin=0, vmax=2, tag="emd_2015_2020", time_mean=False, emd=True)
# plot_basic(compared_to_2010_2015, "emd", cmap='hot_r', title="EMD between FourCastNet and ERA5 (2010-2015)", maskout = keep_land_mask, vmin=0, vmax=2, tag="emd_2010_2015", time_mean=False, emd=True)
# plot_basic(compared_to_2005_2010, "emd", cmap='hot_r', title="EMD between FourCastNet and ERA5 (2005-2010)", maskout = keep_land_mask, vmin=0, vmax=2, tag="emd_2005_2010", time_mean=False, emd=True)
# plot_basic(compared_to_2000_2005, "emd", cmap='hot_r', title="EMD between FourCastNet and ERA5 (2000-2005)", maskout = keep_land_mask, vmin=0, vmax=2, tag="emd_2000_2005", time_mean=False, emd=True)
# plot_basic(compared_to_1995_2000, "emd", cmap='hot_r', title="EMD between FourCastNet and ERA5 (1995-2000)", maskout = keep_land_mask, vmin=0, vmax=2, tag="emd_1995_2000", time_mean=False, emd=True)
# plot_basic(compared_to_1990_1995, "emd", cmap='hot_r', title="EMD between FourCastNet and ERA5 (1990-1995)", maskout = keep_land_mask, vmin=0, vmax=2, tag="emd_1990_1995", time_mean=False, emd=True)
# plot_basic(compared_to_1985_1990, "emd", cmap='hot_r', title="EMD between FourCastNet and ERA5 (1985-1990)", maskout = keep_land_mask, vmin=0, vmax=2, tag="emd_1985_1990", time_mean=False, emd=True)
# plot_basic(compared_to_1980_1985, "emd", cmap='hot_r', title="EMD between FourCastNet and ERA5 (1980-1985)", maskout = keep_land_mask, vmin=0, vmax=2, tag="emd_1980_1985", time_mean=False, emd=True)


