import xarray as xr
import numpy as np

from matplotlib import font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, to_rgb
# Path to your font files
font_path_regular = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-VariableFont_wght.ttf"
font_path_italic = "/home/jlandsbe/ai_weather_to_climate_ats780A8/Fonts/RedHatDisplay-Italic-VariableFont_wght.ttf"

# Register fonts
fm.fontManager.addfont(font_path_regular)
fm.fontManager.addfont(font_path_italic)

# Set global font family to Red Hat Display
mpl.rcParams['font.family'] = 'Red Hat Display'
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
        mean = da.mean(dim='time')
    else:
        # Compute threshold
        thresh = da.quantile(abs(percentile) / 100, dim='time')
        
        if percentile > 0:
            mask = da >= thresh
        else:
            mask = da <= thresh

        # Mask and compute mean over time
        masked = da.where(mask)
        mean = masked.mean(dim='time')
    return mean

def thresholded_yearly_mean(ds, var_name, percentile=0, months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], start_year = 1940, end_year = 2025, chunk_size = 1):
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
        mean = da.mean(dim='time')
        if 'sample' in da.dims:
            mean = mean.mean(dim='sample')
        return mean
    # Define chunk
    years = da['time'].dt.year
    chunk = ((years - years.min()) // chunk_size) * chunk_size + years.min()

    # Now group by chunk
    grouped = da.groupby(chunk)

    results = []

    for y, data in grouped:
        # Collapse 'sample' dimension if it exists

        if percentile == 0:
            # No thresholding: simple mean over time
            mean = data.mean(dim='time')
        else:
            # Compute threshold
            thresh = data.quantile(abs(percentile) / 100, dim='time')
            
            if percentile > 0:
                mask = data >= thresh
            else:
                mask = data <= thresh

            # Mask and compute mean over time
            masked = data.where(mask)
            mean = masked.mean(dim='time')
        # Attach year coordinate
        mean = mean.expand_dims(year=[y])
        results.append(mean)

    # Combine all years
    result = xr.concat(results, dim='year')
    return result


def make_custom_colormap(low_color="#FE5F55", mid_color="#FCDFA6", high_color="#4CB963", n_colors=4, power=2.0):
    """
    Creates a custom colormap with more gradation at the high end.
    
    Parameters:
    - low_color, mid_color, high_color: colors at bottom, middle, and top.
    - n_colors: total number of colors in the colormap.
    - power: controls gradation skew. >1 = more emphasis at top, <1 = more at bottom.
    """
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
                   n_colors=4, variable=None, title="", maskout="ocean", vmin=None, vmax=None,
                   tag="", time_mean=False, pvals=None):


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
    # Optionally average over time
    if time_mean:
        plot_data = plot_data.mean(dim='time', skipna=True)

    # Apply mask if provided
    if maskout is not None:

        ocean_maskout = create_land_ocean_masks(maskout_type="ocean")
        print("Applying maskout:", ocean_maskout)
        plot_data = plot_data * ocean_maskout



    # Compute weighted mean (global)
    weights = np.cos(np.deg2rad(plot_data.lat))
    weights.name = "weights"
    weighted = plot_data.weighted(weights)
    mean_diff = weighted.mean(dim=('lat', 'lon')).values


    # Compute weighted mean over contiguous US
    # Define lat/lon bounds for CONUS (approx: 24-50N, 235-295E)
    lat_min_us, lat_max_us = 25, 42
    lon_min_us, lon_max_us = 265, 290

    # Draw CONUS box on the map in black
    box_lats = [lat_min_us, lat_max_us, lat_max_us, lat_min_us, lat_min_us]
    box_lons = [lon_min_us, lon_min_us, lon_max_us, lon_max_us, lon_min_us]
    ax.plot(box_lons, box_lats, color='black', linewidth=2, transform=ccrs.PlateCarree(), zorder=10)
    if pvals is not None:
        if pvals.shape != plot_data.shape:
            raise ValueError("pvals must have same shape as data being plotted.")

        sig_mask = pvals < 0.1

        # Convert mask → coordinates
        y_idx, x_idx = np.where(sig_mask)

        # Get actual lon/lat arrays
        lats = plot_data.lat.values
        lons = plot_data.lon.values

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
    # ============================================================

    # Subset data for CONUS
    plot_data_us = plot_data.sel(
        lat=slice(lat_min_us, lat_max_us),
        lon=slice(lon_min_us, lon_max_us)
    )

    # Compute weights for CONUS
    weights_us = np.cos(np.deg2rad(plot_data_us.lat))
    weights_us.name = "weights"
    weighted_us = plot_data_us.weighted(weights_us)
    mean_contiguous_us = weighted_us.mean(dim=('lat', 'lon')).values

    print(plot_data)
    # Plot the data
    im = plot_data.plot(
    ax=ax,
    cmap=cmap,
    vmin=0,
    vmax=n_colors,
    add_colorbar=False,
    transform=ccrs.PlateCarree()  
)


    # Colorbar setup
    tick_positions = np.arange(n_colors) + 0.5  # centers of color patches
    tick_labels = ['1980-1985', '1985-1990', '1990-1995', '1995-2000', '2000-2005', '2005-2010', '2010-2015', '2015-2020', '2020-2025']
    tick_labels = ['1951-1965','1966-1980','1981-1995','1996-2010']

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05,
                        boundaries=np.arange(n_colors + 1),
                        ticks=tick_positions)

    cbar.set_label("Best Matching Time Period", fontsize=12)
    # Add mean marker to colorbar
    mean_index = mean_diff  # This should already be in the range [0, n_colors]
    cbar_ax = cbar.ax
    print(mean_index)
    # Plot a triangle marker at the mean index
    mean_index_us = mean_contiguous_us
    print(mean_index_us)
    cbar_ax.plot([mean_index_us + 0.5], .87, marker='v', color='black', clip_on=True, transform=cbar_ax.transData)
    cbar_ax.text(mean_index_us + 0.5, 1.3, "E U.S.", ha='center', va='top', fontsize=10, fontweight=800, transform=cbar_ax.transData)

    cbar_ax.plot([mean_index + 0.5], .87, marker='v', color='black', clip_on=True, transform=cbar_ax.transData)
    # Add label below the marker
    cbar_ax.text(mean_index + 0.5, 1.3, "Global Mean", ha='center', va='top', fontsize=10, fontweight=800 ,transform=cbar_ax.transData)

    cbar.ax.set_xticklabels(tick_labels)
    cbar.ax.tick_params(which='both', length=0)
        # Alternate font size (e.g., large for even indices, small for odd)

    # Set all tick label font sizes to 10
    for label in cbar.ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_ha('center')  # Optional, for horizontal alignment
    plt.setp(cbar.ax.get_xticklabels(), rotation=0, ha='center')
           # Add map features
    ax.set_title(title, fontsize=14)
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")

    # Save figure
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/ace_best_year_plot_{tag}.png",
                bbox_inches='tight', dpi=300)
    plt.close()

def time_series_plot(xlabels, ydata, ylabel, title, tag, vmin=None, vmax=None, abs = 0, xlab="Time"):
    """
    Create a time series plot with specified x-axis labels and y-axis data.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if not abs:
        ax.plot(xlabels, ydata, marker='o', linestyle='-', color='seagreen', linewidth=2, markersize=8, label='Difference in Global Mean Temperature')
    else:
                # Plot the line connecting all points
        ax.plot(xlabels, np.abs(ydata), linestyle='-', color='#102e4a', linewidth=4, label='Absolute Difference in Global Mean Temperature')

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





def plot_trend_map(data, variable="t2m", title="", cmap='RdBu_r', maskout=True, vmin=None, vmax=None, save_name="", time_mean=False, emd=False, trend=False, quantile=False, rectangles=None, pvals=None):
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

    if maskout:
        ocean_maskout = create_land_ocean_masks(maskout_type="ocean")
        plot_data = plot_data*ocean_maskout
    #weight by cosing of latitude and calculate mean
    weights = np.cos(np.deg2rad(plot_data.lat))
    weights.name = "weights"
    difference_weighted = plot_data.weighted(weights)
    mean_diff = difference_weighted.mean(dim=('lat','lon')).values
    #plot the difference
    im = plot_data.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False,transform=ccrs.PlateCarree())
    if pvals is not None:
        if pvals.shape != plot_data.shape:
            raise ValueError("pvals must have same shape as data being plotted.")

        sig_mask = pvals < 0.1

        # Convert mask → coordinates
        y_idx, x_idx = np.where(sig_mask)

        # Get actual lon/lat arrays
        lats = plot_data.lat.values
        lons = plot_data.lon.values

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
    # ============================================================

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
        cbar.set_label(f"Temperature Difference from ERA5", fontsize=12)
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
    if rectangles is not None:
        if isinstance(rectangles[0], (list, tuple)):  # multiple boxes
            boxes = rectangles
        else:  # single box
            boxes = [rectangles]

        for box in boxes:
            lat_min, lat_max, lon_min, lon_max = box
            box_lats = [lat_min, lat_max, lat_max, lat_min, lat_min]
            box_lons = [lon_min, lon_min, lon_max, lon_max, lon_min]
            ax.plot(box_lons, box_lats, color='black', linewidth=2,
                    transform=ccrs.PlateCarree(), zorder=10)
    plt.savefig(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/mean_{save_name}.png", bbox_inches='tight', dpi=300)
    return plot_data


era5_aligned = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ERA5_processed.nc")
ace2_aligned = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ACE2_processed.nc")
ace2_first_member = ace2_aligned.isel(sample=0)


# New chunk_size and year ranges: 10-year chunks, 10-year intervals

# era5_1940s_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1940, end_year=1950, chunk_size=10, months=[1, 2, 12])
# era5_1950s_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1950, end_year=1960, chunk_size=10, months=[1, 2, 12])
# era5_1960s_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1960, end_year=1970, chunk_size=10, months=[1, 2, 12])
# era5_1970s_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1970, end_year=1980, chunk_size=10, months=[1, 2, 12])
# era5_1980s_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1980, end_year=1990, chunk_size=10, months=[1, 2, 12])
# era5_1990s_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1990, end_year=2000, chunk_size=10, months=[1, 2, 12])
# era5_2000s_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=2000, end_year=2010, chunk_size=10, months=[1, 2, 12])
# era5_2010s_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=2010, end_year=2020, chunk_size=10, months=[1, 2, 12])

# ace2_1940s_vals = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, start_year=1940, end_year=1950, chunk_size=10, months=[1, 2, 12])
# ace2_1950s_vals = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, start_year=1950, end_year=1960, chunk_size=10, months=[1, 2, 12])
# ace2_1960s_vals = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, start_year=1960, end_year=1970, chunk_size=10, months=[1, 2, 12])
# ace2_1970s_vals = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, start_year=1970, end_year=1980, chunk_size=10, months=[1, 2, 12])
# ace2_1980s_vals = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, start_year=1980, end_year=1990, chunk_size=10, months=[1, 2, 12])
# ace2_1990s_vals = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, start_year=1990, end_year=2000, chunk_size=10, months=[1, 2, 12])
# ace2_2000s_vals = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, start_year=2000, end_year=2010, chunk_size=10, months=[1, 2, 12])
# ace2_2010s_vals = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, start_year=2010, end_year=2020, chunk_size=10, months=[1, 2, 12])

# # Plotting individual datasets
# for label in ["1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]:
#     plot_trend_map(eval(f"era5_{label}_vals"), cmap='RdBu_r', vmin=None, vmax=None, title=f"ERA5 {label} Mean Winter Temps", save_name=f"JBL_era5_{label}_mean_map.png")
#     plot_trend_map(eval(f"ace2_{label}_vals"), cmap='RdBu_r', vmin=None, vmax=None, title=f"ACE2 {label} Mean Winter Temps", save_name=f"JBL_ace2_{label}_mean_map.png")

# # Plotting ERA5 vs ACE2 differences
# for label in ["1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]:
#     plot_trend_map(eval(f"ace2_{label}_vals - era5_{label}_vals"), cmap='RdBu_r', vmin=-2, vmax=2, title=f"ACE2 - ERA5 {label} Mean Winter Temps", save_name=f"JBL_era5_ace2_{label}_mean_map.png")

# Define time periods and percentiles
time_periods = [(1951,1966),(1966,1981),(1981,1996),(1996,2011), (2000,2011)]
#percentiles = [0,-10, 90]
percentiles = [0,-10, 90]
season = "winter"
if season =="summer":
    months_to_include = [6,7,8]
elif season == "year-round":
    months_to_include = [1,2,3,4,5,6,7,8,9,10,11,12]
elif season == "winter":
    months_to_include = [1,2,12]
    season = "winter"
#Storage for results if needed later
results = {}

# Loop through periods and percentiles
for start_year, end_year in time_periods:
    period_label = f"{start_year}_to_{end_year}"
    results[period_label] = {}

    for percentile in percentiles:
        percentile_label = (
            "all_vals" if percentile == 0 else
            str(abs(percentile)) + "th_percentile"
        )

        # Compute thresholded yearly means
        era5_vals = thresholded_mean(
            era5_aligned, 'surface_temperature',
            percentile=percentile, start_year=start_year, end_year=end_year, months=months_to_include
        )
        ace2_vals = thresholded_mean(
            ace2_aligned, 'surface_temperature',
            percentile=percentile, start_year=start_year, end_year=end_year, months=months_to_include
        )
        # Save results if needed later
        results[period_label][percentile_label] = (era5_vals, ace2_vals)
        # title_percentile = (
        #     "Mean" if percentile == 0 else
        #     "10th Percentile" if percentile == -10 else
        #     "90th Percentile"
        # )
        # print(f"Plotting {period_label} {percentile_label} difference map...")
        # plot_trend_map(
        #     diff, cmap='RdBu_r', vmin=-2, vmax=2,
        #     title=f"ACE2 vs ERA5 {period_label.replace('_', ' ')} {title_percentile} Winter Temps",
        #     save_name=f"JBL_era5_ace2_{period_label}_{percentile_label}_map.png"
        # )

target_period_ace_test = "1996_to_2011"
#target_percentile_list = ["all_vals", "10th_percentile", "90th_percentile"]
target_percentile_list = ["all_vals", "10th_percentile", "90th_percentile"]
diff_list = []
diff_list_means = []

# Extract ace2_vals for the target period
target_ace2 = results[target_period_ace_test]["all_vals"][1]
target_ace2 = target_ace2.mean(dim="sample")

# Build list of diffs in the same order as time_periods
ordered_periods = [
    "1951_to_1966",
    "1966_to_1981",
    "1981_to_1996",
    "1996_to_2011"
]


for period in ordered_periods:
    era5_vals = results[period]["all_vals"][0]
    map_data = target_ace2 - era5_vals
    diff_list.append(map_data)
    ocean_maskout = create_land_ocean_masks(maskout_type="ocean")
    map_data = map_data * ocean_maskout
    weights = np.cos(np.deg2rad(map_data.lat))
    weights.name = "weights"
    weighted = map_data.weighted(weights)
    mean_diff = weighted.mean(dim=('lat', 'lon')).values
    diff_list_means.append(mean_diff)


for target_percentile in target_percentile_list:
    target_ace2 = results[target_period_ace_test][target_percentile][1]
    target_era5 = results[target_period_ace_test][target_percentile][0]
    bias_map = target_ace2 - target_era5
    bias_map = bias_map.mean(dim="sample")
    print(target_percentile)
    print(season)
    if target_percentile == "all_vals":
        if season == "summer":
            title = "ACE2 Summer Mean Temperature Difference (1996-2010)"
        else:
            title = "ACE2 Mean Temperature Difference (1996-2010)"
    elif target_percentile == "10th_percentile":
        title = "ACE2 10th Percentile Temperature Difference (1996-2010)"
    else:
        title = "ACE2 90th Percentile Temperature Difference (1996-2010)"
    print(title)
    if season == "summer":
        sttl = "JJA"
    elif season == "winter":
        sttl = "DJF"
    else:
        print("no season")
    if target_percentile == "all_vals":
        target_percentile_num = 0
    elif target_percentile == "10th_percentile":
        target_percentile_num = 10
    elif target_percentile == "90th_percentile":
        target_percentile_num = 90

    pvals = np.load(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ace2_p_values_{sttl}_{target_percentile_num}.npy")
    pvals = pvals * xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/low_res_poles_mask.nc").__xarray_dataarray_variable__.values
    plot_trend_map(bias_map, cmap='RdBu_r', vmin=-2, vmax=2, title=title, save_name=f"ace2_{season}_months_{target_percentile}_mean_map", pvals=pvals)


##uncomment here on
print("Mean differences for each period:", diff_list_means)
# Assume you have a list of DataArrays with the same lat/lon dims
# Stack them into a new dimension, e.g., 'index'
stacked = xr.concat(diff_list, dim='index')
stacked = np.abs(stacked)


# Create mask where all values are NaN at each gridpoint
all_nan_mask = stacked.isnull().all(dim='index')

# Fill NaNs with +inf before calling argmin (inf will never be selected unless all are inf)
stacked_filled = stacked.fillna(np.inf)

# Compute argmin safely
min_index = stacked_filled.argmin(dim='index')

# Set index to NaN where all were NaN
min_index = min_index.where(~all_nan_mask)
pvals = np.load(f"/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/sig_data/ace2_p_values_{sttl}_{0}.npy")
pvals = pvals*xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/low_res_poles_mask.nc").__xarray_dataarray_variable__.values

plot_emd_years(min_index, variable="surface_temperature", title=f"ACE2 {season.capitalize()} Best Matching Time Periods",
               maskout="Ocean", vmin=0, vmax=3, tag=f"best_matching_time_period_surface_temperature_ace_{season}", pvals=pvals)
xlabs = ['1951-1965','1966-1980','1981-1995','1996-2010']
time_series_plot(xlabels=xlabs, ydata=diff_list_means, ylabel="Mean Difference (K)", title=f"ACE {season.capitalize()} Absolute Global Mean Temperature Difference", tag=f"ace_{season}_time_series_mean_diff_abs", abs=1, vmax=1)


##era5 test

# early_era5_summer = thresholded_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1940, end_year=1979, months=[6,7,8])
# early_era5_winter = thresholded_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1940, end_year=1979, months=[1,2,12])
# early_era5_winter_10 = thresholded_mean(era5_aligned, 'surface_temperature', percentile=-10, start_year=1940, end_year=1979, months=[1,2,12])
# early_era5_winter_90 = thresholded_mean(era5_aligned, 'surface_temperature', percentile=90, start_year=1940, end_year=1979, months=[1,2,12])
# late_era5_summer = thresholded_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1980, end_year=2022, months=[6,7,8])
# late_era5_winter = thresholded_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1980, end_year=2022, months=[1,2,12])
# late_era5_winter_10 = thresholded_mean(era5_aligned, 'surface_temperature', percentile=-10, start_year=1980, end_year=2022, months=[1,2,12])
# late_era5_winter_90 = thresholded_mean(era5_aligned, 'surface_temperature', percentile=90, start_year=1980, end_year=2022, months=[1,2,12])
# early_era5_year_round = thresholded_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1940, end_year=1979, months=[1,2,3,4,5,6,7,8,9,10,11,12])
# late_era5_year_round = thresholded_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1980, end_year=2022, months=[1,2,3,4,5,6,7,8,9,10,11,12])
# year_round_diff = late_era5_year_round - early_era5_year_round
# summer_diff = late_era5_summer - early_era5_summer
# winter_diff = late_era5_winter - early_era5_winter
# winter_diff_10 = late_era5_winter_10 - early_era5_winter_10
# winter_diff_90 = late_era5_winter_90 - early_era5_winter_90
# ###plotting these and their differences
# plot_trend_map(summer_diff, cmap='RdBu_r', vmin=-2.25, vmax=2.25, title="ERA5 Change in Mean Summer Temperature", save_name="ace_summer_era5_climate_change.png")
# plot_trend_map(winter_diff, cmap='RdBu_r', vmin=-2.25, vmax=2.25, title="ERA5 Change in Mean Winter Temperature", save_name="ace_winter_era5_climate_change.png")
# plot_trend_map(winter_diff_10 - winter_diff_90, cmap='RdBu_r', vmin=-2.25, vmax=2.25, title="ERA5 Change in Winter 10th vs 90th Percentile Temperature", save_name="ace_winter_10th_vs_90th_percentile_era5_climate_change.png")
# plot_trend_map(year_round_diff, cmap='RdBu_r', vmin=-2.25, vmax=2.25, title="ERA5 Change in Mean Year-Round Temperature", save_name="ace_year_round_era5_climate_change.png")
# plot_trend_map(summer_diff - year_round_diff, cmap='RdBu_r', vmin=-.5, vmax=.5, title="ERA5 Summer Minus Annual Change in Temperature", save_name="ace_relative_summer_era5_climate_change.png")
# plot_trend_map(winter_diff - year_round_diff, cmap='RdBu_r', vmin=-.5, vmax=.5, title="ERA5 Winter Minus Annual Change in Temperature", save_name="ace_relative_winter_era5_climate_change.png")
#plot_trend_map(winter_diff - summer_diff, cmap='RdBu_r', vmin=-.5, vmax=.5, title="Relative Winter vs. Summer Seasonal Change in Temps", save_name="winter_summer_comparison_climate_change.png")











# era5_40s_to_60s_all_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1940, end_year=1960, chunk_size = 20, months = [1,2,12])
# era5_60s_to_80s_all_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1960, end_year=1980, chunk_size = 20, months = [1,2,12])
# era5_80s_to_00s_all_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1980, end_year=2000, chunk_size = 20, months = [1,2,12])
# era5_00s_to_20s_all_vals = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=2000, end_year=2020, chunk_size = 20, months = [1,2,12])
# ace2_40s_to_60s_all_vals_one_member = thresholded_yearly_mean(ace2_first_member, 'surface_temperature', percentile=0, start_year=1940, end_year=1960, chunk_size = 20, months = [1,2,12])
# ace2_60s_to_80s_all_vals_one_member = thresholded_yearly_mean(ace2_first_member, 'surface_temperature', percentile=0, start_year=1960, end_year=1980, chunk_size = 20, months = [1,2,12])
# ace2_80s_to_00s_all_vals_one_member = thresholded_yearly_mean(ace2_first_member, 'surface_temperature', percentile=0, start_year=1980, end_year=2000, chunk_size = 20, months = [1,2,12])
# ace2_00s_to_20s_all_vals_one_member = thresholded_yearly_mean(ace2_first_member, 'surface_temperature', percentile=0, start_year=2000, end_year=2020, chunk_size = 20, months = [1,2,12])
# ##plotting these and their differences
# plot_trend_map(era5_40s_to_60s_all_vals, cmap='RdBu_r', vmin=None, vmax=None, title="ERA5 40s to 60s mean temps", save_name="JBL_era5_40s_to_60s_mean_map.png")
# plot_trend_map(era5_60s_to_80s_all_vals, cmap='RdBu_r', vmin=None, vmax=None, title="ERA5 60s to 80s mean temps", save_name="JBL_era5_60s_to_80s_mean_map.png")
# plot_trend_map(era5_80s_to_00s_all_vals, cmap='RdBu_r', vmin=None, vmax=None, title="ERA5 80s to 00s mean temps", save_name="JBL_era5_80s_to_00s_mean_map.png")
# plot_trend_map(era5_00s_to_20s_all_vals, cmap='RdBu_r', vmin=None, vmax=None, title="ERA5 00s to 20s mean temps", save_name="JBL_era5_00s_to_20s_mean_map.png")
# plot_trend_map(ace2_40s_to_60s_all_vals_one_member, cmap='RdBu_r', vmin=None, vmax=None, title="ACE2 40s to 60s mean temps", save_name="JBL_ace2_40s_to_60s_mean_map.png")
# plot_trend_map(ace2_60s_to_80s_all_vals_one_member, cmap='RdBu_r', vmin=None, vmax=None, title="ACE2 60s to 80s mean temps", save_name="JBL_ace2_60s_to_80s_mean_map.png")
# plot_trend_map(ace2_80s_to_00s_all_vals_one_member, cmap='RdBu_r', vmin=None, vmax=None, title="ACE2 80s to 00s mean temps", save_name="JBL_ace2_80s_to_00s_mean_map.png")
# plot_trend_map(ace2_00s_to_20s_all_vals_one_member, cmap='RdBu_r', vmin=None, vmax=None, title="ACE2 00s to 20s mean temps", save_name="JBL_ace2_00s_to_20s_mean_map.png")
# # # Plot the differences between the two datasets for each period
# plot_trend_map(era5_40s_to_60s_all_vals - ace2_40s_to_60s_all_vals_one_member, cmap='RdBu_r', vmin=-2, vmax=2, title="ERA5 vs ACE2 40s to 60s mean temps", save_name="JBL_era5_ace2_40s_to_60s_mean_map.png")
# plot_trend_map(era5_60s_to_80s_all_vals - ace2_60s_to_80s_all_vals_one_member, cmap='RdBu_r', vmin=-2, vmax=2, title="ERA5 vs ACE2 60s to 80s mean temps", save_name="JBL_era5_ace2_60s_to_80s_mean_map.png")
# plot_trend_map(era5_80s_to_00s_all_vals - ace2_80s_to_00s_all_vals_one_member, cmap='RdBu_r', vmin=-2, vmax=2, title="ERA5 vs ACE2 80s to 00s mean temps", save_name="JBL_era5_ace2_80s_to_00s_mean_map.png")
# plot_trend_map(era5_00s_to_20s_all_vals - ace2_00s_to_20s_all_vals_one_member, cmap='RdBu_r', vmin=-2, vmax=2, title="ERA5 vs ACE2 00s to 20s mean temps", save_name="JBL_era5_ace2_00s_to_20s_mean_map.png")
# # The following code is commented out because it is not currently being used.
# # It can be uncommented and used for further analysis if needed.

# thresholded_yearly_mean_all_era5 = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1980, chunk_size = 5, months = [1,2,12])
# thresholded_yearly_mean_all_ace2= thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, start_year=1980, chunk_size = 5, months = [1,2,12])
# thresholded_yearly_mean_10_era5 = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=-10, start_year=1980, chunk_size = 5, months = [1,2,12])
# thresholded_yearly_mean_10_ace2 = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=-10, start_year=1980, chunk_size = 5, months = [1,2,12])

# era5_late_relative_10 = thresholded_yearly_mean_10_era5 - thresholded_yearly_mean_all_era5 #how much faster 10th percentile has been warming compared to mean in ERA5 recently
# ace2_late_relative_10 = thresholded_yearly_mean_10_ace2 - thresholded_yearly_mean_all_ace2 #how much faster 10th percentile has been warming compared to mean in ACE2 recently

# early_era5_mean_all = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, end_year=1980, chunk_size = 5, months = [1,2,12])
# early_era5_10 = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=-10, end_year=1980, chunk_size = 5, months = [1,2,12])
# era5_early_relative_10 = early_era5_10.mean(dim="year") - early_era5_mean_all.mean(dim="year")

# early_ace2_mean_all = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, end_year=1980, chunk_size = 5, months = [1,2,12])
# early_ace2_10 = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=-10, end_year=1980, chunk_size = 5, months = [1,2,12])
# ace2_early_relative_10 = early_ace2_10.mean(dim="year") - early_ace2_mean_all.mean(dim="year")



# late_to_early_era5 = era5_late_relative_10 - era5_early_relative_10 
# # print("Late vs. Early ERA5")
# # print(late_to_early_era5)
# late_to_early_ace2 = ace2_late_relative_10 - ace2_early_relative_10

# late_to_early_era5_vs_ace2 = late_to_early_era5 - late_to_early_ace2

# #thresholded_yearly_mean_90_era5 = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=90, start_year = 1980, chunk_size = 5, months = [1,2,12])
# # thresholded_yearly_mean_90_ace2 = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=90, start_year = 1980, chunk_size = 5, months = [1,2,12])


# # relative_trend_era5_10 = thresholded_yearly_mean_10_era5 - thresholded_yearly_mean_all_era5
# # # relative_trend_era5_90 = thresholded_yearly_mean_90_era5 - thresholded_yearly_mean_all_era5
# # relative_trend_ace2_10 = thresholded_yearly_mean_10_ace2 - thresholded_yearly_mean_all_ace2
# # # relative_trend_ace2_90 = thresholded_yearly_mean_90_ace2 - thresholded_yearly_mean_all_ace2

# # era5_to_ace2_10_trend = relative_trend_era5_10 - relative_trend_ace2_10
# # # era5_to_ace2_90_trend = relative_trend_era5_90 - relative_trend_ace2_90

# rectangles_to_plot = [
#     #{'lon_min': 0, 'lon_max': 36, 'lat_min': 42, 'lat_max': 70, 'label': "W Europe"},
#     {'lon_min': 194, 'lon_max': 265, 'lat_min': 55, 'lat_max': 85, 'label': "High Latitudes"},
#     {'lon_min': 236, 'lon_max': 284, 'lat_min': 39, 'lat_max': 53, 'label':'N U.S./S CA'},
#     {'lon_min': 90, 'lon_max': 150, 'lat_min': 50, 'lat_max': 80, 'label':'E Russia'},
#     #{'lon_min': 93, 'lon_max': 120, 'lat_min': 9, 'lat_max': 29, 'label': "SE China", 'color': 'magenta'},
#     #{'lon_min': 285, 'lon_max': 307, 'lat_min': -55, 'lat_max': -18, 'label':'S America','color': 'magenta'},
# ]

# # plot_trend_map(late_to_early_era5, cmap='RdBu_r', vmin=None, vmax=None, title="Late vs. early relative temps", save_name="era5_late_early_map_10.png", rectangles=rectangles_to_plot)
# # plot_trend_map(late_to_early_ace2, cmap='RdBu_r', vmin=None, vmax=None, title="Late vs. early relative temps", save_name="ace2_late_early_map_10.png", rectangles=rectangles_to_plot)
# # plot_trend_map(late_to_early_era5_vs_ace2, cmap='RdBu_r', vmin=None, vmax=None, title="Late vs. early relative temps", save_name="era5_ace2_late_early_map_10.png", rectangles=rectangles_to_plot)
# # plot_trend_map(thresholded_yearly_mean_all_era5, cmap='RdBu_r', vmin=None, vmax=None, title="Mean temps (recent)", save_name="era5_mean_map_recent.png", rectangles=rectangles_to_plot)
# # plot_trend_map(early_era5_mean_all, cmap='RdBu_r', vmin=None, vmax=None, title="Mean temps (old)", save_name="era5_mean_map_old.png", rectangles=rectangles_to_plot)
# # plot_trend_map(thresholded_yearly_mean_10_era5, cmap='RdBu_r', vmin=None, vmax=None, title="10th percentile temps (recent)", save_name="era5_10_map_recent.png", rectangles=rectangles_to_plot)
# # plot_trend_map(early_era5_10, cmap='RdBu_r', vmin=None, vmax=None, title="10th percentile temps (old)", save_name="era5_10_map_old.png", rectangles=rectangles_to_plot)



# # plot_trend_map(thresholded_yearly_mean_all_era5.mean(dim='year') - early_era5_mean_all.mean(dim="year"), cmap='RdBu_r', vmin=-2.5, vmax=2.5, title="Mean temps (recent vs old)", save_name="era_mean_comparison_map_recent.png", rectangles=rectangles_to_plot)
# # plot_trend_map(thresholded_yearly_mean_10_era5.mean(dim='year') - early_era5_10.mean(dim="year"), cmap='RdBu_r', vmin=-2.5, vmax=2.5, title="10th percentile temps (recent vs old)", save_name="era_10_comparison_map_recent.png", rectangles=rectangles_to_plot)


# era5_10_to_mean_relative_change = (thresholded_yearly_mean_10_era5.mean(dim='year') - early_era5_10.mean(dim="year")) - (thresholded_yearly_mean_all_era5.mean(dim='year') - early_era5_mean_all.mean(dim="year"))
# ace2_10_to_mean_relative_change = (thresholded_yearly_mean_10_ace2.mean(dim='year') - early_ace2_10.mean(dim="year")) - (thresholded_yearly_mean_all_ace2.mean(dim='year') - early_ace2_mean_all.mean(dim="year"))
# plot_trend_map(era5_10_to_mean_relative_change, cmap='RdBu_r', vmin=-1, vmax=1, title="Change in 10th percentile - change in mean (recent vs old)", save_name="era_10_to_mean_comparison_map.png", rectangles=rectangles_to_plot)
# plot_trend_map(ace2_10_to_mean_relative_change, cmap='RdBu_r', vmin=-1, vmax=1, title="Change in 10th percentile - change in mean (recent vs old)", save_name="ace2_10_to_mean_comparison_map.png", rectangles=rectangles_to_plot)
# plot_trend_map(era5_10_to_mean_relative_change-ace2_10_to_mean_relative_change, cmap='RdBu_r', vmin=-1, vmax=1, title="Change in 10th percentile - change in mean (ERA5 vs ACE2)", save_name="era5_ace2_10_to_mean_comparison_map.png", rectangles=rectangles_to_plot)

ace_mask = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/ai2ace/low_res_poles_mask.nc").__xarray_dataarray_variable__.values
era5_vals = thresholded_mean(
    era5_aligned, 'surface_temperature',
    percentile=0, start_year=1996, end_year=2011, months=[12,1,2]
)
ace2_vals = thresholded_mean(
    ace2_aligned, 'surface_temperature',
    percentile=0, start_year=1996, end_year=2011, months=[12,1,2]
)
ens = (ace2_vals*ace_mask)
truth = (era5_vals*ace_mask)
# ---- Compute ensemble mean ----
ens_mean = ens.mean(dim="sample")

# ---- Compute bias (systematic error) ----
bias = ens_mean - truth

# ---- Compute ensemble RMSE ----
squared_errors = (ens - truth)**2
rmse = squared_errors.mean(dim="sample")**0.5

#take weighted global mean#
weights = np.cos(np.deg2rad(rmse.lat))
weights.name = "weights"
weighted = rmse.weighted(weights)
rmse = weighted.mean(dim=('lat', 'lon')).values
weighted = bias.weighted(weights)
bias = weighted.mean(dim=('lat', 'lon')).values

# ---- Bias fraction of RMSE ----
bias_fraction = abs(bias) / rmse   # values near 1 → bias-dominated



print(f"Bias fraction: {bias_fraction}")
print(f"Bias: {bias}")
print(f"RMSE: {rmse}")