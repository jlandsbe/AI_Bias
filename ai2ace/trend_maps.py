import xarray as xr
import numpy as np

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
    
    # Define chunk
    years = da['time'].dt.year
    chunk = ((years - years.min()) // chunk_size) * chunk_size + years.min()

    # Now group by chunk
    grouped = da.groupby(chunk)

    results = []

    for y, data in grouped:
        # Collapse 'sample' dimension if it exists
        if 'sample' in data.dims:
            data = data.mean(dim='sample')

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

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_trend_map(thresholded_data, cmap='RdBu_r', vmin=None, vmax=None, title="Trend per Grid Point", save_name="", rectangles=None):
    """
    Compute and plot the linear trend at each grid point using Cartopy.

    Parameters
    ----------
    thresholded_data : xarray.DataArray
        DataArray with dimensions (year, lat, lon).
    cmap : str
        Colormap for plotting.
    vmin, vmax : float or None
        Limits for the color scale.
    title : str
        Title of the plot.
    save_name : str
        Name for saving the plot.
    rectangles : list of dict or None
        List of rectangles to plot, where each rectangle is defined as a dictionary with keys:
        - 'lon_min', 'lon_max', 'lat_min', 'lat_max' for the rectangle bounds.
        - 'color' (optional) for the rectangle edge color.
        - 'label' (optional) for the rectangle label.
    """

    years = thresholded_data['year'].values

    # Stack lat/lon for polyfit
    stacked = thresholded_data.stack(points=("lat", "lon"))

    # Compute trend (slope) at each point
    slopes = np.polyfit(years, stacked.values, deg=1)[0]  # slope only

    # Reshape back to (lat, lon)
    slope_map = xr.DataArray(
        slopes.reshape((thresholded_data.sizes['lat'], thresholded_data.sizes['lon'])),
        coords={"lat": thresholded_data.lat, "lon": thresholded_data.lon},
        dims=["lat", "lon"],
    )
    ocean_maskout = create_land_ocean_masks(maskout_type="ocean")
    slope_map = slope_map*ocean_maskout
    print("ocean masked out")
    # Plot
    fig = plt.figure(figsize=(14, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set color limits if not provided
    if vmin is None:
        vmin = np.nanpercentile(slopes, 5)
    if vmax is None:
        vmax = np.nanpercentile(slopes, 95)

    lim = max(abs(vmin), abs(vmax))

    im = slope_map.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=-lim,
        vmax=lim,
        cbar_kwargs={"label": "Trend (degrees C per year)"}
    )

    # Add coastlines and features
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')

    # Plot rectangles if provided
    if rectangles:
        for rect in rectangles:
            lon_min = rect['lon_min']
            lon_max = rect['lon_max']
            lat_min = rect['lat_min']
            lat_max = rect['lat_max']
            color = rect.get('color', 'springgreen')
            label = rect.get('label', None)

            # Plot rectangle
            ax.add_patch(plt.Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                linewidth=4,
                edgecolor=color,
                facecolor='none',
                transform=ccrs.PlateCarree()
            ))

            # Add label if provided
            if label:
                ax.text(
                    (lon_min + lon_max) / 2,
                    lat_max,
                    label,
                    color='black',
                    fontsize=10,
                    ha='center',
                    va='center',
                    transform=ccrs.PlateCarree(),
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')
                )

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.savefig("/home/jlandsbe/ai_weather_to_climate_ats780A8/output_figures/trend_map_" + save_name, dpi=300)
    return slope_map

era5_aligned = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ERA5_processed.nc")
ace2_aligned = xr.open_dataset("/home/jlandsbe/ai_weather_to_climate_ats780A8/input_data/ACE2_processed.nc")

thresholded_yearly_mean_all_era5 = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=0, start_year=1980, chunk_size = 5, months = [1,2,12])

thresholded_yearly_mean_all_ace2= thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=0, start_year=1980, chunk_size = 5, months = [1,2,12])
thresholded_yearly_mean_10_era5 = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=-10, start_year=1980, chunk_size = 5, months = [1,2,12])
thresholded_yearly_mean_10_ace2 = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=-10, start_year=1980, chunk_size = 5, months = [1,2,12])
thresholded_yearly_mean_90_era5 = thresholded_yearly_mean(era5_aligned, 'surface_temperature', percentile=90, start_year=1980, chunk_size = 5, months = [1,2,12])
thresholded_yearly_mean_90_ace2 = thresholded_yearly_mean(ace2_aligned, 'surface_temperature', percentile=90, start_year=1980, chunk_size = 5, months = [1,2,12])


relative_trend_era5_10 = thresholded_yearly_mean_10_era5 - thresholded_yearly_mean_all_era5
relative_trend_era5_90 = thresholded_yearly_mean_90_era5 - thresholded_yearly_mean_all_era5
relative_trend_ace2_10 = thresholded_yearly_mean_10_ace2 - thresholded_yearly_mean_all_ace2
relative_trend_ace2_90 = thresholded_yearly_mean_90_ace2 - thresholded_yearly_mean_all_ace2

era5_to_ace2_10_trend = relative_trend_era5_10 - relative_trend_ace2_10
era5_to_ace2_90_trend = relative_trend_era5_90 - relative_trend_ace2_90

rectangles_to_plot = [
    #{'lon_min': 0, 'lon_max': 36, 'lat_min': 42, 'lat_max': 70, 'label': "W Europe"},
    {'lon_min': 194, 'lon_max': 265, 'lat_min': 55, 'lat_max': 85, 'label': "High Latitudes"},
    {'lon_min': 236, 'lon_max': 284, 'lat_min': 39, 'lat_max': 53, 'label':'N U.S./S CA'},
    {'lon_min': 90, 'lon_max': 150, 'lat_min': 50, 'lat_max': 80, 'label':'E Russia'},
    #{'lon_min': 93, 'lon_max': 120, 'lat_min': 9, 'lat_max': 29, 'label': "SE China", 'color': 'magenta'},
    #{'lon_min': 285, 'lon_max': 307, 'lat_min': -55, 'lat_max': -18, 'label':'S America','color': 'magenta'},
]

print(relative_trend_era5_10)
# Plot the trends
plot_trend_map(relative_trend_era5_10, cmap='RdBu_r', vmin=None, vmax=None, title="Relative Trend (10th Percentile) ERA5 Winter", save_name = "ERA5_10th_relative", rectangles=rectangles_to_plot)
plot_trend_map(relative_trend_era5_90, cmap='RdBu_r', vmin=None, vmax=None, title="Relative Trend (90th Percentile) ERA5 Winter", save_name = "ERA5_90th_relative", rectangles=rectangles_to_plot)
plot_trend_map(relative_trend_ace2_10, cmap='RdBu_r', vmin=None, vmax=None, title="Relative Trend (10th Percentile) ACE2 Winter", save_name = "ACE2_10th_relative", rectangles=rectangles_to_plot)
plot_trend_map(relative_trend_ace2_90, cmap='RdBu_r', vmin=None, vmax=None, title="Relative Trend (90th Percentile) ACE2 Winter", save_name = "ACE2_90th_relative", rectangles=rectangles_to_plot)
plot_trend_map(era5_to_ace2_10_trend, cmap='RdBu_r', vmin=None, vmax=None, title="Relative Trend (10th Percentile) ERA5 to ACE2 Winter", save_name = "ERA5_to_ACE2_10th_relative", rectangles=rectangles_to_plot)
plot_trend_map(era5_to_ace2_90_trend, cmap='RdBu_r', vmin=None, vmax=None, title="Relative Trend (90th Percentile) ERA5 to ACE2 Winter", save_name = "ERA5_to_ACE2_90th_relative",rectangles=rectangles_to_plot)
plot_trend_map(thresholded_yearly_mean_all_era5 - thresholded_yearly_mean_all_ace2, cmap='RdBu_r', vmin=None, vmax=None, title="Relative Trend (All values) ERA5 to ACE2 Winter", save_name = "ERA5_to_ACE2_all_relative", rectangles=rectangles_to_plot)
# Plot the absolute trends

plot_trend_map(thresholded_yearly_mean_all_era5, cmap='RdBu_r', vmin=None, vmax=None, title="Mean Trend per Grid Point ERA5", save_name = "ERA5_mean", rectangles=rectangles_to_plot)
plot_trend_map(thresholded_yearly_mean_all_ace2, cmap='RdBu_r', vmin=None, vmax=None, title="Mean Trend per Grid Point ACE2", save_name = "ACE2_mean", rectangles=rectangles_to_plot)
# plot_trend_map(thresholded_yearly_mean_10_era5, cmap='RdBu_r', vmin=None, vmax=None, title="5th Percentile Trend per Grid Point ERA5", save_name = "ERA5_5th", rectangles=rectangles_to_plot)
# plot_trend_map(thresholded_yearly_mean_10_ace2, cmap='RdBu_r', vmin=None, vmax=None, title="5th Percentile Trend per Grid Point ACE2", save_name = "ACE2_5th", rectangles=rectangles_to_plot)
# plot_trend_map(thresholded_yearly_mean_90_era5, cmap='RdBu_r', vmin=None, vmax=None, title="95th Percentile Trend per Grid Point ERA5", save_name = "ERA5_95th", rectangles=rectangles_to_plot)
# plot_trend_map(thresholded_yearly_mean_90_ace2, cmap='RdBu_r', vmin=None, vmax=None, title="95th Percentile Trend per Grid Point ACE2", save_name = "ACE2_95th", rectangles=rectangles_to_plot)



