# utils/visualization.py

import numpy as np
import pandas as pd
import time
import geopandas as gpd
from pyproj import Transformer, Proj
import matplotlib.pyplot as plt

def plot_photon_height(sea_photon_dataset, hlims=[-25, 10]):
    fig, ax = plt.subplots()
    ax.scatter(sea_photon_dataset['latitude'], sea_photon_dataset['photon_height'], s=1, c='k', alpha=0.15, edgecolors='none', label='ATL03 Photons')
    ax.set_xlabel('Relative AT Distance')
    ax.set_ylabel('Height (h_ph)')
    ax.set_title('Scatter Plot of ATL03 Photons')
    ax.set_ylim(hlims)
    ax.legend()
    #plt.show()

def plot_filtered_seafloor_photons(filtered_seafloor_subsurface_dataset, sea_photon_dataset, sea_surface_height, output_path):
    """
    Plots the filtered seafloor photon data along with sea surface and seafloor elevation data.

    Parameters:
    - filtered_seafloor_subsurface_dataset: DataFrame containing the filtered subsurface photon data.
    - sea_photon_dataset: DataFrame containing the original sea photon data.    
    - sea_surface_height: Array of sea surface height values.
    - output_path: Path to save the output plot.
    """
    
    # get sea_surface_x_axis_bins
    sea_surface_x_axis_bins = np.linspace(filtered_seafloor_subsurface_dataset['relative_AT_dist'].min(), 
                                          filtered_seafloor_subsurface_dataset['relative_AT_dist'].max(),
                                          len(sea_surface_height))
    
    hlims = [-25, 10]
    fig, ax = plt.subplots()

    # Scatter plot of subsurface photons
    ax.scatter(sea_photon_dataset['relative_AT_dist'], sea_photon_dataset['photon_height'],
               s=1, c='k', alpha=0.15, edgecolors='none', label='Subsurface ATL03 Photons')

    # Overlay the seafloor elevation data as points
    ax.plot(filtered_seafloor_subsurface_dataset['relative_AT_dist'], filtered_seafloor_subsurface_dataset['seafloor_elevation'],
            linewidth=0.8, c='b', alpha=0.4, label='Seafloor Elevation')

    ax.plot(sea_surface_x_axis_bins, [x - 0.5 for x in sea_surface_height], 
            linewidth=0.8, color='#DD571C', alpha=0.4, label='0.5 m Below Surface Peak')

    # Set labels and title
    ax.set_xlabel('Distance From Start of Track (km)')
    ax.set_ylabel('Photon Height (m)')
    ax.set_title('Scatter Plot of Filtered Sea Photon Dataset')
    ax.set_ylim(hlims)

    # Add a legend
    ax.legend()

    # Save the plot
    plt.legend(loc='upper right')
    plt.savefig(output_path, dpi=400, format='jpeg')

    # Show the plot
    #plt.show()

def plot_kd_photons(subsurface_photon_dataset, Kd_DF_MergedDistance):
    hlims = [-45, 5]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(subsurface_photon_dataset['relative_AT_dist'], subsurface_photon_dataset['photon_height'], s=1.5, c='k', alpha=0.2, edgecolors='none', label='Subsurface ATL03 Photon Height')
    ax1.set_xlabel('Relative Along-Track Distance')
    ax1.set_ylabel('Photon Height', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.plot(subsurface_photon_dataset['relative_AT_dist'], subsurface_photon_dataset['seafloor_elevation'], linewidth=0.8, c='b', alpha=0.4, label='Seafloor Elevation')
    ax1.axhline(y=-6, color='blue', linestyle='--', label='y=-6 m')
    ax2 = ax1.twinx()
    ax2.scatter(Kd_DF_MergedDistance['relative_AT_dist'], Kd_DF_MergedDistance['kd'], label='Kd values', color='r', alpha=0.6)
    ax2.set_ylabel('Kd Value', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_ylim(hlims)
    fig.suptitle('Photon Height and Kd Values along Relative Along-Track Distance')
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='lower left', bbox_to_anchor=(0.08, 0.08))
    fig.tight_layout()
    plt.savefig(r'C:\Workstation\ICESat2_HLS\Kd_ComparisionPaper\Results\IS2_subsurface_kd.jpg', dpi=400, format='jpeg')
    #plt.show()

def plot_bin_polygon_data(lat, height, seafloor_height, mask, lat_bins, height_bins, valid_bins, polygons, y_min, y_max):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Height Data")
    plt.scatter(lat, height, c=height, cmap='viridis', s=10)
    plt.plot(np.sort(lat), seafloor_height[np.argsort(lat)], 'r-', label='Seafloor')
    plt.colorbar(label='Height')
    plt.xlabel('Latitude')
    plt.ylabel('Height')
    plt.ylim(y_min, y_max)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Masked Region (ROI)")
    plt.scatter(lat[mask], height[mask], c=height[mask], cmap='viridis', s=10)
    plt.plot(np.sort(lat), seafloor_height[np.argsort(lat)], 'r-', label='Seafloor')

    for polygon in polygons:
        plt.gca().add_patch(plt.Polygon(polygon, edgecolor='red', facecolor='none', linewidth=1))

    for lb in lat_bins:
        plt.axvline(x=lb, color='gray', linestyle='--', linewidth=0.5)
    for hb in height_bins:
        plt.axhline(y=hb, color='gray', linestyle='--', linewidth=0.5)

    plt.colorbar(label='Height')
    plt.xlabel('Latitude')
    plt.ylabel('Height')
    plt.ylim(y_min, y_max)
    plt.legend()

    plt.tight_layout()
    #plt.show()

#
def produce_figures(binned_data, bath_height, sea_height, solo_sea_surface_label,
                    y_limit_top, y_limit_bottom, percentile, file, geo_df,
                    ref_y, ref_z, beam, epsg_num):
    """Create figures"""

    # Create bins for latitude
    bath_x_axis_bins = np.linspace(binned_data.lat.min(),
                              binned_data.lat.max(), len(bath_height))+20

    sea_surface_x_axis_bins = np.linspace(binned_data.lat.min(),
                              binned_data.lat.max(), len(sea_height))+10

    # Create new dataframes for median values
    bath_median_df = pd.DataFrame({'x': bath_x_axis_bins, 'y': bath_height})

    # Create uniform sea surface based on median sea surface values and filter out surface breaching
    sea_height1 = [np.nanmedian(sea_height) if i == i else np.nan for i in sea_height]
    sea_median_df = pd.DataFrame({'x': sea_surface_x_axis_bins, 'y': sea_height1})

    # Create uniform solo sea surface label
    sea_surface_label = solo_sea_surface_label
    sea_surface_label_df = pd.DataFrame({'x': sea_surface_x_axis_bins, 'y': sea_surface_label})
    idx_1 = np.where(sea_surface_label_df.y == 1)
    idx_0 = np.where(sea_surface_label_df.y == 0)

    # Define figure size
    fig = plt.rcParams["figure.figsize"] = (40, 25)

    # Plot raw points
    #     plt.scatter(x=binned_data.lat,
    #     y = binned_data.photon_height, marker='o', lw=0, s=1, alpha = 0.8,
    #     c = 'yellow', label = 'Raw photon height')
    plt.scatter(ref_y, ref_z, s=0.5, alpha=0.1, c='black')
    plt.scatter(geo_df.lat, geo_df.photon_height, s=0.8, marker = 'o',
                alpha=0.1, c='red', label='Classified Photons')

    # plt.scatter(x=geo_df.lat,
    # y = geo_df.photon_height, marker='o', lw=0, s=0.8,
    # alpha = 0.8, c = 'black', label = 'Corrected photon bin')

    # Plot median values
    plt.scatter(bath_median_df.x, bath_median_df.y,
                marker='o', c='r', alpha=0.8, s=2, label='Median bathymetry')

    plt.scatter(sea_median_df.x, sea_median_df.y,
                marker='o', c='b', alpha=1, s=2, label='Median sea surface')

    plt.scatter(sea_surface_label_df.iloc[idx_1].x, sea_surface_label_df.iloc[idx_1].y,
                marker='o', c='pink', alpha=1, s=3, label='solo_sea_surface')
    plt.scatter(sea_surface_label_df.iloc[idx_0].x, sea_surface_label_df.iloc[idx_0].y,
                marker='o', c='g', alpha=1, s=3, label='non_solo_sea_surface')

    # Insert titles and subtitles
    plt.title('Icesat2 Bathymetry\n' + file)
    plt.xlabel('Latitude', fontsize=25)
    plt.ylabel('Photon Height (m)', fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc="upper left", prop={'size': 20})

    # Limit the x and y axes using parameters
    plt.xlim(left=binned_data.lat.min(), right=binned_data.lat.max())
    plt.ylim(top=y_limit_top, bottom=y_limit_bottom)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    file = file.replace('.h5', '')
    # Define where to save file
    plt.tight_layout()
    plt.savefig("C:/Workstation/ICESat2_HLS/" + file + '_gt' +
                str(beam) + '_' + str(percentile) +
                '_EPSG' + str(epsg_num) + '_' + timestr + ".pdf")
    # plt.show()
    # plt.close()

    # convert corrected locations back to wgs84 (useful to contain)
    transformer = Transformer.from_crs("EPSG:" + str(epsg_num),
                                       "EPSG:4326", always_xy=True)
    print(transformer)
    lon_wgs84, lat_wgs84 = transformer.transform(
        geo_df.lon.values, geo_df.lat.values)

    geo_df['lon_wgs84'] = lon_wgs84
    geo_df['lat_wgs84'] = lat_wgs84

    geodf = gpd.GeoDataFrame(geo_df,
                             geometry=gpd.points_from_xy(geo_df.lon_wgs84,
                                                         geo_df.lat_wgs84))

    geodf.set_crs(epsg=4326, inplace=True)

    # geodf.to_file("C:/Workstation/ICESat2_HLS/" +file + '_gt' + '_' + str(percentile) + '_EPSG' +
    #               str(epsg_num) + '_' + timestr + ".gpkg",
    #               driver="GPKG")