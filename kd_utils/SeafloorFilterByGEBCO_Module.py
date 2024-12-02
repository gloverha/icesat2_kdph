import rasterio
import numpy as np
import pandas as pd
from rtree import index
from shapely.geometry import box, Point

# 1. Function to create an R-tree spatial index for raster bounds
def create_spatial_index(gebco_paths):
    """
    Create an R-tree spatial index for raster bounds to quickly find relevant rasters.
    
    Parameters:
        gebco_paths (list): List of paths to GEBCO raster files.
    
    Returns:
        raster_data_dict (dict): Dictionary containing loaded rasters and their respective data.
        spatial_index (rtree.index.Index): R-tree spatial index for raster bounds.
    """
    idx = index.Index()
    raster_data_dict = {}
    for i, path in enumerate(gebco_paths):
        with rasterio.Env(GTIFF_SRS_SOURCE='EPSG'):
            gebco_raster = rasterio.open(path)
            raster_data = gebco_raster.read(1)
            raster_data_dict[path] = (gebco_raster, raster_data)
            bounds = gebco_raster.bounds
            idx.insert(i, (bounds.left, bounds.bottom, bounds.right, bounds.top), obj=path)
    return raster_data_dict, idx

# 2. Function to determine which rasters are needed for a given set of coordinates using spatial index in a batch manner
def get_relevant_rasters_using_index(lons, lats, raster_data_dict, spatial_index):
    """
    Determine which raster datasets are relevant for the given coordinates using spatial index.
    This function uses a more efficient approach by performing a bounding box query for batches of points.
    
    Parameters:
        lons (numpy.ndarray): Array of longitudes.
        lats (numpy.ndarray): Array of latitudes.
        raster_data_dict (dict): Dictionary containing raster datasets and their data.
        spatial_index (rtree.index.Index): R-tree spatial index for raster bounds.
    
    Returns:
        relevant_rasters (list): List of relevant raster datasets and their respective data.
    """
    # Create a bounding box that covers all points
    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()
    bounding_box = box(min_lon, min_lat, max_lon, max_lat)

    # Get all rasters that intersect with the bounding box
    matches = list(spatial_index.intersection((bounding_box.bounds), objects=True))
    relevant_paths = {match.object for match in matches}
    relevant_rasters = [(raster_data_dict[path][0], raster_data_dict[path][1]) for path in relevant_paths]
    return relevant_rasters

# 3. Function to get seafloor elevation from the relevant rasters in a vectorized manner
def get_seafloor_elevation(lons, lats, relevant_rasters):
    """
    Get seafloor elevations for a batch of points based on longitude and latitude from relevant rasters.
    
    Parameters:
        lons (numpy.ndarray): Array of longitudes.
        lats (numpy.ndarray): Array of latitudes.
        relevant_rasters (list): List of relevant raster datasets and their respective data.
    
    Returns:
        seafloor_elevations (numpy.ndarray): Array of seafloor elevations.
    """
    points = np.vstack((lons, lats)).T
    seafloor_elevations = np.full(len(lons), np.nan)
    
    for gebco_raster, raster_data in relevant_rasters:
        # Use rasterio.sample to get values for multiple points in a batch
        values = list(gebco_raster.sample(points))
        for idx, value in enumerate(values):
            if np.isnan(seafloor_elevations[idx]) and value is not None:
                seafloor_elevations[idx] = value[0]
    
    return seafloor_elevations

# 4. The main process function that ties everything together
def process_seafloor_data(gebco_paths, sea_photon_dataset):
    """
    Main function to process the seafloor data.
    
    Parameters:
        gebco_paths (list): List of paths to GEBCO raster files.
        sea_photon_dataset (pandas.DataFrame): Dataset containing longitude, latitude, and photon height.
    
    Returns:
        filtered_sea_photon_dataset (pandas.DataFrame): Filtered dataset containing points above the seafloor.
    """
    # Create spatial index and load GEBCO Raster Data
    raster_data_dict, spatial_index = create_spatial_index(gebco_paths)
    
    # Get the relevant rasters using spatial index
    lons = sea_photon_dataset['longitude'].values
    lats = sea_photon_dataset['latitude'].values
    relevant_rasters = get_relevant_rasters_using_index(lons, lats, raster_data_dict, spatial_index)
    
    # Get seafloor elevation for all points
    sea_photon_dataset['seafloor_elevation'] = get_seafloor_elevation(lons, lats, relevant_rasters)
    
    # Filter out points below the seafloor
    filtered_sea_photon_dataset = sea_photon_dataset[sea_photon_dataset['photon_height'] > sea_photon_dataset['seafloor_elevation']]
    # filtered_sea_photon_dataset.drop(columns=['seafloor_elevation'], inplace=True)
    
    return filtered_sea_photon_dataset