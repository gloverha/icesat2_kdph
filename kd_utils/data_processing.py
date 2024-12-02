# utils/data_processing.py


import re
import os
import io
import time
import math
import h5py
import logging
import netCDF4
import numpy as np
import geopandas as gpd

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial import ConvexHull

import argparse
import subprocess
# import fiona
import utm
import pyproj
from pyproj import Transformer, Proj

# from pyproj import Transformer
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from sklearn.cluster import DBSCAN
from .interpolation import *



#this function from icesat2_toolkit
# PURPOSE: read ICESat-2 ATL03 HDF5 data files
def read_granule(FILENAME, ATTRIBUTES=False, **kwargs):
    """
    Reads ICESat-2 ATL03 Global Geolocated Photons data files

    Parameters
    ----------
    FILENAME: str
        full path to ATL03 file
    ATTRIBUTES: bool, default False
        read file, group and variable attributes

    Returns
    -------
    IS2_atl03_mds: dict
        ATL03 variables
    IS2_atl03_attrs: dict
        ATL03 attributes
    IS2_atl03_beams: list
        valid ICESat-2 beams within ATL03 file
    """
    # Open the HDF5 file for reading
    if isinstance(FILENAME, io.IOBase):
        fileID = h5py.File(FILENAME, 'r')
    else:
        fileID = h5py.File(os.path.expanduser(FILENAME), 'r')

    # Output HDF5 file information
    logging.info(fileID.filename)
    logging.info(list(fileID.keys()))

    # allocate python dictionaries for ICESat-2 ATL03 variables and attributes
    IS2_atl03_mds = {}
    IS2_atl03_attrs = {}

    # read each input beam within the file
    IS2_atl03_beams = []
    for gtx in [k for k in fileID.keys() if bool(re.match(r'gt\d[lr]',k))]:
        # check if subsetted beam contains data
        # check in both the geolocation and heights groups
        try:
            fileID[gtx]['geolocation']['segment_id']
            fileID[gtx]['heights']['delta_time']
        except KeyError:
            pass
        else:
            IS2_atl03_beams.append(gtx)

    # for each included beam
    for gtx in IS2_atl03_beams:
        # get each HDF5 variable
        IS2_atl03_mds[gtx] = {}
        IS2_atl03_mds[gtx]['heights'] = {}
        IS2_atl03_mds[gtx]['geolocation'] = {}
        IS2_atl03_mds[gtx]['bckgrd_atlas'] = {}
        IS2_atl03_mds[gtx]['geophys_corr'] = {}
        # ICESat-2 Measurement Group
        for key,val in fileID[gtx]['heights'].items():
            IS2_atl03_mds[gtx]['heights'][key] = val[:]
        # ICESat-2 Geolocation Group
        for key,val in fileID[gtx]['geolocation'].items():
            IS2_atl03_mds[gtx]['geolocation'][key] = val[:]
        # ICESat-2 Background Photon Rate Group
        for key,val in fileID[gtx]['bckgrd_atlas'].items():
            IS2_atl03_mds[gtx]['bckgrd_atlas'][key] = val[:]
        # ICESat-2 Geophysical Corrections Group: Values for tides (ocean,
        # solid earth, pole, load, and equilibrium), inverted barometer (IB)
        # effects, and range corrections for tropospheric delays
        for key,val in fileID[gtx]['geophys_corr'].items():
            IS2_atl03_mds[gtx]['geophys_corr'][key] = val[:]

        # Getting attributes of included variables
        if ATTRIBUTES:
            # Getting attributes of IS2_atl03_mds beam variables
            IS2_atl03_attrs[gtx] = {}
            IS2_atl03_attrs[gtx]['heights'] = {}
            IS2_atl03_attrs[gtx]['geolocation'] = {}
            IS2_atl03_attrs[gtx]['bckgrd_atlas'] = {}
            IS2_atl03_attrs[gtx]['geophys_corr'] = {}
            # Global Group Attributes
            for att_name,att_val in fileID[gtx].attrs.items():
                IS2_atl03_attrs[gtx][att_name] = att_val
            # ICESat-2 Measurement Group
            for key,val in fileID[gtx]['heights'].items():
                IS2_atl03_attrs[gtx]['heights'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['heights'][key][att_name]=att_val
            # ICESat-2 Geolocation Group
            for key,val in fileID[gtx]['geolocation'].items():
                IS2_atl03_attrs[gtx]['geolocation'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['geolocation'][key][att_name]=att_val
            # ICESat-2 Background Photon Rate Group
            for key,val in fileID[gtx]['bckgrd_atlas'].items():
                IS2_atl03_attrs[gtx]['bckgrd_atlas'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['bckgrd_atlas'][key][att_name]=att_val
            # ICESat-2 Geophysical Corrections Group
            for key,val in fileID[gtx]['geophys_corr'].items():
                IS2_atl03_attrs[gtx]['geophys_corr'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['geophys_corr'][key][att_name]=att_val

    # ICESat-2 spacecraft orientation at time
    IS2_atl03_mds['orbit_info'] = {}
    IS2_atl03_attrs['orbit_info'] = {}
    for key,val in fileID['orbit_info'].items():
        IS2_atl03_mds['orbit_info'][key] = val[:]
        # Getting attributes of group and included variables
        if ATTRIBUTES:
            # Global Group Attributes
            for att_name,att_val in fileID['orbit_info'].attrs.items():
                IS2_atl03_attrs['orbit_info'][att_name] = att_val
            # Variable Attributes
            IS2_atl03_attrs['orbit_info'][key] = {}
            for att_name,att_val in val.attrs.items():
                IS2_atl03_attrs['orbit_info'][key][att_name] = att_val

    # information ancillary to the data product
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    # could alternatively use the Julian day of the ATLAS SDP epoch: 2458119.5
    # and add leap seconds since 2018-01-01T00:00:00Z UTC (ATLAS SDP epoch)
    IS2_atl03_mds['ancillary_data'] = {}
    IS2_atl03_attrs['ancillary_data'] = {}
    ancillary_keys = ['atlas_sdp_gps_epoch','data_end_utc','data_start_utc',
        'end_cycle','end_geoseg','end_gpssow','end_gpsweek','end_orbit',
        'end_region','end_rgt','granule_end_utc','granule_start_utc','release',
        'start_cycle','start_geoseg','start_gpssow','start_gpsweek',
        'start_orbit','start_region','start_rgt','version']
    for key in ancillary_keys:
        # get each HDF5 variable
        IS2_atl03_mds['ancillary_data'][key] = fileID['ancillary_data'][key][:]
        # Getting attributes of group and included variables
        if ATTRIBUTES:
            # Variable Attributes
            IS2_atl03_attrs['ancillary_data'][key] = {}
            for att_name,att_val in fileID['ancillary_data'][key].attrs.items():
                IS2_atl03_attrs['ancillary_data'][key][att_name] = att_val

    # transmit-echo-path (tep) parameters
    IS2_atl03_mds['ancillary_data']['tep'] = {}
    IS2_atl03_attrs['ancillary_data']['tep'] = {}
    for key,val in fileID['ancillary_data']['tep'].items():
        # get each HDF5 variable
        IS2_atl03_mds['ancillary_data']['tep'][key] = val[:]
        # Getting attributes of group and included variables
        if ATTRIBUTES:
            # Variable Attributes
            IS2_atl03_attrs['ancillary_data']['tep'][key] = {}
            for att_name,att_val in val.attrs.items():
                IS2_atl03_attrs['ancillary_data']['tep'][key][att_name] = att_val

    # channel dead time and first photon bias derived from ATLAS calibration
    cal1,cal2 = ('ancillary_data','calibrations')
    for var in ['dead_time','first_photon_bias']:
        IS2_atl03_mds[cal1][var] = {}
        IS2_atl03_attrs[cal1][var] = {}
        for key,val in fileID[cal1][cal2][var].items():
            # get each HDF5 variable
            if isinstance(val, h5py.Dataset):
                IS2_atl03_mds[cal1][var][key] = val[:]
            elif isinstance(val, h5py.Group):
                IS2_atl03_mds[cal1][var][key] = {}
                for k,v in val.items():
                    IS2_atl03_mds[cal1][var][key][k] = v[:]
            # Getting attributes of group and included variables
            if ATTRIBUTES:
                # Variable Attributes
                IS2_atl03_attrs[cal1][var][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[cal1][var][key][att_name] = att_val
                if isinstance(val, h5py.Group):
                    for k,v in val.items():
                        IS2_atl03_attrs[cal1][var][key][k] = {}
                        for att_name,att_val in val.attrs.items():
                            IS2_atl03_attrs[cal1][var][key][k][att_name]=att_val

    # get ATLAS impulse response variables for the transmitter echo path (TEP)
    tep1,tep2 = ('atlas_impulse_response','tep_histogram')
    IS2_atl03_mds[tep1] = {}
    IS2_atl03_attrs[tep1] = {}
    for pce in ['pce1_spot1','pce2_spot3']:
        IS2_atl03_mds[tep1][pce] = {tep2:{}}
        IS2_atl03_attrs[tep1][pce] = {tep2:{}}
        # for each TEP variable
        for key,val in fileID[tep1][pce][tep2].items():
            IS2_atl03_mds[tep1][pce][tep2][key] = val[:]
            # Getting attributes of included variables
            if ATTRIBUTES:
                # Global Group Attributes
                for att_name,att_val in fileID[tep1][pce][tep2].attrs.items():
                    IS2_atl03_attrs[tep1][pce][tep2][att_name] = att_val
                # Variable Attributes
                IS2_atl03_attrs[tep1][pce][tep2][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[tep1][pce][tep2][key][att_name] = att_val

    # Global File Attributes
    if ATTRIBUTES:
        for att_name,att_val in fileID.attrs.items():
            IS2_atl03_attrs[att_name] = att_val

    # Closing the HDF5 file
    fileID.close()
    # Return the datasets and variables
    return (IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams)



# convert_wgs_to_utm function, see https://stackoverflow.com/a/40140326/4556479
def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = 'epsg:326' + utm_band
        return epsg_code
    epsg_code = 'epsg:327' + utm_band
    return epsg_code


def orthometric_correction(lat, lon, Z, epsg):
    # Define the Proj string
    #To transform from WGS84 ellipsoidal height
    # to EGM2008 orthometric height using PyProj
    # proj_string = '+proj=latlong +ellps=WGS84 +datum=WGS84 +vunits=m +no_defs +geoidgrids=egm2008-1.gtx'
    # # Define the Proj string for WGS84 ellipsoidal height
    # wgs84_proj_string = '+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs'
    
    # # Define the Proj string for EGM2008 orthometric height: egm08_25,egm2008-1
    # egm2008_proj_string = \
    #     '+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs ' \
    #     '+geoidgrids=C:/Workstation/ICESat2_HLS/Code/Geoids/egm08_25.gtx'

    # transform ellipsoid (WGS84) height to orthometric height
    # transformer = Transformer.from_crs(wgs84_proj_string, egm2008_proj_string, always_xy=True)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    X_egm08, Y_egm08, Z_egm08 = transformer.transform(lon, lat, Z)

    # transform WGS84 proj to local UTM
    myProj = Proj(epsg)
    X_utm, Y_utm = myProj(lon, lat)

    return Y_utm, X_utm, Z_egm08



def load_data(file_path):
    IS2_atl03_mds, IS2_atl03_attrs, IS2_atl03_beams = read_granule(file_path, ATTRIBUTES=True)
    return IS2_atl03_mds, IS2_atl03_attrs, IS2_atl03_beams



# requires that the input gdf has ranged index values i
# will need to change if index is changed to time or something
# this currently checks point in polygon for EVERY point
# would be significantly sped up if evaluated at 10m or something similar
# maybe later, fine for now
def isolate_sea_land_photons(shoreline_data_path, ICESat2_GDF):
    # try loading the shoreline data
    try:
        ICESat2_GDF.insert(0, 'lat', ICESat2_GDF.geometry.y, False)
        ICESat2_GDF.insert(0, 'lon', ICESat2_GDF.geometry.x, False)

        # allocation of to be used arrays
        zero_int_array = np.int64(np.zeros_like(ICESat2_GDF.geometry.x))

        # Land flag initialized as -1
        # If shorelines downloaded already, will be set to 0 or 1
        ICESat2_GDF.insert(0, 'is_land',
                           zero_int_array - 1, False)

        # set the projection
        ICESat2_GDF.set_crs("EPSG:4326", inplace=True)

        # load shoreline dataset to include only the features that intersect the bounding box
        # bbox can be GeoDataFrame or GeoSeries | shapely Geometry, default None
        # Filter features by given bounding box, GeoSeries, GeoDataFrame or a shapely geometry.
        # engine str, 'fiona' or 'pyogrio'
        # somtime it gives error if using fiona
        # land_polygon_gdf = gpd.read_file(shoreline_data_path, bbox=ICESat2_GDF, engine='fiona')
        land_polygon_gdf = gpd.read_file(shoreline_data_path, bbox=ICESat2_GDF, engine='pyogrio')

        # continue with getting a new array of 0-or-1 labels for each photon
        land_point_labels = np.zeros_like(ICESat2_GDF.is_land.values)

        # update labels for points in the land polygons
        pts_in_land = gpd.sjoin(ICESat2_GDF, land_polygon_gdf, predicate='within')

        # get land or not bool value
        land_loc = ICESat2_GDF.index.isin(pts_in_land.index)

        # asigned them to new numpy array
        land_point_labels[land_loc] = 1
        land_point_labels[~land_loc] = 0

        return land_point_labels

    except Exception as e:

        print(e)

        print("Error loading shoreline data, returning -1s for is_land flag")

        # if the shoreline data is not available
        # return the original label array

        return -np.ones_like(ICESat2_GDF.is_land.values)


def create_photon_dataframe(lat_ph, lon_ph, ref_elev, ref_azimuth, geoid, h_ph, \
                            quality_ph, is_land_label_interp1d, signal_conf_photon, x_atc, relative_AT_dist):
    # Apply geoid correction to the photon heights to convert them from ellipsoidal to orthometric heights
    h_ph_geoid_cor = h_ph[:] - geoid[:]
    
    # Determine the EPSG code for the UTM zone based on the first photon's longitude and latitude   
    epsg_code = convert_wgs_to_utm(lon_ph[0], lat_ph[0])
    
    # Perform orthometric correction to obtain UTM coordinates and corrected heights
    lat_utm, lon_utm, h_ph_cor = orthometric_correction(lat_ph, lon_ph, h_ph, epsg_code)
    
    # Put the data into the dataframe
    sea_photon_dataset = pd.DataFrame({
        'latitude': lat_ph,
        'longitude': lon_ph,
        'lat': lat_utm,
        'lon': lon_utm,
        'photon_height': h_ph_geoid_cor,
        'quality_ph': quality_ph,
        'is_land_label': is_land_label_interp1d,
        'photon_conf': signal_conf_photon,
        'ref_elevation': ref_elev,
        'ref_azimuth': ref_azimuth,
        'relative_AT_dist': relative_AT_dist
    }, columns=['latitude', 'longitude', 'lat', 'lon', 'photon_height', 'quality_ph', 'is_land_label', 'photon_conf', 'ref_elevation', 'ref_azimuth', 'relative_AT_dist'])
    return sea_photon_dataset


def Extract_sea_photons(IS2_atl03_mds, IS2_atl03_beams, shoreline_data_path):
    Segment_ID = {}
    Segment_Index_begin = {}
    Segment_PE_count = {}
    Equator_Segment_Distance = {}
    Segment_Length = {}
    Segment_Is_Land = {}
    Segment_Lon = {}
    Segment_Lat = {}
    Segment_Elev = {}
    Segment_Time = {}
    Segment_ref_elev = {}
    Segment_ref_azimuth = {}
    background_rate = {}
    background_counts = {}
    
    # Initialize a list to store data for each beam
    beam_datasets = []
    
    # Define the indices of the strong beams
    strong_beam_indices = [0, 2, 4]

    # Loop over each strong beam in IS2_atl03_beams using the defined indices
    for i in strong_beam_indices:
        gtx = IS2_atl03_beams[i]
        print('Processing strong beam:', gtx)
        
        # Access the data for the current beam
        IS2_val = IS2_atl03_mds[gtx]
        
        # Initialize dictionaries to store segment data for the beam
        Segment_ID[gtx] = IS2_val['geolocation']['segment_id']
        n_seg = len(Segment_ID[gtx])
        n_pe, = IS2_val['heights']['delta_time'].shape
        Segment_Index_begin[gtx] = IS2_val['geolocation']['ph_index_beg'] - 1
        Segment_PE_count[gtx] = IS2_val['geolocation']['segment_ph_cnt']
        Equator_Segment_Distance[gtx] = IS2_val['geolocation']['segment_dist_x']
        Segment_Length[gtx] = IS2_val['geolocation']['segment_length']
        delta_time = IS2_val['geolocation']['delta_time']
        segment_lat = IS2_val['geolocation']['reference_photon_lat'][:].copy()
        segment_lon = IS2_val['geolocation']['reference_photon_lon'][:].copy()
        ref_elev = IS2_val['geolocation']['ref_elev'][:].copy()
        ref_azimuth = IS2_val['geolocation']['ref_azimuth'][:].copy()
        geoid = IS2_val['geophys_corr']['geoid'][:].copy()
        h_ph = IS2_val['heights']['h_ph'][:].copy()
        lat_ph = IS2_val['heights']['lat_ph'][:].copy()
        lon_ph = IS2_val['heights']['lon_ph'][:].copy()
        signal_conf_photon = IS2_val['heights']['signal_conf_ph'][..., 0].copy()
        x_atc = IS2_val['heights']['dist_ph_along'][:].copy()
        y_atc = IS2_val['heights']['dist_ph_across'][:].copy()
        quality_ph = IS2_val['heights']['quality_ph']

        # Adjust x_atc based on segment distances        
        for seg_index in range(n_seg):
            idx = Segment_Index_begin[gtx][seg_index]
            cnt = Segment_PE_count[gtx][seg_index]
            x_atc[idx:idx + cnt] += Equator_Segment_Distance[gtx][seg_index]

        # Calculate relative distances        
        relative_AT_dist = (x_atc - x_atc[0]) / 1000
        relative_seg_dist = (Equator_Segment_Distance[gtx] - Equator_Segment_Distance[gtx][0]) / 1000
        
        # Create a GeoDataFrame to hold segment data for shoreline check
        Segment_Is_Land['geometry'] = gpd.points_from_xy(segment_lon, segment_lat)
        ICESat2_GDF = gpd.GeoDataFrame(Segment_Is_Land, crs="EPSG:4326")
        
        # Determine if it is land by the land/sea mask
        Segment_Is_Land_Labels = isolate_sea_land_photons(shoreline_data_path, ICESat2_GDF)
        ICESat2_GDF.loc[:, 'is_land'] = Segment_Is_Land_Labels

        # Apply interpolations for required data
        is_land_label_interp1d = apply_interpolation(interpolate_labels(segment_lat, Segment_Is_Land_Labels), lat_ph)
        ph_ref_elev = apply_interpolation(interpolate_labels(segment_lat, ref_elev), lat_ph)
        ph_ref_azimuth = apply_interpolation(interpolate_labels(segment_lat, ref_azimuth), lat_ph)
        ph_geoid = apply_interpolation(interpolate_labels(segment_lat, geoid), lat_ph)


        # Create photon DataFrame for the current beam 
        sea_photon_dataset = create_photon_dataframe(lat_ph=lat_ph, lon_ph=lon_ph,
                                            ref_elev=ph_ref_elev,ref_azimuth=ph_ref_azimuth,
                                            geoid=ph_geoid,h_ph=h_ph,
                                            quality_ph=quality_ph,
                                            is_land_label_interp1d=is_land_label_interp1d,
                                            signal_conf_photon=signal_conf_photon,
                                            x_atc=x_atc,  # Note: x_atc is not used in the function. Remove if unnecessary.
                                            relative_AT_dist=relative_AT_dist)

        # Filter out land photons
        sea_photon_dataset = sea_photon_dataset[sea_photon_dataset['is_land_label'] != 1]
        
        # Add a new column to indicate the beam ID
        sea_photon_dataset['beam_id'] = gtx
        
        # Append the processed dataset for the current beam to the list
        beam_datasets.append(sea_photon_dataset)
        
    # Concatenate all beam data into a single DataFrame
    all_beams_dataset = pd.concat(beam_datasets, ignore_index=True)
    
    return all_beams_dataset


def filter_photon_dataset_by_hull_area(photon_dataset, hull_area_threshold=3000):
    """
    Filters photon dataset based on ConvexHull area threshold and returns the filtered dataset,
    convex hull areas, and convex hull points.
    """
    lat_bins_grouped = photon_dataset.groupby('lat_bins', observed=False)
    filtered_dataset = photon_dataset.copy()
    
    convex_hulls = {}
    convex_hull_areas = {}
    
    for lat_bin, bin_data in lat_bins_grouped:
        if len(bin_data) >= 3:
            points = bin_data[['lat', 'photon_height']].to_numpy()
            hull = ConvexHull(points)
            area = hull.volume
            convex_hull_areas[lat_bin] = area
            
            # Store the ConvexHull points if area meets the threshold
            if area >= hull_area_threshold:
                convex_hulls[lat_bin] = points[hull.vertices]
            else:
                # Remove bins with hull area below the threshold
                filtered_dataset = filtered_dataset[filtered_dataset['lat_bins'] != lat_bin]
                
    return filtered_dataset, convex_hull_areas, convex_hulls




def plot_convex_hulls(photon_dataset, convex_hulls, convex_hull_areas):
    """
    Plots ConvexHull polygons and annotated areas for each lat_bin group in the dataset.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    hlims = [-10, 2]
    
    for lat_bin, hull_points in convex_hulls.items():
        hull = ConvexHull(hull_points)
        ax.fill(hull_points[hull.vertices, 0], hull_points[hull.vertices, 1], alpha=0.3, label=f'Bin {lat_bin}')
        
        # Calculate centroid to place the label
        centroid = np.mean(hull_points[hull.vertices], axis=0)
        ax.text(centroid[0], centroid[1], f'{convex_hull_areas[lat_bin]:.2f}', 
                horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')
    
    ax.scatter(photon_dataset['lat'], photon_dataset['photon_height'],
               s=1, c='k', alpha=0.15, edgecolors='none', label='Subsurface ATL03 Photons')
    ax.set_ylim(hlims)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Photon Height')
    ax.set_title('ConvexHull of Photons within each Lat Bin')
    plt.show()
    



##########################
##Discard Functions Below 
##########################


#  Extracts key information from filenames, 
#  which could include processed status, product ID, timestamps, identifiers, or metadata.
def extract_file_params(file_path):
    '''
    Example: input "processed_ATL06_20231115120000_20231115_001_12_data.h5"
            Output: ('processed_', 'ATL06', '2023', '11', '15', '12', '00', '00', 
            '2023', '11', '15', '001', '12', '_data')
    '''
    # Defines a regex pattern to match strings in the file path
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
                    r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    # Searches for all matches of the regex pattern in the given
    params = rx.findall(file_path).pop()
    return params


def create_mask(binned_data, sea_surface_height, seafloor_height, threshold_height):
    '''
    create a mask for filtering sea surface, sea floor, and threshold_height
    '''
    mask = (binned_data['height'] <= sea_surface_height) & \
           (binned_data['height'] >= seafloor_height) & \
           (binned_data['height'] >= threshold_height)
    return mask


def generate_polygons_from_binned_data(lat, height, mask, lat_interval, height_interval):
    '''
    horizontal_vertical_bin_dataset
    Generate polygons for each bin after masking within a rectangle formed by height and latitude intervals
    '''
    # Create latitude bins based on the specified interval
    lat_bins = np.arange(lat.min(), lat.max() + lat_interval, lat_interval)
    
    # Create height bins within the range [-12, 2] based on the specified interval
    height_bins = np.arange(-12, 2 + height_interval, height_interval)
    
    # Identify which bin each masked latitude and height value belongs to
    lat_bin_indices = np.digitize(lat[mask], lat_bins)
    
    # Determine which height bin each masked point belongs to
    height_bin_indices = np.digitize(height[mask], height_bins)
    
    # Combine latitude and height bin indices for each point
    combined_bins = list(zip(lat_bin_indices, height_bin_indices))
    
    # Find unique bin combinations and count the number of points in each bin    
    unique_bins, counts = np.unique(combined_bins, axis=0, return_counts=True)
    
    # Filter bins to include only those within valid index ranges   
    valid_bins = [(bin[0], bin[1]) for bin in unique_bins if 1 <= bin[0] < len(lat_bins) and 1 <= bin[1] < len(height_bins)]

    polygons = []
    for bin_lat, bin_height in valid_bins:
        if bin_lat > 0 and bin_lat < len(lat_bins) and bin_height > 0 and bin_height < len(height_bins):
            bin_points = np.array([(lat[mask][i], height[mask][i]) for i in range(sum(mask))
                                   if lat_bin_indices[i] == bin_lat and height_bin_indices[i] == bin_height])
            if len(bin_points) > 2:
                hull = ConvexHull(bin_points)
                polygons.append(bin_points[hull.vertices])

    return lat_bins, height_bins, valid_bins, polygons


    
