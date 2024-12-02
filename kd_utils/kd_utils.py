#!/usr/bin/env python
# coding: utf-8

#####
# This group of functions processes ICESat-2 data and creates a bathymetric model.
# To do this, it follows a number of steps in the form of functions, including:
# 1. Reading data (ReadATL03())
# 2. Orthometrically correcting the dataset (OrthometricCorrection())
# 3. Pulling down the data segment ID (getAtl03SegID())
# 4. Bin the data along latitudinal and height gradients (bin_data())
# 5. Calculate sea height (get_sea_height())
# 6. Get water temperature (get_water_temp())
# 7. Correct bathymetric surface for refraction (RefractionCorrection())
# 8. Calculate bathymetric height (get_bath_height())
# 9. Produce figures (produce_figures())
#####
import os
# os.environ["PROJ_LIB"] = r"C:\Users\wayne\anaconda3\Library\share\proj"

import io
import os
import re
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


# Snippet by Eric Guenther (via Amy N.) for assigning photons to a segment
def get_atl03_seg_id(atl03_ph_index_beg, atl03_segment_id, atl03_heights_len):
    # We need to know spacecraft orbit info,
    # which is provided across segments.
    # This first function assigns photons to the segment they belong to.
    # We end up making a new array
    # that has more points to match the photon. Segment is defined as every 100m in the long track.

    # Filter all data where atl03_ph_index starts at 0 (0 indicates errors)
    indsNotZero = atl03_ph_index_beg != 0
    atl03_ph_index_beg = atl03_ph_index_beg[indsNotZero]
    atl03_segment_id = atl03_segment_id[indsNotZero]

    # Subtract 1 from ph_index_beg to start at python 0th pos
    atl03_ph_index_beg = atl03_ph_index_beg - 1

    # Sometimes the ph_index_beg is not at the 0th position, it is not,
    # add it in and then add the associated segment id
    # Warning, this is assuming that the segment id for the points are from
    # the segment id directly before it, this assumption might fail, but I have
    # not come across a case yet where it does.  If you want to play it safe
    # you could comment this section out and then if the first position is not
    # 0 then all photons before the first position will not be assigned a
    # segment id.
    # if atl03_ph_index_beg[0] != 0:
    #     atl03_ph_index_beg = np.append(0,atl03_ph_index_beg)
    #     first_seg_id = atl03_segment_id[0] -1
    #     atl03_segment_id = np.append(first_seg_id,atl03_segment_id)

    # Append atl03_height_len to end of array for final position
    atl03_ph_index_beg = np.append(atl03_ph_index_beg, atl03_heights_len)

    # Make array equal to the length of the atl03_heights photon level data
    ph_segment_id = np.zeros(atl03_heights_len)

    # Iterate through ph_index_beg, from the first to second to last number
    # and set the photons between ph_index_beg i to ph_index_beg i + 1 to
    # segment id i
    for i in range(0, len(atl03_ph_index_beg) - 1):
        ph_segment_id[atl03_ph_index_beg[i]:atl03_ph_index_beg[i + 1]] = atl03_segment_id[i]

    # Return list of segment_id at the photon level
    return ph_segment_id


def ref_linear_interp(x, y):
    # initialize an empty list
    arr = []

    # get unique x values
    ux = np.unique(x)
    for u in ux:
        # get y values for x=u
        idx = y[x == u]

        # try to get the y values for x=u-1 and x=u
        # if this is not possible, set min and max to y values for x=u
        try:
            min = y[x == u - 1][0]
            max = y[x == u][0]
        except:
            min = y[x == u][0]
            max = y[x == u][0]

        # try to get the y values for x=u and x=u+1
        # if this is not possible, set min and max to y values for x=u
        try:
            min = y[x == u][0]
            max = y[x == u + 1][0]
        except:
            min = y[x == u][0]
            max = y[x == u][0]

        # if the min and max values are the same,
        # fill the sub array with the value
        if min == max:
            sub = np.full((len(idx)), min)
            arr.append(sub)

        # if min and max are different,
        # create a sub array using linear interpolation
        else:
            sub = np.linspace(min, max, len(idx))
            arr.append(sub)

    # concatenate all the sub arrays into a single array and return
    return np.concatenate(arr, axis=None).ravel()


# Bin data along vertical and horizontal scales
def horizontal_vertical_bin_dataset(dataset, lat_res, vertical_res):
    """Bin data along vertical and horizontal scales
    for later segmentation"""
    
    # Filter values within the range (-50, 10), because photons elevation outside this range will be real noise
    valid_range = (-50, 10)
    valid_mask = (dataset['photon_height'] > valid_range[0]) & (dataset['photon_height'] < valid_range[1])

    # Apply the valid_mask to filter unwanted values
    filtered_dataset = dataset[valid_mask]

    # Calculate the number of height bins
    height_range = abs(filtered_dataset['photon_height'].max() - filtered_dataset['photon_height'].min())
    height_bin_number = max(1, round(height_range / vertical_res))  # Ensure at least one bin

    # Calculate the number of latitude bins
    lat_range = abs(filtered_dataset['lat'].max() - filtered_dataset['lat'].min())
    lat_bin_number = max(1, round(lat_range / lat_res))  # Ensure at least one bin

    # Create bins for latitude
    lat_bins = pd.cut(filtered_dataset['lat'], bins=lat_bin_number, labels=np.arange(lat_bin_number))

    # Create bins for height
    height_bins = pd.cut(filtered_dataset['photon_height'], bins=height_bin_number,
                         labels=np.round(np.linspace(filtered_dataset['photon_height'].min(),
                                                     filtered_dataset['photon_height'].max(),
                                                     num=height_bin_number), decimals=1))

    # Add bins to dataframe using .loc to avoid SettingWithCopyWarning
    filtered_dataset.loc[:, 'lat_bins'] = lat_bins
    filtered_dataset.loc[:, 'height_bins'] = height_bins
    filtered_dataset = filtered_dataset.reset_index(drop=True)

    return filtered_dataset


# thinking about grid searching to detect bathymetric directly
# rather than bin and then search
def horizontal_vertical_grid_density_cal(dataset, lat_res, vertical_res, density_threshold):
    """Bin data along vertical and horizontal scales
       and calculate high-density points using a grid method"""

    # Calculate the number of bins required both vertically
    lat_bin_number = round(abs(dataset['lat'].min() - dataset['lat'].max()) / lat_res)
    # and horizontally based on resolution size
    height_bin_number = round(abs(dataset['photon_height'].min() - dataset['photon_height'].max()) / vertical_res)

    # Create the grid
    grid = np.zeros((lat_bin_number, height_bin_number))

    # Iterate over the dataset and assign points to cells
    for _, row in dataset.iterrows():
        lat_index = int((row['lat'] - dataset['lat'].min()) / lat_res)
        height_index = int((row['photon_height'] - dataset['photon_height'].min()) / vertical_res)
        grid[lat_index, height_index] += 1

    # Identify high-density cells
    high_density_cells = np.argwhere(grid > density_threshold)

    # Create a copy of the dataset
    dataset_copy = dataset.copy()

    # Assign the cell indices as bins to the dataset
    dataset_copy['lat_bins'] = pd.cut(dataset['lat'], bins=lat_bin_number,
                                      labels=np.arange(lat_bin_number))
    dataset_copy['height_bins'] = pd.cut(dataset['photon_height'], bins=height_bin_number,
                                         labels=np.arange(height_bin_number))

    # Reset the index of the copied dataset
    dataset_copy = dataset_copy.reset_index(drop=True)

    return dataset_copy, high_density_cells


# Bin data along horizontal scale only
def horizontal_bin_dataset(dataset, lat_res):
    """Bin data along the horizontal scale (lat) only
    for later segmentation"""

    # Calculate the number of bins required horizontally based on resolution size
    lat_bin_number = round(abs(dataset['lat_utm'].min() - dataset['lat_utm'].max()) / lat_res)

    # Cut lat bins
    # lat_bins = pd.cut(dataset['lat'], bins=lat_bin_number, labels=np.array(range(lat_bin_number)))
    lat_bins = pd.cut(dataset['lat_utm'], bins=lat_bin_number, labels=np.array(range(lat_bin_number)))

    # Create a copy of the dataset
    dataset_copy = dataset.copy()

    # Add lat bins to the dataframe
    dataset_copy['lat_bins'] = lat_bins

    # Reset the index of the copied dataset
    dataset_copy = dataset_copy.reset_index(drop=True)

    return dataset_copy


# Bin data first, and then throw away only the surface bin
def get_rm_sea_surface_bin(binned_dataset):
    """Calculate mean sea height for easier calculation of depth and cleaner
    figures"""

    # set flag for the df save
    flag = 1

    # group dataset by lat bins
    grouped_data = binned_dataset.groupby(['lat_bins'], group_keys=True)
    data_groups = dict(list(grouped_data))

    # Loop through groups and return average sea height
    for k, v in data_groups.items():

        lat_bin_average = v['lat'].mean()

        # Create new dataframe based on occurrence of photons per height bin
        new_df = pd.DataFrame(v.groupby('height_bins',observed=False).count())

        # Return the bin with the highest count
        largest_h_bin = new_df['lat'].argmax()

        # Select the index of the bin with the highest count
        largest_h_index = new_df.index[largest_h_bin]

        # get all values below this bin
        # Use boolean indexing to select only the values below the peak bin
        new_photon_array_without_peak_bin = v.loc[v['height_bins'] < largest_h_index]

        if flag == 1:
            photon_array_without_peak_bin = new_photon_array_without_peak_bin
            flag = 2

        else:
            photon_array_without_peak_bin = photon_array_without_peak_bin.append(new_photon_array_without_peak_bin)

        del new_df

    return photon_array_without_peak_bin


def get_sea_surface_height(binned_data, threshold):
    """Calculate mean sea height for easier calculation of depth and cleaner figures"""
    
    #set flag for the df save
    firstTimeIndex = True
    
    # Create sea height list
    sea_surface_height = []
    mean_lat_bins_seq = []
    sea_surface_subsurface_photons_ratio = []


    # Group dataset by latitude bins
    grouped_data = binned_data.groupby(['lat_bins'], group_keys=True, observed=False)
    data_groups = dict(list(grouped_data))

    # Loop through groups and return average sea height
    for k, v in data_groups.items():
        # based on lat_utm
        lat_bin_average = v['lat'].mean()

        # Create new dataframe based on occurrence of photons per height bin
        new_df = pd.DataFrame(v.groupby(['height_bins'], observed=False).count())

        # Return the bin with the highest count
        largest_h_bin = new_df['lat'].argmax()

        # Select the index of the bin with the highest count
        largest_h_index = new_df.index[largest_h_bin]

        # Calculate the median value of all photon height values within this bin
        photons_sea_surface = v.loc[v['height_bins'] == largest_h_index, 'photon_height']
        lat_bin_sea_median = photons_sea_surface.median()

        # Append to sea height list
        sea_surface_height.append(lat_bin_sea_median)
        mean_lat_bins_seq.append(lat_bin_average)
        del new_df

        # Get all photons below sea surface
        # to determine segment type of each subsurface water column
        # Use calculated sea height to determine photons at 0.5m below peak
        photons_sea_surface_up = \
            v.loc[(v['photon_height'] > (lat_bin_sea_median - threshold)) &
                  (v['photon_height'] < (lat_bin_sea_median + 2*threshold))]

        # Calculate the photon ratio between surface and whole photons
        if v['photon_height'].shape[0] > 0:
            new_photons_ratio_sea_surface = \
                photons_sea_surface_up.shape[0] / v['photon_height'].shape[0]
        else:
            new_photons_ratio_sea_surface = np.nan

        sea_surface_subsurface_photons_ratio.append((1 - new_photons_ratio_sea_surface))

    # Filter out sea height bin values outside 2 SD of mean.
    mean = np.nanmean(sea_surface_height, axis=0)
    sd = np.nanstd(sea_surface_height, axis=0)

    final_sea_surface_height = np.where((sea_surface_height > (mean + 2 * sd)) | 
                                        (sea_surface_height < (mean - 2 * sd)),
                                        np.nan,
                                        sea_surface_height).tolist()
    
    sea_surface_height_abnormal_label = np.where(np.isnan(final_sea_surface_height), 0, 1)

    # Determine label based on ratio of sea surface photons and subsurface photons
    sea_surface_dominated_label = \
        np.where(np.array(sea_surface_subsurface_photons_ratio) >= 0.2, 0, 1)
        
    # Loop through groups again and return photons below 0.5m of sea height
    PhotonDFBelowThresholdPeak = pd.DataFrame()
    for i, (k, v) in enumerate(data_groups.items()):
        # Get all values below this bin
        NewPhotonDFBelowThresholdPeak = \
            v.loc[v['photon_height'] < (final_sea_surface_height[i] - threshold)]
        
        if firstTimeIndex:
            PhotonDFBelowThresholdPeak = NewPhotonDFBelowThresholdPeak
            firstTimeIndex=False
        else: 
           PhotonDFBelowThresholdPeak=\
               pd.concat([PhotonDFBelowThresholdPeak, NewPhotonDFBelowThresholdPeak])
    
    
    return final_sea_surface_height, \
        sea_surface_height_abnormal_label, \
        sea_surface_dominated_label,\
        PhotonDFBelowThresholdPeak


#
#Arbitrary cutoff below the max value - 0.5 m below peak 
# (we may also use 1 m but we can add that later) (apply to raw photon data, each of 6 beams)
def get_photon_below_sea_surface(binned_data,threshold):
    '''Calculate mean sea height for easier calculation of depth and cleaner figures'''
    
    #set flag for the df save
    firstTimeIndex=1
    
    # Create sea height list
    sea_surface_height = []
#     mean_lat_bins_seq=[]
   
    grouped_data = binned_data.groupby(['lat_bins'], group_keys=True, observed=False)
    data_groups = dict(list(grouped_data))
    
    # Loop through groups and return average sea height
    for k,v in data_groups.items():
        
        lat_bin_average=v['lat_utm'].mean()
        
        # Create new dataframe based on occurance of photons per height bin
        new_df = pd.DataFrame(v.groupby(['height_bins'], observed=False).count())
        
        # Return the bin with the highest count
        largest_h_bin = new_df['lat_utm'].argmax()
        
        # Select the index of the bin with the highest count
        largest_h = new_df.index[largest_h_bin]
        
        # Calculate the median value of all values within this bin
        lat_bin_sea_median = v.loc[v['height_bins']==largest_h, 'photon_height'].median()
        
        # Append to sea height list
        sea_surface_height.append(lat_bin_sea_median)
#         mean_lat_bins_seq.append(lat_bin_average)
        del new_df
        
    # Filter out sea height bin values outside 2 SD of mean.
    mean = np.nanmean(sea_surface_height, axis=0)
    sd = np.nanstd(sea_surface_height, axis=0)
    Final_sea_height = np.where((sea_surface_height > (mean + 2*sd)) | (sea_surface_height < (mean - 2*sd)), np.nan, 
                              sea_surface_height).tolist()
    Abnormal_sea_height_label=np.where(np.isnan(Final_sea_height), 1, 0)
    

    # Loop through groups again and return photons below 0.5m of sea height
    for k,v in data_groups.items():
                
        # get all values below this bin
        # Use calculated sea height to determine photons at 0.5m below peak
        NewPhotonArrayBelowThresholdPeak= \
        v.loc[v['photon_height']<(sea_surface_height[k]-threshold)]
        
        if firstTimeIndex ==1:
            PhotonArrayBelowThresholdPeak=NewPhotonArrayBelowThresholdPeak
            firstTimeIndex=2
            
        else: 
            PhotonArrayBelowThresholdPeak=PhotonArrayBelowThresholdPeak.append(NewPhotonArrayBelowThresholdPeak)
        

    return PhotonArrayBelowThresholdPeak



# Function to get elevation for multiple points
def get_seafloor_bathy_GEBCO_batch(lons, lats, raster, raster_data):
    # Convert geographic coordinates to the raster's coordinate system
    rows, cols = raster.index(lons, lats)
    rows, cols = np.array(rows), np.array(cols)

    # Ensure the indices are within bounds
    valid_mask = (rows >= 0) & (rows < raster.height) & (cols >= 0) & (cols < raster.width)
    elevations = np.full(lons.shape, np.nan)
    elevations[valid_mask] = raster_data[rows[valid_mask], cols[valid_mask]]

    return elevations

def get_water_temp(date_year, date_month, date_day, latitude, longitude):
    """
    Pull down surface water temperature along the track from the JPL GHRSST opendap website.

    The GHRSST data are gridded tiles with dimension 17998 x 35999.
    To get the specific grid tile of the SST, you must convert from lat, lon coordinates
    to the gridded tile ratio of the SST data product using the coordinates of the IS2 data.
    """
    # Get date from data filename
    # data_path[-33:-25]
    date = date_year + date_month + date_day
    # date[0:4]
    year = date_year
    # date[4:6]
    month = date_month
    # date[6:8]
    day = date_day
    day_of_year = str(datetime.strptime(date, '%Y%m%d').timetuple().tm_yday)
    # Add zero in front of day of year string
    zero_day_of_year = day_of_year.zfill(3)

    # Calculate ratio of latitude from mid-point of IS2 track
    old_lat = latitude.mean()
    old_lat_min = -90
    old_lat_max = 90
    new_lat_min = 0
    new_lat_max = 17998

    new_lat = round(((old_lat - old_lat_min) / (old_lat_max - old_lat_min)) *
                    (new_lat_max - new_lat_min) + new_lat_min)

    # Calculate ratio of longitude from mid-point of IS2 track
    old_lon = longitude.mean()
    old_lon_min = -180
    old_lon_max = 180
    new_lon_min = 0
    new_lon_max = 35999

    new_lon = round(((old_lon - old_lon_min) / (old_lon_max - old_lon_min)) *
                    (new_lon_max - new_lon_min) + new_lon_min)

    # Access the SST data using the JPL OpenDap interface
    url = 'https://opendap.jpl.nasa.gov/opendap/OceanTemperature/ghrsst/data/GDS2/L4/GLOB/JPL/MUR/v4.1/' \
          + str(year) + '/' + str(zero_day_of_year) + '/' + str(date) \
          + '090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'

    dataset = netCDF4.Dataset(url)

    # Access the data and convert the temperature from K to C
    water_temp = dataset['analysed_sst'][0, new_lat, new_lon] - 273.15
    return water_temp


def refraction_correction(WTemp, WSmodel, Wavelength,
                          Photon_ref_elev, Ph_ref_azimuth,
                          PhotonZ, PhotonX, PhotonY, Ph_Conf):
    """
    WTemp; there is python library that pulls water temp data
    WSmodel is the value surface height
    Wavelength is fixed
    """

    # Only process photons below water surface model
    PhotonX = PhotonX[PhotonZ <= WSmodel]
    PhotonY = PhotonY[PhotonZ <= WSmodel]
    Photon_ref_elev = Photon_ref_elev[PhotonZ <= WSmodel]
    Ph_ref_azimuth = Ph_ref_azimuth[PhotonZ <= WSmodel]
    Ph_Conf = Ph_Conf[PhotonZ <= WSmodel]
    PhotonZ = PhotonZ[PhotonZ <= WSmodel]

    # water temp for refraction correction
    WaterTemp = WTemp

    # Refraction coefficient #
    a = -0.000001501562500
    b = 0.000000107084865
    c = -0.000042759374989
    d = -0.000160475520686
    e = 1.398067112092424
    wl = Wavelength

    # refractive index of air
    n1 = 1.00029

    # refractive index of water
    n2 = (a * WaterTemp ** 2) + (b * wl ** 2) + (c * WaterTemp) + (d * wl) + e

    # assumption is 0.25416
    # This example is refractionCoef = 0.25449
    # 1.00029 is refraction of air constant
    correction_coef = (1 - (n1 / n2))

    # read photon ref_elev to get theta1
    theta1 = np.pi / 2 - Photon_ref_elev

    # eq 1. Theta2
    theta2 = np.arcsin(((n1 * np.sin(theta1)) / n2))

    # eq 3. S
    # Approximate water Surface = 1.5
    # D  = raw uncorrected depth
    D = WSmodel - PhotonZ

    # For Triangle DTS
    S = D / np.cos(theta1)

    # eq 2. R
    R = (S * n1) / n2
    Gamma = (np.pi / 2) - theta1

    # For triangle RPS
    # phi is an angle needed
    phi = theta1 - theta2

    # P is the difference between raw and corrected YZ location
    P = np.sqrt(R ** 2 + S ** 2 - 2 * R * S * np.cos(phi))

    # alpha is an angle needed
    alpha = np.arcsin((R * np.sin(phi)) / P)

    # Beta angle needed for Delta Y an d Delta Z
    Beta = Gamma - alpha

    # Delta Y
    DY = P * np.cos(Beta)

    # Delta Z
    DZ = P * np.sin(Beta)

    # Delta Easting
    DE = DY * np.sin(Ph_ref_azimuth)

    # Delta Northing
    DN = DY * np.cos(Ph_ref_azimuth)

    outX = PhotonX + DE
    outY = PhotonY + DN
    outZ = PhotonZ + DZ

    '''
        print('For selected Bathy photon:')
        print('lat = ', PhotonY[9000])
        print('long = ', PhotonX[9000])
        print('Raw Depth = ', PhotonZ[9000])
        print('D = ', D[9000])
        
        print('ref_elev = ', Photon_ref_elev[9000])
        
        print('Delta East = ', DE[9000])
        print('Delta North = ', DN[9000])
        print('Delta Z = ', DZ[9000])
        '''
    return (outX, outY, outZ, Ph_Conf, PhotonX, PhotonY, PhotonZ, Ph_ref_azimuth,
            Photon_ref_elev)  # We are most interested in out-x, out-y, out-z


def get_bath_height_percentile_thresh(binned_data, percentile_thresh, sea_surface_height, vertical_res):
    """ Detect bathymetric level per bin based on percentile_thresh """
    # Create sea height list
    bath_height = []

    geo_photon_height = []
    geo_longitude = []
    geo_latitude = []

    # Group data by latitude
    # Filter out surface data that are two bins below median surface value calculated above
    binned_data_bath = binned_data[(binned_data['photon_height'] <
                                    sea_surface_height - (vertical_res * 2))]
    grouped_data = binned_data_bath.groupby(['lat_bins'], group_keys=True)
    data_groups = dict(list(grouped_data))

    # Create a percentile threshold of photon counts in each grid,
    # grouped by both x and y axes.
    count_threshold = np.percentile(
        binned_data.groupby(['lat_bins', 'height_bins']).size().reset_index().groupby('lat_bins')[[0]].max(),
        percentile_thresh)

    # Loop through groups and return average bathy height
    for k, v in data_groups.items():
        new_df = pd.DataFrame(v.groupby('height_bins').count())
        bath_bin = new_df['lat'].argmax()
        bath_bin_h = new_df.index[bath_bin]

        # Set threshold of photon counts per bin
        # here this script determines whether there is bathymetry signals by
        # the photon counts per bin below sea surface height
        if new_df.iloc[bath_bin]['lat'] >= count_threshold:

            geo_photon_height.append(v.loc[v['height_bins'] ==
                                           bath_bin_h, 'cor_photon_height'].values)
            geo_longitude.append(v.loc[v['height_bins'] == bath_bin_h, 'lon'].values)
            geo_latitude.append(v.loc[v['height_bins'] == bath_bin_h, 'lat'].values)

            bath_bin_median = v.loc[v['height_bins'] == bath_bin_h, 'cor_photon_height'].median()
            bath_height.append(bath_bin_median)
            del new_df

        else:
            bath_height.append(np.nan)
            del new_df

    geo_longitude_list = np.concatenate(geo_longitude).ravel().tolist()
    geo_latitude_list = np.concatenate(geo_latitude).ravel().tolist()
    geo_photon_list = np.concatenate(geo_photon_height).ravel().tolist()
    geo_depth = sea_surface_height - geo_photon_list
    geo_df = pd.DataFrame(
        {'lon': geo_longitude_list, 'lat': geo_latitude_list,
         'photon_height': geo_photon_list,
         'depth': geo_depth})

    del geo_longitude_list, geo_latitude_list, geo_photon_list

    return bath_height, geo_df


def get_bath_height_HDBSCAN(lat_binned_data, percentile_thresh, sea_surface_height, vertical_res):
    """ Calculate bathymetric level per lat bin based on horizontal resolution """
    # Create sea height list
    bath_height = []
    geo_photon_height = []
    geo_longitude = []
    geo_latitude = []

    # Group data by latitude
    # Filter out surface data that are two bins (2 times height resolution)
    # below median sea surface value calculated above
    binned_data_bath = lat_binned_data[(lat_binned_data['photon_height'] <
                                    sea_surface_height - (vertical_res * 2))]

    grouped_data = binned_data_bath.groupby(['lat_bins'], group_keys=True)
    data_groups = dict(list(grouped_data))

    # Loop through groups and return average bathymetric height
    for k, v in data_groups.items():

        # assign each group of dataset to a new dataframe
        new_df = pd.DataFrame(v)

        # Check if the DataFrame is empty
        if new_df.empty:
            # If the DataFrame is empty, append null values to the lists and skip to the next iteration
            bath_height.append(np.nan)
            geo_photon_height.append([])
            geo_longitude.append([])
            geo_latitude.append([])
            del new_df
            continue

        lat_height_pairs = list(zip(new_df['lat_utm'], new_df['cor_photon_height']))

        # Convert to a numpy array for sklearn
        lat_height_pairs_array = np.array(lat_height_pairs)

        # Perform HDBSCAN clustering on the photons below sea surface for each lat bin
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(lat_height_pairs_array)

        min_cluster_size = 4

        # Check the number of data points is greater than the initial min_cluster_size
        if len(data_scaled) < min_cluster_size:
            min_cluster_size = len(data_scaled) // 2
            if min_cluster_size < 2:
                bath_height.append(np.nan)
                geo_photon_height.append([])
                geo_longitude.append([])
                geo_latitude.append([])
                del new_df
                continue

        # cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, leaf_size=20)
        cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  min_samples=3, cluster_selection_epsilon=20,
                                  leaf_size=25, core_dist_n_jobs=-1)
        cluster.fit(data_scaled)

        # Get the labels assigned to each point by the HDBSCAN model
        labels = cluster.labels_

        # Count the number of points assigned to each cluster
        unique, counts = np.unique(labels, return_counts=True)

        # Create a dictionary that maps each cluster label to the count of points assigned to it
        clusters_dict = dict(zip(unique, counts))

        # Print the dictionary to see the size of each cluster
        print(clusters_dict)

        # Identify the label of the largest cluster
        max_cluster_label = max(clusters_dict, key=clusters_dict.get)

        # Check if no cluster is found
        if max_cluster_label == -1:
            bath_height.append(np.nan)
        else:
            # Add labels to the DataFrame
            new_df_copy = new_df.copy()
            new_df_copy['cluster'] = labels

            # Subset the DataFrame to get only the data points in the largest cluster
            bath_cluster_data = new_df[new_df_copy['cluster'] == max_cluster_label]

            # calculate the median height value as the average bathymetric height
            bath_bin_median = bath_cluster_data['cor_photon_height'].median()
            bath_height.append(bath_bin_median)

            # Extract the longitude, latitude, and photon height for each lat bin
            # and add it to respective lists
            geo_photon_height.append(bath_cluster_data['cor_photon_height'].values)
            geo_longitude.append(bath_cluster_data['lon_utm'].values)
            geo_latitude.append(bath_cluster_data['lat_utm'].values)

        del new_df

    # Convert the lists to a single list
    geo_longitude_list = np.concatenate(geo_longitude).ravel().tolist()
    geo_latitude_list = np.concatenate(geo_latitude).ravel().tolist()
    geo_photon_list = np.concatenate(geo_photon_height).ravel().tolist()

    # Calculate depth
    geo_depth = sea_surface_height - np.array(geo_photon_list)

    # Create a DataFrame
    geo_df = pd.DataFrame(
        {'lon': geo_longitude_list, 'lat': geo_latitude_list,
         'photon_height': geo_photon_list,
         'depth': geo_depth.tolist()})

    return bath_height, geo_df

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
