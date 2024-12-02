
import numpy as np
import geopandas as gpd
from datetime import datetime
import pandas as pd
import netCDF4


# Function to apply binning beam-by-beam by calling the function of horizontal_vertical_bin_dataset
def process_sea_photon_binning(sea_photon_dataset, horizontal_res, vertical_res):
    # Initialize list to store results from each beam
    binned_beam_datasets = []

    # Group the dataset by 'beam_id' and process each group separately
    for beam_id, beam_data in sea_photon_dataset.groupby('beam_id'):
        print(f'Processing binning for beam: {beam_id}')

        # Apply binning to the current beam dataset
        binned_beam_data = horizontal_vertical_bin_dataset(beam_data, horizontal_res, vertical_res)

        # Append the binned data for the current beam to the list
        binned_beam_datasets.append(binned_beam_data)

    # Combine all binned beam datasets into a single DataFrame
    binned_dataset_sea_surface = pd.concat(binned_beam_datasets, ignore_index=True)
    
    return binned_dataset_sea_surface


# Bin data along vertical and horizontal scales
def horizontal_vertical_bin_dataset(dataset, lat_res, vertical_res):
    """Bin data along vertical and horizontal scales
    for later segmentation"""
    
    # Filter values within the range (-50, 10), because photons elevation outside this range will be real noise
    valid_range = (-70, 5)
    valid_mask = (dataset['photon_height'] > valid_range[0]) & (dataset['photon_height'] < valid_range[1])

    # Apply the valid_mask to filter unwanted values
    # and create a copy to avoid SettingWithCopyWarning
    filtered_dataset = dataset[valid_mask].copy()

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

        # Check if new_df is not empty before finding the bin with the highest photon count
        if not new_df.empty:        
            # Find the vertical bin with the highest photon count
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
            
        else:
            # Append NaNs if the group is empty
            sea_surface_height.append(np.nan)
            mean_lat_bins_seq.append(np.nan)
            sea_surface_subsurface_photons_ratio.append(np.nan)

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
        if not np.isnan(final_sea_surface_height[i]):
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

