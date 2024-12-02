# utils/Kd_analysis.py
# updated to perform a linear fit in log-space, just like MATLAB's polyfitn(zdepth, y, 1) for a first-order polynomial.

import numpy as np
import pandas as pd
import logging
# from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# def log_model(z, kd, e0):
#     return np.log(e0) - kd * z

## This is wrong because the input is already a bined data, 
## it is not necessary to do a histogram again
# def CalculateKdFromFilteredSubsurfacePhoton(df, vertical_res=0.8):
#     if df.empty or 'lat_bins' not in df.columns:
#         return pd.DataFrame({'lat_bins': [np.nan], 'kd': [np.nan], 'e0': [np.nan], 'latitude': [np.nan], 'longitude': [np.nan]})
    
#     # Get the latitude bin value
#     lat_bin_value = df['lat_bins'].iloc[0] if not df['lat_bins'].empty else np.nan
#     latitude = df['latitude'].mean() if 'latitude' in df.columns else df['lat'].mean() if 'lat' in df.columns else np.nan
#     longitude = df['longitude'].mean() if 'longitude' in df.columns else df['lon'].mean() if 'lon' in df.columns else np.nan
    
#     # Calculate photon height range
#     photon_height_min = df['photon_height'].min()
#     photon_height_max = df['photon_height'].max()
    
#     # Check for sufficient data range
#     if np.isnan(photon_height_min) or np.isnan(photon_height_max) or photon_height_min == photon_height_max:
#         return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    
#     height_bins_range = abs(photon_height_max - photon_height_min)    
#     height_bins_number = round(height_bins_range / vertical_res)
    
#     # Ensure there are enough bins
#     if height_bins_number < 5:
#         return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    
#     # Create histogram of photon heights
#     bin_edges = np.linspace(photon_height_min, photon_height_max, num=height_bins_number)
#     counts, _ = np.histogram(df['photon_height'], bins=bin_edges)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     # Store histogram data
#     hist_df = pd.DataFrame({'zdepth': bin_centers, 'photon_counts': counts})
    
#     # x value for model
#     # Reverse zdepth to align with the MATLAB approach
#     hist_df['zdepth'] = hist_df['zdepth'].max() - hist_df['zdepth']
    
#     # Log-transform photon counts, replacing zeros with NaN
#     # y value for model
#     hist_df['log_photon_counts'] = np.log(hist_df['photon_counts'].replace(0, np.nan))
#     hist_df.loc[np.isinf(hist_df['log_photon_counts']), 'log_photon_counts'] = np.nan
    
#     # Filter out rows with NaNs in either column for regression
#     valid_data = hist_df.dropna(subset=['zdepth', 'log_photon_counts'])
    
#     # Check for enough valid data points
#     # Skip the regression if there are fewer than 5 datapoints
#     if valid_data['log_photon_counts'].notna().sum() > 3:
#         # Drop NaNs for regression
#         zdepth_valid = valid_data['zdepth'].values.reshape(-1, 1)
#         log_counts_valid = valid_data['log_photon_counts'].values

#         # Perform linear regression
#         model = LinearRegression()
#         model.fit(zdepth_valid, log_counts_valid)
        
#         # Extract kd as the negative of the slope and e0 from the intercept
#         kd = -model.coef_[0]
        
#         print('zdepth_valid:',zdepth_valid)
#         print('photon_counts:',hist_df['photon_counts'])
#         print('kd:',kd)
        
#         e0 = np.exp(model.intercept_)
        
#         # Set kd to NaN if negative
#         if kd < 0:
#             kd = np.nan
#     else:
#         kd, e0 = np.nan, np.nan
    
#     return pd.DataFrame({
#         'lat_bins': [lat_bin_value],
#         'kd': [kd],
#         'e0': [e0],
#         'latitude': [latitude],
#         'longitude': [longitude]
#     })


# # one solution is to adjust the vertical_res to 0.25
# def CalculateKdFromFilteredSubsurfacePhoton(df, vertical_res=0.25):
#     if df.empty or 'lat_bins' not in df.columns:
#         return pd.DataFrame({'lat_bins': [np.nan], 'kd': [np.nan], 'e0': [np.nan], 'latitude': [np.nan], 'longitude': [np.nan]})
    
#     # Get the latitude bin value
#     lat_bin_value = df['lat_bins'].iloc[0] if not df['lat_bins'].empty else np.nan
#     latitude = df['latitude'].mean() if 'latitude' in df.columns else df['lat'].mean() if 'lat' in df.columns else np.nan
#     longitude = df['longitude'].mean() if 'longitude' in df.columns else df['lon'].mean() if 'lon' in df.columns else np.nan
    
#     # Calculate photon height range
#     photon_height_min = df['photon_height'].min()
#     photon_height_max = df['photon_height'].max()
    
#     # Check for sufficient data range
#     if np.isnan(photon_height_min) or np.isnan(photon_height_max) or photon_height_min == photon_height_max:
#         return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    
#     height_bins_range = abs(photon_height_max - photon_height_min)    
#     height_bins_number = round(height_bins_range / vertical_res)
    
#     # Ensure there are enough bins
#     if height_bins_number < 5:
#         return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    
#     # Create histogram of photon heights
#     bin_edges = np.linspace(photon_height_min, photon_height_max, num=height_bins_number)
#     counts, _ = np.histogram(df['photon_height'], bins=bin_edges)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     # Store histogram data
#     hist_df = pd.DataFrame({'zdepth': bin_centers, 'photon_counts': counts})
    
#     # x value for model
#     # Reverse zdepth to align with the MATLAB approach
#     hist_df['zdepth'] = hist_df['zdepth'].max() - hist_df['zdepth']
    
#     # Log-transform photon counts, replacing zeros with NaN
#     # y value for model
#     hist_df['log_photon_counts'] = np.log(hist_df['photon_counts'].replace(0, np.nan))
#     hist_df.loc[np.isinf(hist_df['log_photon_counts']), 'log_photon_counts'] = np.nan
    
#     # Filter out rows with NaNs in either column for regression
#     valid_data = hist_df.dropna(subset=['zdepth', 'log_photon_counts'])
    
#     # Check for enough valid data points
#     # Skip the regression if there are fewer than 5 datapoints
#     if valid_data['log_photon_counts'].notna().sum() > 3:
#         # Drop NaNs for regression
#         zdepth_valid = valid_data['zdepth'].values.reshape(-1, 1)
#         log_counts_valid = valid_data['log_photon_counts'].values

#         # Perform linear regression
#         model = LinearRegression()
#         model.fit(zdepth_valid, log_counts_valid)
        
#         # Extract kd as the negative of the slope and e0 from the intercept
#         kd = -model.coef_[0]
        
#         print('zdepth_valid:',zdepth_valid)
#         print('photon_counts:',hist_df['photon_counts'])
#         print('kd:',kd)
        
#         e0 = np.exp(model.intercept_)
        
#         # Set kd to NaN if negative
#         if kd < 0:
#             kd = np.nan
#     else:
#         kd, e0 = np.nan, np.nan
    
#     return pd.DataFrame({
#         'lat_bins': [lat_bin_value],
#         'kd': [kd],
#         'e0': [e0],
#         'latitude': [latitude],
#         'longitude': [longitude]
#     })


# another solution is to calculate kd without hist
def CalculateKdFromFilteredSubsurfacePhoton(df, vertical_res=0.8):
    # Early exit if DataFrame is empty or missing required column
    if df.empty or 'lat_bins' not in df.columns:
        return pd.DataFrame({'lat_bins': [np.nan], 'kd': [np.nan], 'e0': [np.nan], 'latitude': [np.nan], 'longitude': [np.nan]})
    
    # Retrieve latitude and longitude
    lat_bin_value = df['lat_bins'].iloc[0] if not df['lat_bins'].empty else np.nan
    latitude = df['latitude'].mean() if 'latitude' in df.columns else df['lat'].mean() if 'lat' in df.columns else np.nan
    longitude = df['longitude'].mean() if 'longitude' in df.columns else df['lon'].mean() if 'lon' in df.columns else np.nan
    
    # Use value_counts to get photon counts in each height bin
    height_counts = df['height_bins'].value_counts().sort_index()
    bin_centers = height_counts.index.astype(float)
    
    # Create a DataFrame for the height bins and counts
    hist_df = pd.DataFrame({'zdepth': bin_centers, 'photon_counts': height_counts.values})
    
    # Reverse zdepth for model alignment
    hist_df['zdepth'] = hist_df['zdepth'].max() - hist_df['zdepth']
    
    # Log-transform photon counts, replacing zeros with NaN for regression
    hist_df['log_photon_counts'] = np.log(hist_df['photon_counts'].replace(0, np.nan))
    hist_df.loc[np.isinf(hist_df['log_photon_counts']), 'log_photon_counts'] = np.nan
    
    # Filter for rows without NaNs for regression
    valid_data = hist_df.dropna(subset=['zdepth', 'log_photon_counts'])
    
    # Perform regression if there are sufficient valid data points
    if valid_data['log_photon_counts'].notna().sum() > 3:
        zdepth_valid = valid_data['zdepth'].values.reshape(-1, 1)
        log_counts_valid = valid_data['log_photon_counts'].values

        # Perform linear regression
        model = LinearRegression()
        model.fit(zdepth_valid, log_counts_valid)
        
        # Calculate kd and e0
        kd = -model.coef_[0]
        e0 = np.exp(model.intercept_)
        
        # Set kd to NaN if negative
        if kd < 0:
            kd = np.nan
    else:
        kd, e0 = np.nan, np.nan
    
    return pd.DataFrame({
        'lat_bins': [lat_bin_value],
        'kd': [kd],
        'e0': [e0],
        'latitude': [latitude],
        'longitude': [longitude]
    })


# Original kd calculation function remains unchanged
def calculate_kd(filtered_seafloor_subsurface_photon_dataset):
    logging.info("Calculating Kd from filtered subsurface photon dataset")
    SubsurfacePhotonDFAddedKd = filtered_seafloor_subsurface_photon_dataset\
                                .groupby('lat_bins', observed=False)\
                                .apply(CalculateKdFromFilteredSubsurfacePhoton, include_groups=True)
                                

    # Remove the index without resetting it if 'lat_bins' already exists    
    SubsurfacePhotonDFAddedKd = SubsurfacePhotonDFAddedKd.droplevel(0)
    return SubsurfacePhotonDFAddedKd


# Updated function to apply kd calculation beam-by-beam
def process_kd_calculation(Final_filtered_subsurface_photon_dataset):
    # Initialize list to store results for each beam
    kd_beam_datasets = []

    # Group by 'beam_id' to process each beam independently
    for beam_id, beam_data in Final_filtered_subsurface_photon_dataset.groupby('beam_id'):
        logging.info(f"Calculating Kd for beam: {beam_id}")

        # Apply the calculate_kd function to the current beam's dataset
        SubsurfacePhotonDFAddedKd = calculate_kd(beam_data)

        # Add a column to track the beam_id in the results
        SubsurfacePhotonDFAddedKd['beam_id'] = beam_id

        # Append the result to the list
        kd_beam_datasets.append(SubsurfacePhotonDFAddedKd)

    # Combine results from all beams into a single DataFrame
    combined_kd_dataset = pd.concat(kd_beam_datasets, ignore_index=True)

    return combined_kd_dataset