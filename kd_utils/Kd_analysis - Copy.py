# utils/Kd_analysis.py

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def log_model(z, kd, e0):
    return np.log(e0) - kd * z

def CalculateKdFromFilteredSubsurfacePhoton(df, vertical_res=0.8):
    if df.empty or 'lat_bins' not in df.columns:
        return pd.DataFrame({'lat_bins': [np.nan], 'kd': [np.nan], 'e0': [np.nan]})
    lat_bin_value = df['lat_bins'].iloc[0] if not df['lat_bins'].empty else np.nan
    photon_height_min = df['photon_height'].min()
    photon_height_max = df['photon_height'].max()
    if np.isnan(photon_height_min) or np.isnan(photon_height_max) or photon_height_min == photon_height_max:
        return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    height_bins_range = abs(photon_height_max - photon_height_min)
    if height_bins_range == 0:
        return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    height_bins_number = round(height_bins_range / vertical_res)
    if height_bins_number < 5:
        return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    bin_edges = np.linspace(photon_height_min, photon_height_max, num=height_bins_number)
    counts, _ = np.histogram(df['photon_height'], bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_df = pd.DataFrame({'zdepth': bin_centers, 'photon_counts': counts})
    # x value for model
    hist_df['zdepth'] = hist_df['zdepth'].max() - hist_df['zdepth']
    # y value for model
    hist_df['log_photon_counts'] = np.log(hist_df['photon_counts'].replace(0, np.nan))
    
    hist_df.loc[np.isinf(hist_df['log_photon_counts']), 'log_photon_counts'] = np.nan
    
    # Skip the regression if there are fewer than 5 datapoints
    if hist_df['log_photon_counts'].notna().sum() > 3:
        try:
            popt, _ = curve_fit(log_model, hist_df['zdepth'].dropna(), hist_df['log_photon_counts'].dropna(), p0=[1, np.exp(1)])
            kd, e0 = popt
            if kd < 0:
                kd = np.nan
        except Exception as e:
            print(f"Error in curve fitting: {e}")
            kd, e0 = np.nan, np.nan
    else:
        kd, e0 = np.nan, np.nan
    return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [kd], 'e0': [e0]})
