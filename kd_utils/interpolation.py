# utils/interpolation.py

import scipy.interpolate

def interpolate_labels(segment_lat, labels):
    model = scipy.interpolate.interp1d(segment_lat, labels, fill_value="extrapolate")
    return model

def apply_interpolation(model, lat_ph):
    return model(lat_ph)

def geoid_correction(lat_ph, segment_lat, geoid):
    model = interpolate_labels(segment_lat, geoid)
    return apply_interpolation(model, lat_ph)

def refraction_correction(lat_ph, segment_lat, ref_elev, ref_azimuth):
    elev_model = interpolate_labels(segment_lat, ref_elev)
    azimuth_model = interpolate_labels(segment_lat, ref_azimuth)
    return apply_interpolation(elev_model, lat_ph), apply_interpolation(azimuth_model, lat_ph)
