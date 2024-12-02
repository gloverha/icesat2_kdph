# config.py

workspace_path = './'

atl03_path = 'data/'

#US, Orgon
# ATL03_h5_file = "processed_ATL03_20181206092124_10500106_005_01.h5"#
#South America
# ATL03_h5_file = "processed_ATL03_20220530041141_10391513_005_02.h5"#


#Alaska
# ATL03_h5_file = "processed_ATL03_20210801125753_05941205_005_01.h5"#


# Hawaii line without afterpulses
# ATL03_h5_file = "processed_ATL03_20220122044818_04721407_006_01.h5"#

# ChesapeakeBay
# atl03_file = 'processed_ATL03_20230825074121_10102002_006_02.h5'

# India
# ATL03_h5_file = "processed_ATL03_20200331204156_00810707_006_01.h5"#

#China
# ATL03_h5_file = "processed_ATL03_20191128115256_09560502_006_01.h5"#
# ATL03_h5_file = "processed_ATL03_20190530203316_09560302_006_02.h5"#

# ATL03_h5_file = "processed_ATL03_20190829161305_09560402_006_02.h5"#

other_data_path = 'data/'

# shoreline_data = 'Shorelines/GeoPkgGlobalShoreline.gpkg'
shoreline_data = 'ne_10m_land.shp'


#load the Global bathy dataset
gebco_path = 'gebco_2024_geotiff/'


horizontal_res = 500

vertical_res = 0.25

# Use calculated sea height to determine photons at 0.5m below peak
subsurface_thresh = 0.5

# subsurface distance below the sea surface beginning to account
ignore_subsurface_height_thres = -1

output_path = 'results/'

######################################################################
# import argparse

# def get_args():
#     """
#     Parses arguments for the ICESat-2 HLS analysis script.

#     Args:
#         args (list, optional): A list of arguments to parse. If None, arguments are taken from the command line.
    
#     Returns:
#         argparse.Namespace: Parsed arguments.
#     """
#     parser = argparse.ArgumentParser(description="Configure ICESat-2 HLS analysis script.")
    
#     # General paths
#     parser.add_argument("--workspace_path", type=str, default='.IS2_kd_py/',
#                         help="Root path for all data processing.")
#     parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/',
#                         help="Base path for ICESat-2 ATL03 data.")
#     parser.add_argument("--atl03_file", type=str, default="processed_ATL03_20230825074121_10102002_006_02.h5",
#                         help="Name of the ATL03 H5 file to process.")
#     parser.add_argument("--other_data_path", type=str, default='Dataset/',
#                         help="Path to the current working directory.")
#     parser.add_argument("--output_path", type=str, default='Results/',
#                         help="Directory for saving results.")
    
#     # Shoreline data
#     parser.add_argument("--shoreline_data", type=str, 
#                         default='Shorelines/GeoPkgGlobalShoreline.gpkg',
#                         help="Path to the global shoreline dataset.")
    
#     # Bathymetry datasets
#     parser.add_argument("--gebco_path", nargs='+', type=str, 
#                         default='Bathy/gebco_2024_geotiff/',
#                         help="Path to GEBCO bathymetry datasets.")
    
#     # Resolution settings
#     parser.add_argument("--horizontal_res", type=int, default=500,
#                         help="Horizontal resolution for the analysis (in meters).")
#     parser.add_argument("--vertical_res", type=float, default=0.25,
#                         help="Vertical resolution for the analysis.")
    
#     # Analysis parameters
#     parser.add_argument("--subsurface_thresh", type=float, default=0.5,
#                         help="Threshold for photons below the sea surface.")
#     parser.add_argument("--ignore_subsurface_height_thres", type=float, default=-1,
#                         help="Maximum depth thres for Kd calculations (in meters).")
    
#     # If args is None, argparse will use sys.argv. Otherwise, use provided args.
#     return parser.parse_args(args)

# if __name__ == "__main__":
#     args = get_args()
#     # Access parameters like this:
#     atl03_file_path = args.workspace_path + args.atl03_file
    
#     print("Processing file:", atl03_file_path)
#     print("Output path:", args.output_path)
