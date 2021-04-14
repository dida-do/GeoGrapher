""" 
Utils for handling remote sensing datasets.

convert_assoc_dataset_tif2npy converts a dataset of GeoTiffs to .npys.
imgs_df_from_tif_dir generates an imgs_df GeoDataFrame from a directory of GeoTiffs.
"""

from rs_tools.utils.assoc_tif2npy import convert_assoc_dataset_tif2npy
from rs_tools.utils.imgs_df_from_tif_dir import imgs_df_from_tif_dir