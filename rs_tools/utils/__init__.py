""" 
Utils for handling remote sensing datasets.

convert_assoc_dataset_tif2npy converts a dataset of GeoTiffs to .npys.
imgs_df_from_tif_dir generates an imgs_df GeoDataFrame from a directory of GeoTiffs.
"""

from rs_tools.utils.imgs_df_from_tif_dir import (
    default_read_in_img_for_img_df_function, imgs_df_from_imgs_dir)
from rs_tools.utils.utils import deepcopy_gdf, transform_shapely_geometry
