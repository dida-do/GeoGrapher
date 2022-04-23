"""Utils for handling remote sensing datasets.

convert_assoc_dataset_tif2npy converts a dataset of GeoTiffs to .npys.
img_data_from_tif_dir generates an img_data GeoDataFrame from a directory
of GeoTiffs.
"""

from rs_tools.utils.img_data_from_tif_dir import (
    default_read_in_img_for_img_df_function, img_data_from_imgs_dir)
from rs_tools.utils.utils import deepcopy_gdf, transform_shapely_geometry

