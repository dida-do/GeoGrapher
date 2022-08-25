"""Utils for handling remote sensing datasets.

convert_connector_dataset_tif2npy converts a dataset of GeoTiffs to
.npys. raster_imgs_from_tif_dir generates an raster_imgs GeoDataFrame
from a directory of GeoTiffs.
"""

from geographer.utils.raster_imgs_from_tif_dir import (
    default_read_in_img_for_img_df_function, raster_imgs_from_imgs_dir)
from geographer.utils.utils import deepcopy_gdf, transform_shapely_geometry
