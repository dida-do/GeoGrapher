"""Utils for handling remote sensing datasets.

- convert_connector_dataset_tif2npy converts a dataset of GeoTiffs to
    .npys.
- rasters_from_tif_dir generates an rasters GeoDataFrame from a
    directory of GeoTiffs.
"""

from geographer.utils.rasters_from_tif_dir import (
    default_read_in_raster_for_raster_df_function,
    rasters_from_rasters_dir,
)
from geographer.utils.utils import deepcopy_gdf, transform_shapely_geometry
