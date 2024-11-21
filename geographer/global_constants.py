"""Global constants."""

from pathlib import Path

STANDARD_CRS_EPSG_CODE = 4326  # WGS84
DATA_DIR_SUBDIRS = [
    Path("rasters"),
    Path("labels"),
]
RASTER_IMGS_INDEX_NAME = "raster_name"
VECTOR_FEATURES_INDEX_NAME = "vector_name"

DUMMY_VALUE = "__DUMMY_VALUE__"
