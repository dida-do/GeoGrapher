from pathlib import Path

STANDARD_CRS_EPSG_CODE = 4326  # WGS84
DATA_DIR_SUBDIRS = [Path("images"),
                    Path("labels")]  # for sentinel-2, also Path("safe_files")
RASTER_FEATURES_INDEX_NAME = "img_name"
VECTOR_DATA_INDEX_NAME = "geom_name"
