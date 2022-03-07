from pathlib import Path

STANDARD_CRS_EPSG_CODE = 4326 # WGS84
DATA_DIR_SUBDIRS = [Path("images"), Path("labels")] # for sentinel-2, also Path("safe_files")
IMGS_DF_INDEX_NAME="img_name"
POLYGONS_DF_INDEX_NAME="polygon_name"

# MAX_PERCENT_CLOUD_COVERAGE=10
# PRODUCTTYPE='L2A' # or 'L1C'
# STANDARD_CRS_EPSG_CODE = 4326 # WGS84
# RESOLUTION = 10 # possible values for Sentinel-2 L2A: 10, 20, 60 (in meters). See here https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
# DATA_DIR_SUBDIRS = [Path("images"), Path("labels"), Path("safe_files")]
# LABEL_TYPE = 'categorical'
