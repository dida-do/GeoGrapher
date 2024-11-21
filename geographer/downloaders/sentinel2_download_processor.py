"""RasterDownloadProcessor for Sentinel-2 data from Copernicus Sci-hub.

Should be easily extendable to Sentinel-1.
"""

import logging
import shutil
from pathlib import Path

from geographer.downloaders.base_download_processor import RasterDownloadProcessor
from geographer.downloaders.sentinel2_safe_unpacking import (
    NO_DATA_VAL,
    safe_to_geotif_L2A,
)
from geographer.utils.utils import transform_shapely_geometry

log = logging.getLogger(__name__)


# TODO Rename? Change docstring? Works for SAFE files/dirs
# TODO Used to be called Sentinel2Processor. Adapt documentation!
# TODO only works for L2A...
class Sentinel2SAFEProcessor(RasterDownloadProcessor):
    """Processes downloads of Sentinel-2 products from Copernicus Sci-hub."""

    def process(
        self,
        raster_name: str,
        download_dir: Path,
        rasters_dir: Path,
        return_bounds_in_crs_epsg_code: int,
        resolution: int,
        delete_safe: bool,  # TODO better name, uniformly usable for all processors?
        nodata_val: int = NO_DATA_VAL,
        **kwargs,
    ) -> dict:
        """Process Sentinel-2 download.

        Extract downloaded sentinel-2 zip file to a .SAFE directory, then
        process/convert to a GeoTiff raster, delete the zip file, put the
        GeoTiff raster in the right directory, and return information about the
        raster in a dict.

        Args:
            raster_name:
                The name of the raster.
            in_dir:
                The directory containing the zip file.
            out_dir:
                The directory to save the
            convert_to_crs_epsg:
                The EPSG code to use to create the raster bounds
                property.  # TODO: this name might not be appropriate as it
                suggests that the raster geometries will be converted into that crs.
            resolution:
                resolution.
            nodata_val:
                The nodata value to fill. Defaults to 0.

        Returns:
            return_dict: Contains information about the downloaded product.
        """
        log.info("Processing %s to a .tif file. This might take a while..")
        safe_path = download_dir / Path(raster_name).with_suffix(".SAFE")
        conversion_dict = safe_to_geotif_L2A(
            safe_root=safe_path,
            resolution=resolution,
            outdir=rasters_dir,
            nodata_val=nodata_val,
        )

        if delete_safe:
            shutil.rmtree(safe_path, ignore_errors=True)

        orig_crs_epsg_code = int(conversion_dict["crs_epsg_code"])
        raster_bounding_rectangle_orig_crs = conversion_dict[
            "raster_bounding_rectangle"
        ]
        raster_bounding_rectangle = (
            transform_shapely_geometry(  # convert to standard crs
                raster_bounding_rectangle_orig_crs,
                from_epsg=orig_crs_epsg_code,
                to_epsg=return_bounds_in_crs_epsg_code,
            )
        )
        return {
            "raster_name": raster_name,
            "geometry": raster_bounding_rectangle,
            "orig_crs_epsg_code": orig_crs_epsg_code,
            "raster_processed?": True,
        }
