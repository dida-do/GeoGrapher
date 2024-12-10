"""RasterDownloadProcessor for JAXA downloads."""

import logging
import shutil
from pathlib import Path

import rasterio as rio
from shapely.geometry import box

from geographer.downloaders.base_download_processor import RasterDownloadProcessor
from geographer.utils.utils import transform_shapely_geometry

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class JAXADownloadProcessor(RasterDownloadProcessor):
    """RasterDownloadProcessor for JAXA downloads."""

    def process(
        self,
        raster_name: str,
        download_dir: Path,
        rasters_dir: Path,
        return_bounds_in_crs_epsg_code: int,
    ) -> dict:
        """Process a downloaded JAXA file.

        Args:
            raster_name: raster name
            download_dir: download directory
            rasters_dir: rasters directory
            return_bounds_in_crs_epsg_code: EPSG code of crs to return raster bounds in

        Returns:
            raster_info_dict containing information about the raster
        """
        geotif_filename = download_dir / raster_name
        with rio.open(geotif_filename) as src:
            orig_crs_epsg_code = src.crs.to_epsg()
            raster_bounding_rectangle = box(*src.bounds)
        raster_bounding_rectangle_in_correct_crs = transform_shapely_geometry(
            raster_bounding_rectangle, orig_crs_epsg_code, 4326
        )

        shutil.move(download_dir / raster_name, rasters_dir / raster_name)

        raster_info_dict = {
            "orig_crs_epsg_code": orig_crs_epsg_code,
            "raster_name": raster_name,
            "raster_processed?": True,
            "geometry": raster_bounding_rectangle_in_correct_crs,
        }

        return raster_info_dict
