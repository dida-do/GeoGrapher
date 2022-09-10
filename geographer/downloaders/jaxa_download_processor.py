"""ImgDownloadProcessor for JAXA downloads."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import rasterio as rio
from shapely.geometry import box

from geographer.downloaders.base_download_processor import ImgDownloadProcessor
from geographer.utils.utils import transform_shapely_geometry

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class JAXADownloadProcessor(ImgDownloadProcessor):
    """ImgDownloadProcessor for JAXA downloads."""

    def process(
        self,
        img_name: str,
        download_dir: Path,
        images_dir: Path,
        return_bounds_in_crs_epsg_code: int,
        **kwargs,
    ) -> dict:
        """Process a downloaded JAXA file.

        Args:
            img_name: image name
            download_dir: download directory
            images_dir: images directory
            return_bounds_in_crs_epsg_code: EPSG code of crs to return image bounds in

        Returns:
            img_info_dict containing information about the image
        """
        geotif_filename = download_dir / img_name
        with rio.open(geotif_filename) as src:
            orig_crs_epsg_code = src.crs.to_epsg()
            img_bounding_rectangle = box(*src.bounds)
        img_bounding_rectangle_in_correct_crs = transform_shapely_geometry(
            img_bounding_rectangle, orig_crs_epsg_code, 4326
        )

        shutil.move(download_dir / img_name, images_dir / img_name)

        img_info_dict = {
            "orig_crs_epsg_code": orig_crs_epsg_code,
            "img_name": img_name,
            "img_processed?": True,
            "geometry": img_bounding_rectangle_in_correct_crs,
        }

        return img_info_dict
