"""RasterDownloadProcessor for Sentinel-2 data from Copernicus Sci-hub.

Should be easily extendable to Sentinel-1.
"""

import os
from pathlib import Path
from zipfile import ZipFile

from geographer.downloaders.base_download_processor import RasterDownloadProcessor
from geographer.downloaders.sentinel2_safe_unpacking import safe_to_geotif_L2A
from geographer.utils.utils import transform_shapely_geometry


class Sentinel2Processor(RasterDownloadProcessor):
    """Processes downloads of Sentinel-2 products from Copernicus Sci-hub."""

    def process(
        self,
        raster_name: str,
        download_dir: Path,
        rasters_dir: Path,
        return_bounds_in_crs_epsg_code: int,
        resolution: int,
        **kwargs,
    ) -> dict:
        """Process Sentinel-2 download.

        Extract downloaded sentinel-2 zip file to a .SAFE directory, then
        process/convert to a GeoTiff raster, delete the zip file, put the
        GeoTiff raster in the right directory, and return information about the
        raster in a dict.

        Args:
            raster_name: The name of the raster.
            in_dir: The directory containing the zip file.
            out_dir: The directory to save the
            convert_to_crs_epsg: The EPSG code to use to create the raster bounds
                property.  # TODO: this name might not be appropriate as it
                suggests that the raster geometries will be converted into that crs.
            resolution: resolution.

        Returns:
            return_dict: Contains information about the downloaded product.
        """
        filename_no_extension = Path(raster_name).stem
        zip_filename = filename_no_extension + ".zip"
        safe_path = download_dir / f"safe_files/{filename_no_extension}.SAFE"
        zip_path = download_dir / zip_filename

        # extract zip to SAFE
        with ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(download_dir / Path("safe_files/"))
        os.remove(zip_path)
        # convert SAFE to GeoTiff
        conversion_dict = safe_to_geotif_L2A(
            safe_root=Path(safe_path), resolution=resolution, outdir=rasters_dir
        )

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
