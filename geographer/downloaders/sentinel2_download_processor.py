"""SingleImgDownloader for downloading Sentinel-2 images form Copernicus Sci-
hub.

Should be easily extendable to Sentinel-1.
"""

import os
from pathlib import Path
from zipfile import ZipFile

from geographer.downloaders.base_download_processor import ImgDownloadProcessor
from geographer.downloaders.sentinel2_safe_unpacking import safe_to_geotif_L2A
from geographer.utils.utils import transform_shapely_geometry


class Sentinel2Processor(ImgDownloadProcessor):
    """Processes downloads of Sentinel-2 products from Copernicus Sci-hub."""

    def process(
        self,
        img_name: str,
        download_dir: Path,
        images_dir: Path,
        return_bounds_in_crs_epsg_code: int,
        resolution: int,
        **kwargs,
    ) -> dict:
        """Extracts downloaded sentinel-2 zip file to a .SAFE directory, then
        processes/converts to a GeoTiff image, deletes the zip file, puts the
        GeoTiff image in the right directory, and returns information about the
        img in a dict.

        Args:
            img_name: The name of the image.
            in_dir: The directory containing the zip file.
            out_dir: The directory to save the
            convert_to_crs_epsg: The EPSG code to use to create the image bounds
                property.  # TODO: this name might not be appropriate as it
                suggests that the image geometries will be converted into that crs.
            resolution: resolution.

        Returns:
            return_dict: Contains information about the downloaded product.
        """

        filename_no_extension = Path(img_name).stem
        zip_filename = filename_no_extension + ".zip"
        safe_path = download_dir / f"safe_files/{filename_no_extension}.SAFE"
        zip_path = download_dir / zip_filename

        # extract zip to SAFE
        with ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(download_dir / Path("safe_files/"))
        os.remove(zip_path)
        # convert SAFE to GeoTiff
        conversion_dict = safe_to_geotif_L2A(safe_root=Path(safe_path),
                                             resolution=resolution,
                                             outdir=images_dir)

        orig_crs_epsg_code = int(conversion_dict["crs_epsg_code"])
        img_bounding_rectangle_orig_crs = conversion_dict[
            "img_bounding_rectangle"]
        img_bounding_rectangle = transform_shapely_geometry(  # convert to standard crs
            img_bounding_rectangle_orig_crs,
            from_epsg=orig_crs_epsg_code,
            to_epsg=return_bounds_in_crs_epsg_code)
        return {
            'img_name': img_name,
            'geometry': img_bounding_rectangle,
            'orig_crs_epsg_code': orig_crs_epsg_code,
            'img_processed?': True,
        }
