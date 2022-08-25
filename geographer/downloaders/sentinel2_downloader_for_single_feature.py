"""SingleImgDownloader for downloading Sentinel-2 images form Copernicus Sci-
hub.

Should be easily extendable to Sentinel-1.
"""

import configparser
import logging
import os
from pathlib import Path
from typing import Any, Set, Union
from zipfile import ZipFile

from sentinelsat import SentinelAPI
from sentinelsat.exceptions import ServerError, UnauthorizedError
from shapely import wkt
from shapely.geometry import Polygon

from geographer.downloaders.base_downloader_for_single_feature import \
    ImgDownloaderForSingleVectorFeature
from geographer.downloaders.sentinel2_safe_unpacking import safe_to_geotif_L2A
from geographer.errors import NoImgsForVectorFeatureFoundError
from geographer.utils.utils import transform_shapely_geometry

# logger
log = logging.getLogger(__name__)


class SentinelDownloaderForSingleVectorFeature(
        ImgDownloaderForSingleVectorFeature):
    """Downloader for Sentinel-2 images.

    Requires environment variables sentinelAPIusername and
    sentinelAPIpassword to set up the sentinel API. Assumes raster_imgs
    has columns 'geometry', 'timestamp', 'orig_crs_epsg_code', and
    'img_processed?'. Subclass/modify if you need other columns.
    """

    def download(
        self,
        feature_name: Union[str, int],
        feature_geom: Polygon,
        download_dir: Path,
        previously_downloaded_imgs_set: Set[str],
        producttype: str,
        resolution: int,
        max_percent_cloud_coverage: int,
        date:
        Any,  # See https://sentinelsat.readthedocs.io/en/latest/api_reference.html
        area_relation: str,
        credentials_ini_path: Path,
        **kwargs,
    ) -> dict:
        """Downloads a sentinel-2 image fully containing the vector feature,
        returns a dict in the format needed by the associator.

        Note:
            If not given, the username and password for the Copernicus Sentinel-2 OpenAPI
            will be read from an s2_copernicus_credentials.ini in self.associator_dir.

        Args:
            feature_name: name of vector feature
            feature_geom: geometry of vector feature
            download_dir: Directory Sentinel-2 products will be downloaded to.
            previously_downloaded_imgs_set: Set of already downloaded products.
            producttype: One of 'L1C'/'S2MSI1C' or 'L2A'/'S2MSI2A'
            resolution: One of 10, 20, or 60.
            max_percent_cloud_coverage: Integer between 0 and 100.
            date:  See https://sentinelsat.readthedocs.io/en/latest/api_reference.html
            area_relation : See https://sentinelsat.readthedocs.io/en/latest/api_reference.html
            credentials_ini_path: Path to ini file containing API credentials.

        Returns:
            A dictionary containing information about the images. ({'list_img_info_dicts': [img_info_dict]})

        Raises:
            ValueError: Raised if an unkknown product type is given.
            NoImgsForPolygonFoundError: Raised if no downloadable images with cloud coverage less than or equal to max_percent_cloud_coverage could be found for the vector feature.
        """

        self._check_args_are_valid(producttype, resolution,
                                   max_percent_cloud_coverage)

        # Determine missing args for the sentinel query.
        rectangle_wkt: str = wkt.dumps(feature_geom.envelope)
        producttype = self._get_longform_producttype(producttype)

        api = self._get_api(credentials_ini_path)

        try:

            # Query, remember results
            products = api.query(
                area=rectangle_wkt,
                date=date,
                area_relation=area_relation,
                producttype=producttype,
                cloudcoverpercentage=(0, max_percent_cloud_coverage))

            products = {k: v for k, v in products.items() if api.is_online(k)}

        except (UnauthorizedError, ServerError) as exc:
            log.exception(str(exc))
            raise

        # If we couldn't find anything, remember that, so we can deal with it later.
        if len(products) == 0:
            raise NoImgsForVectorFeatureFoundError(
                f"No images for vector feature {feature_name} found with cloud coverage less than or equal to {max_percent_cloud_coverage}!"
            )

        # Return dicts with values to be collected in calling associator.
        img_info_dict = {}

        # If the query was succesful, ...
        products_list = list(products.keys())
        products_list = sorted(
            products_list, key=lambda x: products[x]["cloudcoverpercentage"])
        # ... iterate over the products ordered by cloud coverage
        for product_id in products_list:

            product_metadata = api.get_product_odata(product_id, full=True)

            try:
                # (this key might have to be 'filename' (minus the .SAFE at the end) for L1C products?)
                img_name = product_metadata['title'] + ".tif"
            except:
                raise Exception(
                    "Couldn't get the filename. Are you trying to download L1C products? Try changing the key for the products dict in the line of code above this..."
                )

            if img_name not in previously_downloaded_imgs_set:
                try:
                    api.download(product_id, directory_path=download_dir)
                    zip_path = download_dir / (product_metadata['title'] +
                                               ".zip")
                    with ZipFile(zip_path) as zip_ref:
                        assert zip_ref.testzip() is None

                    # And assemble the information to be updated in the returned img_info_dict:
                    img_info_dict['img_name'] = img_name
                    img_info_dict['img_processed?'] = False
                    img_info_dict['timestamp'] = product_metadata[
                        'Date'].strftime("%Y-%m-%d-%H:%M:%S")

                    return {'list_img_info_dicts': [img_info_dict]}
                except Exception as exc:
                    log.warning("Failed to download or unzip %s: %s",
                                product_metadata['title'], str(exc))

        raise NoImgsForVectorFeatureFoundError(
            f"All images for {feature_name} failed to download.")

    def _get_longform_producttype(self, producttype):
        """Return producttype in longform as needed by the sentinel API."""
        if producttype in {'L2A', 'S2MSI2A'}:
            producttype = 'S2MSI2A'
        elif producttype in {'L1C', 'S2MSI1C'}:
            producttype = 'S2MSI1C'
        else:
            raise ValueError(f"Unknown producttype: {producttype}")

        return producttype

    @staticmethod
    def _check_args_are_valid(producttype, resolution,
                              max_percent_cloud_coverage):
        """Run some safety checks on the arg values."""
        if resolution not in {10, 20, 60}:
            raise ValueError(f"Unknown resolution: {resolution}")
        if max_percent_cloud_coverage < 0 or max_percent_cloud_coverage > 100:
            raise ValueError(
                f"Unknown max_percent_cloud_coverage: {max_percent_cloud_coverage}"
            )
        if producttype not in {'L1C', 'S2MSI1C', 'L2A', 'S2MSI2A'}:
            raise ValueError(f"Unknown producttype: {producttype}")

    def _get_api(self, config_path: Path):
        # Get username and password to set up the sentinel API ...
        try:
            if config_path is None:
                raise ValueError(
                    "Need username and password or config_path to .ini file containing username and password"
                )
            config = configparser.ConfigParser()
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Can't find .ini file containing username and password in {config_path}"
                )
            config.read(config_path)
            username = config["login"]["username"]
            password = config["login"]["password"]
        except KeyError as exc:
            log.error(
                "Missing entry in 'sentinel_scihub.ini' file. Need API credentials. %s",
                exc)

        # ... and instantiate the API.
        api = SentinelAPI(username, password)

        return api

    def _process_downloaded_img_file_sentinel2(self, img_name: str,
                                               in_dir: Union[str, Path],
                                               out_dir: Union[str, Path],
                                               convert_to_crs_epsg: int,
                                               resolution: int,
                                               **kwargs) -> dict:
        """Extracts downloaded sentinel-2 zip file to a .SAFE directory, then
        processes/converts to a GeoTiff image, deletes the zip file, puts the
        GeoTiff image in the right directory, and returns information about the
        img in a dict.

        Args:
            img_name: The name of the image.
            in_dir: The directory containing the zip file.
            out_dir: The directory to save the
            convert_to_crs_epsg: The EPSG code to use to create the image bounds property.  # TODO: this name might not be appropriate as it suggests that the image geometries will be converted into that crs.
            resolution: resolution

        Returns:
            return_dict: Contains information about the downloaded product.
        """

        filename_no_extension = Path(img_name).stem
        zip_filename = filename_no_extension + ".zip"
        safe_path = Path(in_dir) / f"safe_files/{filename_no_extension}.SAFE"
        zip_path = Path(in_dir) / zip_filename

        # extract zip to SAFE
        with ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(in_dir / Path("safe_files/"))
        os.remove(zip_path)
        # convert SAFE to GeoTiff
        conversion_dict = safe_to_geotif_L2A(safe_root=Path(safe_path),
                                             resolution=resolution,
                                             outdir=out_dir)

        orig_crs_epsg_code = int(conversion_dict["crs_epsg_code"])
        img_bounding_rectangle_orig_crs = conversion_dict[
            "img_bounding_rectangle"]
        img_bounding_rectangle = transform_shapely_geometry(  # convert to standard crs
            img_bounding_rectangle_orig_crs,
            from_epsg=orig_crs_epsg_code,
            to_epsg=convert_to_crs_epsg)
        return {
            'img_name': img_name,
            'geometry': img_bounding_rectangle,
            'orig_crs_epsg_code': orig_crs_epsg_code,
            'img_processed?': True,
        }
