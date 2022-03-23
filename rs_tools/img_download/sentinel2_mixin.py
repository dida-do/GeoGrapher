"""MixIn that implements downloading sentinel-2 images."""

import configparser
import itertools
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import rasterio as rio
from geopandas import GeoSeries
from scipy.ndimage import zoom
from sentinelsat import SentinelAPI
from sentinelsat.exceptions import UnauthorizedError, ServerError, QueryLengthError, QuerySyntaxError
from shapely import wkt
from shapely.geometry import box

from rs_tools.errors import NoImgsForPolygonFoundError
from rs_tools.img_download.sentinel2_safe_unpacking import safe_to_geotif_L2A
from rs_tools.utils.utils import transform_shapely_geometry

NO_DATA_VAL = 0  # No data value for sentinel 2

# logger
log = logging.getLogger(__name__)


class Sentinel2DownloaderMixIn:
    """Downloader for Sentinel-2 images.

    Requires environment variables sentinelAPIusername and
    sentinelAPIpassword to set up the sentinel API. Assumes imgs_df has
    columns 'geometry', 'timestamp', 'orig_crs_epsg_code', and
    'img_processed?'. Subclass/modify if you need other columns.
    """

    @property
    def sentinel2_producttype(self) -> str:
        return self._params_dict['sentinel2_producttype']

    @sentinel2_producttype.setter
    def sentinel2_producttype(self, new_sentinel2_producttype: str):
        self._params_dict['sentinel2_producttype'] = new_sentinel2_producttype

    @property
    def sentinel2_resolution(self):
        return self._params_dict['sentinel2_resolution']

    @sentinel2_resolution.setter
    def sentinel2_resolution(self, new_sentinel2_resolution: str):
        self._params_dict['sentinel2_resolution'] = new_sentinel2_resolution

    @property
    def sentinel2_max_percent_cloud_coverage(self):
        return self._params_dict['sentinel2_max_percent_cloud_coverage']

    @sentinel2_max_percent_cloud_coverage.setter
    def sentinel2_max_percent_cloud_coverage(
            self, new_sentinel2_max_percent_cloud_coverage: str):
        self._params_dict[
            'sentinel2_max_percent_cloud_coverage'] = new_sentinel2_max_percent_cloud_coverage

    def _download_imgs_for_polygon_sentinel2(
            self,
            polygon_name: Union[str, int],
            polygon_geometry: GeoSeries,
            download_dir: Union[str, Path],
            previously_downloaded_imgs_set: List[str],
            producttype: Optional[str] = None,
            resolution: Optional[int] = None,
            max_percent_cloud_coverage: Optional[int] = None,
            date:
        Optional[
            Any] = None,  # See here for type https://sentinelsat.readthedocs.io/en/latest/api_reference.html
            area_relation: Optional[str] = None,
            username: Optional[str] = None,
            password: Optional[str] = None,
            **kwargs) -> dict:
        """Downloads a sentinel-2 image fully containing the polygon, returns a
        dict in the format needed by the associator.

        Note:
            If not given, the username and password for the Copernicus Sentinel-2 OpenAPI
            will be read from an s2_copernicus_credentials.ini in self.associator_dir.
            TODO::::

        Args:
            polygon_name: The name of the polygon, only relevant for print statements and errors.
            polygon_geometry: The areas the images shall be downloaded for.
            download_dir: Directory to save the downloaded Sentinel-2 products.
            previously_downloaded_imgs_set: A list of already downloaded products, will be used prevent double downloads.
            producttype (str): One of 'L1C'/'S2MSI1C' or 'L2A'/'S2MSI2A'
            resolution (int): One of 10, 20, or 60.
            max_percent_cloud_coverage (int): Integer between 0 and 100.
            date (Any):  See https://sentinelsat.readthedocs.io/en/latest/api_reference.html
            area_relation : See https://sentinelsat.readthedocs.io/en/latest/api_reference.html
            username: Username for Sentinel-2 API. Defaults to None, i.e. read from case s2_copernicus_credentials.inis2_copernicus_credentials.ini (see note).
            password: Password for Sentinel-2 API. Defaults to None, i.e. read from case s2_copernicus_credentials.ini (see note).

        Returns:
            info_dicts: A dictionary containing information about the images and polygons. ({'list_img_info_dicts': [img_info_dict], 'polygon_info_dict': polygon_info_dict})

        Raises:
            ValueError: Raised if an unkknown product type is given.
            NoImgsForPolygonFoundError: Raised if no downloadable images with cloud coverage less than or equal to max_percent_cloud_coverage could be found for the polygon.
        """

        # (Save args / replace by saved value if None)
        producttype, resolution, max_percent_cloud_coverage, date, area_relation = \
            self._get_and_remember_args(producttype, resolution, max_percent_cloud_coverage, date, area_relation)
        self._check_args_are_valid(producttype, resolution,
                                   max_percent_cloud_coverage)

        # Determine missing args for the sentinel query.
        rectangle_wkt: str = wkt.dumps(polygon_geometry.envelope)
        producttype = self._get_longform_producttype(producttype)

        api = self._get_api(username, password)

        try:

            # Query, remember results
            products = api.query(
                area=rectangle_wkt,
                date=date,
                area_relation=area_relation,
                producttype=producttype,
                cloudcoverpercentage=(0, max_percent_cloud_coverage))

            products = {k: v for k, v in products.items() if api.is_online(k)}

        except (UnauthorizedError, ServerError) as e:
            log.exception(str(e))
            raise
        # # The sentinelsat API can throw an exception if there are no results for a query instead of returning an empty dict ...
        # except Exception as e:
        #     # ... so in that case we set the result by hand:
        #     log.exception(str(e))
        #     products = {}

        # If we couldn't find anything, remember that, so we can deal with it later.
        if len(products) == 0:
            raise NoImgsForPolygonFoundError(
                f"No images for polygon {polygon_name} found with cloud coverage less than or equal to {max_percent_cloud_coverage}!"
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
                    f"Couldn't get the filename. Are you trying to download L1C products? Try changing the key for the products dict in the line of code above this..."
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
                except Exception as e:
                    log.warn(
                        f"Failed to download or unzip {product_metadata['title']}: {str(e)}"
                    )

        raise NoImgsForPolygonFoundError(
            f"All images for {polygon_name} failed to download.")

    def _get_longform_producttype(self, producttype):
        """Return producttype in longform as needed by the sentinel API"""
        if producttype in {'L2A', 'S2MSI2A'}:
            producttype = 'S2MSI2A'
        elif producttype in {'L1C', 'S2MSI1C'}:
            producttype = 'S2MSI1C'
        else:
            raise ValueError(f"Unknown producttype: {producttype}")

        return producttype

    def _check_args_are_valid(self, producttype, resolution,
                              max_percent_cloud_coverage):
        """Run some safety checks on the arg values"""
        if resolution not in {10, 20, 60}:
            raise ValueError(f"Unknown resolution: {resolution}")
        if max_percent_cloud_coverage < 0 or max_percent_cloud_coverage > 100:
            raise ValueError(
                f"Unknown max_percent_cloud_coverage: {max_percent_cloud_coverage}"
            )
        if producttype not in {'L1C', 'S2MSI1C', 'L2A', 'S2MSI2A'}:
            raise ValueError(f"Unknown producttype: {producttype}")

    def _get_api(self, username, password):
        # Get username and password to set up the sentinel API ...
        try:
            if None in {username, password}:
                config = configparser.ConfigParser()
                config_path = self.assoc_dir / "sentinel_scihub.ini"
                if not config_path.is_file():
                    raise FileNotFoundError
                config.read(self.assoc_dir / "sentinel_scihub.ini")
            if username is None:
                username = config["login"]["username"]
            if password is None:
                password = config["login"]["password"]
        except FileNotFoundError:
            # TODO: raise ValueError ?
            log.error(
                f"Missing 'sentinel_scihub.ini' file in {self.assoc_dir}. Need API credentials (username and password)"
            )
        except KeyError as e:
            log.error(
                f"{e}: Missing entry in 'sentinel_scihub.ini' file. Need API credentials."
            )

        # ... and instantiate the API.
        api = SentinelAPI(username, password)

        return api

    def _get_and_remember_args(self, producttype, resolution,
                               max_percent_cloud_coverage, date,
                               area_relation):
        """Replace saved arg values for those values that are None and save arg values that are not None"""
        for s2_specific_keyword_arg, value in {
            ('producttype', producttype), ('resolution', resolution),
            ('max_percent_cloud_coverage', max_percent_cloud_coverage),
            ('area_relation', area_relation), ('date', date)
        }:
            if value is None:
                try:
                    # Use saved value
                    value = getattr(self,
                                    f"sentinel2_{s2_specific_keyword_arg}")
                except (AttributeError, KeyError):
                    raise ValueError(
                        f"Need to set {s2_specific_keyword_arg} keyword argument for the sentinel2 downloader."
                    )
            else:
                # Remember value
                setattr(self, f"sentinel2_{s2_specific_keyword_arg}", value)

        return producttype, resolution, max_percent_cloud_coverage, date, area_relation

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
            resolution: int

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
        # try using rio.warp.transform_bounds instead?? maybe more accurate when comparing to APIfootprint??
        return {
            'img_name': img_name,
            'geometry': img_bounding_rectangle,
            'orig_crs_epsg_code': orig_crs_epsg_code,
            'img_processed?': True,
        }
