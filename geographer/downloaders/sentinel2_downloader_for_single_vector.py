"""SingleRasterDownloader for Sentinel-2 rasters from Copernicus Sci-hub.

Should be easily extendable to Sentinel-1.
"""

import configparser
import logging
from pathlib import Path
from typing import Any, Union
from zipfile import ZipFile

from sentinelsat import SentinelAPI
from sentinelsat.exceptions import ServerError, UnauthorizedError
from shapely import wkt
from shapely.geometry import Polygon

from geographer.downloaders.base_downloader_for_single_vector import (
    RasterDownloaderForSingleVector,
)
from geographer.errors import NoRastersForVectorFoundError

# logger
log = logging.getLogger(__name__)


class SentinelDownloaderForSingleVector(RasterDownloaderForSingleVector):
    """Downloader for Sentinel-2 rasters.

    Requires environment variables sentinelAPIusername and
    sentinelAPIpassword to set up the sentinel API. Assumes rasters has
    columns 'geometry', 'timestamp', 'orig_crs_epsg_code', and
    'raster_processed?'. Subclass/modify if you need other columns.

    See
    https://sentinelsat.readthedocs.io/en/latest/api_reference.html
    for details on args passed to the API (e.g. date).
    """

    def download(  # type: ignore
        self,
        vector_name: Union[str, int],
        vector_geom: Polygon,
        download_dir: Path,
        previously_downloaded_rasters_set: set[str],
        producttype: str,
        resolution: int,
        max_percent_cloud_coverage: int,
        date: Any,
        area_relation: str,
        credentials: Union[tuple[str, str], Path, str],
        **kwargs,
    ) -> dict:
        """Download a S-2 raster for a vector feature.

        Download a sentinel-2 raster fully containing the vector feature,
        returns a dict in the format needed by the associator.

        Note:
            If not given, the username and password for the Copernicus Sentinel-2
            OpenAPI will be read from an s2_copernicus_credentials.ini in
            self.associator_dir.

        Args:
            vector_name: name of vector feature
            vector_geom: geometry of vector feature
            download_dir: Directory Sentinel-2 products will be downloaded to.
            previously_downloaded_rasters_set: Set of already downloaded products.
            producttype: One of 'L1C'/'S2MSI1C' or 'L2A'/'S2MSI2A'
            resolution: One of 10, 20, or 60.
            max_percent_cloud_coverage: Integer between 0 and 100.
            date:  See https://sentinelsat.readthedocs.io/en/latest/api_reference.html
            area_relation : See
                https://sentinelsat.readthedocs.io/en/latest/api_reference.html
            credentials: Tuple of username and password or
                Path or str to ini file containing API credentials.

        Returns:
            A dictionary containing information about the rasters.
            ({'list_raster_info_dicts': [raster_info_dict]})

        Raises:
            ValueError: Raised if an unkknown product type is given.
            NoRastersForPolygonFoundError: Raised if no downloadable rasters with cloud
            coverage less than or equal to max_percent_cloud_coverage could be found
            for the vector feature.
        """
        self._check_args_are_valid(producttype, resolution, max_percent_cloud_coverage)

        # Determine missing args for the sentinel query.
        rectangle_wkt: str = wkt.dumps(vector_geom.envelope)
        producttype = self._get_longform_producttype(producttype)

        api = self._get_api(credentials)

        try:

            # Query, remember results
            products = api.query(
                area=rectangle_wkt,
                date=date,
                area_relation=area_relation,
                producttype=producttype,
                cloudcoverpercentage=(0, max_percent_cloud_coverage),
            )

            products = {k: v for k, v in products.items() if api.is_online(k)}

        except (UnauthorizedError, ServerError) as exc:
            log.exception(str(exc))
            raise

        # If we couldn't find anything, remember that, so we can deal with it later.
        if len(products) == 0:
            raise NoRastersForVectorFoundError(
                f"No rasters for vector feature {vector_name} found with "
                f"cloud coverage less than or equal to {max_percent_cloud_coverage}!"
            )

        # Return dicts with values to be collected in calling associator.
        raster_info_dict = {}

        # If the query was succesful, ...
        products_list = list(products.keys())
        products_list = sorted(
            products_list, key=lambda x: products[x]["cloudcoverpercentage"]
        )
        # ... iterate over the products ordered by cloud coverage
        for product_id in products_list:

            product_metadata = api.get_product_odata(product_id, full=True)

            try:
                # (this key might have to be 'filename'
                # (minus the .SAFE at the end) for L1C products?)
                raster_name = product_metadata["title"] + ".tif"
            except Exception as exc:
                raise Exception(
                    "Couldn't get the filename. Are you trying to download L1C "
                    "products? Try changing the key for the products dict in the "
                    "line of code above this..."
                ) from exc

            if raster_name not in previously_downloaded_rasters_set:
                try:
                    api.download(product_id, directory_path=download_dir)
                    zip_path = download_dir / (product_metadata["title"] + ".zip")
                    with ZipFile(zip_path) as zip_ref:
                        assert zip_ref.testzip() is None

                    # And assemble the information to be updated
                    # in the returned raster_info_dict:
                    raster_info_dict["raster_name"] = raster_name
                    raster_info_dict["raster_processed?"] = False
                    raster_info_dict["timestamp"] = product_metadata["Date"].strftime(
                        "%Y-%m-%d-%H:%M:%S"
                    )

                    return {"list_raster_info_dicts": [raster_info_dict]}
                except Exception as exc:
                    log.warning(
                        "Failed to download or unzip %s: %s",
                        product_metadata["title"],
                        str(exc),
                    )

        raise NoRastersForVectorFoundError(
            f"All rasters for {vector_name} failed to download."
        )

    def _get_longform_producttype(self, producttype: str):
        """Return producttype in longform as needed by the sentinel API."""
        if producttype in {"L2A", "S2MSI2A"}:
            producttype = "S2MSI2A"
        elif producttype in {"L1C", "S2MSI1C"}:
            producttype = "S2MSI1C"
        else:
            raise ValueError(f"Unknown producttype: {producttype}")

        return producttype

    @staticmethod
    def _check_args_are_valid(
        producttype: str,
        resolution: int,
        max_percent_cloud_coverage: int,
    ):
        """Run some safety checks on the arg values."""
        if resolution not in {10, 20, 60}:
            raise ValueError(f"Unknown resolution: {resolution}")
        if max_percent_cloud_coverage < 0 or max_percent_cloud_coverage > 100:
            raise ValueError(
                f"Unknown max_percent_cloud_coverage: {max_percent_cloud_coverage}"
            )
        if producttype not in {"L1C", "S2MSI1C", "L2A", "S2MSI2A"}:
            raise ValueError(f"Unknown producttype: {producttype}")

    def _get_api(self, credentials: Union[tuple[str, str], Path, str]):
        # Get username and password to set up the sentinel API ...
        if (
            isinstance(credentials, tuple)
            and len(credentials) == 2
            and all(isinstance(cred, str) for cred in credentials)
        ):
            username, password = credentials
        elif isinstance(credentials, (str, Path)):
            try:
                config = configparser.ConfigParser()
                if not credentials.is_file():
                    raise FileNotFoundError(
                        "Can't find .ini file containing username and password in "
                        f"{credentials}"
                    )
                config.read(credentials)
                username = config["login"]["username"]
                password = config["login"]["password"]
            except KeyError as exc:
                log.error(
                    "Missing entry in 'sentinel_scihub.ini' file. "
                    "Need API credentials. %s",
                    exc,
                )
        else:
            raise TypeError(
                "Need username and password or config_path to .ini file "
                "containing username and password"
            )

        # ... and instantiate the API.
        api = SentinelAPI(username, password)

        return api
