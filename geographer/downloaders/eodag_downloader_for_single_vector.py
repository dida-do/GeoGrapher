"""SingleRasterDownloader for all providers supported by eodag.

In particular, this downloader can be used to obtain Sentinel-2 L2A
data.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import eodag
import pandas as pd
import shapely
from eodag import EODataAccessGateway
from eodag.api.search_result import SearchResult
from eodag.utils import sanitize
from pydantic import Field, PrivateAttr
from shapely.geometry import Polygon

from geographer.downloaders.base_downloader_for_single_vector import (
    RasterDownloaderForSingleVector,
)
from geographer.errors import NoRastersForVectorFoundError
from geographer.global_constants import DUMMY_VALUE

log = logging.getLogger(__name__)


class SearchParams(dict):
    """Parameters for the `search_all` method of an EODataAccessGateway.

    Note:
        The `geom` parameter of the EODataAccessGateway.search_all method is
        omitted, because its value is determined as a `geographer` argument.

        See See https://eodag.readthedocs.io/en/latest/api_reference/core.html#eodag.api.core.EODataAccessGateway.search_all. # noqa
        for more details on most of the arguments below.

    This dictionary may include the following keys:
    - `start` (str | None):
    Start sensing time in ISO 8601 format (e.g. “1990-11-26”,
    “1990-11-26T14:30:10.153Z”, “1990-11-26T14:30:10+02:00”, …).
    If no time offset is given, the time is assumed to be given in UTC.
    - `end` (str | None):
    End sensing time in ISO 8601 format (e.g. “1990-11-26”,
    “1990-11-26T14:30:10.153Z”, “1990-11-26T14:30:10+02:00”, …).
    If no time offset is given, the time is assumed to be given in UTC.
    - `provider` (str | None):
    The provider to be used. If set, search fallback will be disabled.
    If not set, the configured preferred provider will be used at first
    before trying others until finding results. See
    https://eodag.readthedocs.io/en/stable/_modules/eodag/api/core.html#EODataAccessGateway.search. # noqa
    - `items_per_page` (int | None):
    Number of items to retrieve per page.
    - `locations` (dict[str, str] | None):
    Location filtering by name using locations configuration
    {"<location_name>"="<attr_regex>"}. For example, {"country"="PA."}
    will use the geometry of the features having the property ISO3 starting
    with 'PA' such as Panama and Pakistan in the shapefile configured with
    name=country and attr=ISO3.
    - In addition, the dictionary may contain any other keys (except ``geom``)
    compatible with the provider.
    """

    pass


class DownloadParams(dict):
    """Parameters for the `download` method of an EOProduct.

    Refer to the EOProduct documentation for more details:
    https://eodag.readthedocs.io/en/stable/api_reference/eoproduct.html

    Some parameters of the EOProduct.download method should not be used:
        - `product`: Omitted because the value is determined by `geographer`.
        - `progress_callback`: Omitted because its values cannot easily
            be JSON serialized.
        - `extract`: Omitted because geographer requires the value of this
            kwarg to be True.
        - `output_dir`: Omitted because the value is determined by `geographer`.
        - `asset`: Omitted because it does not make sense for a downloader
            for a single vector.
        - `output_extension`: Omitted for simplicity's sake.

    This dictionary may include any of the following keys:
    - `wait` (int): The wait time in minutes between two download attempts.
    - `timeout` (int): The max time in minutes to retry downloading before stopping.
    - `dl_url_params` (dict[str, str]): Additional URL parameters to pass to
    the download URL.
    - `delete_archive` (bool): Whether to delete the downloaded archives
    after extraction.
    """

    pass


FORBIDDEN_DOWNLOAD_KWARGS_KEYS = [
    "product",
    "progress_callback",
    "extract",
    "output_dir",
    "asset",
]


ASC_OR_DESC = Literal["ASC", "DESC"]
ASC_OR_DESC_VALUES = ["ASC", "DESC"]


class EodagDownloaderForSingleVector(RasterDownloaderForSingleVector):
    """Downloader for providers supported by eodag.

    Refer to the eodag documentation at
    https://eodag.readthedocs.io/en/stable/
    for more details on eodag.
    """

    eodag_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional kwargs defining an EODataAccessGateway instance. "
            "Possible keys are 'user_conf_file_path' to define a Path to "
            "the user configuration file and locations_conf_path to define "
            "a Path to the locations configuration file. "
            "See https://eodag.readthedocs.io/en/stable/api_reference/core.html#eodag.api.core.EODataAccessGateway."  # noqa
        ),
    )

    eodag_setup_logging_kwargs: dict[str, Any] = Field(
        default_factory=lambda: dict(verbose=1),
        description=(
            "Kwargs to be passed to eodag.utils.logging.setup_logging "
            "to set up eodag logging. See "
            "https://eodag.readthedocs.io/en/stable/api_reference/utils.html#eodag.utils.logging.setup_logging"  # noqa
        ),
    )

    # Note that eodag as is not defined as a field.
    # This is so the pydantic fields are json serializable.
    _eodag: EODataAccessGateway = PrivateAttr()

    def model_post_init(self, __context):
        """Perform additional initialization."""
        eodag.setup_logging(**self.eodag_setup_logging_kwargs)

        self._eodag = EODataAccessGateway(**self.eodag_kwargs)

    @property
    def eodag(self) -> EODataAccessGateway:
        """Get eodag."""
        return self._eodag

    def download(  # type: ignore
        self,
        vector_name: str | int,
        vector_geom: Polygon,
        download_dir: Path,
        previously_downloaded_rasters_set: set[str],
        *,  # downloader_params of RasterDownloaderForVectors.download start below
        search_kwargs: SearchParams | None = None,
        download_kwargs: DownloadParams | None = None,
        properties_to_save: list[str] | None = None,
        filter_property: dict[str, Any] | list[dict[str, Any]] | None = None,
        filter_online: bool = True,
        sort_by: str | tuple[str, ASC_OR_DESC] | None = None,
        suffix_to_remove: str | None = None,
    ) -> dict:
        """Download a raster for a vector feature using eodag.

        Download a raster fully containing the vector feature,
        returns a dict in the format needed by the associator.

        Note:
            The start, end, provider, items_per_page, and locations arguments correspond
            to kwargs of EODataAccessGateway.search_all (though the provider kwarg is only
            documented for the EODataAccessGateway.search). The descriptions are adapted
            from the official eodag documentation at
            https://eodag.readthedocs.io/en/latest/api_reference/core.html#eodag.api.core.EODataAccessGateway.

        Args:
            vector_name:
                name of vector feature
            vector_geom:
                Geometry of vector feature
            download_dir:
                Directory Sentinel-2 products will be downloaded to.
            previously_downloaded_rasters_set:
                Set of already downloaded products.
            search_kwargs:
                Keyword arguments for the `search_all` method of an EODataAccessGateway,
                excluding "geom". Refer to the docstring of SearchParams for more details.
            download_kwargs:
                Keyword arguments for the download` method of an EOProduct, excluding
                certain keys. Refer to the docstring of DownloadParams for more details.
            properties_to_save:
                List of property keys to extract and save from an EOProduct's
                properties dictionary. Values that cannot be stored in a
                GeoDataFrame will be replaced with the string "__DUMMY_VALUE__".
            filter_property:
                Kwargs or list of kwargs defining criteria according to which products
                should be filtered. These correspond exactly to kwargs for the
                EODataAccessGateway.filter_property method. Refer to
                https://eodag.readthedocs.io/en/stable/plugins_reference/generated/eodag.plugins.crunch.filter_property.FilterProperty.html#eodag.plugins.crunch.filter_property.FilterProperty  # noqa
                for more details.
            filter_online:
                Whether to filter the results to include only products that are online.
            sort_by:
                (Optional) A string or tuple like ("key", "ASC"|"DESC") by which to sort the results.
                If a string is provided, it will be interpreted as ("key", "ASC").
            suffix_to_remove:
                (Optional) A suffix to strip from the downloaded EOProduct's file name.
                The resulting .tif raster will use the modified file name (if applicable)
                with ".tif" appended.

        Returns:
            A dictionary containing information about the rasters.
            ({'list_raster_info_dicts': [raster_info_dict]})

        Raises:
            ValueError: Raised if an unkknown product type is given.
            NoRastersForPolygonFoundError: Raised if no downloadable rasters
            could be found for the vector feature.
        """
        search_kwargs = search_kwargs or {}
        download_kwargs = download_kwargs or {}
        properties_to_save = properties_to_save or []
        filter_property = filter_property or {}
        if isinstance(filter_property, dict):
            filter_property = [filter_property]
        sort_by = sort_by or []
        if isinstance(sort_by, (str, tuple)):
            sort_by = [sort_by]
        sort_by = [
            (entry, "ASC") if isinstance(entry, str) else entry for entry in sort_by
        ]

        self._validate_download_args(download_kwargs=download_kwargs, sort_by=sort_by)

        search_criteria = search_kwargs | {
            "geom": vector_geom,
        }

        result: SearchResult = self.eodag.search_all(**search_criteria)

        # Only keep results that contain the geometry
        result.filter_overlap(geometry=vector_geom, contains=True)

        for filter_kwargs in filter_property:
            result.filter_property(**filter_kwargs)

        if filter_online:
            result.filter_online()

        if len(result) == 0:
            raise NoRastersForVectorFoundError(
                f"No rasters for vector feature {vector_name} found with "
                f"search criteria {search_criteria}!"
            )

        if sort_by:
            # Currently only support sorting by a single key.
            # In the future, we may implement hierarchical
            # sorting by multiple keys.
            key, asc_or_desc = sort_by[0]
            if asc_or_desc == "ASC":
                reverse = False
            elif asc_or_desc == "DESC":
                reverse = True
            else:
                raise ValueError(
                    f"sort_by is {sort_by[0]}, second tuple entry must be "
                    f"one of 'ASC' or 'DESC'"
                )
            result = SearchResult(
                products=sorted(
                    result, key=lambda product: product.properties[key], reverse=reverse
                ),
                number_matched=result.number_matched,
                errors=result.errors,
            )

        # Return dicts with values to be collected in calling associator.
        raster_info_dict = {}

        for eo_product in result:

            # For the next couple of lines we are essentially following
            # the _prepare_download method of the
            # eodag.plugins.download.base.Download class to extract
            # the name of the extracted product.
            sanitized_title = sanitize(eo_product.properties["title"])
            if sanitized_title == eo_product.properties["title"]:
                collision_avoidance_suffix = ""
            else:
                collision_avoidance_suffix = "-" + sanitize(eo_product.properties["id"])
            extracted_product_file_name = sanitized_title + collision_avoidance_suffix

            if suffix_to_remove is not None:
                raster_name = (
                    extracted_product_file_name.removesuffix(suffix_to_remove) + ".tif"
                )
            else:
                raster_name = extracted_product_file_name + ".tif"

            if raster_name not in previously_downloaded_rasters_set:

                download_params = download_kwargs | dict(
                    product=eo_product,
                    output_dir=download_dir,
                    extract=True,
                )

                try:
                    location = self.eodag.download(**download_params)

                    location_name = Path(location).name
                    if location_name != extracted_product_file_name:
                        msg = (
                            "The name of the downloaded file (%s) does not "
                            "match the expected name (%s). eodag must have "
                            "changed the way they determine the file name. "
                            "Unfortunately, `geographer` relies on being able "
                            "to determine the name of the extracted file "
                            "without downloading the product. The "
                            "`EodagDownloaderForSingleVector` will have to be "
                            "updated to work with the new naming convention of"
                            "eodag. Sorry!"
                        )
                        log.error(msg, location_name, extracted_product_file_name)
                        raise RuntimeError(
                            msg % (msg, location_name, extracted_product_file_name)
                        )

                    # And assemble the information to be updated
                    # in the returned raster_info_dict:
                    properties_to_save_dict = {}
                    for key in properties_to_save:
                        if key in eo_product.properties:
                            val = eo_product.properties.get(key)
                            definitely_accepted_types = (
                                str,
                                int,
                                float,
                                type(None),
                                date,
                                datetime,
                                shapely.geometry.base.BaseGeometry,
                            )
                            if not isinstance(val, definitely_accepted_types):
                                try:
                                    pd.Series([val])
                                except (TypeError, ValueError):
                                    val = DUMMY_VALUE
                            properties_to_save_dict[key] = val
                    raster_info_dict.update(properties_to_save_dict)
                    raster_info_dict["raster_name"] = raster_name
                    raster_info_dict["raster_processed?"] = False

                    return {"list_raster_info_dicts": [raster_info_dict]}

                except Exception as exc:
                    log.warning(
                        "Failed to download, extract, or process %s: %s",
                        eo_product,
                        str(exc),
                    )

        raise NoRastersForVectorFoundError(
            f"All rasters for {vector_name} failed to download."
        )

    def _validate_download_args(self, download_kwargs: DownloadParams, sort_by: list):
        """Validate download arguments."""
        for key in download_kwargs:
            if key in FORBIDDEN_DOWNLOAD_KWARGS_KEYS:
                msg = "The key '%s' is forbidden and cannot be used."
                log.error(msg, key)
                raise ValueError(msg % key)

        for key, asc_or_desc in sort_by:
            if asc_or_desc not in ASC_OR_DESC_VALUES:
                msg = (
                    "Found %s as second entry of a sort_by pair. "
                    "Must be one of 'ASC' or 'DESC'!"
                )
                log.error(msg, asc_or_desc)
                raise ValueError(msg % asc_or_desc)

        if len(sort_by) > 1:
            msg = (
                "At the moment sorting is only supported for a single key at a time. "
                "The length of the sort_by list must be at most 1."
            )
            log.error(msg)
            raise ValueError(msg)
