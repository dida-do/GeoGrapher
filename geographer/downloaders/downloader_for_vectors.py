"""Download a targeted number of rasters per vector feature."""

from __future__ import annotations

import logging
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Union

from geopandas import GeoDataFrame
from pydantic import BaseModel
from shapely.ops import unary_union
from tqdm.auto import tqdm

from geographer import Connector
from geographer.base_model_dict_conversion.save_load_base_model_mixin import (
    SaveAndLoadBaseModelMixIn,
)
from geographer.downloaders.base_download_processor import RasterDownloadProcessor
from geographer.downloaders.base_downloader_for_single_vector import (
    RasterDownloaderForSingleVector,
)
from geographer.errors import (
    NoRastersForVectorFoundError,
    RasterAlreadyExistsError,
    RasterDownloadError,
)
from geographer.utils.utils import concat_gdfs

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


class RasterDownloaderForVectors(BaseModel, SaveAndLoadBaseModelMixIn):
    """Class that downloads a targeted number of rasters per vector feature."""

    downloader_for_single_vector: RasterDownloaderForSingleVector
    download_processor: RasterDownloadProcessor
    temp_dir_relative_path: Union[Path, str] = "temp_download_dir"

    def download(
        self,
        connector: Path | str | Connector,
        vector_names: str | int | list[int] | list[str] | None = None,
        target_raster_count: int = 1,
        filter_out_vectors_contained_in_union_of_intersecting_rasters: bool = False,
        shuffle: bool = True,
        downloader_params: dict[str, Any] | None = None,
        processor_params: dict[str, Any] | None = None,
    ):
        """Download a targeted number of rasters per vector feature.

        For each vector feature with fewer than `target_raster_count`
        rasters fully containing it, this function attempts to download
        additional rasters to meet the target. The new rasters are integrated
        into the dataset/connector immediately after downloading, updating
        the  raster count for the vector feature before proceeding to the
        next feature.

        Warning:
            The target number of downloads depends on `target_raster_count`
            and the current `raster_count` (number of rasters fully containing
            the vector feature). For vector features (e.g., polygons) too large to
            be fully contained in any raster, the `raster_count` will remain zero,
            and every call to this method will attempt to download `target_raster_count`
            rasters (or raster series). To avoid this, use the 
            `filter_out_vectors_contained_in_union_of_intersecting_rasters` argument.

        Args:
            vector_names:
                Optional vector_name or list of vector_names to download
                rasters for. Defaults to None, i.e. consider all vector features in
                connector.vectors.
            downloader:
                One of 'sentinel2' or 'jaxa'. Defaults, if possible, to
                previously used downloader.
            target_raster_count:
                Target for number of rasters per vector feature in
                the dataset after downloading. The actual number of rasters for each
                vector feature P that fully contain it could be lower if there
                are not enough rasters available or higher if after downloading
                num_target_rasters_per_vector rasters for P P is also contained
                in rasters downloaded for other vector features.
            filter_out_vectors_contained_in_union_of_intersecting_rasters:
                Useful when dealing with 'large' vector features. Defaults to False.
            shuffle:
                Whether to shuffle order of vector features for which rasters
                will be downloaded. Might in practice prevent an uneven distribution
                of the raster count for repeated downloads. Defaults to True.
            downloader_params:
                (Optional) keyword arguments to pass to the
                downloader_for_single_vector.download. Corresponds to ``**params`` of
                download method of the the abstract base class
                RasterDownloaderForSingleVector. In particular, the keywords
                vector_name, vector_geom, download_dir, and
                previously_downloaded_rasters_set corresponding to the other
                arguments are not allowed.
            processor_params:
                Optional additional keyword arguments passed to
                download_processor.process as ``**params``. In particular, the keywords
                raster_name, download_dir, rasters_dir, and
                return_bounds_in_crs_epsg_code are not allowed.

        Returns:
            None

        Warning:
            In the case that the vector vector features are polygons it's easy to come
            up with examples where the raster count distribution (i.e. distribution of
            rasters per polygon) becomes unbalanced particularly if
            num_target_rasters_per_vector is large. These scenarios are not necessarily
            very likely, but possible. As an example, if one wants to download say 5
            rasters rasters for a polygon that is not fully contained in any raster in
            the dataset and if there does not exist a raster we can download that fully
            contains it but there are 20 disjoint sets of rasters we can download that
            jointly cover the polygon then these 20 disjoint sets will all be
            downloaded.
        """
        downloader_params = downloader_params or {}
        processor_params = processor_params or {}
        if not isinstance(connector, Connector):
            connector = Connector.from_data_dir(connector)
        connector.rasters_dir.mkdir(parents=True, exist_ok=True)
        temp_download_dir = connector.data_dir / self.temp_dir_relative_path
        temp_download_dir.mkdir(parents=True, exist_ok=True)

        vectors_for_which_to_download = self._get_vectors_for_which_to_download(
            vector_names=vector_names,
            target_raster_count=target_raster_count,
            connector=connector,
            filter_out_vectors_contained_in_union_of_intersecting_rasters=filter_out_vectors_contained_in_union_of_intersecting_rasters,  # noqa: E501
        )

        if shuffle:
            random.shuffle(vectors_for_which_to_download)

        previously_downloaded_rasters_set = set(connector.rasters.index)
        # (Will be used to make sure no attempt is made to download a raster more
        # than once.)

        # Dict to keep track of rasters we've downloaded. We'll append this to
        # connector.rasters as a (geo)dataframe later
        new_raster_dicts_list = []

        pbar = tqdm(
            enumerate(
                connector.vectors[["geometry"]]
                .loc[vectors_for_which_to_download]
                .itertuples(),
                start=1,
            )
        )
        for count, (vector_name, vector_geom) in pbar:
            # vector_geom = connector.vectors.loc[vector_name, 'geometry']

            pbar.set_description(
                f"Polygon {count}/{len(vectors_for_which_to_download)}",
            )
            log.debug(
                "download_missing_rasters_for_vectors: considering "
                "vector feature %s.",
                vector_name,
            )

            # Since we process and connect each raster after downloading it, we might
            # not need to download a raster for a vector feature that earlier was
            # lacking a raster if it is now contained in one of the already downloaded
            # rasters, so need to check again that there are not enough rasters for the
            # vector feature (since the iterator above is set when it is called and
            # won't know if the self.raster_count_col_name column value has been changed
            # in the meanwhile).
            num_raster_series_to_download = (
                target_raster_count
                - connector.vectors.loc[vector_name, connector.raster_count_col_name]
            )
            if num_raster_series_to_download <= 0:
                log.debug(
                    "Skipping %s since there now enough rasters fully containing it.",
                    vector_name,
                )
                continue

            while num_raster_series_to_download > 0:
                # Try downloading a raster series and save returned dict (of dicts)
                # containing information for vectors, connector.rasters...
                try:
                    # DEBUG INFO
                    log.debug(
                        "attempting to download raster for vector feature. %s",
                        vector_name,
                    )

                    # the previously_downloaded_rasters_set argument should be used by
                    # downloader_for_single_vector should use this to make sure no
                    # attempt at downloading an already downloaded raster is made.
                    return_dict = self.downloader_for_single_vector.download(
                        vector_name=vector_name,
                        vector_geom=vector_geom,
                        download_dir=temp_download_dir,
                        previously_downloaded_rasters_set=previously_downloaded_rasters_set,  # noqa: E501
                        **downloader_params,
                    )

                # WHY DOES THIS NOT WORK?
                # except TypeError as exc:
                #     log.exception("Probably missing kwargs for\
                #     downloader_for_single_vector: {exc}")
                #     raise

                # ... unless either no rasters could be found ...
                except NoRastersForVectorFoundError as exc:
                    # ... in which case we save it in connector.vectors, ...
                    connector.vectors.loc[vector_name, "download_exception"] = repr(exc)

                    # ... log a warning, ...
                    log.warning(exc, exc_info=True)

                    # ... and break the while loop, ...
                    break

                # ... or a download error occured, ...
                except RasterDownloadError as exc:
                    connector.vectors.loc[vector_name, "download_exception"] = repr(exc)
                    log.warning(exc, exc_info=True)

                # ... or downloader_for_single_vector tried downloading a previously
                # downloaded raster.
                except RasterAlreadyExistsError:
                    log.exception(
                        "downloader_for_single_vector tried "
                        "downloading a previously downloaded raster!"
                    )

                # If the download_method call was successful ...
                else:
                    # ... we first extract the information to be appended to
                    # connector.rasters.
                    list_raster_info_dicts = return_dict["list_raster_info_dicts"]
                    # (each raster_info_dict contains the information for a new
                    # row of connector.rasters)

                    # DEBUG INFO
                    log.debug(
                        "list_raster_info_dicts is %s \n\n", list_raster_info_dicts
                    )

                    # If no rasters were downloaded, ...
                    if list_raster_info_dicts == []:
                        # ... we break the loop, since no further rasters can be found
                        break
                    # ... else ...
                    else:
                        self._run_safety_checks_on_downloaded_rasters(
                            previously_downloaded_rasters_set,
                            vector_name,
                            list_raster_info_dicts,
                        )

                        # For each download ...
                        for raster_info_dict in list_raster_info_dicts:
                            # ... process it to a raster ...
                            raster_name = raster_info_dict["raster_name"]
                            single_raster_processed_return_dict = (
                                self.download_processor.process(
                                    raster_name,
                                    temp_download_dir,
                                    connector.rasters_dir,
                                    connector.crs_epsg_code,
                                    **processor_params,
                                )
                            )

                            # ... and update the raster_info_dict with the returned
                            # information from processing. (This modifies
                            # list_raster_info_dicts, too).
                            raster_info_dict.update(single_raster_processed_return_dict)

                            # Connect the raster: Add a raster vertex to the graph,
                            # connect to all vectors vertices for which
                            # the intersection is non-empty and modify
                            # connector.vectors where necessary ...
                            connector._add_raster_to_graph_modify_vectors(
                                raster_name=raster_name,
                                raster_bounding_rectangle=raster_info_dict["geometry"],
                            )

                            # Finally, remember we downloaded the raster.
                            previously_downloaded_rasters_set.add(raster_name)

                        # update new_raster_dicts_list
                        new_raster_dicts_list += list_raster_info_dicts

                        num_raster_series_to_download -= 1

        if len(new_raster_dicts_list) > 0:
            new_rasters = self._get_new_rasters(
                new_raster_dicts_list, connector.crs_epsg_code
            )
            connector.rasters = concat_gdfs([connector.rasters, new_rasters])
            connector.save()

        # clean up
        if not list(temp_download_dir.iterdir()):
            shutil.rmtree(temp_download_dir)

    def save(self, file_path: Path | str):
        """Save downloader.

        By convention, the downloader should be saved to the connector
        subdirectory of the data directory it is supposed to operate on.
        """
        self._save(file_path)

    @staticmethod
    def _run_safety_checks_on_downloaded_rasters(
        previously_downloaded_rasters_set: set[str | int],
        vector_name: str | int,
        list_raster_info_dicts: list[dict],
    ):
        """Check no rasters have been downloaded more than once.

        Args:
            previously_downloaded_rasters_set: previously downloaded rasters
            vector_name: name of vector feature
            list_raster_info_dicts: raster_info_dicts

        Raises:
            Exception: _description_
            Exception: _description_
        """
        # Extract the new raster names ...
        new_raster_names_list = [
            raster_info_dict["raster_name"]
            for raster_info_dict in list_raster_info_dicts
        ]

        # ... and make sure we have not downloaded a raster twice
        # for the same vector feature.
        if len(new_raster_names_list) != len(set(new_raster_names_list)):
            duplicate_rasters_dict = {
                raster_name: raster_count
                for raster_name, raster_count in Counter(new_raster_names_list).items()
                if raster_count > 1
            }

            log.error(
                "Something is wrong with downloader_for_single_vector: it attempted "
                "to download the following rasters multiple times for vector feature "
                "%s: %s",
                vector_name,
                duplicate_rasters_dict,
            )

            raise Exception(
                "Something is wrong with downloader_for_single_vector: it attempted "
                "to download the following rasters multiple times for vector feature "
                f"{vector_name}: {duplicate_rasters_dict}"
            )

            # Make sure we haven't downloaded a raster that's already in the dataset.
            # (the downloader_for_single_vector method should have thrown an
            # RasterAlreadyExistsError exception in this case, but we're checking
            # again ourselves that this hasn't happened. )
        if set(new_raster_names_list) & previously_downloaded_rasters_set:
            log.error(
                "Something is wrong with downloader_for_single_vector: it downloaded "
                "raster(s) that have already been downloaded: %s",
                set(new_raster_names_list) & previously_downloaded_rasters_set,
            )

            raise Exception(
                "Something is wrong with downloader_for_single_vector: it downloaded "
                "raster(s) that have already been downloaded: "
                f"{set(new_raster_names_list) & previously_downloaded_rasters_set}"
            )

    def _get_vectors_for_which_to_download(
        self,
        vector_names: str | int | list[int] | list[str],
        target_raster_count: int,
        connector: Connector,
        filter_out_vectors_contained_in_union_of_intersecting_rasters: bool,
    ) -> list[int | str]:
        if vector_names is None:
            vectors_for_which_to_download = list(
                connector.vectors.loc[
                    connector.vectors[connector.raster_count_col_name]
                    < target_raster_count
                ].index
            )
        elif isinstance(vector_names, (str, int)):
            vectors_for_which_to_download = [vector_names]
        elif isinstance(vector_names, list) and all(
            isinstance(element, (str, int)) for element in vector_names
        ):
            vectors_for_which_to_download = vector_names
        else:
            raise TypeError(
                "The vector_names argument should be a list of vector feature names"
            )

        if not set(vectors_for_which_to_download) <= set(connector.vectors.index):
            missing = set(vectors_for_which_to_download) - set(connector.vectors.index)
            raise ValueError(f"Polygons {missing} missing from connector.vectors")

        vectors_for_which_to_download = self._filter_out_vectors_with_null_geometry(
            vectors_for_which_to_download, connector
        )

        if filter_out_vectors_contained_in_union_of_intersecting_rasters:
            vectors_for_which_to_download = (
                self._filter_out_vectors_contained_in_union_of_intersecting_rasters(
                    vectors_for_which_to_download, connector
                )
            )

        return vectors_for_which_to_download

    def _filter_out_vectors_with_null_geometry(
        self,
        vector_names: str | int | list[int] | list[str],
        connector: Connector,
    ) -> None:
        vectors_w_null_geometry_mask = (
            connector.vectors.geometry.values == None  # noqa: E711
        )
        vectors_w_null_geometry = connector.vectors[
            vectors_w_null_geometry_mask
        ].index.tolist()
        if vectors_w_null_geometry != []:
            log.info(
                "download_rasters: skipping vector features with null geometry: %s.",
                vectors_w_null_geometry,
            )
            return [
                vector_name
                for vector_name in vector_names
                if vector_name not in vectors_w_null_geometry
            ]
        else:
            return vector_names

    def _filter_out_vectors_contained_in_union_of_intersecting_rasters(
        self,
        vector_names: str | int | list[int] | list[str],
        connector: Connector,
    ) -> None:
        vector_names = [
            vector_name
            for vector_name in vector_names
            if not unary_union(
                connector.rasters.loc[
                    connector.rasters_intersecting_vector(vector_name)
                ].geometry.tolist()
            ).contains(connector.vectors.loc[vector_name].geometry)
        ]
        return vector_names

    def _get_new_rasters(
        self,
        new_raster_dicts_list: list[dict[str, Any]],
        rasters_crs_epsg_code: int,
    ) -> GeoDataFrame:
        """Build and return new rasters gdf from new_raster_dicts_list."""
        new_rasters = GeoDataFrame.from_records(new_raster_dicts_list)
        new_rasters.set_geometry("geometry", inplace=True)
        new_rasters.set_crs(epsg=rasters_crs_epsg_code, inplace=True)
        new_rasters.set_index("raster_name", inplace=True)
        new_rasters = new_rasters.convert_dtypes(
            infer_objects=True,
            convert_string=True,
            convert_integer=True,
            convert_boolean=True,
            convert_floating=False,
        )
        return new_rasters
