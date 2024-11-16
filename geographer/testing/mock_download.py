"""Mock RasterDownloadProcessor for testing.

Virtually 'download' from a dataset of rasters in a source directory.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field
from shapely.geometry import Polygon

from geographer.connector import Connector
from geographer.downloaders.base_download_processor import RasterDownloadProcessor
from geographer.downloaders.base_downloader_for_single_vector import (
    RasterDownloaderForSingleVector,
)
from geographer.errors import (
    NoRastersForVectorFoundError,
    RasterAlreadyExistsError,
    RasterDownloadError,
)


class MockDownloadProcessor(RasterDownloadProcessor):
    """Mock raster raster download processor.

    Just return the raster_dict, since no actual raster data is
    downloaded.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_connector: Connector = Field(exclude=True)

    def process(
        self,
        raster_name: str,
        download_dir: Path,
        rasters_dir: Path,
        return_bounds_in_crs_epsg_code: int,
        **kwargs: Any,
    ) -> dict[
        Literal["raster_name", "geometry", "orig_crs_epsg_code"] | str, Any
    ]:
        """Process "downloaded" file, i.e. does nothing.

        Returns:
            return dict
        """
        return {
            "raster_name": raster_name,
            "geometry": self.source_connector.rasters.loc[raster_name, "geometry"],
            "orig_crs_epsg_code": self.source_connector.rasters.loc[
                raster_name, "orig_crs_epsg_code"
            ],
            "raster_processed?": True,
        }


class MockDownloaderForSingleVector(RasterDownloaderForSingleVector):
    """Mock downloader for single vector feature.

    Just return the information from a source dataset's rasters from a
    source directory. No actual raster data is copied.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_connector: Connector = Field(exclude=True)
    probability_of_download_error: float = 0.1
    probability_raster_already_downloaded: float = 0.1

    def download(
        self,
        vector_name: int | str,
        vector_geom: Polygon,
        download_dir: Path,
        previously_downloaded_rasters_set: set[str | int],
        **kwargs,
    ) -> dict[Literal["raster_name", "raster_processed?"] | str, Any]:
        """Mock download an raster.

        Mock download an raster fully containing a vector feature or
        several rasters jointly containing it from the source_connector and
        return a dict with information to be updated in the connector, see
        below for details.

        Args:
            vector_name: name of vector feature
            vector_geom: shapely geometry of vector feature
            download_dir: directory that the raster file should be 'downloaded' to.
            previously_downloaded_rasters_set: previously downloaded raster_names.
                In some use cases when it can't be guaranteed that an raster
                can be downloaded that fully contains the vector feature it
                can happen that attempts will be made to download an raster
                that is already in the connector. Passing this argument
                allows the download function to make sure it doesn't try
                downloading an raster that is already in the dataset.
            **kwargs: optional keyword arguments depending on the application.

        Returns:
             A dict with keys and values:
                'list_raster_info_dicts': a list of dicts containing the information
                to be included in each row in the rasters of the calling connector,
                one for each newly downloaded raster. The keys should be the index and
                column names of the rasters and the values the indices or entries
                of those columns in row that will correspond to the new raster.
        """
        # Make sure the vector feature is in self.source_connector.
        # This should be true by construction.
        if vector_name not in self.source_connector.vectors.index:
            raise Exception(
                f"Vector feature {vector_name} not in source connector. "
                "This shouldn't have happened, since the source connector "
                "should contain all vector features of the mock test connector."
            )

        # Find the rasters in self.source_connector containing the vector feature
        rasters_containing_vector = list(
            self.source_connector.rasters_containing_vector(vector_name)
        )

        # If there isn't such an raster ...
        if rasters_containing_vector == []:
            # ... inform the calling download_missing_rasters_for_vectors
            # by raising an error.
            raise NoRastersForVectorFoundError(
                f"No rasters containing vector feature {vector_name} "
                "found in source dataset"
            )

        # Else, there is an raster in the source dataset
        # containing the vector feature.
        else:
            # With some probability the API answers our query with
            # an raster that has already been downloaded...
            if (
                rasters_containing_vector
                and random.random() < self.probability_raster_already_downloaded
            ):
                # ... in which case we raise an error.
                raise RasterAlreadyExistsError(
                    "random.random() was less than "
                    "self.probability_raster_already_downloaded= "
                    f"{self.probability_raster_already_downloaded}."
                )

            # Else, from not previously downloaded rasters ...
            remaining_rasters = [
                raster
                for raster in rasters_containing_vector
                if raster not in previously_downloaded_rasters_set
            ]

            if remaining_rasters:
                # ... choose one to 'download'.
                raster_name = random.choice(remaining_rasters)

                # With some probabibility  ...
                if random.random() < self.probability_of_download_error:
                    # ... an error occurs when downloading,
                    # so we raise an RasterDownloadError.
                    raise RasterDownloadError(
                        "random.random() was less than "
                        "self.probability_of_download_error= "
                        f"{self.probability_of_download_error}."
                    )

                # ... 'download' it, i.e. return the corresponding return dict.
                raster_info_dict = {
                    "raster_name": raster_name,
                    "raster_processed?": False,
                }

                return {
                    "list_raster_info_dicts": [raster_info_dict],
                }

            else:
                raise NoRastersForVectorFoundError(
                    "No new rasters containing vector feature "
                    f"{vector_name} found in source dataset"
                )
