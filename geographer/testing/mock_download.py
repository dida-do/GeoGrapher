"""Mock ImgDownloaderForSingleVectorFeature and ImgDownloadProcessor for
testing.

Virtually 'download' from a dataset of images in a source directory.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

import shapely
from geopandas.geodataframe import GeoDataFrame
from pydantic import Field
from shapely.geometry import Polygon

from geographer.connector import Connector
from geographer.downloaders.base_download_processor import ImgDownloadProcessor
from geographer.downloaders.base_downloader_for_single_feature import \
    ImgDownloaderForSingleVectorFeature
from geographer.errors import (ImgAlreadyExistsError, ImgDownloadError,
                               NoImgsForVectorFeatureFoundError)


class MockDownloadProcessor(ImgDownloadProcessor):
    """Mock raster image download processor.

    Just return the img_dict, since no actual image data is downloaded.
    """

    source_connector: Connector = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def process(
        self,
        img_name: str,
        download_dir: Path,
        images_dir: Path,
        return_bounds_in_crs_epsg_code: int,
        **kwargs: Any,
    ) -> Dict[Union[Literal['img_name', 'geometry', 'orig_crs_epsg_code'],
                    str], Any]:
        return {
            'img_name':
            img_name,
            'geometry':
            self.source_connector.raster_imgs.loc[img_name, 'geometry'],
            'orig_crs_epsg_code':
            self.source_connector.raster_imgs.loc[img_name,
                                                  'orig_crs_epsg_code'],
            'img_processed?':
            True
        }


class MockDownloaderForSingleFeature(ImgDownloaderForSingleVectorFeature):
    """Mock downloader for single feature.

    Just return the information from a source dataset's raster_imgs from
    a source directory. No actual image data is copied.
    """

    source_connector: Connector = Field(exclude=True)
    probability_of_download_error: float = 0.1
    probability_img_already_downloaded: float = 0.1

    class Config:
        arbitrary_types_allowed = True

    def download(
        self,
        feature_name: Union[int, str],
        feature_geom: Polygon,
        download_dir: Path,
        previously_downloaded_imgs_set: Set[Union[str, int]],
        **kwargs,
    ) -> Dict[Union[Literal['img_name', 'img_processed?'], str], Any]:
        """'Download' an image fully containing a vector vector feature or
        several images jointly containing it from the source_connector and
        return a dict with information to be updated in the connector, see
        below for details.

        Args:
            feature_name: name of vector feature
            feature_geom: shapely geometry of vector feature
            download_dir: directory that the image file should be 'downloaded' to.
            previously_downloaded_imgs_set: previously downloaded img_names.
                In some use cases when it can't be guaranteed that an image
                can be downloaded that fully contains the vector feature it
                can happen that attempts will be made to download an image
                that is already in the connector. Passing this argument
                allows the download function to make sure it doesn't try
                downloading an image that is already in the dataset.
            **kwargs: optional keyword arguments depending on the application.
        Returns:
             A dict with keys and values:
                'list_img_info_dicts': a list of dicts containing the information
                to be included in each row in the raster_imgs of the calling connector,
                one for each newly downloaded image. The keys should be the index and
                column names of the raster_imgs and the values the indices or entries
                of those columns in row that will correspond to the new image.
        """

        # Make sure the vector feature is in self.source_connector.
        # This should be true by construction.
        if not feature_name in self.source_connector.vector_features.index:
            raise Exception(
                f"Polygon {feature_name} not in source connector. This shouldn't have happened, since the source connector should contain all vector features of the mock test connector."
            )

        # Find the images in self.source_connector containing the vector feature
        imgs_containing_vector_feature = list(
            self.source_connector.imgs_containing_vector_feature(feature_name))

        # If there isn't such an image ...
        if imgs_containing_vector_feature == []:

            # ... inform the calling download_missing_imgs_for_vector_features
            # by raising an error.
            raise NoImgsForVectorFeatureFoundError(
                f"No images containing vector feature {feature_name} found in source dataset"
            )

        # Else, there is an image in the source dataset
        # containing the vector feature.
        else:

            # With some probability the API answers our query with
            # an image that has already been downloaded...
            if imgs_containing_vector_feature and random.random(
            ) < self.probability_img_already_downloaded:

                # ... in which case we raise an error.
                raise ImgAlreadyExistsError(
                    f"random.random() was less than self.probability_img_already_downloaded= {self.probability_img_already_downloaded}."
                )

            # Else, from not previously downloaded images ...
            remaining_imgs = [
                img for img in imgs_containing_vector_feature
                if img not in previously_downloaded_imgs_set
            ]

            if remaining_imgs:

                # ... choose one to 'download'.
                img_name = random.choice(remaining_imgs)

                # With some probabibility  ...
                if random.random() < self.probability_of_download_error:

                    # ... an error occurs when downloading,
                    # so we raise an ImgDownloadError.
                    raise ImgDownloadError(
                        f"random.random() was less than self.probability_of_download_error={self.probability_of_download_error}."
                    )

                # ... 'download' it, i.e. return the corresponding return dict.
                img_info_dict = {
                    'img_name': img_name,
                    'img_processed?': False,
                }

                return {
                    'list_img_info_dicts': [img_info_dict],
                }

            else:

                raise NoImgsForVectorFeatureFoundError(
                    f"No new images containing vector feature {feature_name} found in source dataset"
                )
