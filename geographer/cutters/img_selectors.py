"""Callable classes for selecting a sublist from a list of images.

Used by cutting functions.
"""

import random
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Literal, Union

from geopandas import GeoSeries
from pandas import Series
from pydantic import BaseModel

from geographer.connector import Connector


class ImgSelector(Callable, BaseModel):
    """Abstract base class for selecting from a list of images. Subclasses are
    used by DSCutterIterOverFeatures.

    Subclasses should implement a __call__method that has the arguments
    and behavior given below.
    """

    @abstractmethod
    def __call__(
        self,
        img_names_list: List[str],
        target_connector: Connector,
        new_imgs_dict: dict,
        source_connector: Connector,
        cut_imgs: Dict[str, List[str]],
        **kwargs: Any,
    ) -> List[str]:
        """Select rasters to create cutouts from a list of rasters.

        Args:
            img_names_list: list of images to be selected from
            target_connector: connector of target dataset.
            new_imgs_dict: dict with keys index or column names of
                target_connector.raster_imgs and values lists of entries correspondong
                to images
            source_connector: connector of source dataset that new images are being cut
                out from
            cut_imgs: dict containing for each raster in the target dataset
                the list of rasters in the source from which cutouts have been
                created for it
            kwargs: Optional keyword arguments

        Returns:
            sublist of img_names_list

        Note:
            - Override to subclass. If img_names_list is empty an empty list
            should be returned.

            - The new_vector_features and new_graph arguments contain all the
            information available to decide which images to select. They should not be
            modified by this method.

            It should be possible for the returned sublist to depend on all the
            information in the source and target connectors. The ImgSelector used by
            the cutting function create_or_update_tif_dataset_from_iter_over_features
            in geographer.cut.cut_iter_over_features. This function does not
            concatenate the information about the new images that have been cut to the
            target_connector.raster_imgs until after all vector features have been
            iterated over. We want to use the vector features filter
            predicate _during_ this iteration, so we allow the call function to also
            depend on a new_imgs_dict argument which contains the information about
            the new images that have been cut. Unlike the target_connector.raster_imgs,
            the target_connector.vector_features and graph are updated during the
            iteration. One should thus think of the target_connector and new_imgs_dict
            arguments together as the actual the target connector argument.
        """


class RandomImgSelector(ImgSelector):
    """ImgSelector that randomly selects randomly from a list of images."""

    target_img_count: int = 1

    def __call__(
        self,
        feature_name: Union[str, int],
        img_names_list: List[str],
        target_connector: Connector,
        new_imgs_dict: dict,
        source_connector: Connector,
        cut_imgs: Dict[str, List[str]],
        **kwargs: Any,
    ) -> List[str]:
        """Randomly select images from a list of images.

        Select target_img_count - #{img_count of vector feature in target_connector}
        images (or if not possible less) from img_names_list.
        """

        target_num_imgs_to_sample = self.target_img_count \
            - len(target_connector.imgs_containing_vector_feature(feature_name)) \
            - len(cut_imgs[feature_name])

        # can only sample a non-negative number of images
        target_num_imgs_to_sample = max(0, target_num_imgs_to_sample)

        # can only sample from img_names_list
        num_imgs_to_sample = min(len(img_names_list),
                                 target_num_imgs_to_sample)

        return random.sample(img_names_list, num_imgs_to_sample)


random_img_selector = RandomImgSelector()
