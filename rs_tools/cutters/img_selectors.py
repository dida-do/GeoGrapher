"""
TODO: remove reference to _update_from_source_dataset_dict!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Callable classes for selecting a sublist from a list of images.
Used by cutting functions.
"""

import random
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Union

from pydantic import BaseModel
from rs_tools import ImgPolygonAssociator


class ImgSelector(Callable, BaseModel):
    """Abstract base class for selecting from a list of images. Subclasses are
    used by create_or_update_dataset_from_iter_over_polygons.

    Subclasses should implement a __call__method that has the arguments
    and behavior given below.
    """

    @abstractmethod
    def __call__(
        self,
        img_names_list: List[str],
        target_assoc: ImgPolygonAssociator,
        new_imgs_dict: dict,
        source_assoc: ImgPolygonAssociator,
        cut_imgs: Dict[str, List[str]],
        **kwargs: Any,
    ) -> List[str]:
        """Override to subclass. If img_names_list is empty an empty list
        should be returned.

        The new_polygons_df and new_graph arguments contain all the information available to decide which images to select. They should not be modified by this method.

        Args:
            img_names_list (List[str]): list of images to be selected from
            target_assoc (ImgPolygonAssociator): associator of target dataset.
            new_imgs_dict (dict): dict with keys index or column names of target_assoc.imgs_df and values lists of entries correspondong to images
            source_assoc (ImgPolygonAssociator): associator of source dataset that new images are being cut out from
            kwargs (Any): Optional keyword arguments

        Returns:
            List[str]: sublist of img_names_list

        Note:
            It should be possible for the returned sublist to depend on all the information in the source and target associators. The ImgSelector used by the cutting function create_or_update_tif_dataset_from_iter_over_polygons in rs_tools.cut.cut_iter_over_polygons. This function does not concatenate the information about the new images that have been cut to the target_assoc.imgs_df until after all polygons have been iterated over. We want to use the polygon filter predicate _during_ this iteration, so we allow the call function to also depend on a new_imgs_dict argument which contains the information about the new images that have been cut. Unlike the target_assoc.imgs_df, the target_assoc.polygons_df and graph are updated during the iteration. One should thus think of the target_assoc and new_imgs_dict arguments together as the actual the target associator argument.
        """


class RandomImgSelector(ImgSelector):
    """ImgSelector that randomly selects randomly from a list of images."""

    target_img_count: int = 1

    def __call__(
        self,
        polygon_name: Union[str, int],
        img_names_list: List[str],
        target_assoc: ImgPolygonAssociator,
        new_imgs_dict: dict,
        source_assoc: ImgPolygonAssociator,
        cut_imgs: Dict[str, List[str]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Select target_img_count - #{img_count of polygon in target_assoc} images (or if not possible less) from img_names_list.
        """

        target_num_imgs_to_sample = self.target_img_count \
            - len(target_assoc.imgs_containing_polygon(polygon_name)) \
            - len(cut_imgs[polygon_name])

        # can only sample a non-negative number of images
        target_num_imgs_to_sample = max(0, target_num_imgs_to_sample)

        # can only sample from img_names_list
        num_imgs_to_sample = min(len(img_names_list),
                                 target_num_imgs_to_sample)

        return random.sample(img_names_list, num_imgs_to_sample)


random_img_selector = RandomImgSelector()
