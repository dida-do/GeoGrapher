"""Predicates for filtering vector features."""

from __future__ import annotations

import collections
from abc import abstractmethod
from typing import Any, Literal, Union

from geopandas import GeoSeries
from pandas import Series
from pydantic import BaseModel

from geographer.connector import Connector


class FeatureFilterPredicate(BaseModel, collections.abc.Callable):
    """ABC for predicates for filtering vector features.

    To be used in cutting functions.

    Subclasses should implement a __call__method that has the arguments
    and behavior given below.
    """

    @abstractmethod
    def __call__(
        self,
        feature_name: Union[str, int],
        target_connector: Connector,
        new_imgs_dict: dict,
        source_connector: Connector,
        **kwargs: Any,
    ) -> bool:
        """Return True if the vector feature is to be kept, else False.

        Args:
            feature_name: vector feature identifier
            target_connector: connector of target dataset.
            new_imgs_dict: dict with keys index or column names of
                target_connector.raster_imgs and values lists of entries correspondong
                to images
            source_connector: connector of source dataset that new images are being cut
                out from
            kwargs: Optional keyword arguments

        Returns:
            True should mean feature is to be kept, False that it is to be filtered out

        Note:
            The feature filter predicate should be able to return a boolean answer for
            a given feature depending on all the information in the source and target
            connectors. It is used by the cutting function
            create_or_update_dataset_from_iter_over_vector features in
            geographer.cut.cut_iter_over_vector features. This function does not
            concatenate the information about the new images that have been cut to the
            target_connector.raster_imgs until after all vector features have been
            iterated over. We want to use the feature filter predicate _during_
            this iteration, so we allow the call function to also depend on a
            new_imgs_dict argument which contains the information about the new images
            that have been cut. Unlike the target_connector.raster_imgs, the
            target_connector.vector_features and graph are updated during the
            iteration. One should thus think of the target_connector and new_imgs_dict
            arguments together as the actual the target connector argument.
        """
        raise NotImplementedError


class IsFeatureMissingImgs(FeatureFilterPredicate):
    """FeatureFilterPredicate that uses image counts as criterion.

    Checks whether a feature has fewer images than a specified target
    image count.
    """

    target_img_count: int = 1

    def __call__(
        self,
        feature_name: Union[str, int],
        target_connector: Connector,
        new_imgs_dict: dict,
        source_connector: Connector,
        **kwargs: Any,
    ) -> bool:
        """Return True if image count < target_img_count, else False.

        Return True if the image count of the vector feauture is stricitly less than
        the target_img_count, else False.

        Args:
            feature_name: feature identifier
            target_connector: connector of target dataset.
            new_imgs_dict: dict with keys index or column names of
                target_connector.raster_imgs and values lists of entries correspondong
                to images
            source_connector: connector of source dataset that new images are being cut
                out from
            kwargs: Optional keyword arguments

        Returns:
            answer
        """
        return (
            target_connector.vector_features.loc[feature_name, "img_count"]
            < self.target_img_count
        )


class AlwaysTrue(FeatureFilterPredicate):
    """Simple vector feature filter predicate that always returns True."""

    def __call__(
        self,
        feature_name: Union[str, int],
        target_connector: Connector,
        new_imgs_dict: dict,
        source_connector: Connector,
        **kwargs: Any,
    ) -> bool:
        """Return True."""
        return True


class OnlyThisVectorFeature(FeatureFilterPredicate):
    """Filter out all vector features except a given one.

    Simple vector feature filter initialized with a feature
    this_feature_name.

    Returns True if and only if the feature under consideration is equal
    to this_feature_name.
    """

    def __init__(self, this_feature_name: Union[str, int]) -> None:
        """Initialize OnlyThisVectorFeature.

        Args:
            this_feature_name (str): (name of) vector feature to be compared to.
        """
        super().__init__()
        self.this_feature_name = this_feature_name

    def __call__(
        self,
        feature_name: Union[str, int],
        target_connector: Connector,
        new_imgs_dict: dict,
        source_connector: Connector,
        **kwargs: Any,
    ) -> bool:
        """Return True if the feature_name matches."""
        return feature_name == self.this_feature_name


class FilterVectorFeatureByRowCondition(FeatureFilterPredicate):
    """Simple GeomFilterPredicate that uses a predicate on rows.

    Applies a predicate to the row in the source or target
    vector_features corresponding to the vector feature name in
    question.
    """

    def __init__(
        self,
        row_series_predicate: collections.abc.Callable[
            [Union[GeoSeries, Series]], bool
        ],
        mode: Literal["source", "target"],
    ) -> None:
        """Initialize FilterVectorFeatureByRowCondition.

        Args:
            row_series_predicate (Callable[Union[[GeoSeries, Series]], bool]):
                predicate to apply to the row corresponding to a vector feature in
                vector_features in source_connector or target_connector.
            mode (str) : Which GeoDataFrame the predicate should be applied to.
                One of 'source' or 'target'
        """
        super().__init__()

        self.row_series_predicate = row_series_predicate
        assert mode in {
            "source",
            "target",
        }, f"Unknown mode: {mode}. Should be one of 'source' or 'target'"
        self.mode = mode

    def __call__(
        self,
        feature_name: Union[str, int],
        target_connector: Connector,
        new_imgs_dict: dict,
        source_connector: Connector,
        **kwargs: Any,
    ) -> bool:
        """Return results of applying predicate to row."""
        if self.mode == "target":
            connector = target_connector
        elif self.mode == "source":
            connector = source_connector

        vector_features = connector.vector_features
        row_series: Union[GeoSeries, Series] = vector_features.loc[feature_name]
        answer = self.row_series_predicate(row_series)

        return answer
