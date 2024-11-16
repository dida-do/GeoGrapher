"""Predicates for filtering vector features."""

from __future__ import annotations

import collections
from abc import abstractmethod
from typing import Any, Literal

from geopandas import GeoSeries
from pandas import Series
from pydantic import BaseModel

from geographer.connector import Connector


class VectorFilterPredicate(BaseModel, collections.abc.Callable):
    """ABC for predicates for filtering vector features.

    To be used in cutting functions.

    Subclasses should implement a __call__method that has the arguments
    and behavior given below.
    """

    @abstractmethod
    def __call__(
        self,
        vector_name: str | int,
        target_connector: Connector,
        new_rasters_dict: dict,
        source_connector: Connector,
        **kwargs: Any,
    ) -> bool:
        """Return True if the vector feature is to be kept, else False.

        Args:
            vector_name: vector feature identifier
            target_connector: connector of target dataset.
            new_rasters_dict: dict with keys index or column names of
                target_connector.rasters and values lists of entries correspondong
                to rasters
            source_connector: connector of source dataset that new rasters are being cut
                out from
            kwargs: Optional keyword arguments

        Returns:
            True should mean the vector feature is to be kept,
            False that it is to be filtered out

        Note:
            The feature filter predicate should be able to return a boolean answer for
            a given feature depending on all the information in the source and target
            connectors. It is used by the cutting class in
            geographer.cut.cut_iter_over_vectors. This function does not
            concatenate the information about the new rasters that have been cut to the
            target_connector.rasters until after all vector features have been
            iterated over. We want to use the vector feature filter predicate _during_
            this iteration, so we allow the call function to also depend on a
            new_rasters_dict argument which contains the information about the new
            rasters that have been cut. Unlike the target_connector.rasters, the
            target_connector.vectors and graph are updated during the
            iteration. One should thus think of the target_connector and
            new_rasters_dict arguments together as the actual the target connector
            argument.
        """
        raise NotImplementedError


class IsVectorMissingRasters(VectorFilterPredicate):
    """VectorFilterPredicate that uses raster counts as criterion.

    Checks whether a vector feature has fewer rasters than a specified
    target raster count.
    """

    target_raster_count: int = 1

    def __call__(
        self,
        vector_name: str | int,
        target_connector: Connector,
        new_rasters_dict: dict,
        source_connector: Connector,
        **kwargs: Any,
    ) -> bool:
        """Return True if raster count < target_raster_count, else False.

        Return True if the raster count of the vector feauture is stricitly less than
        the target_raster_count, else False.

        Args:
            vector_name: vector feature identifier
            target_connector: connector of target dataset.
            new_rasters_dict: dict with keys index or column names of
                target_connector.rasters and values lists of entries correspondong
                to rasters
            source_connector: connector of source dataset that new rasters are being
                cut out from
            kwargs: Optional keyword arguments

        Returns:
            answer
        """
        return (
            target_connector.vectors.loc[
                vector_name, target_connector.raster_count_col_name
            ]
            < self.target_raster_count
        )


class AlwaysTrue(VectorFilterPredicate):
    """Simple vector feature filter predicate that always returns True."""

    def __call__(
        self,
        vector_name: str | int,
        target_connector: Connector,
        new_rasters_dict: dict,
        source_connector: Connector,
        **kwargs: Any,
    ) -> bool:
        """Return True."""
        return True


class OnlyThisVector(VectorFilterPredicate):
    """Filter out all vector features except a given one.

    Simple vector feature filter initialized with a vector feature
    this_vector_name.

    Returns True if and only if the vector feature under consideration
    is equal to this_vector_name.
    """

    def __init__(self, this_vector_name: str | int) -> None:
        """Initialize OnlyThisVector.

        Args:
            this_vector_name (str): (name of) vector feature to be compared to.
        """
        super().__init__()
        self.this_vector_name = this_vector_name

    def __call__(
        self,
        vector_name: str | int,
        target_connector: Connector,
        new_rasters_dict: dict,
        source_connector: Connector,
        **kwargs: Any,
    ) -> bool:
        """Return True if the vector_name matches."""
        return vector_name == self.this_vector_name


class FilterVectorByRowCondition(VectorFilterPredicate):
    """Simple GeomFilterPredicate that uses a predicate on rows.

    Applies a predicate to the row in the source or target vectors
    corresponding to the vector feature name in question.
    """

    def __init__(
        self,
        row_series_predicate: collections.abc.Callable[
            [GeoSeries | Series], bool
        ],
        mode: Literal["source", "target"],
    ) -> None:
        """Initialize FilterVectorByRowCondition.

        Args:
            row_series_predicate:
                predicate to apply to the row corresponding to a vector feature in
                vectors in source_connector or target_connector.
            mode:
                Which GeoDataFrame the predicate should be applied to.
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
        vector_name: str | int,
        target_connector: Connector,
        new_rasters_dict: dict,
        source_connector: Connector,
        **kwargs: Any,
    ) -> bool:
        """Return results of applying predicate to row."""
        if self.mode == "target":
            connector = target_connector
        elif self.mode == "source":
            connector = source_connector

        vectors = connector.vectors
        row_series: GeoSeries | Series = vectors.loc[vector_name]
        answer = self.row_series_predicate(row_series)

        return answer
