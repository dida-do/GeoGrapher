"""ABC for predicates for filtering rasters.

Used in cutting functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from geopandas import GeoSeries
from pandas import Series
from pydantic import BaseModel

from geographer.connector import Connector


class RasterFilterPredicate(ABC, Callable, BaseModel):
    """ABC for predicates used to filter rasters in cutting functions.

    Subclasses should implement a __call__method that has the arguments
    and behavior given below.
    """

    @abstractmethod
    def __call__(
        self,
        raster_name: str,
        target_connector: Connector,
        new_raster_dict: dict,
        source_connector: Connector,
        cut_rasters: list[str],
    ) -> bool:
        """Return if the raster is to be kept, else False.

        Args:
            raster_name: raster identifier
            target_connector: connector of target dataset.
            new_rasters_dict: dict with keys index or column names of
                target_connector.rasters and values lists of entries
                correspondong to rasters
            source_connector: connector of source dataset that new rasters are being
                cut out from
            cut_rasters: list of (names of) cut rasters

        Returns:
            True should mea raster is to be kept, False that it is to be filtered out

        Note:
            The new_rasters_dict should be viewed as part of the target connector.
            See vector_filter_predicates.py for an explanation.
        """
        raise NotImplementedError

    def save(self, json_path: Path) -> None:
        """Save the predicate."""
        json_path.parent.mkdir(exist_ok=True)
        with open(json_path, "w") as f:
            f.write(self.json(indent=2))


class AlwaysTrue(RasterFilterPredicate):
    """Default raster filter predicate that always returns True.

    Used when filtering is not desired.
    """

    def __call__(
        self,
        raster_name: str,
        target_connector: Connector,
        new_raster_dict: dict,
        source_connector: Connector,
        cut_rasters: list[str],
    ) -> bool:
        """Return True."""
        return True


class RastersNotPreviouslyCutOnly(RasterFilterPredicate):
    """Select rasters not previously cut."""

    def __call__(
        self,
        raster_name: str,
        target_connector: Connector,
        new_raster_dict: dict,
        source_connector: Connector,
        cut_rasters: list[str],
    ) -> bool:
        """Return True if the raster was not previously cut, else False."""
        return raster_name not in cut_rasters


class RowSeriesPredicate(ABC, BaseModel):
    """Row series predicate.

    Apply to series, i.e. single row.
    """

    @abstractmethod
    def __call__(*args, **kwargs):
        """Return evaluation of predicate."""
        pass


class RasterFilterRowCondition(RasterFilterPredicate):
    """Simple RasterFilter based on row condition.

    Applies a given predicate to the row in source_connector.rasters
    corresponding to the raster name in question.
    """

    row_series_predicate: RowSeriesPredicate

    def __init__(
        self, row_series_predicate: Callable[[GeoSeries | Series], bool]
    ) -> None:
        """Initialize an instance of RasterFilterRowCondition.

        Args:
            row_series_predicate:
                predicate to apply to the row corresponding to a raster
                (i.e. source_connector.rasters.loc[raster_name])
        """
        super().__init__()
        self.row_series_predicate = row_series_predicate

    def __call__(
        self,
        raster_name: str,
        target_connector: Connector,
        new_raster_dict: dict,
        source_connector: Connector,
        cut_rasters: list[str],
    ) -> bool:
        """Apply predicate to row.

        Apply self.row_series_predicate to
        source_connector.rasters[raster_name]

        Args:

            raster_name: raster name
            target_connector: connector of target dataset.
            new_rasters_dict: dict with keys index or column names of
                target_connector.rasters and values lists of entries
                correspondong to rasters
            source_connector: source connector

        Returns:
            result of aplying self.row_series_predicate to
            source_connector.rasters[raster_name]
        """
        row_series: GeoSeries | Series = source_connector.rasters.loc[raster_name]
        answer = self.row_series_predicate(row_series)

        return answer


def wrap_function_as_RowSeriesPredicate(
    fun: Callable[[GeoSeries | Series], bool],
) -> RowSeriesPredicate:
    """Wrap a function as a RowSeriesPredicate.

    Args:
        fun:
            Function to wrap.

    Returns:
        RowSeriesPredicate.
    """

    class WrappedAsRowSeriesPredicate(RowSeriesPredicate):
        def __call__(*args, **kwargs):
            return fun(*args, **kwargs)

    return WrappedAsRowSeriesPredicate
