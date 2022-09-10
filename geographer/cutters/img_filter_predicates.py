"""ABC for predicates for filtering images.

Used in cutting functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Union

from geopandas import GeoSeries
from pandas import Series
from pydantic import BaseModel

from geographer.connector import Connector


class ImgFilterPredicate(ABC, Callable, BaseModel):
    """ABC for predicates used to filter images in cutting functions.

    Subclasses should implement a __call__method that has the arguments
    and behavior given below.
    """

    @abstractmethod
    def __call__(
        self,
        img_name: str,
        target_connector: Connector,
        new_img_dict: dict,
        source_connector: Connector,
        cut_imgs: list[str],
    ) -> bool:
        """Return if the image is to be kept, else False.

        Args:
            img_name: img identifier
            target_connector: connector of target dataset.
            new_imgs_dict: dict with keys index or column names of
                target_connector.raster_imgs and values lists of entries
                correspondong to images
            source_connector: connector of source dataset that new images are being
                cut out from
            cut_imgs: list of (names of) cut images

        Returns:
            True should mean image is to be kept, False that it is to be filtered out

        Note:
            The new_imgs_dict should be viewed as part of the target connector.
            See feature_filter_predicates.py for an explanation.
        """
        raise NotImplementedError

    def save(self, json_path: Path) -> None:
        """Save the predicate."""
        json_path.parent.mkdir(exist_ok=True)
        with open(json_path, "w") as f:
            f.write(self.json(indent=2))


class AlwaysTrue(ImgFilterPredicate):
    """Default image filter predicate that always returns True.

    Used when filtering is not desired.
    """

    def __call__(
        self,
        img_name: str,
        target_connector: Connector,
        new_img_dict: dict,
        source_connector: Connector,
        cut_imgs: list[str],
    ) -> bool:
        """Return True."""
        return True


class ImgsNotPreviouslyCutOnly(ImgFilterPredicate):
    """Select images not previously cut."""

    def __call__(
        self,
        img_name: str,
        target_connector: Connector,
        new_img_dict: dict,
        source_connector: Connector,
        cut_imgs: list[str],
    ) -> bool:
        """Return True if the image was not previously cut, else False."""
        return img_name not in cut_imgs


class RowSeriesPredicate(ABC, BaseModel):
    """Row series predicate."""

    @abstractmethod
    def __call__(*args, **kwargs):
        """Return evaluation of predicate."""
        pass


class ImgFilterRowCondition(ImgFilterPredicate):
    """Simple ImgFilter based on row condition.

    Applies a given predicate to the row in source_connector.raster_imgs
    corresponding to the image name in question.
    """

    row_series_predicate: RowSeriesPredicate

    def __init__(
        self, row_series_predicate: Callable[[Union[GeoSeries, Series]], bool]
    ) -> None:
        """Initialize an instance of ImgFilterRowCondition.

        Args:
            row_series_predicate (Callable[[Union[GeoSeries, Series]], bool]):
                predicate to apply to the row corresponding to an image
                (i.e. source_connector.raster_imgs.loc[img_name])
        """
        super().__init__()
        self.row_series_predicate = row_series_predicate

    def __call__(
        self,
        img_name: str,
        target_connector: Connector,
        new_img_dict: dict,
        source_connector: Connector,
        cut_imgs: list[str],
    ) -> bool:
        """Apply predicate to row.

        Apply self.row_series_predicate to
        source_connector.raster_imgs[img_name]

        Args:

            img_name: image name
            target_connector: connector of target dataset.
            new_imgs_dict: dict with keys index or column names of
                target_connector.raster_imgs and values lists of entries
                correspondong to images
            source_connector: source connector

        Returns:
            result of aplying self.row_series_predicate to
            source_connector.raster_imgs[img_name]
        """
        row_series: Union[GeoSeries, Series] = source_connector.raster_imgs.loc[
            img_name
        ]
        answer = self.row_series_predicate(row_series)

        return answer


def wrap_function_as_RowSeriesPredicate(
    fun: Callable[[Union[GeoSeries, Series]], bool]
) -> RowSeriesPredicate:
    """Wrap a function as a RowSeriesPredicate.

    Args:
        fun (Callable[[Union[GeoSeries, Series]], bool]):

    Returns:
        RowSeriesPredicate.
    """

    class WrappedAsRowSeriesPredicate(RowSeriesPredicate):
        def __call__(*args, **kwargs):
            return fun(*args, **kwargs)

    return WrappedAsRowSeriesPredicate
