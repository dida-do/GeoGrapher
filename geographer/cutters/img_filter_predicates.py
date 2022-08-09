"""Abstract base class for predicates used to filter images in cutting
functions."""
from pathlib import Path
from abc import abstractmethod, ABC
from collections.abc import Callable
from typing import List, Union

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
        cut_imgs: List[str],
    ) -> bool:
        """
        Args:
            img_name (str): img identifier
            target_connector (Connector): connector of target dataset.
            new_imgs_dict (dict): dict with keys index or column names of target_connector.raster_imgs and values lists of entries correspondong to images 
            source_connector (Connector): connector of source dataset that new images are being cut out from

        Returns:
            bool: True should mean image is to be kept, False that it is to be filtered out

        Note:
            The new_imgs_dict should be viewed as part of the target connector. See feature_filter_predicates.py for an explanation.
        """
        raise NotImplementedError

    def save(self, json_path: Path) -> None:
        """Save the predicate."""
        json_path.parent.mkdir(exist_ok=True)
        with open(json_path, 'w') as f:
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
        cut_imgs: List[str],
    ) -> bool:
        """Return True."""
        return True

class ImgsNotPreviouslyCutOnly(ImgFilterPredicate):
    """Select images not previously cut"""

    def __call__(
        self,
        img_name: str,
        target_connector: Connector,
        new_img_dict: dict,
        source_connector: Connector,
        cut_imgs: List[str],
    ) -> bool:
        return img_name not in cut_imgs


class RowSeriesPredicate(ABC, BaseModel):

    @abstractmethod
    def __call__(*args, **kwargs):
        pass

class ImgFilterRowCondition(ImgFilterPredicate):
    """Simple ImgFilter that applies a given predicate to the row in
    source_connector.raster_imgs corresponding to the image name in question."""

    row_series_predicate: RowSeriesPredicate

    def __init__(
        self, row_series_predicate: Callable[[Union[GeoSeries, Series]],
                                             bool]) -> None:
        """
        Args:
            row_series_predicate (Callable[[Union[GeoSeries, Series]], bool]): predicate to apply to the row corresponding to an image (i.e. source_connector.raster_imgs.loc[img_name])
        """

        super().__init__()
        self.row_series_predicate = row_series_predicate

    def __call__(
        self,
        img_name: str,
        target_connector: Connector,
        new_img_dict: dict,
        source_connector: Connector,
        cut_imgs: List[str],
    ) -> bool:
        """Apply self.row_series_predicate to source_connector.raster_imgs[img_name]

        Args:

            img_name (str): image name
            target_connector (Connector): connector of target dataset.
            new_imgs_dict (dict): dict with keys index or column names of target_connector.raster_imgs and values lists of entries correspondong to images
            source_connector (Connector): source connector

        Returns:
            bool: result of aplying self.row_series_predicate to source_connector.raster_imgs[img_name]
        """

        row_series: Union[GeoSeries,
                          Series] = source_connector.raster_imgs.loc[img_name]
        answer = self.row_series_predicate(row_series)

        return answer


def wrap_function_as_RowSeriesPredicate(
        fun: Callable[[Union[GeoSeries, Series]], bool]) -> RowSeriesPredicate:

    class WrappedAsRowSeriesPredicate(RowSeriesPredicate):

        def __call__(*args, **kwargs):
            return fun(*args, **kwargs)

    return WrappedAsRowSeriesPredicate
