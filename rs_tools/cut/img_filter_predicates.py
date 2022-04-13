"""Abstract base class for predicates used to filter images in cutting
functions."""
from pathlib import Path
from abc import abstractmethod, ABC
from collections.abc import Callable
from typing import Union

from geopandas import GeoSeries
from pandas import Series
from pydantic import BaseModel

from rs_tools import ImgPolygonAssociator


class ImgFilterPredicate(ABC, Callable, BaseModel):
    """Abstract base class for predicates used to filter images in cutting
    functions.

    Subclasses should implement a __call__method that has the arguments
    and behavior given below.
    """

    @abstractmethod
    def __call__(
        self,
        img_name: str,
        target_assoc: ImgPolygonAssociator,
        new_img_dict: dict,
        source_assoc: ImgPolygonAssociator,
    ) -> bool:
        """
        Args:
            img_name (str): img identifier
            target_assoc (ImgPolygonAssociator): associator of target dataset.
            new_imgs_dict (dict): dict with keys index or column names of target_assoc.imgs_df and values lists of entries correspondong to images 
            source_assoc (ImgPolygonAssociator): associator of source dataset that new images are being cut out from

        Returns:
            bool: True should mean image is to be kept, False that it is to be filtered out

        Note:
            The new_imgs_dict should be viewed as part of the target associator. See polygon_filter_predicates.py for an explanation.
        """
        raise NotImplementedError

    def save(self, save_dir: Path) -> None:
        """Save the predicate to a given directory."""
        save_dir.mkdir(exist_ok=True)
        with open(save_dir, 'w') as f:
            f.write(self.json(indent=2))


class AlwaysTrue(ImgFilterPredicate):
    """Default image filter predicate that always returns True.

    Used when filtering is not desired.
    """

    def __call__(
        self,
        img_name: str,
        target_assoc: ImgPolygonAssociator,
        new_img_dict: dict,
        source_assoc: ImgPolygonAssociator,
    ) -> bool:
        """Return True."""
        return True


class ImgFilterRowCondition(ImgFilterPredicate):
    """Simple ImgFilter that applies a given predicate to the row in
    source_assoc.imgs_df corresponding to the image name in question."""

    row_series_predicate: RowSeriesPredicate

    def __init__(
        self, row_series_predicate: Callable[[Union[GeoSeries, Series]],
                                             bool]) -> None:
        """
        Args:
            row_series_predicate (Callable[[Union[GeoSeries, Series]], bool]): predicate to apply to the row corresponding to an image (i.e. source_assoc.imgs_df.loc[img_name])
        """

        super().__init__()
        self.row_series_predicate = row_series_predicate

    def __call__(
        self,
        img_name: str,
        target_assoc: ImgPolygonAssociator,
        new_img_dict: dict,
        source_assoc: ImgPolygonAssociator,
    ) -> bool:
        """Apply self.row_series_predicate to source_assoc.imgs_df[img_name]

        Args:

            img_name (str): image name
            target_assoc (ImgPolygonAssociator): associator of target dataset.
            new_imgs_dict (dict): dict with keys index or column names of target_assoc.imgs_df and values lists of entries correspondong to images
            source_assoc (ImgPolygonAssociator): source associator

        Returns:
            bool: result of aplying self.row_series_predicate to source_assoc.imgs_df[img_name]
        """

        row_series: Union[GeoSeries,
                          Series] = source_assoc.imgs_df.loc[img_name]
        answer = self.row_series_predicate(row_series)

        return answer


class RowSeriesPredicate(ABC, BaseModel):

    @abstractmethod
    def __call__(*args, **kwargs):
        pass


def wrap_function_as_RowSeriesPredicate(
        fun: Callable[[Union[GeoSeries, Series]], bool]) -> RowSeriesPredicate:

    class WrappedAsRowSeriesPredicate(RowSeriesPredicate):

        def __call__(*args, **kwargs):
            return fun(*args, **kwargs)

    return WrappedAsRowSeriesPredicate
