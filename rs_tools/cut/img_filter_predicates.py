from typing import Any, Callable, Union
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series
from collections.abc import Callable
from abc import abstractmethod
from rs_tools import ImgPolygonAssociator


class ImgFilterPredicate(Callable):
    """
    Abstract base class for predicates used to filter images in cutting functions. 

    Subclasses should implement a __call__method that has the arguments and behavior given below.
    """

    @abstractmethod
    def __call__(self, 
                img_name: str, 
                source_assoc: ImgPolygonAssociator
                ) -> bool:
        """
        Args:
            img_name (str): img identifier
            source_assoc (ImgPolygonAssociator): associator of source dataset that new images are being cut out from 

        Returns:
            bool: True should mean image is to be kept, False that it is to be filtered out
        """
        raise NotImplementedError


class AlwaysTrue(ImgFilterPredicate):
    """
    Default polygon filter predicate that always returns True. Used when filtering is not desired. 
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, 
                    img_name: str, 
                    source_assoc: ImgPolygonAssociator) -> bool:
        """Return True"""
        return True


class ImgFilterRowCondition(ImgFilterPredicate):
    """
    Simple ImgFilter that applies a given predicate to the information in the row corresponding to the image name in question. 
    """

    def __init__(self, 
        row_series_predicate : Callable[[Union[GeoSeries, Series]], bool]
        ) -> None:
        """
        Args:
            row_series_predicate (Callable[[Union[GeoSeries, Series]], bool]): predicate to apply to the row corresponding to an image (i.e. source_assoc.imgs_df.loc[img_name])
        """
        
        super().__init__()
        self.row_series_predicate = row_series_predicate

    def __call__(self, 
            img_name: str, 
            source_assoc: ImgPolygonAssociator
            ) -> bool:
        """
        Apply self.row_series_predicate to source_assoc.imgs_df[img_name]

        Args:
            img_name (str): image name
            source_assoc (ImgPolygonAssociator): source associator

        Returns:
            bool: result of aplying self.row_series_predicate to source_assoc.imgs_df[img_name]
        """

        row_series : Union[GeoSeries, Series] = source_assoc.imgs_df.loc[img_name]
        answer = self.row_series_predicate(row_series) 
        
        return answer