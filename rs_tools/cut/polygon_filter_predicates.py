""" 
Predicates for filtering polygons. 
"""

from typing import Any, Callable, Union
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series
from collections.abc import Callable
from abc import abstractmethod
from rs_tools import ImgPolygonAssociator


class PolygonFilterPredicate(Callable):
    """
    Abstract base class for predicates used to filter polygons in cutting functions. 

    Subclasses should implement a __call__method that has the arguments and behavior given below.
    """

    @abstractmethod
    def __call__(self, 
            polygon_name : str, 
            target_assoc : ImgPolygonAssociator,
            new_imgs_dict : dict, 
            source_assoc : ImgPolygonAssociator, 
            **kwargs : Any
            ) -> bool:
        """
        Args:
            polygon_name (str): polygon identifier
            target_assoc (ImgPolygonAssociator): associator of target dataset. 
            new_imgs_dict (dict): dict with keys index or column names of target_assoc.imgs_df and values lists of entries correspondong to images 
            source_assoc (ImgPolygonAssociator): associator of source dataset that new images are being cut out from 
            kwargs (Any): Optional keyword arguments

        Returns:
            bool: True should mean polygon is to be kept, False that it is to be filtered out

        Note:
            The polygon filter predicate should be able to return a boolean answer for a given polygon depending on all the information in the source and target associators. It is used by the cutting function create_or_update_dataset_from_iter_over_polygons in rs_tools.cut.cut_iter_over_polygons. This function does not concatenate the information about the new images that have been cut to the target_assoc.imgs_df until after all polygons have been iterated over. We want to use the polygon filter predicate _during_ this iteration, so we allow the call function to also depend on a new_imgs_dict argument which contains the information about the new images that have been cut. Unlike the target_assoc.imgs_df, the target_assoc.polygons_df and graph are updated during the iteration. One should thus think of the target_assoc and new_imgs_dict arguments together as the actual the target associator argument. 
        """
        raise NotImplementedError


class IsPolygonMissingImgs(PolygonFilterPredicate):
    """
    Simple polygon filter predicate that tests whether a polygon has fewer images than a specified target image count. 
    """

    def __init__(self, target_img_count : int = 1) -> None:
        """
        Args:
            target_img_count (int, optional): Target image count. Defaults to 1.
        """
        self.target_img_count = target_img_count
        super().__init__()

    def __call__(self, 
            polygon_name : str, 
            target_assoc : ImgPolygonAssociator,
            new_imgs_dict : dict, 
            source_assoc : ImgPolygonAssociator, 
            **kwargs : Any
            ) -> bool:
        """
        Return True if the image count of the polygon under consideration is strictly less than target_img_count, False otherwise. 

        Args:
            polygon_name (str): polygon identifier
            target_assoc (ImgPolygonAssociator): associator of target dataset. 
            new_imgs_dict (dict): dict with keys index or column names of target_assoc.imgs_df and values lists of entries correspondong to images 
            source_assoc (ImgPolygonAssociator): associator of source dataset that new images are being cut out from 
            kwargs (Any): Optional keyword arguments

        Returns:
            answer (bool)
        """
        
        return target_assoc.polygons_df.loc[polygon_name, 'img_count'] < self.target_img_count


class AlwaysTrue(PolygonFilterPredicate):
    """
    Simple polygon filter predicate that always returns True. 
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self,
            polygon_name : str, 
            target_assoc : ImgPolygonAssociator,
            new_imgs_dict : dict, 
            source_assoc : ImgPolygonAssociator, 
            **kwargs : Any
            ) -> bool:
        """Return True"""
        return True


class OnlyThisPolygon(PolygonFilterPredicate):
    """
    Simple polygon filter initialized with a polygon this_polygon_name. Returns True if and only if the polygon under consideration is equal to this_polygon_name. 
    """
    def __init__(self, this_polygon_name: str) -> None:
        """
        Args:
            this_polygon (str): (name of) polygon to be compared to.
        """
        super().__init__()
        self.this_polygon_name = this_polygon_name
    
    def __call__(self, 
            polygon_name : str, 
            target_assoc : ImgPolygonAssociator,
            new_imgs_dict : dict, 
            source_assoc : ImgPolygonAssociator, 
            **kwargs : Any
            ) -> bool:

        return polygon_name == self.this_polygon_name


class PolygonFilterRowCondition(PolygonFilterPredicate):
    """
    Simple PolygonFilterPredicate that applies a given predicate to the row in the source or target polygons_df corresponding to the polygon name in question. 
    """

    def __init__(self, 
        row_series_predicate : Callable[[Union[GeoSeries, Series]], bool], 
        mode : str 
        ) -> None:
        """
        Args:
            row_series_predicate (Callable[Union[[GeoSeries, Series]], bool]): predicate to apply to the row corresponding to a polygon in polygons_df in source_assoc or target_assoc.
            mode (str) : Which GeoDataFrame the predicate should be applied to. One of 'source_assoc' or 'target_assoc'
        """
        
        super().__init__()

        self.row_series_predicate = row_series_predicate
        assert mode in {'source_assoc', 'target_assoc'}, f"Unknown mode: {mode}. Should be one of 'source_assoc' or 'target_assoc'"
        self.mode = mode

    def __call__(self, 
            polygon_name : str, 
            target_assoc : ImgPolygonAssociator,
            new_imgs_dict : dict, 
            source_assoc : ImgPolygonAssociator, 
            **kwargs : Any
            ) -> bool:

        if self.mode == 'target_assoc':
            assoc = target_assoc
        elif self.mode == 'source_assoc':
            assoc = source_assoc

        polygons_df = assoc.polygons_df
        row_series : Union[GeoSeries, Series] = polygons_df.loc[polygon_name]
        answer = self.row_series_predicate(row_series) 

        return answer