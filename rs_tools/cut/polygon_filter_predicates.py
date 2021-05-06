from geopandas import GeoDataFrame
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
                polygon_name: str, 
                new_polygons_df: GeoDataFrame, 
                source_assoc: ImgPolygonAssociator) -> bool:
        """
        Args:
            polygon_name (str): polygon identifier
            new_polygons_df (GeoDataFrame): polygons_df of new associator/dataset to be created for the new images created by the cutting function
            source_assoc (ImgPolygonAssociator): associator of source dataset that new images are being cut out from 

        Returns:
            bool: True should mean polygon is to be kept, False that it is to be filtered out
        """
        raise NotImplementedError


class DoesPolygonNotHaveImg(PolygonFilterPredicate):
    """
    Simple polygon filter predicate that tests whether a polygon has already been created for it, i.e. whether there is an image for it in new_polgons_df.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, 
                polygon_name: str, 
                new_polygons_df: GeoDataFrame, 
                source_assoc: ImgPolygonAssociator) -> bool:
        """
        Return True if an image has been created for the polygon, i.e. if the polygon has an image in new_polygons_df, False otherwise. 

        Args:
            polygon_name (str): polygon identifier
            new_polygons_df (GeoDataFrame): new polygons_df of associator to be created
            source_assoc (ImgPolygonAssociator): source associator

        Returns:
            bool: 
        """
        
        return new_polygons_df.loc[polygon_name, 'have_img?'] == False


class AlwaysTrue(PolygonFilterPredicate):
    """
    Simple polygon filter predicate that always returns True. 
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, 
                    polygon_name: str, 
                    new_polygons_df: GeoDataFrame, 
                    source_assoc: ImgPolygonAssociator) -> bool:
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
                polygon_name: str, 
                new_polygons_df: GeoDataFrame, 
                source_assoc: ImgPolygonAssociator) -> bool:

        return polygon_name == self.this_polygon_name