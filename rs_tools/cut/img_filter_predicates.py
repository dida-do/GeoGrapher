from geopandas import GeoDataFrame
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
                source_assoc: ImgPolygonAssociator) -> bool:
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
    Simple polygon filter predicate that always returns True. 
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, 
                    img_name: str, 
                    source_assoc: ImgPolygonAssociator) -> bool:
        """Return True"""
        return True
