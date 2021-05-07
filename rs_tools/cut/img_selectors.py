"""
Callable classes for selecting a sublist from a list of images used by cutting functions. 
"""

from typing import List
import random
from geopandas import GeoDataFrame
from collections.abc import Callable
from abc import abstractmethod
from rs_tools import ImgPolygonAssociator


class ImgSelector(Callable):
    """
    Abstract base class for selecting from a list of images. Subclasses are used by create_or_update_dataset_from_iter_over_polygons.

    Subclasses should implement a __call__method that has the arguments and behavior given below.
    """

    @abstractmethod
    def __call__(self, img_names_list: List[str], 
                new_polygons_df: GeoDataFrame, 
                source_assoc: ImgPolygonAssociator) -> List[str]:
        """
        Override to subclass. If img_names_list is empty an empty list should be returned. 

        The new_polygons_df and new_graph arguments contain all the information available to decide which images to select. They should not be modified by this method. 
        
        Args:
            img_names_list (List[str]): list of images to be selected from
            new_polygons_df (GeoDataFrame): polygons_df of new associator/dataset to be created for the new images created by the cutting function
            source_assoc (ImgPolygonAssociator): associator of source dataset that new images are being cut out from 

        Returns:
            List[str]: sublist of img_names_list
        """        
        pass

class RandomImgSelector(ImgSelector):
    """
    ImgSelector that randomly selects a single image from a list of images.
    """
    def __init__(self):
        return

    def __call__(self, img_names_list: List[str], 
                new_polygons_df: GeoDataFrame, 
                source_assoc: ImgPolygonAssociator) -> List[str]:
        """
        Args:
            img_names_list (List[str]): list of images to be selected from
            new_polygons_df (GeoDataFrame): polygons_df of new associator/dataset to be created for the new images created by the cutting function
            source_assoc (ImgPolygonAssociator): associator of source dataset that new images are being cut out from 

        Returns:
            List[str]: sublist of img_names_list
        """
        return [random.choice(img_names_list)] if img_names_list != [] else []    

random_img_selector = RandomImgSelector()