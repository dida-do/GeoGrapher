"""
Mix-in that implements creating a new dataset by combining and/or removing segmentation classes from an existing one.     
"""

from __future__ import annotations
from typing import List, Optional, Union, TYPE_CHECKING
from pathlib import Path
if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator
from rs_tools.convert_dataset.combine_remove_seg_classes import create_dataset_by_combining_or_removing_seg_classes_from_existing_dataset

class CreateNewDatasetCombiningRemovingSegClassesMixIn:

    def create_new_dataset_by_combining_and_or_removing_seg_classes(
            seg_classes : List[Union[str, List[str]]], 
            target_data_dir : Union[Path, str], 
            new_seg_classes : Optional[List[str]] = None, 
            class_separator : str = "+", 
            new_background_class : Optional[str] = None, 
            remove_imgs : bool = True
            ) -> ImgPolygonAssociator:

        """
        TODO

        Args:
            seg_classes (List[Union[str, List[str]]]):  
            target_data_dir (Union[Path, str]):
            new_seg_classes (Optional[List[str]]):
            class_separator (str):
            new_background_class (Optional[str]):
            remove_imgs (bool):

        Returns: 
            ImgPolygonAssociator
        """
    
        return create_dataset_by_combining_or_removing_seg_classes_from_existing_dataset(
                    seg_classes=seg_classes, 
                    target_data_dir=target_data_dir, 
                    source_assoc=self, 
                    new_seg_classes=new_seg_classes, 
                    class_separator=class_separator, 
                    new_background_class=new_background_class, 
                    remove_imgs=remove_imgs
        )