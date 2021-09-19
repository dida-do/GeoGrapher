""" 
Mix-in that implements creating a new dataset by cutting every img in the source dataset to a grid of images. 
"""


from __future__ import annotations
import assoc
from typing import List, Optional, Union, TYPE_CHECKING
from pathlib import Path
if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator
from rs_tools.cut.type_aliases import ImgSize
from rs_tools.cut.cut_every_img_to_grid import cut_dataset_every_img_to_grid

class CreateNewDatasetByCuttingEveryImgToGridOfImgsMixIn:

    def create_new_dataset_by_cutting_every_img_to_grid_of_imgs(self, 
            target_data_dir : Union[str, Path], 
            new_img_size : ImgSize = 512, 
            img_bands : Optional[List[int]]=None, 
            label_bands : Optional[List[int]]=None
            ) -> ImgPolygonAssociator:
        """
        TODO

        Note:
            Works only for datasets of GeoTiffs.

        Args:
            target_data_dir (Union[str, Path]): [description]
            new_img_size (ImgSize, optional): [description]. Defaults to 512.
            img_bands (Optional[List[int]], optional): [description]. Defaults to None.
            label_bands (Optional[List[int]], optional): [description]. Defaults to None.

        Returns:
            ImgPolygonAssociator: [description]
        """

        return cut_dataset_every_img_to_grid(
                    target_data_dir=target_data_dir, 
                    new_img_size=new_img_size, 
                    source_assoc=self, 
                    img_bands=img_bands, 
                    label_bands=label_bands
        )
        
        
