""" 
Mix-in that implements creating a new dataset by cutting an image (or a grid of images) around every polygon. 
"""


from __future__ import annotations
import assoc
from typing import List, Optional, Union, TYPE_CHECKING
from pathlib import Path
if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator
from rs_tools.cut.type_aliases import ImgSize
from rs_tools.cut.cut_imgs_around_every_polygon import cut_dataset_imgs_around_every_polygon

class CreateNewDatasetByCuttingImgsAroundEveryPolygonMixIn:

    def create_new_dataset_by_cutting_imgs_around_every_polygon(self, 
            target_data_dir : Union[str, Path], 
            new_img_size : Optional[ImgSize] = 512, 
            min_new_img_size : Optional[ImgSize] = 64, 
            scaling_factor : Union[None, float] = 1.2,
            target_img_count : int = 1,
            img_bands : Optional[List[int]]=None, 
            label_bands : Optional[List[int]]=None, 
            mode : str = 'random', 
            random_seed : int = 10
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

        return cut_dataset_imgs_around_every_polygon(
                    target_data_dir=target_data_dir, 
                    source_assoc=self, 
                    new_img_size=new_img_size, 
                    min_new_img_size=min_new_img_size, 
                    scaling_factor=scaling_factor, 
                    target_img_count=target_img_count, 
                    img_bands=img_bands, 
                    label_bands=label_bands, 
                    mode=mode, 
                    random_seed=random_seed
        )
        
        
