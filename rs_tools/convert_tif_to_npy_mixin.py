"""
Mix-in that implements creating a new dataset from self by converting the images from GeoTiff to .npy format. 
"""

from __future__ import annotations
from typing import Union, Optional, List
from typing import TYPE_CHECKING
from pathlib import Path

from rs_tools.convert_dataset.convert_or_update_dataset_from_tif_to_npy import convert_dataset_from_tif_to_npy

if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator

class CreateNewDatasetOfNpysFromDatsetOfTifsMixIn:
    """
    Mix-in that implements creating a new dataset from self by converting the images from GeoTiff to .npy format. 
    """

    def convert_dataset_from_tif_to_npy(self, 
            target_data_dir : Union[Path, str], 
            img_bands : Optional[List[int]] = None, 
            label_bands : Optional[List[int]] = None, 
            squeeze_label_channel_dim_if_single_channel : bool = True, 
            channels_first_or_last_in_npy : str = 'last'
            ) -> ImgPolygonAssociator:
        """
        TODO 

        Args:
            target_data_dir (Union[Path, str]): [description]
            img_bands (Optional[List[int]], optional): [description]. Defaults to None.
            label_bands (Optional[List[int]], optional): [description]. Defaults to None.
            squeeze_label_channel_dim_if_single_channel (bool, optional): [description]. Defaults to True.
            channels_first_or_last_in_npy (str, optional): [description]. Defaults to 'last'.

        Returns:
            ImgPolygonAssociator: [description]
        """

        return convert_dataset_from_tif_to_npy(
                    target_data_dir=target_data_dir, 
                    source_assoc=self, 
                    img_bands=img_bands, 
                    label_bands=label_bands, 
                    squeeze_label_channel_dim_if_single_channel=squeeze_label_channel_dim_if_single_channel, 
                    channels_first_or_last_in_npy=channels_first_or_last_in_npy
        )