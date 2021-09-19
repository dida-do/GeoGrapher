"""
Mix-in that implements creating a new dataset from self by converting the label_type. 
"""

from __future__ import annotations
from typing import Union, Optional, List
from typing import TYPE_CHECKING
from pathlib import Path

from rs_tools.convert_dataset.soft_to_categorical import create_categorical_from_soft_categorical_dataset

if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator

class CreateNewDatasetWithNewLabelTypeMixIn:
    """
    Mix-in that implements creating a new dataset from self converting the label_type. 
    """

    def create_new_dataset_with_label_type(self,
            label_type : str, 
            target_data_dir : Union[Path, str], 
            label_dim : int = 0
            ) -> ImgPolygonAssociator:

        """
        TODO 

        Args:
            label_type (str): TODO
            target_data_dir (Union[Path, str]): TODO. 
            label_dim (int): TODO. Defaults to 0.

        Returns:
            [type]: [description]
        """

        if label_type not in {'categorical'}:
            raise NotImplementedError(f"Unknown label_type: {label_type}")

        if not (self.label_type == 'soft-categorical' and label_type == 'categorical'):
            raise NotImplementedError(f"Conversion from label_type {self.label_type} to {label_type} not implemented.")

        if self.label_type == 'soft-categorical' and label_type == 'categorical':
            target_assoc =  create_categorical_from_soft_categorical_dataset(
                                target_data_dir=target_data_dir, 
                                source_assoc=self, 
                                label_dim=label_dim
            )

        return target_assoc
