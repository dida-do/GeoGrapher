"""
Create dataset with categorical labels from dataset with soft labels
"""

from typing import Union
from pathlib import Path
import numpy as np
from geopandas import GeoDataFrame
from tqdm import tqdm
import os
import shutil
from rs_tools.utils.utils import deepcopy_gdf
from rs_tools import ImgPolygonAssociator
from rs_tools.labels.label_type_conversion_utils import convert_polygons_df_soft_cat_to_cat

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def assoc_tifs_soft_cat_to_cat(
        source_data_dir : Union[Path, str], 
        target_data_dir : Union[Path, str], 
        label_dim=0
        ) -> None:
    """
    Create a new dataset of GeoTiffs (and associator) with categorical labels from an existing dataset with soft-categorical labels by taking label with the highest probability (with random tiebreaking).

    :param source_data_dir: source data directory
    :type source_data_dir: Union[Path, str]
    :param target_dir: target data directory
    :type target_dir: Union[Path, str]
    :param label_dim: segmentation class dimension. Defaults to 0, i.e. channels first.
    :type label_dim: int
    """

    # basically, copy over a lot of code from assoc_tif2npy to make sure folders exist etc. 

    # MAKE SURE TIEBREAKING IS CONSISTENT BETWEEN IMGS_DF AND LABELS

    source_data_dir = Path(source_data_dir)
    source_assoc = ImgPolygonAssociator.from_data_dir(source_data_dir)

    # set target_data_dir and create if necessary
    if target_data_dir == None:
        
        target_data_dir = source_data_dir
        # remember we're converting a dataset in place. We'll need to delete the old npys in this case.
        
        inplace=True
    
    else:
        
        inplace=False

        # build target associator
        target_assoc = source_assoc.empty_assoc_same_format_as(target_data_dir)
        target_assoc.integrate_new_polygons_df(source_assoc.polygons_df.copy())
        target_assoc.integrate_new_imgs_df(source_assoc.imgs_df.copy())

        if inplace == False:
            for dir in target_assoc.image_data_dirs:
                                
                # Make sure subdir exists ...
                dir.mkdir(parents=True, exist_ok=True)
                
                # ... but is empty.
                if list(dir.iterdir()) != []:

                    raise Exception(f"{dir} should be empty!")

            # Make sure no associator files already exist.
            if list(target_assoc.assoc_dir.iterdir()) != []:

                raise Exception(f"The assoc_dir in {target_assoc.assoc_dir} should be empty!")

    target_assoc._params_dict['label_type'] = 'categorical'
    
    target_assoc.polygons_df = convert_polygons_df_soft_cat_to_cat(target_assoc.polygons_df)

    target_assoc.save()

    # copy images to target data dir
    tqdm(shutil.copytree(source_assoc.images_dir, target_assoc.images_dir, dirs_exist_ok=True))

    # make missing labels (now categorical) and if possible masks
    target_assoc.make_missing_labels()

    log.info(f"assoc_soft_cat_npys_to_cat: done!")


