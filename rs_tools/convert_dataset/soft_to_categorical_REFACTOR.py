"""
Create dataset with categorical labels from dataset with soft labels
"""

from typing import Union, Optional
from pathlib import Path
import numpy as np
from geopandas import GeoDataFrame
from tqdm import tqdm
import shutil
from rs_tools.utils.utils import deepcopy_gdf
from rs_tools import ImgPolygonAssociator
from rs_tools.labels.label_type_conversion_utils import convert_polygons_df_soft_cat_to_cat

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def convert_dataset_soft_categorical_to_categorical(
        target_data_dir : Union[Path, str], 
        source_data_dir : Optional[Union[Path, str]] = None, 
        source_assoc : Optional[ImgPolygonAssociator] = None, 
        label_dim : int = 0
        ) -> ImgPolygonAssociator:

    if not ((source_data_dir is not None) ^ (source_assoc is not None)):
        raise ValueError(f"Exactly one of the source_data_dir or source_assoc arguments needs to be set (i.e. not None).")

    if source_assoc is None:
        source_assoc = ImgPolygonAssociator.from_data_dir(source_data_dir)

    return _convert_or_update_dataset_soft_categorical_to_categorical(
                create_or_update='create', 
                source_assoc=source_assoc, 
                target_data_dir=target_data_dir, 
                label_dim=label_dim
    )
    
def update_dataset_soft_categorical_to_categorical(
        data_dir : Optional[Union[Path, str]] = None, 
        assoc : Optional[ImgPolygonAssociator] = None
        ) -> ImgPolygonAssociator:
    pass

        
    


def _convert_or_update_dataset_soft_categorical_to_categorical(
        create_or_update : str, 
        source_assoc : ImgPolygonAssociator, 
        target_data_dir : Optional[Union[Path, str]], 
        label_dim=0
        ) -> ImgPolygonAssociator:
    """
    Create a new dataset of GeoTiffs (and associator) with categorical labels from an existing dataset with soft-categorical labels by taking label with the highest probability (with random tiebreaking).

    Args:
        create_or_update (str): One of 'create' or 'update'. 
        source_assoc (ImgPolygonAssociator): associator of source dataset
        target_dir (Union[Path, str]): target data directory
        label_dim (int): segmentation class dimension. Defaults to 0, i.e. channels first.

    """
    # MAKE SURE TIEBREAKING IS CONSISTENT BETWEEN IMGS_DF AND LABELS

    if create_or_update not in {'create', 'update'}:
        raise ValueError(f"create_or_update argument must be one of 'create' or 'update'")

    if target_data_dir is None:
        
        target_assoc = source_assoc
        inplace=True # remember we're converting a dataset in place. We'll need to delete the old GeoTiffs in that case.
        if not create_or_update == 'create':
            raise ValueError(f"When converting in place (i.e. when target_data_dir arg not given) create_or_update argument must be 'create', but is {create_or_update}.")

    else:
        inplace=False

        if create_or_update == 'create':
            target_assoc = source_assoc.empty_assoc_same_format_as(target_data_dir)
        elif create_or_update == 'update':
            target_assoc = ImgPolygonAssociator.from_data_dir(target_data_dir)

    if not inplace:
        
        # create data dirs if necessary
        for dir in target_assoc.image_data_dirs:
            dir.mkdir(parents=True, exist_ok=True)
            if list(dir.iterdir()) != []:
                raise Exception(f"{dir} should be empty!")
        # create associator dir
        target_assoc.assoc_dir.mkdir(parents=True, exist_ok=True)

        # Make sure no associator files already exist.
        if list(target_assoc.assoc_dir.iterdir()) != []:
            raise Exception(f"The assoc_dir in {target_assoc.assoc_dir} should be empty!")

    target_assoc._params_dict['label_type'] = 'categorical'

    soft_cat_source_polygons_df = convert_polygons_df_soft_cat_to_cat(source_assoc.polygons_df)

    new_polygons = set(soft_cat_source_polygons_df.index) - set(target_assoc.polygons_df.index)
    new_imgs = set(source_assoc.imgs_df.index) - set(target_assoc.imgs_df.index)

    previously_existing_imgs_in_target = set(target_assoc.imgs_df.index).copy()

    # add polygons
    if inplace:
        source_assoc.polygons_df = convert_polygons_df_soft_cat_to_cat(source_assoc.polygons_df)
    else:
        target_assoc.add_to_polygons_df(soft_cat_source_polygons_df)
    
    new_polygons_intersecting_already_existing_target_imgs = {polygon_name for polygon_name in new_polygons if target_assoc.imgs_intersecting_polygon(polygon_name) != []}

    target_assoc.add_to_imgs_df(source_assoc.imgs_df)

    target_assoc.save()

    if not inplace:
        
        imgs_in_target_images_dir = {img_path.name for img_path in target_assoc.images_dir.iterdir()}
        imgs_in_source_images_dir = {img_path.name for img_path in source_assoc.images_dir.iterdir()}
        imgs_in_source_that_are_not_in_target = imgs_in_source_images_dir - imgs_in_target_images_dir

        for img_name in tqdm(imgs_in_source_that_are_not_in_target):
            source_img_path = source_assoc.images_dir / img_name
            target_img_path = target_assoc.images_dir / img_name
            shutil.copyfile(source_img_path, target_img_path)

    # For each image that already existed in the target dataset ...
    for img_name in previously_existing_imgs_in_target:
        # ... if among the polygons intersecting it ...
        polygons_intersecting_img = set(target_assoc.polygons_intersecting_img(img_name))
        # ... there is a new polygon ...
        if polygons_intersecting_img & new_polygons != set():
            # ... then we need to update the label for it, so we delete the label ...
            (target_assoc.labels_dir / img_name).unlink(missing_ok=True)

    if inplace == True:
        for label_path in source_assoc.labels_dir.iterdir():
            label_path.unlink(missing_ok=True)

    # Finally, we make all missing categorical labels in target dataset.
    target_assoc.make_labels()

    log.info(f"assoc_soft_cat_npys_to_cat: done!")

    return target_assoc
