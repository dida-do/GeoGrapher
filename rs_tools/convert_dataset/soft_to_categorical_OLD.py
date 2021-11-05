"""
Create or update dataset with categorical labels from a source dataset with soft-categorical labels.
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

def create_categorical_from_soft_categorical_dataset(
        target_data_dir : Union[Path, str],
        source_data_dir : Optional[Union[Path, str]] = None,
        source_assoc : Optional[ImgPolygonAssociator] = None,
        label_dim : int = 0
        ) -> ImgPolygonAssociator:
    """
    TODO:

    Assumes the dataset are in standard data directories (i.e. in one directory containing an 'images', 'labels', and 'associator' subdirectory).

    Args:
        target_data_dir (Union[Path, str]): [description]
        source_data_dir (Optional[Union[Path, str]], optional): [description]. Defaults to None.
        source_assoc (Optional[ImgPolygonAssociator], optional): [description]. Defaults to None.
        label_dim (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns:
        ImgPolygonAssociator: [description]
    """

    if not ((source_data_dir is not None) ^ (source_assoc is not None)):
        raise ValueError(f"Exactly one of the source_data_dir or source_assoc arguments needs to be set (i.e. not None).")

    if source_assoc is None:
        source_assoc = ImgPolygonAssociator.from_data_dir(source_data_dir)

    return _create_or_update_categorical_from_soft_categorical_dataset(
                create_or_update='create',
                source_assoc=source_assoc,
                target_data_dir=target_data_dir,
                label_dim=label_dim
    )

def update_categorical_from_soft_categorical_dataset(
        data_dir : Optional[Union[Path, str]] = None,
        assoc : Optional[ImgPolygonAssociator] = None
        ) -> ImgPolygonAssociator:
    """
    Update an existing dataset created using create_categorical_from_soft_categorical_dataset.

    !!!!!!!!!WRITE DESCRIPTION!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Assumes the dataset are in standard data directories (i.e. in one directory containing an 'images', 'labels', and 'associator' subdirectory).

    Args:
        data_dir (Optional[Union[Path, str]], optional): data directory of dataset to be updated. Defaults to None.
        assoc (Optional[ImgPolygonAssociator], optional): associator of dataset to be updated. Defaults to None.

    Returns:
        ImgPolygonAssociator: associator of updated dataset.
    """

    if not ((data_dir is not None) ^ (assoc is not None)):
        raise ValueError(f"Exactly one of the data_dir or assoc arguments needs to be set (i.e. not None).")

    if assoc is None:
        assoc = ImgPolygonAssociator.from_data_dir(data_dir)

    source_data_dir = assoc._update_from_source_dataset_dict['source_data_dir']
    label_dim = assoc._update_from_source_dataset_dict['label_dim']

    source_assoc = ImgPolygonAssociator.from_data_dir(source_data_dir)

    return _create_or_update_categorical_from_soft_categorical_dataset(
                create_or_update='update',
                source_assoc=source_assoc,
                target_assoc=assoc,
                label_dim=label_dim
    )


def _create_or_update_categorical_from_soft_categorical_dataset(
        create_or_update : str,
        source_assoc : ImgPolygonAssociator,
        target_data_dir : Optional[Union[Path, str]] = None,
        target_assoc : Optional[ImgPolygonAssociator] = None,
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

    # argument consistency check
    if create_or_update not in {'create', 'update'}:
        raise ValueError(f"create_or_update argument must be one of 'create' or 'update'")

    # load target associator
    if create_or_update == 'update':

        if target_assoc is None or target_data_dir is not None:
            raise ValueError(f"When updating, the target_assoc arg needs to be set (not None), and the target_data_dir arg should not be set (i.e. be None).")

        if target_data_dir is not None:
            target_assoc = ImgPolygonAssociator.from_data_dir(target_data_dir)

        if not target_assoc.label_type == 'categorical':
            raise ValueError(f"label_type of target dataset should be 'categorical', but is {target_assoc.label_type}")

    elif create_or_update == 'create':
        target_assoc = source_assoc.empty_assoc_same_format_as(target_data_dir)
        target_assoc.label_type = 'categorical' # converts cols of empty polygons_df as well

        # Create image data dirs ...
        for dir in target_assoc.image_data_dirs:
            dir.mkdir(parents=True, exist_ok=True)
            if list(dir.iterdir()) != []:
                raise Exception(f"{dir} should be empty!")
        # ... and the associator dir.
        target_assoc.assoc_dir.mkdir(parents=True, exist_ok=True)

        # Make sure no associator files already exist.
        if list(target_assoc.assoc_dir.iterdir()) != []:
            raise Exception(f"The assoc_dir in {target_assoc.assoc_dir} should be empty!")

    # need this later
    polygons_that_will_be_added_to_target_dataset = set(source_assoc.polygons_df.index) - set(target_assoc.polygons_df.index)

    # add polygons to target dataset
    source_polygons_df_converted_to_soft_categorical_format = convert_polygons_df_soft_cat_to_cat(source_assoc.polygons_df)
    target_assoc.add_to_polygons_df(source_polygons_df_converted_to_soft_categorical_format)

    # add images to target dataset
    target_assoc.add_to_imgs_df(source_assoc.imgs_df)

    # Determine which images to copy to target dataset
    imgs_that_already_existed_in_target_dataset = {img_name for img_name in target_assoc.imgs_df.index if (target_assoc.images_dir / img_name).is_file()}
    imgs_in_source_images_dir = {img_name for img_name in source_assoc.imgs_df.index}
    imgs_in_source_that_are_not_in_target = imgs_in_source_images_dir - imgs_that_already_existed_in_target_dataset

    # Copy those images
    for img_name in tqdm(imgs_in_source_that_are_not_in_target, desc='Copying images'):
        source_img_path = source_assoc.images_dir / img_name
        target_img_path = target_assoc.images_dir / img_name
        shutil.copyfile(source_img_path, target_img_path)

    # For each image that already existed in the target dataset ...
    for img_name in imgs_that_already_existed_in_target_dataset:
        # ... if among the polygons intersecting it in the target dataset ...
        polygons_intersecting_img = set(target_assoc.polygons_intersecting_img(img_name))
        # ... there is a *new* polygon ...
        if polygons_intersecting_img & polygons_that_will_be_added_to_target_dataset != set():
            # ... then we need to update the label for it, so we delete the current label.
            (target_assoc.labels_dir / img_name).unlink(missing_ok=True)

    # Finally, we make all missing categorical labels in target dataset.
    target_assoc.make_labels()

    # Remember the cutting params and save the associator.
    target_assoc._update_from_source_dataset_dict.update(
        {
            'update_method' : 'update_dataset_soft_categorical_to_categorical',
            'source_data_dir' : source_assoc.images_dir.parent, # !!! Assuming standard directory format here.
            'label_dim' : label_dim
        }
    )
    target_assoc.save()

    log.info(f"_convert_or_update_dataset_soft_categorical_to_categorical: done!")

    return target_assoc
