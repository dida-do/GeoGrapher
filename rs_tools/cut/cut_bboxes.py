"""
TODO: Include as method in ImgPolygonAssociator.

Functions to cut datasets of GeoTiffs (or update previously cut datasets) by cutting each image in the source dataset to a grid of images.
    - cut_dataset_img_to_grid_of_imgs. Updates a dataset of GeoTiffs that was created with new_tif_dataset_img2grid_imgs. 
    - update_dataset_img_to_grid_of_imgs: customizable general function to create or update datasets of GeoTiffs from existing ones by iterating over polygons.
"""

from __future__ import annotations
from rs_tools.cut.single_img_cutter_bbox import ToImgBBoxCutter
from rs_tools.global_constants import DATA_DIR_SUBDIRS
from typing import TYPE_CHECKING, Union, List, Tuple, Optional
from geopandas import GeoDataFrame
from rs_tools.cut.type_aliases import ImgSize
import logging
from pathlib import Path

if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.cut.single_img_cutter_grid import ImgToGridCutter
from rs_tools.cut.img_filter_predicates import AlwaysTrue 
from rs_tools.cut.cut_iter_over_imgs import create_or_update_dataset_iter_over_imgs

logger = logging.getLogger(__name__)


def create_dataset_cut_bboxes(
        create_or_update : str, 
        bounding_boxes : GeoDataFrame, 
        source_assoc : Optional[ImgPolygonAssociator] = None, 
        target_data_dir : Union[str, Path] = None, 
        target_assoc : Optional[ImgPolygonAssociator] = None, 
        new_img_size : ImgSize = 512, 
        img_bands : Optional[List[int]]=None, 
        label_bands : Optional[List[int]]=None
        ) -> ImgPolygonAssociator:
    """
    TODO

    Warning: 
        TODO! update is not going to work because should be iter over polygons but uses iter over imgs. 


    Args:
        source_data_dir (Union[str, Path], optional): data directory (images, labels, associator) containing the GeoTiffs to be cut from.
        source_assoc (ImgPolygonAssociator, optional): associator of dataset containing the GeoTiffs to be cut from.
        target_data_dir (Union[str, Path]): path to data directory where the new dataset (images, labels, associator) will be created. If the directory does not exist it will be created. 
        target_assoc (ImgPolygonAssociator, optional): associator of target dataset.
        new_img_size (ImgSize): size of new images (side length or (rows, col)) for 'centered' and 'random' modes. Defaults to 512.
        img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
    
    Returns:
        ImgPolygonAssociator: associator of new dataset in target_data_dir
    """

    target_data_dir = Path(target_data_dir)

    bbox_cutter = ToImgBBoxCutter(
                            source_assoc=source_assoc, 
                            target_images_dir=target_data_dir / 'images', 
                            target_labels_dir=target_data_dir / 'labels',
                            new_img_size=new_img_size, 
                            bounding_boxes=bounding_boxes, 
                            img_bands=img_bands, 
                            label_bands=label_bands)
    always_true = AlwaysTrue()
    
    target_assoc = create_or_update_dataset_iter_over_imgs(
                        create_or_update=create_or_update, 
                        source_assoc=source_assoc, 
                        target_data_dir=target_data_dir, 
                        target_assoc=target_assoc, 
                        img_cutter=bbox_cutter, 
                        img_filter_predicate=always_true)

    # throw out images with duplicate bboxes:
    # First, find a subset of images without duplicate bboxes ...
    imgs_to_keep = [] 
    for count, img_name in enumerate(target_assoc.imgs_df.index):
        img_bbox = target_assoc.imgs_df.loc[img_name, 'geometry']
        if {img_name_ for img_name_ in imgs_to_keep if img_bbox.equals(target_assoc.imgs_df.loc[img_name_, 'geometry'])} == set():
            imgs_to_keep += [img_name]
    # ... and delete the remaining images, which have duplicate bboxes
    imgs_to_delete = [img_name for img_name in target_assoc.imgs_df.index if img_name not in imgs_to_keep]
    target_assoc.drop_imgs(imgs_to_delete, remove_imgs_from_disk=True)

    # remember the cutting params.
    target_assoc._update_from_source_dataset_dict.update(
        {
            'update_method' : 'update_dataset_imgs_around_every_polygon', 
            'source_data_dir' : source_assoc.images_dir.parent, # Assuming standard data directory format
            'new_img_size' :  new_img_size, 
            'img_bands' : img_bands, 
            'label_bands' : label_bands,
        }
    )
    target_assoc._params_dict['img_size'] = new_img_size
    target_assoc.save()

    return target_assoc