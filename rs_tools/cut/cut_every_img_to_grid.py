"""
Functions to cut datasets of GeoTiffs (or update previously cut datasets) by cutting each image in the source dataset to a grid of images.
    - cut_dataset_img_to_grid_of_imgs. Updates a dataset of GeoTiffs that was created with new_tif_dataset_img2grid_imgs. 
    - update_dataset_img_to_grid_of_imgs: customizable general function to create or update datasets of GeoTiffs from existing ones by iterating over polygons.
"""

from rs_tools.global_constants import DATA_DIR_SUBDIRS
from typing import Union, List, Tuple, Optional
from rs_tools.cut.type_aliases import ImgSize
import logging
from pathlib import Path

import rs_tools.img_polygon_associator as ipa
from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.cut.single_img_cutter_grid import ImgToGridCutter
from rs_tools.cut.img_filter_predicates import AlwaysTrue 
from rs_tools.cut.cut_iter_over_imgs import create_or_update_dataset_iter_over_imgs

logger = logging.getLogger(__name__)


def cut_dataset_every_img_to_grid(
        target_data_dir : Union[str, Path], 
        new_img_size : ImgSize = 512, 
        source_data_dir : Optional[Union[str, Path]] = None, 
        source_assoc : Optional[ImgPolygonAssociator] = None, 
        img_bands : Optional[List[int]]=None, 
        label_bands : Optional[List[int]]=None
        ) -> ImgPolygonAssociator:
    """
    Create a new dataset of GeoTiffs (images, labels, and associator) where each image is cut into a grid of images.
    
    Exactly one of source_data_dir or source_assoc should be set (i.e. not None).

    Args:
        source_data_dir (Union[str, Path], optional): data directory (images, labels, associator) containing the GeoTiffs to be cut from.
        source_assoc (ImgPolygonAssociator, optional): associator of source dataset to be cut from.
        target_data_dir (Union[str, Path]): path to data directory where the new dataset (images, labels, associator) will be created. If the directory does not exist it will be created. 
        new_img_size (ImgSize): size of new images (side length or (rows, col)) for 'centered' and 'random' modes. Defaults to 512.
        img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
    
    Returns:
        ImgPolygonAssociator: associator of new dataset in target_data_dir

    Warning:
        Currently only works if the source associator component files are in the standard locations determined by the source_data_dir arg. 
    """

    if not ((source_data_dir is not None) ^ (source_assoc is not None)):
        raise ValueError(f"Exactly one of the source_data_dir or source_assoc arguments needs to be set (i.e. not None).")

    if source_assoc is None:
        source_assoc = ImgPolygonAssociator.from_data_dir(source_data_dir)

    return _create_or_update_dataset_every_img_to_grid(
                create_or_update='create',
                source_assoc=source_assoc, 
                target_data_dir=target_data_dir, 
                new_img_size=new_img_size, 
                img_bands=img_bands, 
                label_bands=label_bands
    )


def update_dataset_every_img_to_grid(
        data_dir: Optional[Union[str, Path]] = None, 
        assoc : Optional[ImgPolygonAssociator] = None, 
        ) -> ImgPolygonAssociator:
    """
    Update a dataset of GeoTiffs (images, labels, and associator) where each image is cut into a grid of images.
    
    Either the data_dir or the assoc argument should be given (i.e. not None). 

    Args:
        data_dir (Union[str, Path], optional): data directory to be updated. 
        assoc (ImgPolygonAssociator, optional): associator of dataset to be updated. 

    Returns:
        associator of updated data directory

    Warning:
        Make sure this does exactly what you want when updating an existing data_dir (e.g. if new polygons have been addded to the source_data_dir that overlap with existing labels in the target_data_dir these labels will not be updated. This should be fixed!). It might be safer to just recut the source_data_dir. 
    """
    
    if not ((data_dir is not None) ^ (assoc is not None)):
        raise ValueError(f"Exactly one of the data_dir or assoc arguments should be set (i.e. not None).")

    required_cut_params = {
        'source_data_dir', 
        'new_img_size', 
        'img_bands', 
        'label_bands'
    }

    if assoc is None:    
        assoc = ImgPolygonAssociator.from_data_dir(data_dir)
    
    if not required_cut_params <= set(assoc._update_from_source_dataset_dict.keys()):
        raise KeyError(f"The associator in {data_dir} is missing the following cut params: {set(assoc._update_from_source_dataset_dict.keys()) - required_cut_params}")

    return _create_or_update_dataset_every_img_to_grid(
        create_or_update='update',
        source_data_dir=assoc._update_from_source_dataset_dict['source_data_dir'], 
        target_assoc=assoc, 
        new_img_size=assoc._update_from_source_dataset_dict['new_img_size'], 
        img_bands=assoc._update_from_source_dataset_dict['img_bands'], 
        label_bands=assoc._update_from_source_dataset_dict['label_bands']
    )
    

def _create_or_update_dataset_every_img_to_grid(
        create_or_update : str, 
        source_assoc : Optional[ImgPolygonAssociator] = None, 
        target_data_dir : Union[str, Path] = None, 
        target_assoc : Optional[ImgPolygonAssociator] = None, 
        new_img_size : ImgSize = 512, 
        img_bands : Optional[List[int]]=None, 
        label_bands : Optional[List[int]]=None
        ) -> ImgPolygonAssociator:
    """
    Create a new dataset of GeoTiffs (images, labels, and associator) where each image is cut into a grid of images.
    
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

    img2grid_cutter = ImgToGridCutter(
                            source_assoc=source_assoc, 
                            target_images_dir=target_data_dir / 'images', 
                            target_labels_dir=target_data_dir / 'labels',
                            new_img_size=new_img_size, 
                            img_bands=img_bands, 
                            label_bands=label_bands)
    always_true = AlwaysTrue()
    
    target_assoc = create_or_update_dataset_iter_over_imgs(
                        create_or_update=create_or_update, 
                        source_assoc=source_assoc, 
                        target_data_dir=target_data_dir, 
                        target_assoc=target_assoc, 
                        img_cutter=img2grid_cutter, 
                        img_filter_predicate=always_true)

    # remember the cutting params.
    target_assoc._update_from_source_dataset_dict.update(
        {
            'update_method' : 'update_dataset_every_img_to_grid', 
            'source_data_dir' : source_assoc.images_dir.parent, # Assuming standard data directory format
            'new_img_size' :  new_img_size, 
            'img_bands' : img_bands, 
            'label_bands' : label_bands,
        }
    )
    target_assoc._params_dict['img_size'] = new_img_size
    target_assoc.save()

    return target_assoc