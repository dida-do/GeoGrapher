"""
Create or update a dataset of GeoTiffs by cutting images around polygons from a source dataset.
    - cut_dataset_imgs_for_each_polygon: Creates a new dataset of GeoTiffs from an existing one by cutting out images around polygons.  
    - update_dataset_imgs_for_each_polygon. Updates a dataset of GeoTiffs that was created with cut_dataset_imgs_for_each_polygon. 
"""

from rs_tools.cut.cut_dataset_iter_over_polygons import create_or_update_dataset_iter_over_polygons
from typing import Union, List, Optional
from rs_tools.cut.type_aliases import ImgSize
import logging
from pathlib import Path

from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.cut.single_img_cutter_around_polygons import ImgsAroundPolygonCutter
from rs_tools.cut.polygon_filter_predicates import IsPolygonMissingImgs
from rs_tools.cut.img_selectors import RandomImgSelector 


logger = logging.getLogger(__name__)


def cut_dataset_imgs_around_every_polygon(
        source_data_dir : Union[str, Path], 
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
    Create a dataset of GeoTiffs so that it contains (if possible) for each polygon in the target (or source) dataset a number target_img_count of images cut from images in the source dataset. 
    
    Note:
        If a polygon is too large to be contained in a single target image grids of images (with the property that the images in each grid should jointly contain the polygon and such that the grid is the minimal grid satisfying this property) will be cut from the target dataset.  
    
    Args:
        source_data_dir (Union[str, Path]): data directory (images, labels, associator) containing the GeoTiffs to be cut from.
        target_data_dir (Union[str, Path]): path to data directory where the new dataset (images, labels, associator) will be created. If the directory does not exist it will be created. 
        mode (str, optional): One of 'random', 'centered', or 'variable'. If 'random' images (or minimal image grids) will be randomly chose subject to constraint that they fully contain the polygons, if 'centered' will be centered on the polygons. If 'variable', the images will be centered but of variable size determined by the scaling_factor and min_new_img_size arguments. Defaults to 'random'.
        new_img_size (Optional[ImgSize]): size of new images (side length or (rows, col)) for 'centered' and 'random' modes. Defaults to 512.
        min_new_img_size (Optional[ImgSize]): minimum size of new images (side length or (rows, col)) for 'variable' mode. Defaults to 64.
        scaling_factor (float): scaling factor for 'variable' mode. Defaults to 1.2.
        target_img_count (int): image count (number of images per polygon) in target data set to aim for
        img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
        random_seed (int, optional): random seed.

    Returns:
        ImgPolygonAssociator: associator of target dataset

    Warning:
        Currently only works if the source associator component files are in the standard locations determined by the source_data_dir arg. 
    """

    return _create_or_update_dataset_imgs_around_every_polygon(
                source_data_dir=source_data_dir,
                target_data_dir=target_data_dir,
                new_img_size=new_img_size,
                min_new_img_size=min_new_img_size,
                scaling_factor=scaling_factor,
                target_img_count=target_img_count,
                img_bands=img_bands,
                label_bands=label_bands,
                mode=mode,
                random_seed=random_seed)
                

def update_dataset_imgs_around_every_polygon(
        data_dir : Union[str, Path]
        ) -> ImgPolygonAssociator:
    
    """
    Update a dataset of GeoTiffs created by new_tif_dataset_small_imgs_for_each_polygon. 
    
    Adds polygons from source_data_dir not contained in data_dir to data_dir and then iterates over all polygons in data_dir that do not have an image and creates a cutout from source_data_dir for them if one exists. 

    Args:
        data_dir (Union[str, Path]): data directory (images, labels, associator) containing the GeoTiffs to be cut from. This is the only argument needed. 
        
    Returns:
        ImgPolygonAssociator: associator of updated dataset
    
    Warning: 
        If the (new_)img_size of the images in data_dir is smaller than the size of an polygon in data_dir then that polygon will not have an image associated with it and so new images will be created for it from the source_data_dir!

    Warning:
        If new polygons have been addded to the source_data_dir that overlap with existing labels in the target_data_dir these labels will not be updated. 

    Warning:
        Make sure this does exactly what you want when updating an existing data_dir. If in doubt it might be easier or safer to just recut the source_data_dir. 
    """

    required_cut_params = {
        'mode', 
        'source_data_dir', 
        'new_img_size', 
        'min_new_img_size', 
        'scaling_factor', 
        'target_img_count',
        'img_bands', 
        'label_bands', 
        'random_seed' 
    }

    assoc = ImgPolygonAssociator.from_data_dir(data_dir)
    if not required_cut_params <= set(assoc._cut_params_dict.keys()):
        raise KeyError(f"The associator in {data_dir} is missing the following cut params: {set(assoc._cut_params_dict.keys()) - required_cut_params}")

    return _create_or_update_dataset_imgs_around_every_polygon(
        source_data_dir=assoc._cut_params_dict['source_data_dir'], 
        target_data_dir=data_dir, 
        new_img_size=assoc._cut_params_dict['new_img_size'],
        min_new_img_size=assoc._cut_params_dict['min_new_img_size'], 
        scaling_factor=assoc._cut_params_dict['scaling_factor'], 
        target_img_count=assoc._cut_params_dict['target_img_count'],
        img_bands=assoc._cut_params_dict['img_bands'], 
        label_bands=assoc._cut_params_dict['label_bands'], 
        mode=assoc._cut_params_dict['mode']
    )


def _create_or_update_dataset_imgs_around_every_polygon(
        source_data_dir : Union[str, Path], 
        target_data_dir : Union[str, Path], 
        mode : str = 'random', 
        new_img_size : Optional[ImgSize] = 512, 
        min_new_img_size : Optional[ImgSize] = 64, 
        scaling_factor : Union[None, float] = 1.2,
        target_img_count : int = 1,
        img_bands : Optional[List[int]]=None, 
        label_bands : Optional[List[int]]=None, 
        random_seed : int = 10
        ) -> ImgPolygonAssociator:
    """
    Create or update a dataset of GeoTiffs so that it contains (if possible) for each polygon in the target (or source) dataset a number target_img_count of images cut from images in the source dataset. 
    
    Note:
        If a polygon is too large to be contained in a single target image grids of images (with the property that the images in each grid should jointly contain the polygon and such that the grid is the minimal grid satisfying this property) will be cut from the target dataset.  
    
    Args:
        source_data_dir (Union[str, Path]): data directory (images, labels, associator) containing the GeoTiffs to be cut from.
        target_data_dir (Union[str, Path]): path to data directory where the new dataset (images, labels, associator) will be created. If the directory does not exist it will be created. 
        mode (str, optional): One of 'random', 'centered', or 'variable'. If 'random' images (or minimal image grids) will be randomly chose subject to constraint that they fully contain the polygons, if 'centered' will be centered on the polygons. If 'variable', the images will be centered but of variable size determined by the scaling_factor and min_new_img_size arguments. Defaults to 'random'.
        new_img_size (Optional[ImgSize]): size of new images (side length or (rows, col)) for 'centered' and 'random' modes. Defaults to 512.
        min_new_img_size (Optional[ImgSize]): minimum size of new images (side length or (rows, col)) for 'variable' mode. Defaults to 64.
        scaling_factor (float): scaling factor for 'variable' mode. Defaults to 1.2.
        target_img_count (int): image count (number of images per polygon) in target data set to aim for
        img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
        random_seed (int, optional): random seed.

    Returns:
        ImgPolygonAssociator: associator of target dataset
    """

    source_assoc = ImgPolygonAssociator.from_data_dir(source_data_dir)

    # Create the polygon_filter_predicate, img_selector, and single img cutter ...
    is_polygon_missing_imgs = IsPolygonMissingImgs(target_img_count)
    random_img_selector = RandomImgSelector(target_img_count)
    small_imgs_around_polygons_cutter = ImgsAroundPolygonCutter(
                                            source_assoc=source_assoc, 
                                            target_images_dir=target_data_dir / "images", 
                                            target_labels_dir=target_data_dir / "labels", 
                                            mode=mode, 
                                            new_img_size=new_img_size, 
                                            min_new_img_size=min_new_img_size, 
                                            scaling_factor=scaling_factor,
                                            img_bands=img_bands, 
                                            label_bands=label_bands, 
                                            random_seed=random_seed)

    # ... cut the dataset (and return target associator) ...
    target_assoc = create_or_update_dataset_iter_over_polygons(
                        source_data_dir=source_data_dir, 
                        target_data_dir=target_data_dir, 
                        img_cutter=small_imgs_around_polygons_cutter, 
                        img_selector=random_img_selector, 
                        polygon_filter_predicate=is_polygon_missing_imgs)

    # ... and remember the cutting params.
    target_assoc._cut_params_dict.update(
        {
            'mode' : mode,    
            'source_data_dir' : source_data_dir,
            'new_img_size' :  new_img_size, 
            'min_new_img_size' : min_new_img_size, 
            'scaling_factor' : scaling_factor, 
            'target_img_count' : target_img_count,
            'img_bands' : img_bands, 
            'label_bands' : label_bands,
            'random_seed' : random_seed, 
        }
    )
    target_assoc.save()
    
    return target_assoc