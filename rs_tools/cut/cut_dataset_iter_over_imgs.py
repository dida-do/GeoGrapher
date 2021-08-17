"""
Functions to cut datasets of GeoTiffs by iterating over images.

Contains: 
    - new_tif_dataset_img2grid_imgs: Creates a new dataset of GeoTiffs form an existing one by cutting each image into a grid of images.
    - update_tif_dataset_img2grid_imgs. Updates a dataset of GeoTiffs that was created with new_tif_dataset_img2grid_imgs. 
    - create_or_update_tif_dataset_from_iter_over_polygons: customizable general function to create or update datasets of GeoTiffs from existing ones by iterating over polygons.
"""

from typing import Union, Callable, List, Tuple, Optional, Any
import logging
import os
import copy
from pathlib import Path
import random
from tqdm import tqdm
import rasterio as rio
import pandas as pd 
from geopandas import GeoDataFrame
from rs_tools.cut import polygon_filter_predicates

import rs_tools.img_polygon_associator as ipa
from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.utils.utils import transform_shapely_geometry
from rs_tools.cut.single_img_cutter_base import SingleImgCutter
from rs_tools.cut.single_img_cutter_grid import ToImgGridCutter
from rs_tools.cut.polygon_filter_predicates import PolygonFilterPredicate, AlwaysTrue
from rs_tools.cut.polygon_filter_predicates import AlwaysTrue as AlwaysTruePolygons
from rs_tools.cut.img_filter_predicates import ImgFilterPredicate
from rs_tools.cut.img_filter_predicates import AlwaysTrue as AlwaysTrueImgs
from assoc.utils.utils import deepcopy_gdf

logger = logging.getLogger(__name__)


def new_tif_dataset_img2grid_imgs(source_data_dir: Union[str, Path], 
                                    target_data_dir: Union[str, Path], 
                                    new_img_size: Union[None, int, Tuple[int, int]] = 512, 
                                    img_bands: Optional[List[int]]=None, 
                                    label_bands: Optional[List[int]]=None
                                    ) -> ImgPolygonAssociator:
    """
    Create a new dataset of GeoTiffs (images, labels, and associator) where each image is cut into a grid of images.
    
    Args:
        source_data_dir (Union[str, Path]): data directory (images, labels, associator) containing the GeoTiffs to be cut from.
        target_data_dir (Union[str, Path]): path to data directory where the new dataset (images, labels, associator) will be created. If the directory does not exist it will be created. 
        new_img_size (Union[int, Tuple[int, int], optional): size of new images (side length or (rows, col)) for 'centered' and 'random' modes. Defaults to 512.
        img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
    
    Returns:
        ImgPolygonAssociator: associator of new dataset in target_data_dir
    """

    # make sure dir args are Path objects
    source_data_dir = Path(source_data_dir)
    target_data_dir = Path(target_data_dir)

    source_assoc = ImgPolygonAssociator(source_data_dir)

    # Create new target_data_dir and subdirectories if necessary.
    for subdir in ipa.DATA_DIR_SUBDIRS:
        (target_data_dir / subdir).mkdir(parents=True, exist_ok=True)

    img2grid_cutter = ToImgGridCutter(
                            source_assoc=source_assoc, 
                            target_data_dir=target_data_dir, 
                            polygon_filter_predicate=AlwaysTruePolygons(), # ignored
                            new_img_size=new_img_size, 
                            img_bands=img_bands, 
                            label_bands=label_bands)

    target_assoc = create_or_update_tif_dataset_from_iter_over_imgs(
                        source_data_dir=source_data_dir, 
                        target_data_dir=target_data_dir, 
                        img_cutter=img2grid_cutter, 
                        polygon_filter_predicate=AlwaysTruePolygons(),
                        img_bands=img_bands, 
                        label_bands=label_bands,
                        new_img_size=new_img_size)

    return target_assoc

def update_tif_dataset_img2grid_imgs(data_dir: Union[str, Path]) -> ImgPolygonAssociator:
    """
    Update a dataset of GeoTiffs (images, labels, and associator) where each image is cut into a grid of images.
    
    Args:
        data_dir (Union[str, Path]): data directory to be updated

    Returns:
        associator of updated data directory
    """

    data_dir = Path(data_dir)
    assoc = ImgPolygonAssociator(data_dir)
    
    # get remaining arguments from file
    cut_params_dict = assoc._params_dict['cut_params']
    
    source_data_dir = Path(cut_params_dict['source_data_dir'])
    new_img_size = cut_params_dict['new_img_size']
    img_bands = cut_params_dict['img_bands']
    label_bands = cut_params_dict['label_bands']

    img2grid_cutter = ToImgGridCutter(
                            source_assoc=ImgPolygonAssociator(source_data_dir), 
                            target_data_dir=data_dir, 
                            polygon_filter_predicate=AlwaysTruePolygons(), # ignored
                            new_img_size=new_img_size, 
                            img_bands=img_bands, 
                            label_bands=label_bands)

    target_assoc = create_or_update_tif_dataset_from_iter_over_imgs(
                        source_data_dir=source_data_dir, 
                        target_data_dir=data_dir, 
                        img_cutter=img2grid_cutter, 
                        polygon_filter_predicate=AlwaysTruePolygons(),
                        img_bands=img_bands, 
                        label_bands=label_bands,
                        new_img_size=new_img_size)

    return target_assoc
    

def create_or_update_tif_dataset_from_iter_over_imgs(
        source_data_dir: Union[str, Path], 
        target_data_dir: Union[str, Path], 
        img_cutter:SingleImgCutter, 
        polygon_filter_predicate: PolygonFilterPredicate = AlwaysTruePolygons(), 
        img_filter_predicate: ImgFilterPredicate = AlwaysTrueImgs(), 
        img_bands: List[int] = None, 
        label_bands: List[int] = None, 
        **kwargs: Any) -> ImgPolygonAssociator:
    """
    Create or update a data set of GeoTiffs by iterating over images in the source dataset. 

    Create or update a data set of GeoTiffs (images, labels, and associator) in target_data_dir from the data set of GeoTiffs in source_data_dir by iterating over the polygons in the source dataset/associator, selecting a subset of the images in the source dataset containing the polygon (using img_selector) and cutting each selected img using an img_cutter which could could depend e.g. on information in the source associator. We can restrict to a subset of the polygons in the source data_dir by filtering using the img_filter_predicate, which can depend on information in the source associator.

    Args:
        source_data_dir (Union[str, Path]): data directory (images, labels, associator) containing the GeoTiffs to be cut from.
        target_data_dir (Union[str, Path]): data directory of target dataset to be created or updated. 
        img_cutter (SingleImgCutter): single image cutter. 
        polygon_filter_predicate (PolygonFilterPredicate, optional): predicate to filter polygons. Defaults to AlwaysTruePolygons().
        imgs_filter_predicate (ImgsFilterPredicate, optional): predicate to filter images. Defaults to AlwaysTrueImgs().
        img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
        **kwargs (Any): additional arguments for the img_cutter. 
        
    Returns:
        ImgPolygonAssociator: associator of newly created or updated dataset
    """
    
    # Make sure dir args are Path objects.
    source_data_dir = Path(source_data_dir)
    target_data_dir = Path(target_data_dir)

    # Load source associator.
    source_assoc = ImgPolygonAssociator(data_dir = source_data_dir)

    # Load target associator, if it exists already, ...
    if (target_data_dir / "imgs_df.geojson").is_file() and (target_data_dir / "polygons_df.geojson").is_file() and (target_data_dir / "graph.json").is_file(): # try target_assoc = ImgPolygonAssociator(data_dir = target_data_dir) except ... didn't work for reason I don't understand, so to test if an associator exists, I just check if the associator files exist
        target_assoc = ImgPolygonAssociator(data_dir = target_data_dir)    
    # ... else, create empty target associator.
    else:
        target_assoc = ipa.empty_assoc_same_format_as(target_data_dir=target_data_dir, 
                                                        source_assoc=source_assoc)

    new_polygons_df = deepcopy_gdf(source_assoc.polygons_df)

    target_assoc.integrate_new_polygons_df(new_polygons_df)

    if 'cut_params' not in target_assoc._params_dict:
        target_assoc._params_dict['cut_params'] = {'names_of_cut_imgs': []}

    previously_cut_imgs = target_assoc._params_dict['cut_params']['names_of_cut_imgs']
    mask_imgs_not_in_target_imgs_df = ~source_assoc.imgs_df.index.isin(previously_cut_imgs)
    imgs_not_in_target_imgs_df = source_assoc.imgs_df.loc[mask_imgs_not_in_target_imgs_df]
    imgs_not_in_target_imgs_df = deepcopy_gdf(imgs_not_in_target_imgs_df)

    # dict to keep track of information which will be appended to target_assoc's imgs_df after cutting
    new_imgs_dict = {index_or_col_name: [] for index_or_col_name in [source_assoc.imgs_df.index.name] + list(source_assoc.imgs_df.columns)}

    names_of_cut_imgs = []

    # ... and iterate over them:
    for img_name in tqdm(imgs_not_in_target_imgs_df.index):

        # If filter condition is satisfied, (if not, don't do anything) ...
        if img_filter_predicate(img_name, source_assoc):

            # ... and cut/cut the images (and their labels) and remember information to be appended to target_assoc imgs_df in return dict
            imgs_from_single_cut_dict = img_cutter(
                                            img_name=img_name, 
                                            new_polygons_df=target_assoc.polygons_df, 
                                            new_graph=target_assoc._graph)

            # Make sure img_cutter returned dict with same keys as needed by new_imgs_dict.
            assert set(imgs_from_single_cut_dict.keys()) == set(target_assoc.imgs_df.columns) | {target_assoc.imgs_df.index.name}, f"dict returned by img_cutter doesn't contain the same keys as needed by new_imgs_dict!"

            # Accumulate information for the new imgs in new_imgs_dict.
            for key in new_imgs_dict.keys(): 
                        new_imgs_dict[key] += (imgs_from_single_cut_dict[key])

            # Rememer image names:
            names_of_cut_imgs.append(img_name)

    # Extract accumulated information about the imgs we've downloaded from new_imgs into a dataframe...
    new_imgs_df = GeoDataFrame(new_imgs_dict)
    new_imgs_df.set_crs(epsg=target_assoc.imgs_df.crs.to_epsg(), inplace=True) # copy crs
    new_imgs_df.set_index(target_assoc.imgs_df.index.name, inplace=True)

    # ... and append it to self.imgs_df.
    data_frames_list = [target_assoc.imgs_df, new_imgs_df]  
    target_assoc.imgs_df = GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)

    # Save args/information needed for updating the target dataset from the source dataset in the future
    kwargs.update({
        'source_data_dir': str(source_data_dir), 
        'img_bands': img_bands, 
        'label_bands': label_bands})
    for key, val in kwargs.items():
        if key in target_assoc._params_dict['cut_params'] and target_assoc._params_dict['cut_params'][key] != val:
            logger.warning(f"updating value for key {key} to {val}")
        else:
            target_assoc._params_dict['cut_params'][key] = val
    target_assoc._params_dict['cut_params']['names_of_cut_imgs'] += names_of_cut_imgs

    # make masks if possible
    if target_assoc._params_dict['mask_class'] is not None and target_assoc._params_dict['label_type'] == 'categorical':
        target_assoc.make_missing_masks()

    # Save associator to disk.
    target_assoc.save()

    return target_assoc