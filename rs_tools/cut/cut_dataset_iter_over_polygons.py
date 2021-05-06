"""
Functions to cut datasets of GeoTiffs by iterating over polygons.

Customizable general function create_or_update_dataset_from_iter_over_polygons to create or update a new remote sensing imagery dataset (images, labels, and associator) from an existing one by iterating over (a subset of) the polygons and cutting (a subset of) the images containing the polygons, as well as a specialization to functions new_dataset_one_small_img_for_each_polygon and update_dataset_one_small_img_for_each_polygon that create and update exactly one new small image for each polygon in the old dataset.
"""
from typing import Union, Callable, List, Tuple, Optional
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
from rs_tools.cut.single_img_cutters import SingleImgCutter, SmallImgsAroundPolygonsCutter
from rs_tools.cut.polygon_filter_predicates import PolygonFilterPredicate, AlwaysTrue, DoesPolygonNotHaveImg, OnlyThisPolygon
from rs_tools.cut.img_selectors import ImgSelector, RandomImgSelector 

          

def new_tif_dataset_small_imgs_for_each_polygon(source_data_dir: Union[str, Path], 
                                        target_data_dir: Union[str, Path], 
                                        new_img_size: Union[int, Tuple[int, int]] = 512, 
                                        img_bands: Optional[List[int]]=None, 
                                        label_bands: Optional[List[int]]=None, 
                                        mode: str = 'random', 
                                        random_seed: int = 10) -> ImgPolygonAssociator:
    """
    Create a new dataset of GeoTiffs (images, labels, and associator) consisting of one 'small' image for each polygon (or a grid of such images if a polygon is too large to be contained in a single small image) from the dataset of GeoTiffs in source_data_dir in target_data_dir. 
    
    Args:
        source_data_dir (Union[str, Path]): data directory (images, labels, associator) containing the GeoTiffs to be cut from.
        target_data_dir (Union[str, Path]): path to data directory where the new dataset (images, labels, associator) will be created. If the directory does not exist it will be created. 
        new_img_size (Union[int, Tuple[int, int], optional): size of new images (side length or (rows, col)). Defaults to 512.
        img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
        mode (str, optional): One of 'random' or 'centered'. If 'random' images (or minimal image grids) will be randomly chose subject to constraint that they fully contain the polygons, if 'centered' will be centered on the polygons. Defaults to 'random'.
        random_seed (int, optional): random seed.

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

    does_polygon_not_have_img = DoesPolygonNotHaveImg()

    random_img_selector = RandomImgSelector()

    small_imgs_around_polygons_cutter = SmallImgsAroundPolygonsCutter(source_assoc=source_assoc, 
                                                                        target_data_dir=target_data_dir, 
                                                                        polygon_filter_predicate=does_polygon_not_have_img,
                                                                        new_img_size=new_img_size, 
                                                                        img_bands=img_bands, 
                                                                        labels_bands=label_bands, 
                                                                        mode=mode, 
                                                                        random_seed=random_seed)

    target_assoc = create_or_update_tif_dataset_from_iter_over_polygons(source_data_dir=source_data_dir, 
                                                            target_data_dir=target_data_dir, 
                                                            polygon_filter_predicate=does_polygon_not_have_img,
                                                            img_cutter=small_imgs_around_polygons_cutter, 
                                                            img_selector=random_img_selector, 
                                                            new_img_size=new_img_size, 
                                                            img_bans=img_bands, 
                                                            label_bands=label_bands,
                                                            mode=mode)

    return target_assoc


def update_tif_dataset_small_imgs_for_each_polygon(source_data_dir: Union[str, Path], 
                                        target_data_dir: Union[str, Path], 
                                        new_img_size: Optional[Union[int, Tuple[int, int]]] = None, 
                                        img_bands: Optional[List[int]]=None, 
                                        label_bands: Optional[List[int]]=None, 
                                        mode: str = 'random', 
                                        random_seed: int = 10) -> ImgPolygonAssociator:
    """
    Update a dataset created by new_tif_dataset_small_imgs_for_each_polygon. 
    
    Update a dataset of GeoTiffs (images, labels, and associator) in target_data_dir that was created using new_dataset_small_imgs_for_polygons from an updated version of the dataset of GeoTiffs in source_data_dir by adding those polygons from the source dataset that are not yet in the target dataset and then adding small images and labels for those polygons from the updated source dataset as well as for the polygons in the target dataset that had (i.e. were contained in) no images in the pre-update version of source dataset but now have an image in the updated source dataset. 

    Args:
        source_data_dir (Union[str, Path]): data directory (images, labels, associator) containing the GeoTiffs to be cut from.
        target_data_dir (Union[str, Path]): path to data directory of dataset to be updated.
        new_img_size (Union[int, Tuple[int, int], optional): size of new images (side length  or (rows, cols)). If None, will infer from randomly chosen image in target dataset. 
        img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
        mode (str, optional): One of 'random' or 'centered'. If 'random' images (or minimal image grids) will be randomly chose subject to constraint that they fully contain the polygons, if 'centered' will be centered on the polygons. Defaults to 'random'.
        random_seed (int, optional): random seed.
        
    Returns:
        ImgPolygonAssociator: associator of updated dataset
    """

    if new_img_size is None:
        img_name = random.choice(os.listdir(target_data_dir / "images"))
        with rio.open(target_data_dir / "images" / img_name) as src:
            new_size_rows = src.height
            new_img_size_cols = src.width

    new_img_size = (new_size_rows, new_img_size_cols)

    # make sure dir args are Path objects
    source_data_dir = Path(source_data_dir)
    target_data_dir = Path(target_data_dir)

    source_assoc = ImgPolygonAssociator(source_data_dir)

    # Create new target_data_dir and subdirectories if necessary.
    for subdir in ipa.DATA_DIR_SUBDIRS:
        (target_data_dir / subdir).mkdir(parents=True, exist_ok=True)

    does_polygon_not_have_img = DoesPolygonNotHaveImg()

    random_img_selector = RandomImgSelector()

    small_imgs_around_polygons_cutter = SmallImgsAroundPolygonsCutter(source_assoc=source_assoc, 
                                                                        target_data_dir=target_data_dir, 
                                                                        polygon_filter_predicate=does_polygon_not_have_img,
                                                                        new_img_size=new_img_size, 
                                                                        img_bands=img_bands, 
                                                                        labels_bands=label_bands, 
                                                                        mode=mode, 
                                                                        random_seed=random_seed)
    
    target_assoc = create_or_update_tif_dataset_from_iter_over_polygons(source_data_dir=source_data_dir, 
                                                            target_data_dir=target_data_dir, 
                                                            polygon_filter_predicate=does_polygon_not_have_img,
                                                            img_cutter=small_imgs_around_polygons_cutter, 
                                                            img_selector=random_img_selector, 
                                                            new_img_size=new_img_size, 
                                                            img_bans=img_bands, 
                                                            label_bands=label_bands,
                                                            mode=mode)

    return target_assoc


def create_or_update_tif_dataset_from_iter_over_polygons(source_data_dir: Union[str, Path], 
                                                    target_data_dir: Union[str, Path], 
                                                    img_cutter:SingleImgCutter, 
                                                    img_selector:ImgSelector,
                                                    polygon_filter_predicate: PolygonFilterPredicate = AlwaysTrue(), 
                                                    img_bands: List[int] = None, 
                                                    label_bands: List[int] = None, 
                                                    **kwargs) -> ImgPolygonAssociator:
    """
    Create or update a data set of GeoTiffs by iterating over polygons in the source dataset. 

    Create or update a data set of GeoTiffs (images, labels, and associator) in target_data_dir from the data set of GeoTiffs in source_data_dir by iterating over the polygons in the source dataset/associator, selecting a subset of the images in the source dataset containing the polygon (using img_selector) and cutting each selected img using an img_cutter which could could depend e.g. on information in the source associator. We can restrict to a subset of the polygons in the source data_dir by filtering using the polygon_filter_predicate, which can depend on information in the source associator.

    Args:
        source_data_dir (Union[str, Path]): data directory (images, labels, associator) containing the GeoTiffs to be cut from.
        target_data_dir (Union[str, Path]): data directory of target dataset to be created or updated. 
        img_cutter (SingleImgCutter): single image cutter. Should take a polygon_filter_predicate argument. 
        img_selector (ImgSelector): image selector.
        polygon_filter_predicate (PolygonFilterPredicate, optional): predicate to filter polygons. Defaults to AlwaysTrue().
        img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
        
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

    # Select subdf of new polygons in source's polygons_df (i.e. polygons that are not in the target dataset) ...
    mask_polygons_not_in_target_polygons_df = ~source_assoc.polygons_df.index.isin(target_assoc.polygons_df.index)
    polygons_not_in_target_polygons_df = source_assoc.polygons_df.loc[mask_polygons_not_in_target_polygons_df]
    # (deepcopy, just to be safe)
    polygons_not_in_target_polygons_df = GeoDataFrame(columns=source_assoc.polygons_df.columns, 
                                        data=copy.deepcopy(polygons_not_in_target_polygons_df.values), 
                                        crs=source_assoc.polygons_df.crs)
    # (make sure types are set correctly)
    polygons_not_in_target_polygons_df = polygons_not_in_target_polygons_df.astype(source_assoc.polygons_df.dtypes)                                        
    # (set index)
    polygons_not_in_target_polygons_df.set_index(source_assoc.polygons_df.index, inplace=True)

    # ... and modify df to reflect that in target_assoc we don't yet have imgs for these new polygons.
    polygons_not_in_target_polygons_df['have_img?'] = False # will fill as we cut images from old dataset
    polygons_not_in_target_polygons_df['have_img_downloaded?'] = False

    # Integrate new polygons into target_assoc
    target_assoc.integrate_new_polygons_df(polygons_not_in_target_polygons_df)

    # dict to keep track of information which will be appended to target_assoc's imgs_df after cutting
    new_imgs_dict = {index_or_col_name: [] for index_or_col_name in [source_assoc.imgs_df.index.name] + list(source_assoc.imgs_df.columns)}

    # select polygons in target associator without an image (in particular the new ones) ...
    target_assoc_polygons_without_imgs = target_assoc.polygons_df.loc[target_assoc.polygons_df['have_img?'] == False]

    # ... and iterate over them:
    for polygon_name in tqdm(target_assoc_polygons_without_imgs.index):

        # If filter condition is satisfied, (if not, don't do anything) ...
        if polygon_filter_predicate(polygon_name, target_assoc.polygons_df, source_assoc):

            # ... then from the images in the source dataset containing the polygon ...
            source_imgs_containing_polygon = source_assoc.imgs_containing_polygon(polygon_name)

            # (polygon filter predicate to be used in img_cutter)
            only_this_polygon = OnlyThisPolygon(polygon_name)

            # ... select the images we want to cut/cut from ...
            for img_name in img_selector(source_imgs_containing_polygon, target_assoc.polygons_df, source_assoc):

                # ... and cut/cut the images (and their labels) and remember information to be appended to target_assoc imgs_df in return dict
                imgs_from_single_cut_dict = img_cutter(img_name=img_name, 
                                                        source_assoc=source_assoc, 
                                                        # (the least obvious argument: only consider this polygon for cutting the image)
                                                        polygon_filter_predicate=only_this_polygon,
                                                        new_polygons_df=target_assoc.polygons_df, 
                                                        new_graph=target_assoc._graph, 
                                                        target_data_dir=target_data_dir, 
                                                        img_bands=img_bands, 
                                                        label_bands=label_bands,
                                                        **kwargs)

                # Make sure img_cutter returned dict with same keys as needed by new_imgs_dict.
                assert set(imgs_from_single_cut_dict.keys()) == set(target_assoc.imgs_df.columns) | {target_assoc.imgs_df.index.name}, f"dict returned by img_cutter doesn't contain the same keys as needed by new_imgs_dict!"

                # Accumulate information for the new imgs in new_imgs_dict.
                for key in new_imgs_dict.keys(): 
                            new_imgs_dict[key] += (imgs_from_single_cut_dict[key])

    # Extract accumulated information about the imgs we've downloaded from new_imgs into a dataframe...
    new_imgs_df = GeoDataFrame(new_imgs_dict)
    new_imgs_df.set_crs(epsg=target_assoc.imgs_df.crs.to_epsg(), inplace=True) # copy crs
    new_imgs_df.set_index(target_assoc.imgs_df.index.name, inplace=True)

    # ... and append it to self.imgs_df.
    data_frames_list = [target_assoc.imgs_df, new_imgs_df]  
    target_assoc.imgs_df = GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)

    # Save associator to disk.
    target_assoc.save()

    return target_assoc

