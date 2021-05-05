"""
Functions to cut datasets of GeoTiffs by iterating over polygons.

Customizable general function create_or_update_dataset_from_iter_over_polygons to create or update a new remote sensing imagery dataset (images, labels, and associator) from an existing one by iterating over (a subset of) the polygons and cutting (a subset of) the images containing the polygons, as well as a specialization to functions new_dataset_one_small_img_for_each_polygon and update_dataset_one_small_img_for_each_polygon that create and update exactly one new small image for each polygon in the old dataset.
"""
from typing import Union, Callable, List
import copy
from pathlib import Path
import random
from tqdm import tqdm
import rasterio as rio
import pandas as pd 
from geopandas import GeoDataFrame

import rs_tools.img_polygon_associator as ipa
from rs_tools.utils.utils import transform_shapely_geometry
from rs_tools.cut.cut_dataset_utils import small_imgs_centered_around_polygons_cutter, have_no_img_for_polygon, alwaystrue, Filter, ImgCutter
          

def new_dataset_one_small_img_for_each_polygon(source_data_dir: Union[str, Path], target_data_dir: Union[str, Path], img_size: int = 256, centered: bool = False) -> ipa.ImgPolygonAssociator:
    """Create a new dataset (images, labels, and associator) in target_data_dir from a dataset in source_data_dir by cutting out a small square window around each polygon in the source dataset.

    Args:
        source_data_dir: a data directory, i.e. a directory conforming to the conventions of an ImgPolygonAssociator. 
        target_data_dir: directory where the new dataset will be created. Will be created if it doesn't exist already.
        img_size: size in pixels of the new square images to be created (length in pixels).
        centered: whether to center the polygon around which a small image is cut or select random window fully containing the polygon. 

    Returns:
        target_assoc: the updated associator of the target dataset.

    """

    # make sure dir args are Path objects
    source_data_dir = Path(source_data_dir)
    target_data_dir = Path(target_data_dir)

    # Create new target_data_dir and subdirectories if necessary.
    for subdir in ipa.DATA_DIR_SUBDIRS:
        (target_data_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Update new empty associator/dataset from source dataset/associator, return associator of target dataset.
    return create_or_update_dataset_from_iter_over_polygons(source_data_dir=source_data_dir, 
                                        target_data_dir=target_data_dir, 
                                        polygon_filter_predicate=have_no_img_for_polygon,
                                        img_cutter=small_imgs_centered_around_polygons_cutter, 
                                        img_selector=random_img_selector, 
                                        img_size=img_size, 
                                        centered=centered)

def update_dataset_one_small_img_for_each_polygon(source_data_dir: Union[str, Path], target_data_dir: Union[str, Path], img_size: int = 256, centered: bool = False) -> ipa.ImgPolygonAssociator:
    """Update a dataset (images, labels, and associator) in target_data_dir that was created using new_dataset_one_small_img_for_each_polygon from an updated version of the dataset in source_data_dir by adding the new polygons from the source dataset that are not yet in the target dataset and then adding small images and labels for those polygons from the updated source dataset as well as for the polygons in the target dataset that had (i.e. were contained in) no images in the pre-update version of source dataset but now have an image in the updated source dataset. 

    Args:
        source_data_dir: a data directory, i.e. a directory conforming to the conventions of an ImgPolygonAssociator. 
        target_data_dir: data_dir of dataset to be updated.
        img_size: size in pixels of the new square images to be created (length in pixels)
        centered: whether to center the polygon around which a small image is cut or select random window fully containing the polygon.

    Returns:
        target_assoc: the updated associator of the target dataset

    """

    # make sure dir args are Path objects
    source_data_dir = Path(source_data_dir)
    target_data_dir = Path(target_data_dir)

    # Update new empty associator/dataset from source dataset/associator, return associator of target_dataset
    return create_or_update_dataset_from_iter_over_polygons(source_data_dir=source_data_dir, 
                                        target_data_dir=target_data_dir, 
                                        polygon_filter_predicate=have_no_img_for_polygon,
                                        img_cutter=small_imgs_centered_around_polygons_cutter, 
                                        img_selector=random_img_selector, 
                                        img_size=img_size, 
                                        centered=centered)

ImgSelector = Callable[[List[str], GeoDataFrame, ipa.ImgPolygonAssociator], List[str]]
"""Type alias for the image selector functions. Will be made obsolete when we refactor these functions to inherit from an ABC.
Selects an image from a list of images. An example is given by random_img_selector. 
            
    Args:
        img_names_list: list of imags containing a polygon
        new_polygons_df: 
        source_assoc:
    Returns:
        a sublist of img_names_list. If img_names_list is empty should return the empty list. 
"""

def random_img_selector(img_names_list: List[str], new_polygons_df: GeoDataFrame, source_assoc: ipa.ImgPolygonAssociator) -> List[str]:
    """Randomly selects an image from a list of images if it is non-empty.

    Args:
        img_names_list: list of image names
        new_polygons_df: not used
        source_assoc: not used

    Returns:
        A list with one image name or an empty list if the original list was empty.
    """

    return [random.choice(img_names_list)] if img_names_list != [] else []

def create_or_update_dataset_from_iter_over_polygons(source_data_dir: Union[str, Path], target_data_dir: Union[str, Path], img_cutter:ImgCutter, img_selector:ImgSelector,
polygon_filter_predicate: Filter = alwaystrue, img_bands: List[int] = None, label_bands: List[int] = None, **kwargs) -> ipa.ImgPolygonAssociator:
    """Create or update a data set (images, labels, and associator) in target_data_dir from the data set in source_data_dir by iterating over the polygons in the
    source dataset/associator, selecting a subset of the images in the source dataset containing the polygon (using img_selector) and cutting each selected img
    using an img_cutter which could could depend e.g. on information in the source associator. We can restrict to a subset of the polygons in the source data_dir
    by filtering using the polygon_filter_predicate, which can depend on information in the source associator.

    Args:
        source_data_dir: a data directory, i.e. a directory conforming to the conventions of an ImgPolygonAssociator. 
        target_data_dir: data directory of target dataset to be created or updated. 
        img_cutter: Function that cuts a single image (e.g. small_imgs_centered_around_polygons_cutter in cut.cut_dataset_utils)
        img_selector: selects an image from a list of images. An example is given by random_img_selector. 
        target_assoc: the target associator. This is optional and we only allow it as an argument so that we can [TODO: do what?]
        polygon_filter_predicate: predicate to filter polygons. See above in args for img_cutter.
        kwargs: additional keyword arguments that might be needed by an img_cutter.
    
    Returns:
        target_assoc: the updated associator of the target dataset.

    """

    # Make sure dir args are Path objects.
    source_data_dir = Path(source_data_dir)
    target_data_dir = Path(target_data_dir)

    # Load source associator.
    source_assoc = ipa.ImgPolygonAssociator(data_dir = source_data_dir)

    # Load target associator, if it exists already, ...
    if (target_data_dir / "imgs_df.geojson").is_file() and (target_data_dir / "polygons_df.geojson").is_file() and (target_data_dir / "graph.json").is_file(): # try target_assoc = ipa.ImgPolygonAssociator(data_dir = target_data_dir) except ... didn't work for reason I don't understand, so to test if an associator exists, I just check if the associator files exist
        target_assoc = ipa.ImgPolygonAssociator(data_dir = target_data_dir)    
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

        # DEBUG INFO
        # print(f"new_assoc_from_iter_over_polygons: considering {polygon_name} ")
        
        # If filter condition is satisfied, (if not, don't do anything) ...
        if polygon_filter_predicate(polygon_name, target_assoc.polygons_df, source_assoc):

            # ... then from the images in the source dataset containing the polygon ...
            source_imgs_containing_polygon = source_assoc.imgs_containing_polygon(polygon_name)

            # DELETE? READABILITY OF USING DEF DIRECTLY BELOW?
            # to be used below in img_cutter to select for each image exactly one subimage, namely one for _this_ polygon
            # only_this_polygon = lambda p_name, new_p_df, src_assoc: p_name == polygon_name

            # DEBUG INFO
            # print(f"new_assoc_from_iter_over_polygons: for polygon {polygon_name} selecting images from ")

            # DEBUG INFO
            # print(f"polygon: {polygon_name}, selecting from source imgs {imgs_containing_polygon}") 

            # ... select the images we want to cut/cut from ...
            for img_name in img_selector(source_imgs_containing_polygon, target_assoc.polygons_df, source_assoc):

                # ... and cut/cut the images (and their labels) and remember information to be appended to target_assoc imgs_df in return dict
                imgs_from_single_cut_dict = img_cutter(img_name=img_name, 
                                                            source_assoc=source_assoc, 

                                                            # (the least obvious argument: only consider this polygon for cutting the image)
                                                            polygon_filter_predicate=lambda p, df, assoc: p == polygon_name, 

                                                            new_polygons_df=target_assoc.polygons_df, 
                                                            new_graph=target_assoc._graph, 
                                                            target_data_dir=target_data_dir, 
                                                            img_bands=img_bands, 
                                                            label_bands=label_bands,
                                                            **kwargs)

                # Make sure img_cutter returned dict with same keys as needed by new_imgs_dict.
                assert set(imgs_from_single_cut_dict.keys()) == set(target_assoc.imgs_df.columns) | {target_assoc.imgs_df.index.name}

                # DEBUG INFO
                # print(f"\nimgs_from_single_cut_dict: {imgs_from_single_cut_dict}")

                # Accumulate information for the new imgs in new_imgs_dict.
                for key in new_imgs_dict.keys(): 
                            new_imgs_dict[key] += (imgs_from_single_cut_dict[key])

                # DEBUG INFO
                # print(f"new_imgs_dict after considering {polygon_name}: \n{new_imgs_dict}")

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

