"""
Customizable general function to create or update datasets of GeoTiffs from existing ones by iterating over polygons.
"""

from typing import Union, List
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd 
from geopandas import GeoDataFrame

from rs_tools.global_constants import DATA_DIR_SUBDIRS
from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.cut.single_img_cutter_base import SingleImgCutter
from rs_tools.cut.polygon_filter_predicates import PolygonFilterPredicate, AlwaysTrue
from rs_tools.cut.img_selectors import ImgSelector 

logger = logging.getLogger(__name__)


def create_or_update_dataset_iter_over_polygons( 
        source_data_dir : Union[str, Path],
        target_data_dir : Union[str, Path],
        img_cutter : SingleImgCutter, 
        img_selector : ImgSelector, 
        polygon_filter_predicate : PolygonFilterPredicate = AlwaysTrue()
        ) -> ImgPolygonAssociator:
    """
    Create or update a dataset of GeoTiffs by iterating over polygons. 

    Add all polygons in the source dataset to the target dataset and iterate over all polygons in the target dataset. For each polygon if the polygon_filter_predicate is met use the img_selector to select a subset of the images in the source dataset for which no images for this polygon have previously been cut from. Cut each image using the img_cutter, and add the new images to the target dataset/associator. 
    
    Args:
        source_data_dir (Union[str, Path]): data directory (images, labels, associator) containing the GeoTiffs to be cut from. 
        target_data_dir (Union[str, Path]): data directory of target dataset to be created or updated.
        img_cutter (SingleImgCutter): single image cutter used to cut selected images in the source dataset. 
        img_selector (ImgSelector): image selector.
        polygon_filter_predicate (PolygonFilterPredicate, optional): predicate to filter polygons. Defaults to AlwaysTrue().

    Returns:
        ImgPolygonAssociator: associator of newly created or updated dataset
    """

    source_assoc = ImgPolygonAssociator.from_data_dir(source_data_dir)
    try: 
        target_assoc = ImgPolygonAssociator.from_data_dir(target_data_dir)
    except FileNotFoundError:
        target_assoc = source_assoc.empty_assoc_same_format(target_data_dir)
        target_assoc._cut_params_dict['cut_imgs'] = {}
            
        # Create new target_data_dir and subdirectories if necessary.
        for dir in target_assoc.image_data_dirs:
            dir.mkdir(parents=True, exist_ok=True)

    # Remember information to determine for which images to generate new labels
    imgs_in_target_dataset_before_update = set(target_assoc.imgs_df.index)
    added_polygons = []  # updated as we iterate

    # dict to temporarily store information which will be appended to target_assoc's imgs_df after cutting
    new_imgs_dict = {index_or_col_name: [] for index_or_col_name in [source_assoc.imgs_df.index.name] + list(source_assoc.imgs_df.columns)}

    # Add all polygons on source dataset to target dataset
    target_assoc.integrate_new_polygons_df(source_assoc.polygons_df)

    # For each polygon ...
    for polygon_name in tqdm(target_assoc.polygons_df.index): #!!!!!!!!! all_polygons????????? 

        # ... if we want to create new images for it ... 
        if polygon_filter_predicate(
                polygon_name=polygon_name, 
                target_assoc=target_assoc, 
                new_imgs_dict=new_imgs_dict, 
                source_assoc=source_assoc):

            # ... remember it ...
            added_polygons += [polygon_name]

            # ... and then from the images in the source dataset that containing the polygon ...
            potential_source_images = source_assoc.imgs_containing_polygon(polygon_name)
            # ... but from which an image for that polygon has not yet been cut ...
            potential_source_images = _filter_src_imgs_containing_polygon(
                                        polygon_name=polygon_name, 
                                        src_imgs_containing_polygon=potential_source_images, 
                                        target_assoc=target_assoc)

            # ... select the images we want to cut from. 
            for img_name in img_selector(
                                polygon_name=polygon_name,
                                img_names_list=potential_source_images, 
                                target_assoc=target_assoc, 
                                new_imgs_dict=new_imgs_dict, 
                                source_assoc=source_assoc):

                # Cut each image (and label) and remember the information to be appended to target_assoc imgs_df in return dict
                imgs_from_single_cut_dict = img_cutter(
                                                img_name=img_name, 
                                                polygon_name=polygon_name, 
                                                target_assoc=target_assoc, 
                                                new_imgs_dict=new_imgs_dict)

                # Make sure img_cutter returned dict with same keys as needed by new_imgs_dict.
                assert set(imgs_from_single_cut_dict.keys()) == set(target_assoc.imgs_df.columns) | {target_assoc.imgs_df.index.name}, f"dict returned by img_cutter doesn't contain the same keys as needed by new_imgs_dict!"

                # Accumulate information for the new imgs in new_imgs_dict.
                for key in new_imgs_dict.keys(): 
                            new_imgs_dict[key] += (imgs_from_single_cut_dict[key])

                new_img_names = imgs_from_single_cut_dict[source_assoc.imgs_df.index.name]
                img_bounding_rectangles = imgs_from_single_cut_dict['geometry']
                for new_img_name, img_bounding_rectangle in zip(new_img_names, img_bounding_rectangles):

                    # Update target_assoc._cut_params_dict
                    target_assoc._cut_params_dict['cut_imgs'][new_img_name] = img_name # remember img new_img_name in target was cut from img img_name 

                    # Update graph and modify polygons_df in target_assoc
                    target_assoc._add_img_to_graph_modify_polygons_df(
                        img_name=new_img_name, 
                        img_bounding_rectangle=img_bounding_rectangle)

    # Extract accumulated information about the imgs we've created in the target dataset into a dataframe...
    new_imgs_df = GeoDataFrame(
                    new_imgs_dict, 
                    crs=target_assoc.imgs_df.crs)
    new_imgs_df.set_index(target_assoc.imgs_df.index.name, inplace=True)

    # ... and append it to self.imgs_df.
    data_frames_list = [target_assoc.imgs_df, new_imgs_df]  
    target_assoc.imgs_df = GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)

    # For those images that existed before the update and now intersect with newly added polygons ...
    imgs_w_new_polygons = [img_name for polygon_name in added_polygons for img_name in target_assoc.imgs_intersecting_polygon(polygon_name) if img_name in imgs_in_target_dataset_before_update]
    # Delete the old labels (since they won't show the new polygons)...
    for img_name in imgs_w_new_polygons: 
        label_path = target_assoc.labels_dir / img_name
        label_path.unlink(missing_ok=True)
    # ... and generate new ones. 
    target_assoc.make_missing_labels(img_names=imgs_w_new_polygons)

    # Finally, save associator to disk.
    target_assoc.save()

    return target_assoc


def _filter_src_imgs_containing_polygon(
        polygon_name : str,
        src_imgs_containing_polygon : List[str], 
        target_assoc : ImgPolygonAssociator
        ) -> List[str]:
    """
    Filter out source images from which cutouts containing a polygon have already been created. 

    Args:
        polygon_name (str): name/id of polygon 
        src_imgs_containing_polygon (List[str]): list of images in source dataset containing the polygon
        target_assoc (ImgPolygonAssociator): associator of target dataset

    Returns:
        List[str]: [description]
    """
    
    previously_cut_imgs_dict = target_assoc._cut_params_dict['cut_imgs']
    target_imgs_containing_polygon = target_assoc.imgs_containing_polygon(polygon_name)
    cut_src_imgs_containing_polygon = [previously_cut_imgs_dict[img_name] for img_name in target_imgs_containing_polygon]    
    answer = [img_name for img_name in src_imgs_containing_polygon if img_name not in cut_src_imgs_containing_polygon]

    return answer