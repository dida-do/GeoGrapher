"""
Create a new dataset from an existing one by combining and/or removing segmentation classes.
"""

from rs_tools.global_constants import DATA_DIR_SUBDIRS
from typing import List, Union, Optional
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

import rasterio as rio 

from rs_tools import ImgPolygonAssociator

log = logging.Logger(__name__)

def combine_remove_seg_classes(
        source_data_dir : Union[Path, str], 
        target_data_dir : Union[Path, str], 
        seg_classes : List[Union[str, List[str]]], 
        new_seg_classes : Optional[List[str]] = None, 
        class_separator : str = "+", 
        new_background_class : Optional[str] = None, 
        new_mask_class : Optional[str] = None, 
        remove_imgs : bool = True
        ) -> ImgPolygonAssociator:
    """
    Create a new dataset/associator from an existing one by combining and/or removing segmentation classes. Works for both categorical and soft-categorical label types. 

    Args:
        source_data_dir (pathlib.Path or str): data_dir of source dataset/associator
        target_data_dir (pathlib.Path or str, optional): data_dir of target dataset/associator. If None (default value), will convert in place, i.e. overwrite source dataset and associator of tifs.
        seg_classes: (List[Union[List[str], str]]) segmentation classes in existing dataset/associator to be kept and combined in new dataset/associator. E.g. [['ct', 'ht'], 'wr', ['h']] will combine the 'ct' and 'ht' classes, and also keep the 'wr' and 'h' classes. Along with the regular segmentation classes one may also use the mask or background classes here.  
        new_seg_class_names: (Optional[List[str]]) optional list of names of new segmentation classes corresponding to seg_classes. Defaults to joining the names of existing using the class_separator (which defaults to class_separator).
        class_separator: (str) used if the new_seg_class_names argument is not provided to join the names of existing segmentation classes that are to be kept. Defaults to class_separator.
        new_background_class (Optional[str]): optional new background class, defaults to None, i.e. old background class
        new_mask_class (Optional[str]): optional new ignore class, defaults to None, i.e. old ignore class
        remove_imgs: (bool). If True, remove images not containing polygons of the segmentation classes to be kept.

    Returns:
        The ImgPolygonAssociator of the new dataset. 

    Note: 
        For the purposes of this function the background and mask classes will be treated as regular segmentation classes. In particular, if you do not include them in the seg_classes argument, polygons of the background or mask class will be lost. 

    Example:

        Suppose the dataset in the SOURCE_DATA_DIR directory has the following segmentation classes (including the background class 'bg' and mask class 'ig'): 
        
        'ct', 'ht', 'wr', 'h', 'pt', 'ig', 'bg'

        Then the command

        >>> new_assoc_remove_combine_seg_classes(
                source_data_dir=SOURCE_DATA_DIR, 
                target_data_dir=TARGET_DATA_DIR, 
                seg_classes=[['ct', 'ht'], 'wr', ['h']]) 

        will create a new dataset in TARGET_DATA_DIR where only the polygons labelled as (or having non-zero probability if the labels are soft-categorical) 'ct', 'ht', 'wr', and 'h' have been retained, and the classes 'ct' and 'ht' have been combined to a class 'ct+ht'. If the label type is soft-categorical, the new dataset will still retain any existing prob_seg_class columns for the background and mask class, though all the entries will in this case be zero, since neither 'bg' nor 'ig' were in the seg_classes argument.

    Warning:
        Make sure this works as desired for the edge cases pertaining to the ignore or background classes! 

    """

    # load source assoc
    source_assoc = ImgPolygonAssociator.from_data_dir(source_data_dir)

    seg_classes = list(map(lambda x: x if isinstance(x,list) else [x], seg_classes))
    classes_to_keep = [class_ for list_of_classes in seg_classes for class_ in list_of_classes]
    
    assert len(classes_to_keep) == len(set(classes_to_keep)), "a segmentation class in the source dataset can only be in at most one of the new segmentation classes"

    if new_seg_classes is None:
        new_seg_classes = list(map(lambda x: class_separator.join(x), seg_classes))
    else:
        assert len(new_seg_classes) == len(set(new_seg_classes)), "new_seg_class_names need to be distinct!"
        assert len(new_seg_classes) == len(seg_classes), "there should be as many new_seg_class_names as there are segmentation classes to be kept!"

    assert source_data_dir != target_data_dir, "Source_data_dir and target_data_dir should not be equal!"

    if new_background_class is not None:
        assert new_background_class in new_seg_classes

    if new_mask_class is not None:
        assert new_mask_class in new_seg_classes

    def get_new_class(class_ : str) -> str:
        for classes_ in seg_classes: 
            if class_ in classes_:
                return class_separator.join(classes_)

    for subdir in DATA_DIR_SUBDIRS:
        
        subdir_path = Path(target_data_dir / subdir)
        
        # Make sure subdir exists ...
        subdir_path.mkdir(parents=True, exist_ok=True)
        
        # ... but is empty.
        if os.listdir(subdir_path) != []:
                raise Exception(f"{subdir} subdirectory of target_data_dir {target_data_dir} should be empty!")

    # Make sure no associator files already exist.
    if (target_data_dir / "imgs_df.geojson").is_file():
        raise Exception(f"target_data_dir {target_data_dir} already contains associator file imgs_df.geojson")

    if (target_data_dir / "polygons_df.geojson").is_file():
        raise Exception(f"target_data_dir {target_data_dir} already contains associator file polygons_df.geojson")

    if (target_data_dir / "graph.json").is_file():
        raise Exception(f"target_data_dir {target_data_dir} already contains associator file graph.json")

    if (target_data_dir / "params_dict.json").is_file():
        raise Exception(f"target_data_dir {target_data_dir} already contains associator file params_dict.json")
    
    # build empty target associator
    target_assoc = source_assoc.empty_assoc_same_format(target_data_dir)

    new_polygons_df = source_assoc.polygons_df

    if source_assoc.label_type == 'categorical':
        
        # keep only polygons belonging to segmentation we want to keep
        new_polygons_df = new_polygons_df.loc[new_polygons_df['type'].apply(lambda class_: class_ in classes_to_keep)]
        # rename to new classes
        new_polygons_df['type'] = new_polygons_df['type'].apply(get_new_class) 
        # integrate into target_assoc
        target_assoc.integrate_new_polygons_df(new_polygons_df)
    
    elif source_assoc.label_type == 'soft-categorical':
        
        # drop cols for probabilities of classes not to be kept
        all_classes = source_assoc._params_dict['segmentation_classes'] + [x for x in [source_assoc._params_dict['background_class'], source_assoc._params_dict['mask_class']] if x is not None]
    
        classes_to_drop = [class_ for class_ in all_classes if class_ not in classes_to_keep]
        cols_to_drop = list(map(lambda class_: f"prob_seg_class_{class_}", classes_to_drop))
        new_polygons_df = new_polygons_df.drop(columns=cols_to_drop)

        # adjust columns of target_assoc.polygons_df
        cols_of_classes_to_keep = list(map(lambda class_: f"prob_seg_class_{class_}", classes_to_keep))
        target_assoc.polygons_df = target_assoc.polygons_df.drop(
                                        columns=cols_of_classes_to_keep + cols_to_drop)
        for class_ in new_seg_classes:
            target_assoc.polygons_df[f"prob_seg_class_{class_}"] = None

        # create temporary dataframe to avoid column name conflicts when renaming/deleting etc
        temp_polygons_df = pd.DataFrame()

        # for each row/polygon find sum of probabilities for the remaining segmentation classes         
        cols_probs_remaining_classes = list(map(lambda class_: f"prob_seg_class_{class_}", classes_to_keep))        
        sum_probs_remaining_classes = pd.DataFrame(new_polygons_df[cols_probs_remaining_classes].sum(axis=1), columns=['sum'], index=new_polygons_df.index)
        sum_is_zero = (sum_probs_remaining_classes['sum'] == 0)

        # remove rows/polygons which do not belong to remaining classes
        new_polygons_df = new_polygons_df.loc[~sum_is_zero]
        sum_probs_remaining_classes = sum_probs_remaining_classes.loc[~sum_is_zero]

        # renormalize probabilities to sum to 1
        new_polygons_df.loc[:, cols_probs_remaining_classes] = new_polygons_df[cols_probs_remaining_classes].div(sum_probs_remaining_classes['sum'], axis=0)

        # combine probabilities of new_classes and drop old classes
        for classes_to_combine, new_seg_class_name in zip(seg_classes, new_seg_classes):
            cols_to_add = list(map(lambda class_: f"prob_seg_class_{class_}", classes_to_combine))
            temp_polygons_df[f"prob_seg_class_{new_seg_class_name}"] = new_polygons_df[cols_to_add].sum(axis=1)
            new_polygons_df = new_polygons_df.drop(columns=cols_to_add)
        
        new_polygons_df = pd.concat(
                            [
                                new_polygons_df, 
                                temp_polygons_df
                            ], 
                            axis=1) # i.e. concatenate along column axis

        # Recompute most likely type column.
        new_polygons_df["most_likely_class"] = new_polygons_df[temp_polygons_df.columns].apply(
            lambda s: 
                ",".join(
                    map(
                        lambda col_name: col_name[15:], 
                        s[(s == s.max()) & (s!=0)].index.values)), 
            axis=1) 

        target_assoc.integrate_new_polygons_df(new_polygons_df)

    else:
        raise ValueError(f"Unknown label type: {source_assoc._label_type}")

    # set information about background, mask, and segmentation classes in target_assoc
    if new_background_class is not None:
        target_assoc._params_dict['background_class'] = new_background_class
    elif source_assoc._params_dict['background_class'] not in new_seg_classes:
        target_assoc._params_dict['background_class'] = None

    if new_mask_class is not None:
        target_assoc._params_dict['mask_class'] = new_mask_class
    elif source_assoc._params_dict['mask_class'] not in new_seg_classes:
        target_assoc._params_dict['mask_class'] = None

    target_assoc._params_dict['segmentation_classes'] = [class_ for class_ in new_seg_classes if class_ not in {target_assoc._params_dict['background_class'], target_assoc._params_dict['mask_class']}]

    # integrate the imgs_df and optionally keep only those imgs connected to remaining polygons
    target_assoc.integrate_new_imgs_df(source_assoc.imgs_df)

    if remove_imgs == True:
        imgs_to_remove = [img_name for img_name in target_assoc.imgs_df.index if len(target_assoc.polygons_intersecting_img(img_name)) == 0]
        target_assoc.drop_imgs(imgs_to_remove, remove_imgs_from_disk=False)

    # save associator
    target_assoc.save()

    # copy images to target data dir
    for img_name in tqdm(target_assoc.imgs_df.index):
        shutil.copy(source_data_dir / f"images/{img_name}", target_data_dir / f"images/{img_name}")

    # make missing labels and if possible masks
    target_assoc.make_missing_geotif_labels()
    if target_assoc._params_dict['mask_class'] is not None and target_assoc._params_dict['label_type'] == 'categorical':
        target_assoc.make_missing_masks()

    log.info(f"new_cat_assoc_remove_classes: done!")

    return target_assoc

            




