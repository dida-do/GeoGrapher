"""
Convert a dataset (images, labels, and associator) of GeoTiffs to a dataset of npy files.
"""
from __future__ import annotations
from typing import Union, Optional, List
from typing import TYPE_CHECKING
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

import rasterio as rio

#if TYPE_CHECKING:
from rs_tools import ImgPolygonAssociator

log = logging.Logger(__name__)

def convert_dataset_from_tif_to_npy(
        target_data_dir : Union[Path, str],
        source_data_dir : Optional[Union[Path, str]] = None,
        source_assoc : Optional[ImgPolygonAssociator] = None,
        img_bands : Optional[List[int]] = None,
        label_bands : Optional[List[int]] = None,
        squeeze_label_channel_dim_if_single_channel : bool = True,
        channels_first_or_last_in_npy : str = 'last'
        ) -> ImgPolygonAssociator:
    """
    TODO:

    Args:
        target_data_dir (Union[Path, str]): [description]
        source_data_dir (Optional[Union[Path, str]], optional): [description]. Defaults to None.
        source_assoc (Optional[ImgPolygonAssociator], optional): [description]. Defaults to None.
        img_bands (Optional[List[int]], optional): [description]. Defaults to None.
        label_bands (Optional[List[int]], optional): [description]. Defaults to None.
        squeeze_label_channel_dim_if_single_channel (bool, optional): [description]. Defaults to True.
        channels_first_or_last_in_npy (str, optional): [description]. Defaults to 'last'.

    Returns:
        ImgPolygonAssociator: [description]
    """

    if not ((source_data_dir is not None) ^ (source_assoc is not None)):
        raise ValueError(f"Exactly one of the source_data_dir or source_assoc arguments needs to be set (i.e. not None).")

    if source_assoc is None:
        source_assoc = ImgPolygonAssociator.from_data_dir(source_data_dir)

    _convert_or_update_dataset_from_tif_to_npy(
        create_or_update='create',
        source_assoc=source_assoc,
        target_data_dir=target_data_dir,
        img_bands=img_bands,
        label_bands=label_bands,
        squeeze_label_channel_dim_if_single_channel=squeeze_label_channel_dim_if_single_channel,
        channels_first_or_last_in_npy=channels_first_or_last_in_npy)


def update_dataset_converted_from_tif_to_npy(
        data_dir : Optional[Union[Path, str]] = None,
        assoc : Optional[ImgPolygonAssociator] = None
        ) -> ImgPolygonAssociator:
    """
    TODO:

    Args:
        data_dir (Optional[Union[Path, str]], optional): [description]. Defaults to None.
        assoc (Optional[ImgPolygonAssociator], optional): [description]. Defaults to None.

    Returns:
        ImgPolygonAssociator: [description]
    """

    if not ((data_dir is not None) ^ (assoc is not None)):
        raise ValueError(f"Exactly one of the data_dir or assoc arguments needs to be set (i.e. not None).")

    if assoc is None:
        assoc = ImgPolygonAssociator.from_data_dir(data_dir)

    # Read args from update dict
    source_data_dir = assoc._update_from_source_dataset_dict['source_data_dir']
    img_bands = assoc._update_from_source_dataset_dict['img_bands']
    label_bands = assoc._update_from_source_dataset_dict['label_bands']
    squeeze_label_channel_dim_if_single_channel = assoc._update_from_source_dataset_dict['squeeze_label_channel_dim_if_single_channel']
    channels_first_or_last_in_npy = assoc._update_from_source_dataset_dict['channels_first_or_last_in_npy']

    source_assoc = ImgPolygonAssociator.from_data_dir(source_data_dir)

    return _convert_or_update_dataset_from_tif_to_npy(
        create_or_update='update',
        source_assoc=source_assoc,
        target_assoc=assoc,
        img_bands=img_bands,
        label_bands=label_bands,
        squeeze_label_channel_dim_if_single_channel=squeeze_label_channel_dim_if_single_channel,
        channels_first_or_last_in_npy=channels_first_or_last_in_npy
    )


def _convert_or_update_dataset_from_tif_to_npy(
        create_or_update : str,
        source_assoc : ImgPolygonAssociator,
        target_data_dir : Optional[Union[Path, str]] = None,
        target_assoc : Optional[ImgPolygonAssociator] = None,
        img_bands : Optional[List[int]] = None,
        label_bands : Optional[List[int]] = None,
        squeeze_label_channel_dim_if_single_channel : bool = True,
        channels_first_or_last_in_npy : str = 'last'
        ) -> ImgPolygonAssociator:
    """
    TODO

    Remark:
        If the resulting labels have one channel will by default squeeze the channel dimension/axis, so that the resulting image has shape (height, width) instead of (1, height, width).

    Args:
        create_or_update (str): One of 'create' or 'update'.
        source_assoc (ImgPolygonAssociator): associator of dataset of GeoTiffs that are to be converted
        target_data_dir (Optional[Union[Path, str]], optional): data directory of .npys to be created. Defaults to None.
        target_assoc (Optional[ImgPolygonAssociator], optional): associator of dataset of .npys to be created. Defaults to None.
        img_bands ([List[int], optional): List of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional): List of bands to extract from source labels. Defaults to None (i.e. all bands).
        squeeze_label_channel_dim_if_single_channel (bool, optional): If True squeeze the label channel dimension/axis if possible, so that it has shape (height, width) instead of (1, height, width) or (height, width, 1).  Defaults to True.
        channels_first_or_last_in_npy (str, optional): One of 'last' or 'first'. If 'last' the resulting arrays (if they are not a label with one channel and squeeze_label_channel_dim_if_single_channel is True) should have shape (height, width, channels), if 'first' (channels, height, width). Defaults to 'last' because albumentations expects this format.

    Raises:
        Exception: [description]

    Returns:
        ImgPolygonAssociator: associator of newly created or updated dataset
    """

    def tif_filename_to_npy(tif_filename):
        return tif_filename[:-4] + ".npy"

    # argument consistency checks
    if create_or_update not in {'create', 'update'}:
        raise ValueError(f"create_or_update argument must be one of 'create' or 'update'")
    if not ((target_data_dir is not None) ^ (target_assoc is not None)):
        raise ValueError(f"Exactly one of the target_data_dir and target_assoc arguments needs to be given (i.e. not None)")

    # source associator
    source_tif_assoc = source_assoc

    # target associator
    if create_or_update == 'update':

        if target_assoc is None or target_data_dir is not None:
            raise ValueError(f"When updating, the target_assoc arg needs to be set (not None), and the target_data_dir arg should not be set (i.e. be None).")

        if target_data_dir is not None:
            target_npy_assoc = ImgPolygonAssociator.from_data_dir(target_data_dir)
        elif target_assoc is not None:
            target_npy_assoc = target_assoc

    elif create_or_update == 'create':
        # Create target assoc, ...
        target_npy_assoc = source_tif_assoc.empty_assoc_same_format_as(target_data_dir, copy_update_from_source_dataset_dict=True)
        # ..., image data dirs, ...
        for dir in target_npy_assoc.img_data_dirs:
            dir.mkdir(parents=True, exist_ok=True)
            if list(dir.iterdir()) != []:
                 raise Exception(f"{dir} should be empty!")

        # ... and the associator dir.
        target_npy_assoc.assoc_dir.mkdir(parents=True, exist_ok=True)

        # Make sure no associator files already exist.
        if list(target_npy_assoc.assoc_dir.iterdir()) != []:
            raise Exception(f"The assoc_dir in {target_npy_assoc.assoc_dir} should be empty!")

    # need this later
    polygons_that_will_be_added_to_target_dataset = set(source_tif_assoc.polygons_df.index) - set(target_npy_assoc.polygons_df.index)

    # add polygons to target dataset
    target_npy_assoc.add_to_polygons_df(source_tif_assoc.polygons_df)

    # Generate imgs_df of npy_assoc ...
    npy_imgs_df = source_tif_assoc.imgs_df
    tif_assoc_imgs_df_index_name = source_tif_assoc.imgs_df.index.name # the next line destroys the index name of tif_assoc.imgs_df, so we remember it ...
    tif_img_name_list = npy_imgs_df.index.tolist().copy() # (it's either the .tolist() or .copy() operation, don't understand why)
    npy_imgs_df.index = list(map(tif_filename_to_npy, tif_img_name_list))
    npy_imgs_df.index.name = tif_assoc_imgs_df_index_name # ... and then set it by hand

    # ... and add to target dataset of .npys
    target_npy_assoc.add_to_imgs_df(npy_imgs_df)

    # We're done creating/updating the associator. Let's turn to the images and labels:

    # Determine which images to copy to target dataset
    imgs_that_already_existed_in_target_images_dir = {img_path.name for img_path in target_npy_assoc.images_dir.iterdir()}
    imgs_in_source_images_dir = {img_path.name for img_path in source_tif_assoc.images_dir.iterdir()}
    imgs_in_source_that_are_not_in_target = {tif_filename_to_npy(img_name) for img_name in imgs_in_source_images_dir} - imgs_that_already_existed_in_target_images_dir

    # For each image that already existed in the target dataset ...
    for img_name in imgs_that_already_existed_in_target_images_dir:
        # ... if among the polygons intersecting it in the target dataset ...
        polygons_intersecting_img = set(target_npy_assoc.polygons_intersecting_img(img_name))
        # ... there is a *new* polygon ...
        if polygons_intersecting_img & polygons_that_will_be_added_to_target_dataset != set():
            # ... then we need to update the label for it, so we delete the current label.
            (target_npy_assoc.labels_dir / img_name).unlink(missing_ok=True)

    # For the images_dir and labels_dir of the source tif and target npy dataset ...
    for tif_dir, npy_dir in zip(source_tif_assoc.img_data_dirs, target_npy_assoc.img_data_dirs):
        # ... go through all tif files. ...
        for tif_img_name in tqdm(tif_img_name_list, desc=f"Converting {tif_dir.name}"):
            # If the corresponding npy in the target image data dir does not exist ...
            if not (npy_dir / tif_filename_to_npy(tif_img_name)).is_file():
                # ... convert the tif: Open the tif file ...
                with rio.open(tif_dir / tif_img_name) as src:

                    # ... set the bands ...
                    if str(tif_dir.name) == "images":
                        if img_bands is None:
                            bands = list(range(1, src.count + 1)) # default to all bands
                        else:
                            bands = img_bands
                    if str(tif_dir.name) == "labels":
                        if label_bands is None:
                            bands = list(range(1, src.count + 1)) # default to all bands
                        else:
                            bands = label_bands

                    # extract bands to list of arrays
                    seq_extracted_np_bands = [src.read(band) for band in bands]

                    # new img path
                    new_npy_img_path = npy_dir / tif_filename_to_npy(tif_img_name)

                    # axis along which to stack
                    if channels_first_or_last_in_npy == 'last':
                        axis = 2
                    else: # 'first'
                        axis = 0

                    # stack band arrays into single tensor
                    np_img = np.stack(seq_extracted_np_bands, axis = axis)

                    # squeeze np_img if necessary
                    if str(tif_dir.name) == "labels" and squeeze_label_channel_dim_if_single_channel == True:
                        if len(bands) == 1:
                            np_img = np.squeeze(np_img, axis=axis)

                    # save numpy array
                    np.save(new_npy_img_path, np_img)

    # Remember the cutting params ...
    target_npy_assoc._update_from_source_dataset_dict.update(
        {
            'update_method' : 'update_dataset_soft_categorical_to_categorical',
            'source_data_dir' : source_tif_assoc.images_dir.parent, # !!! Assuming standard directory format here.
            'img_bands' : img_bands,
            'label_bands' : label_bands,
            'squeeze_label_channel_dim_if_single_channel' : squeeze_label_channel_dim_if_single_channel,
            'channels_first_or_last_in_npy' : channels_first_or_last_in_npy
        }
    )
    # ... and save the associator.
    target_npy_assoc.save()

    log.info(f"convert_dataset_tif2numpy: done!")






