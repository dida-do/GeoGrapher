"""
Convert a dataset (images, labels, and associator) of GeoTiffs to a dataset of npy files.
"""
from __future__ import annotations
from typing import Union, Optional, List
from typing import TYPE_CHECKING
import os
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

import rasterio as rio 

import rs_tools.img_polygon_associator as ipa
if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator

log = logging.Logger(__name__)


def convert_assoc_dataset_tif2npy(source_data_dir: Union[Path, str], 
        target_data_dir: Optional[Union[Path, str]] = None, 
        img_bands: Optional[List[int]] = None, 
        label_bands: Optional[List[int]] = None, 
        squeeze_label_channel_dim_if_single_channel: bool = True, 
        channels_first_or_last_in_npy: str = 'last') -> ImgPolygonAssociator:
    """
    Convert images and labels in dataset/associator from GeoTiffs to .npy format. If the resulting labels have one channel will by default squeeze the channel dimension/axis, so that the resulting image has shape (height, width) instead of (1, height, width).

    Args:
        source_data_dir (Union[Path, str]): data directory of GeoTiffs to be converted
        target_data_dir (Union[Path, str], optional): data directory of .npys to be created. Defaults to None (i.e. convert in place). 
        img_bands ([List[int], optional): List of bands to extract from source images. Defaults to None (i.e. all bands).
        label_bands (List[int], optional): List of bands to extract from source labels. Defaults to None (i.e. all bands).
        squeeze_label_channel_dim_if_single_channel (bool, optional): If True squeeze the label channel dimension/axis if possible, so that it has shape (height, width) instead of (1, height, width) or (height, width, 1).  Defaults to True.
        channels_first_or_last_in_npy (str, optional): One of 'last' or 'first'. If 'last' the resulting arrays (if they are not a label with one channel and squeeze_label_channel_dim_if_single_channel is True) should have shape (height, width, channels), if 'first' (channels, height, width). Defaults to 'last' because albumentations expects this format.

    Raises:
        Exception: If the images or labels subdirectory of the target_data_dir exist but are not empty. 
        FileExistsError: If the target_data_dir already contains an associator file.
        
    Returns:
        ImgPolygonAssociator: the associator of the newly created dataset of .npys
    """
        
    def filename_tif2npy(tif_filename):
        return tif_filename[:-4] + ".npy"

    # load tif assoc
    tif_assoc = ipa.ImgPolygonAssociator(data_dir=source_data_dir)
    
    source_data_dir = Path(source_data_dir)

    # set target_data_dir and create if necessary
    if target_data_dir == None:
        
        target_data_dir = source_data_dir
        # remember we're converting a dataset in place. We'll need to delete the GeoTiffs in this case.
        
        inplace=True
    
    else:
        
        inplace=False

        for subdir in ipa.DATA_DIR_SUBDIRS:
            
            subdir_path = Path(target_data_dir / subdir)
            
            # Make sure subdir exists ...
            subdir_path.mkdir(parents=True, exist_ok=True)
            
            # ... but is empty.
            if os.listdir(subdir_path) != []:

                 raise Exception(f"{subdir} subdirectory of target_data_dir {target_data_dir} should be empty!")

        # Make sure no associator files already exist.
        if (target_data_dir / "imgs_df.geojson").is_file():

            raise FileExistsError(f"target_data_dir {target_data_dir} already contains associator file imgs_df.geojson")

        if (target_data_dir / "polygons_df.geojson").is_file():

            raise FileExistsError(f"target_data_dir {target_data_dir} already contains associator file polygons_df.geojson")

        if (target_data_dir / "graph.json").is_file():

            raise FileExistsError(f"target_data_dir {target_data_dir} already contains associator file graph.json")

        if (target_data_dir / "params_dict.json").is_file():

            raise FileExistsError(f"target_data_dir {target_data_dir} already contains associator file params_dict.json")

    # build new .npy empty associator
    new_npy_assoc = ipa.empty_assoc_same_format_as(target_data_dir=target_data_dir, source_assoc=tif_assoc)
    # add polygons to empty assoc
    new_npy_assoc.integrate_new_polygons_df(tif_assoc.polygons_df)

    # add imgs to new .npy assoc
    new_npy_imgs_df = tif_assoc.imgs_df
    tif_assoc_imgs_df_index_name = tif_assoc.imgs_df.index.name # the next line destroys the index name of tif_assoc.imgs_df, so we remember it ...
    tif_img_name_list = new_npy_imgs_df.index.tolist().copy() # (it's either the .tolist() or .copy() operation, don't understand why)
    new_npy_imgs_df.index = list(map(filename_tif2npy, tif_img_name_list))
    new_npy_imgs_df.index.name = tif_assoc_imgs_df_index_name # ... and then set it by hand like this.

    new_npy_assoc.integrate_new_imgs_df(new_npy_imgs_df)

    # convert all images:
    for subdir in ipa.DATA_DIR_SUBDIRS:
        for tif_img_name in tqdm(tif_img_name_list, desc=f"Converting {subdir}"):
            with rio.open(Path(tif_assoc.data_dir) / subdir / tif_img_name) as src:
                
                # set bands
                if str(subdir) == "images":
                    # default to all bands
                    if img_bands is None:
                        bands = list(range(1, src.count + 1)) 
                    else:
                        bands = img_bands 
                else: # labels
                    # default to all bands
                    if label_bands is None:
                        bands = list(range(1, src.count + 1)) 
                    else:
                        bands = label_bands 
                
                # extract bands to list of arrays
                seq_extracted_np_bands = [src.read(band) for band in bands]

                # new img path
                new_npy_img_path = target_data_dir / Path(subdir) / filename_tif2npy(tif_img_name)

                # axis along which to stack
                if channels_first_or_last_in_npy == 'last':
                    axis = 2
                else: # 'first' 
                    axis = 0

                # stack band arrays into single tensor
                np_img = np.stack(seq_extracted_np_bands, axis = axis)

                # squeeze np_img if necessary
                if str(subdir) == "labels" and squeeze_label_channel_dim_if_single_channel == True:
                    if len(bands) == 1:
                        np_img = np.squeeze(np_img, axis=axis)

                # save numpy array
                np.save(new_npy_img_path, np_img)

            if inplace==True:
                # delete tif
                (Path(tif_assoc.data_dir) / subdir / tif_img_name).unlink()

    # save associator
    new_npy_assoc.save()

    log.info(f"convert_assoc_dataset_tif2numpy: done!")

            



