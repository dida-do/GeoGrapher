from rs_tools.img_polygon_associator import ImgPolygonAssociator
from typing import Optional, Sequence, Union, List, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
from geopandas import GeoDataFrame
from rs_tools.cut.cut_every_img_to_grid import update_dataset_every_img_to_grid
from rs_tools.cut.cut_imgs_around_every_polygon import update_dataset_imgs_around_every_polygon
from rs_tools.convert_dataset.soft_to_categorical import update_dataset_soft_categorical_to_categorical
from rs_tools.convert_dataset.convert_or_update_dataset_from_tif_to_npy import update_dataset_converted_from_tif_to_npy




# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)


UPDATE_METHODS = {
    'update_dataset_every_img_to_grid' : update_dataset_every_img_to_grid, 
    'update_dataset_imgs_around_every_polygon' : update_dataset_imgs_around_every_polygon, 
    'update_dataset_soft_categorical_to_categorical' : update_dataset_soft_categorical_to_categorical, 
    'update_dataset_converted_from_tif_to_npy' : update_dataset_converted_from_tif_to_npy    
}

class UpdateFromSourceDatasetMixIn(object):
    """
    Mix-in that implements updating the dataset from the source dataset (which itself is recursively updated first) it was created from. 
    """

    def update_from_source_dataset(self):
        """ 
        Recursively update the dataset (and associator) from the source dataset (if any) that it was created from. 
        """
        
        try:
            update_method_key = self._update_from_source_dataset_dict['update_method']
        except KeyError:
            log.info(f"Dataset was not created from a source dataset.")
        else:
            # (Recursively) update source dataset first.
            source_data_dir = self._update_from_source_dataset_dict['source_data_dir']
            log.info(f"Updating source dataset in {source_data_dir}")
            source_assoc = self.__class__.from_data_dir(source_data_dir)
            source_assoc.update()
            log.info(f"Completed update of source dataset in {source_data_dir}")

            update_method = UPDATE_METHODS[update_method_key]

            return update_method(assoc=self)
