from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Sequence, Union, List, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
from geopandas import GeoDataFrame
if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator


# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)



class UpdateFromSourceDatasetMixIn(object):
    """
    Mix-in that implements updating the dataset from the source dataset (which itself is recursively updated first) it was created from. 
    """

    def update_from_source_dataset(self):
        """ 
        Recursively update the dataset (and associator) from the source dataset (if any) that it was created from. 
        """
        
        try:
            update_method = getattr(self, self._update_from_source_dataset_dict['update_method'])
        except KeyError:
            log.info(f"Unknown or missing update method: {self._update_from_source_dataset_dict['update_method']}")
        else:
            # (Recursively) update source dataset first.
            source_data_dir = self._update_from_source_dataset_dict['source_data_dir']
            log.info(f"Updating source dataset in {source_data_dir}")
            source_assoc = self.__class__.from_data_dir(source_data_dir)
            source_assoc.update()
            log.info(f"Completed update of source dataset in {source_data_dir}")

            return update_method()
