from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

from geopandas import GeoDataFrame
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import Connector

# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)


class UpdateFromSourceDSMixIn(object):
    """Mix-in that implements updating the dataset organized by the associator
    from the source dataset (which itself is recursively updated first) it was
    created from."""

    @property
    def source_data_dir(self) -> Path:
        return Path(self._update_from_source_dataset_dict['source_data_dir'])

    @source_data_dir.setter
    def source_data_dir(self, new_source_data_dir: Union[Path, str]) -> None:
        self._update_from_source_dataset_dict['source_data_dir'] = str(
            new_source_data_dir)

    def update_from_source_dataset(self):
        """Recursively update the dataset (and associator) from the source
        dataset (if any) that it was created from."""

        try:
            update_method = getattr(
                self, self._update_from_source_dataset_dict['update_method'])
        except KeyError:
            log.info("Unknown or missing update method.")
        else:
            # (Recursively) update source dataset first.
            source_data_dir = self._update_from_source_dataset_dict[
                'source_data_dir']
            log.info(f"Updating source dataset in {source_data_dir}")
            source_assoc = self.__class__.from_data_dir(source_data_dir)
            source_assoc.update_from_source_dataset()
            log.info(
                "Completed update of source dataset in %s", source_data_dir)

            update_method()  # might return a value????

            # return update_method()