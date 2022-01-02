"""
Mixin to cut datasets of GeoTiffs (or update previously cut datasets)
by cutting each image in the source dataset to a grid of images.
"""

from __future__ import annotations
from rs_tools.global_constants import DATA_DIR_SUBDIRS
from typing import TYPE_CHECKING, Union, List, Tuple, Optional
from rs_tools.cut.type_aliases import ImgSize
import logging
from pathlib import Path

if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.cut.single_img_cutter_grid import ImgToGridCutter
from rs_tools.cut.img_filter_predicates import AlwaysTrue

logger = logging.getLogger(__name__)

class CreateDSCutEveryImgToGridMixIn(object):

    def cut_every_img_to_grid(
            self,
            target_data_dir : Union[str, Path],
            new_img_size : ImgSize = 512,
            img_bands : Optional[List[int]]=None,
            label_bands : Optional[List[int]]=None
            ) -> ImgPolygonAssociator:
        """
        Create a new dataset of GeoTiffs (images, labels, and associator) where each image is cut into a grid of images.

        Exactly one of source_data_dir or source_assoc should be set (i.e. not None).

        Args:
            target_data_dir (Union[str, Path]): path to data directory where the new dataset (images, labels, associator) will be created. If the directory does not exist it will be created.
            new_img_size (ImgSize): size of new images (side length or (rows, col)) for 'centered' and 'random' modes. Defaults to 512.
            img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
            label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).

        Returns:
            ImgPolygonAssociator: associator of new dataset in target_data_dir

        Warning:
            Currently only works if the source associator component files are in the standard locations determined by the source_data_dir arg.
        """

        return self._create_or_update_cut_every_img_to_grid(
                    create_or_update='create',
                    target_data_dir=target_data_dir,
                    new_img_size=new_img_size,
                    img_bands=img_bands,
                    label_bands=label_bands
        )


    def _update_cut_every_img_to_grid(self) -> None:
        """
        Update a dataset of GeoTiffs (images, labels, and associator) where each image is cut into a grid of images.

        Warning:
            Make sure this does exactly what you want when updating an existing data_dir (e.g. if new polygons have been addded to the source_data_dir that overlap with existing labels in the target_data_dir these labels will not be updated. This should be fixed!). It might be safer to just recut the source_data_dir.
        """

        required_cut_params = {
            'source_data_dir',
            'new_img_size',
            'img_bands',
            'label_bands'
        }

        if not required_cut_params <= set(self._update_from_source_dataset_dict.keys()):
            raise KeyError(f"The following cut params are missing: {set(self._update_from_source_dataset_dict.keys()) - required_cut_params}")

        self._create_or_update_cut_every_img_to_grid(
            create_or_update='update',
            source_data_dir=self._update_from_source_dataset_dict['source_data_dir'],
            new_img_size=self._update_from_source_dataset_dict['new_img_size'],
            img_bands=self._update_from_source_dataset_dict['img_bands'],
            label_bands=self._update_from_source_dataset_dict['label_bands']
        )


    def _create_or_update_cut_every_img_to_grid(
            self,
            create_or_update : str,
            source_data_dir : Union[str, Path] = None,
            target_data_dir : Union[str, Path] = None,
            new_img_size : ImgSize = 512,
            img_bands : Optional[List[int]]=None,
            label_bands : Optional[List[int]]=None
            ) -> Optional[ImgPolygonAssociator]:
        """
        Create a new dataset of GeoTiffs (images, labels, and associator) where each image is cut into a grid of images.

        Args:
            source_data_dir (Union[str, Path], optional): data directory (images, labels, associator) containing the GeoTiffs to be cut from.
            source_assoc (ImgPolygonAssociator, optional): associator of dataset containing the GeoTiffs to be cut from.
            target_data_dir (Union[str, Path]): path to data directory where the new dataset (images, labels, associator) will be created. If the directory does not exist it will be created.
            target_assoc (ImgPolygonAssociator, optional): associator of target dataset.
            new_img_size (ImgSize): size of new images (side length or (rows, col)) for 'centered' and 'random' modes. Defaults to 512.
            img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
            label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).

        Returns:
            ImgPolygonAssociator: associator of new dataset in target_data_dir
        """

        # TODO factor out, code same for iter over imgs
        if not create_or_update in {'create', 'update'}:
            raise ValueError(f"Unknown create_or_update arg {create_or_update}, should be one of 'create', 'update'.")

        if create_or_update == 'create':

            # Check args
            if source_data_dir is not None:
                raise ValueError(f"TODOTODO")
            if target_data_dir is None:
                raise ValueError("TODOTODO")

            source_assoc = self
            target_data_dir = Path(target_data_dir)

        elif create_or_update == 'update':

            # Check args
            if target_data_dir is not None:
                raise ValueError(f"TODOTODO")
            if source_data_dir is None:
                raise ValueError("TODOTODO")

            source_assoc = self.__class__.from_data_dir(source_data_dir)
            target_data_dir = self.images_dir.parent

        img2grid_cutter = ImgToGridCutter(
                                source_assoc=source_assoc,
                                target_images_dir=target_data_dir / 'images',
                                target_labels_dir=target_data_dir / 'labels',
                                new_img_size=new_img_size,
                                img_bands=img_bands,
                                label_bands=label_bands)
        always_true = AlwaysTrue()

        if create_or_update == 'update':
            target_data_dir = None # we needed the value for the ImgCutter above, but if it's not None and we're updating create_or_update_dataset_iter_over_imgs will complain

        target_assoc = self.create_or_update_dataset_iter_over_imgs(
                            create_or_update=create_or_update,
                            source_data_dir=source_data_dir,
                            target_data_dir=target_data_dir,
                            img_cutter=img2grid_cutter,
                            img_filter_predicate=always_true)

        # remember the cutting params.
        target_assoc._update_from_source_dataset_dict.update(
            {
                'update_method' : '_update_cut_every_img_to_grid',
                'source_data_dir' : source_assoc.images_dir.parent, # Assuming standard data directory format
                'new_img_size' :  new_img_size,
                'img_bands' : img_bands,
                'label_bands' : label_bands,
            }
        )
        target_assoc._params_dict['img_size'] = new_img_size
        target_assoc.save()

        if create_or_update == 'create':
            return target_assoc