"""
REWRITE DOCSTRINGS

Mixin that implements creating or updating a dataset of GeoTiffs
by cutting images around polygons from a source dataset.
"""

from __future__ import annotations
from typing import Union, List, Optional, TYPE_CHECKING
from rs_tools.cut.type_aliases import ImgSize
import logging
from pathlib import Path

if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.cut.single_img_cutter_around_polygon import ImgsAroundPolygonCutter
from rs_tools.cut.polygon_filter_predicates import IsPolygonMissingImgs
from rs_tools.cut.img_selectors import RandomImgSelector


logger = logging.getLogger(__name__)


class CreateDSCutImgsAroundEveryPolygonMixIn(object):

    def cut_imgs_around_every_polygon(
            self,
            target_data_dir : Union[str, Path],
            new_img_size : Optional[ImgSize] = 512,
            min_new_img_size : Optional[ImgSize] = 64,
            scaling_factor : Union[None, float] = 1.2,
            target_img_count : int = 1,
            img_bands : Optional[List[int]]=None,
            label_bands : Optional[List[int]]=None,
            mode : str = 'random',
            random_seed : int = 10
            ) -> ImgPolygonAssociator:
        """
        TODO: rewrite!
        Create a dataset of GeoTiffs so that it contains (if possible) for each polygon in the target (or source) dataset a number target_img_count of images cut from images in the source dataset.

        Note:
            If a polygon is too large to be contained in a single target image grids of images (with the property that the images in each grid should jointly contain the polygon and such that the grid is the minimal grid satisfying this property) will be cut from the target dataset.

        Args:
            target_data_dir (Union[str, Path]): path to data directory where the new dataset (images, labels, associator) will be created. If the directory does not exist it will be created.
            mode (str, optional): One of 'random', 'centered', or 'variable'. If 'random' images (or minimal image grids) will be randomly chose subject to constraint that they fully contain the polygons, if 'centered' will be centered on the polygons. If 'variable', the images will be centered but of variable size determined by the scaling_factor and min_new_img_size arguments. Defaults to 'random'.
            new_img_size (Optional[ImgSize]): size of new images (side length or (rows, col)) for 'centered' and 'random' modes. Defaults to 512.
            min_new_img_size (Optional[ImgSize]): minimum size of new images (side length or (rows, col)) for 'variable' mode. Defaults to 64.
            scaling_factor (float): scaling factor for 'variable' mode. Defaults to 1.2.
            target_img_count (int): image count (number of images per polygon) in target data set to aim for
            img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
            label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
            random_seed (int, optional): random seed.

        Returns:
            ImgPolygonAssociator: associator of target dataset

        Warning:
            Currently only works if the source associator component files are in the standard locations determined by the source_data_dir arg.
        """

        return self._create_or_update_cut_imgs_around_every_polygon(
                    create_or_update='create',
                    target_data_dir=target_data_dir,
                    new_img_size=new_img_size,
                    min_new_img_size=min_new_img_size,
                    scaling_factor=scaling_factor,
                    target_img_count=target_img_count,
                    img_bands=img_bands,
                    label_bands=label_bands,
                    mode=mode,
                    random_seed=random_seed)


    def _update_cut_imgs_around_every_polygon(self) -> None:

        """
        Update a dataset of GeoTiffs created by cut_imgs_around_every_polygon.

        Adds polygons from source_data_dir not contained in data_dir to data_dir and then iterates over all polygons in data_dir that do not have an image and creates a cutout from source_data_dir for them if one exists.

        Warning:
            If the (new_)img_size of the images in data_dir is smaller than the size of an polygon in data_dir then that polygon will not have an image associated with it and so new images will be created for it from the source_data_dir!

        Warning:
            If new polygons have been addded to the source_data_dir that overlap with existing labels in the target_data_dir these labels will not be updated.

        Warning:
            Make sure this does exactly what you want when updating an existing data_dir. If in doubt it might be easier or safer to just recut the source_data_dir.
        """

        required_cut_params = {
            'mode',
            'source_data_dir',
            'new_img_size',
            'min_new_img_size',
            'scaling_factor',
            'target_img_count',
            'img_bands',
            'label_bands',
            'random_seed'
        }

        if not required_cut_params <= set(self._update_from_source_dataset_dict.keys()):
            raise KeyError(f"Can't update, the following cut params are missing: {set(self._update_from_source_dataset_dict.keys()) - required_cut_params}")

        source_data_dir = self._update_from_source_dataset_dict['source_data_dir']

        self._create_or_update_cut_imgs_around_every_polygon(
            create_or_update='update',
            source_data_dir=source_data_dir,
            new_img_size=self._update_from_source_dataset_dict['new_img_size'],
            min_new_img_size=self._update_from_source_dataset_dict['min_new_img_size'],
            scaling_factor=self._update_from_source_dataset_dict['scaling_factor'],
            target_img_count=self._update_from_source_dataset_dict['target_img_count'],
            img_bands=self._update_from_source_dataset_dict['img_bands'],
            label_bands=self._update_from_source_dataset_dict['label_bands'],
            mode=self._update_from_source_dataset_dict['mode']
        )


    def _create_or_update_cut_imgs_around_every_polygon(
            self,
            create_or_update : str,
            target_data_dir : Union[str, Path] = None,
            source_data_dir : Union[str, Path] = None,
            mode : str = 'random',
            new_img_size : Optional[ImgSize] = 512,
            min_new_img_size : Optional[ImgSize] = 64,
            scaling_factor : Union[None, float] = 1.2,
            target_img_count : int = 1,
            img_bands : Optional[List[int]]=None,
            label_bands : Optional[List[int]]=None,
            random_seed : int = 10
            ) -> Optional[ImgPolygonAssociator]:
        """
        Create or update a dataset of GeoTiffs so that it contains (if possible) for each polygon in the target (or source) dataset a number target_img_count of images cut from images in the source dataset.

        Note:
            Exactly one of the target_data_dir and target_assoc arguments needs to be set (i.e. not None).

        Note:
            If a polygon is too large to be contained in a single target image grids of images (with the property that the images in each grid should jointly contain the polygon and such that the grid is the minimal grid satisfying this property) will be cut from the target dataset.

        Args:
            create_or_update (str): Whether to create or update the target dataset.
            target_data_dir (Union[str, Path], optional): path to data directory where the new dataset (images, labels, associator) will be created. If the directory does not exist it will be created.
            mode (str, optional): One of 'random', 'centered', or 'variable'. If 'random' images (or minimal image grids) will be randomly chose subject to constraint that they fully contain the polygons, if 'centered' will be centered on the polygons. If 'variable', the images will be centered but of variable size determined by the scaling_factor and min_new_img_size arguments. Defaults to 'random'.
            new_img_size (Optional[ImgSize]): size of new images (side length or (rows, col)) for 'centered' and 'random' modes. Defaults to 512.
            min_new_img_size (Optional[ImgSize]): minimum size of new images (side length or (rows, col)) for 'variable' mode. Defaults to 64.
            scaling_factor (float): scaling factor for 'variable' mode. Defaults to 1.2.
            target_img_count (int): image count (number of images per polygon) in target data set to aim for
            img_bands (List[int], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
            label_bands (List[int], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
            random_seed (int, optional): random seed.

        Returns:
            ImgPolygonAssociator: associator of target dataset
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

        elif create_or_update == 'update':

            # Check args
            if target_data_dir is not None:
                raise ValueError(f"TODOTODO")
            else:
                target_data_dir = self.images_dir.parent
            if source_data_dir is None:
                raise ValueError("TODOTODO")

            source_assoc = self.__class__.from_data_dir(source_data_dir)

        # Create the polygon_filter_predicate, img_selector, and single img cutter ...
        is_polygon_missing_imgs = IsPolygonMissingImgs(target_img_count)
        random_img_selector = RandomImgSelector(target_img_count)
        small_imgs_around_polygons_cutter = ImgsAroundPolygonCutter(
                                                source_assoc=source_assoc,
                                                target_images_dir=Path(target_data_dir) / "images",
                                                target_labels_dir=Path(target_data_dir) / "labels",
                                                mode=mode,
                                                new_img_size=new_img_size,
                                                min_new_img_size=min_new_img_size,
                                                scaling_factor=scaling_factor,
                                                img_bands=img_bands,
                                                label_bands=label_bands,
                                                random_seed=random_seed)

        # clean this up
        if create_or_update == 'update':
            target_data_dir = None

        # ... cut the dataset and return the target associator.
        # Note that target_assoc is self if we are updating.
        target_assoc = self.create_or_update_dataset_iter_over_polygons(
                            create_or_update=create_or_update,
                            source_data_dir=source_data_dir,
                            target_data_dir=target_data_dir,
                            img_cutter=small_imgs_around_polygons_cutter,
                            img_selector=random_img_selector,
                            polygon_filter_predicate=is_polygon_missing_imgs)

        # Remember the cutting params.
        target_assoc._update_from_source_dataset_dict.update(
            {
                'update_method' : '_update_cut_imgs_around_every_polygon',
                'mode' : mode,
                'source_data_dir' : source_data_dir,
                'new_img_size' :  new_img_size,
                'min_new_img_size' : min_new_img_size,
                'scaling_factor' : scaling_factor,
                'target_img_count' : target_img_count,
                'img_bands' : img_bands,
                'label_bands' : label_bands,
                'random_seed' : random_seed,
            }
        )

        if mode != 'variable':
            target_assoc._params_dict['img_size'] = new_img_size
        target_assoc.save()

        if create_or_update == 'create':
            return target_assoc