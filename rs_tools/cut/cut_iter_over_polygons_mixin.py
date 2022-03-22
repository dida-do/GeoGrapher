"""
TODO: fmg count

Mix-in that implements a higher order general-purpose method to create
or update datasets of GeoTiffs from existing ones by iterating over polygons.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, DefaultDict, Dict, Hashable,
                    List, Optional, Set, Union)

import pandas as pd
from geopandas import GeoDataFrame
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator

from rs_tools.cut.img_selectors import ImgSelector
from rs_tools.cut.polygon_filter_predicates import (AlwaysTrue,
                                                    PolygonFilterPredicate)
from rs_tools.cut.single_img_cutter_base import SingleImgCutter
from rs_tools.utils.utils import map_dict_values

logger = logging.getLogger(__name__)


class CreateDSCutIterOverPolygonsMixIn(object):

    def create_or_update_dataset_iter_over_polygons(
        self,
        create_or_update: str,
        img_cutter: SingleImgCutter,
        img_selector: ImgSelector,
        source_data_dir: Optional[Union[str, Path]] = None,
        target_data_dir: Optional[Union[str, Path]] = None,
        polygon_filter_predicate: PolygonFilterPredicate = AlwaysTrue()
    ) -> ImgPolygonAssociator:
        """Higher order general purpose method to create or update a dataset of
        GeoTiffs by iterating over polygons.

        Add all polygons in the source dataset to the target dataset and
        iterate over all polygons in the target dataset. For each polygon if
        the polygon_filter_predicate is met use the img_selector to select a
        subset of the images in the source dataset for which no images for
        this polygon have previously been cut from. Cut each image using the
        img_cutter, and add the new images to the target dataset/associator.

        Args:
            create_or_update (str) : One of 'create' or 'update'.
            source_data_dir (Union[str, Path]): data directory (images, labels,
                associator) containing the GeoTiffs to be cut from.
                Use only if updating a dataset.
            target_data_dir (Union[str, Path]): data directory of target
                dataset. Use only if creating a dataset.
            img_cutter (SingleImgCutter): single image cutter used to cut
                selected images in the source dataset.
            img_selector (ImgSelector): image selector.
            polygon_filter_predicate (PolygonFilterPredicate, optional):
                predicate to filter polygons. Defaults to AlwaysTrue().

        Returns:
            ImgPolygonAssociator: associator of newly created or updated dataset
        """

        # TODO factor out, code same for iter over imgs
        if not create_or_update in {'create', 'update'}:
            raise ValueError(
                f"Unknown create_or_update arg {create_or_update}, should be one of 'create', 'update'."
            )

        if create_or_update == 'create':

            # Check args
            if source_data_dir is not None:
                raise ValueError(f"TODOTODO")
            if target_data_dir is None:
                raise ValueError("TODOTODO")

            # Create source_assoc
            source_assoc = self

            # Create target assoc, ...
            target_assoc = self.empty_assoc_same_format_as(target_data_dir)
            target_assoc._update_from_source_dataset_dict[
                'cut_imgs'] = defaultdict(list)

            # ..., image data dirs, ...
            for dir in target_assoc.image_data_dirs:
                dir.mkdir(parents=True, exist_ok=True)

            # ... and the associator dir.
            target_assoc.assoc_dir.mkdir(parents=True, exist_ok=True)

            # Make sure no associator files already exist.
            if list(target_assoc.assoc_dir.iterdir()) != []:
                raise Exception(
                    f"The assoc_dir in {target_assoc.assoc_dir} should be empty!"
                )

        elif create_or_update == 'update':

            # Check args
            if target_data_dir is not None:
                raise ValueError(f"TODOTODO")
            if source_data_dir is None:
                raise ValueError("TODOTODO")

            target_assoc = self
            source_assoc = self.__class__.from_data_dir(source_data_dir)

            target_assoc._update_from_source_dataset_dict[
                'cut_imgs'] = defaultdict(
                    list,
                    target_assoc._update_from_source_dataset_dict['cut_imgs'])

        # Remember information to determine for which images to generate new labels
        imgs_in_target_dataset_before_update = set(target_assoc.imgs_df.index)
        added_polygons = []  # updated as we iterate

        # dict to temporarily store information which will be appended to target_assoc's imgs_df after cutting
        new_imgs_dict = {
            index_or_col_name: []
            for index_or_col_name in [source_assoc.imgs_df.index.name] +
            list(source_assoc.imgs_df.columns)
        }

        # Add all polygons in source dataset to target dataset
        target_assoc.add_to_polygons_df(source_assoc.polygons_df)

        # For each polygon ...
        for polygon_name in tqdm(
                target_assoc.polygons_df.index,
                desc='Cutting dataset: '):  #!!!!!!!!! all_polygons?????????

            # ... if we want to create new images for it ...
            if polygon_filter_predicate(polygon_name=polygon_name,
                                        target_assoc=target_assoc,
                                        new_imgs_dict=new_imgs_dict,
                                        source_assoc=source_assoc):

                # ... remember it ...
                added_polygons += [polygon_name]

                # ... and then from the images in the source dataset that containing the polygon ...
                potential_source_images = source_assoc.imgs_containing_polygon(
                    polygon_name)
                # ... but from which an image for that polygon has not yet been cut ...
                potential_source_images = self._filter_out_previously_cut_imgs(
                    polygon_name=polygon_name,
                    src_imgs_containing_polygon=set(potential_source_images),
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
                    assert {
                        'img_name', 'geometry', 'orig_crs_epsg_code'
                    } <= set(
                        imgs_from_single_cut_dict.keys()
                    ), f"dict returned by img_cutter needs the following keys: 'img_name', 'geometry', 'orig_crs_epsg_code'."
                    # if not set(imgs_from_single_cut_dict.keys()) == set(target_assoc.imgs_df.columns) | {target_assoc.imgs_df.index.name}, f"dict returned by img_cutter doesn't contain the same keys as needed by new_imgs_dict!"

                    # Accumulate information for the new imgs in new_imgs_dict.
                    for key in new_imgs_dict.keys():
                        new_imgs_dict[key] += (imgs_from_single_cut_dict[key])

                    new_img_names = imgs_from_single_cut_dict[
                        source_assoc.imgs_df.index.name]
                    img_bounding_rectangles = imgs_from_single_cut_dict[
                        'geometry']
                    for new_img_name, img_bounding_rectangle in zip(
                            new_img_names, img_bounding_rectangles):

                        # Update graph and modify polygons_df in target_assoc
                        target_assoc._add_img_to_graph_modify_polygons_df(
                            img_name=new_img_name,
                            img_bounding_rectangle=img_bounding_rectangle)

                        # Update target_assoc._update_from_source_dataset_dict
                        for polygon_name_ in target_assoc.polygons_contained_in_img(
                                new_img_name):
                            target_assoc._update_from_source_dataset_dict[
                                'cut_imgs'][polygon_name_] += [img_name]

                    # In case the polygon polygon_name is not contained in any of the new_imgs:
                    if img_name not in target_assoc._update_from_source_dataset_dict[
                            'cut_imgs'][polygon_name]:
                        target_assoc._update_from_source_dataset_dict[
                            'cut_imgs'][polygon_name] += [img_name]

        # Extract accumulated information about the imgs we've created in the target dataset into a dataframe...
        new_imgs_df = GeoDataFrame(new_imgs_dict, crs=target_assoc.imgs_df.crs)
        new_imgs_df.set_index(target_assoc.imgs_df.index.name, inplace=True)

        # log warning if columns don't agree
        if set(new_imgs_df.columns) - set(target_assoc.imgs_df.columns) != set(
        ) or set(target_assoc.imgs_df.columns) - set(
                new_imgs_df.columns) != set():
            logger.warning("columns of source and target datasets don't agree")

        # ... and append it to self.imgs_df.
        data_frames_list = [target_assoc.imgs_df, new_imgs_df]
        target_assoc.imgs_df = GeoDataFrame(pd.concat(data_frames_list),
                                            crs=data_frames_list[0].crs)

        # For those images that existed before the update and now intersect with newly added polygons ...
        imgs_w_new_polygons = [
            img_name for polygon_name in added_polygons for img_name in
            target_assoc.imgs_intersecting_polygon(polygon_name)
            if img_name in imgs_in_target_dataset_before_update
        ]
        # Delete the old labels (since they won't show the new polygons)...
        for img_name in imgs_w_new_polygons:
            label_path = target_assoc.labels_dir / img_name
            label_path.unlink(missing_ok=True)
        # ... and generate new ones.
        target_assoc.make_labels(img_names=imgs_w_new_polygons)

        # Remove duplicates from cut_imgs lists
        target_assoc._update_from_source_dataset_dict[
            'cut_imgs'] = map_dict_values(
                remove_duplicates,
                target_assoc._update_from_source_dataset_dict['cut_imgs'])

        # Finally, save associator to disk.
        target_assoc.save()

        return target_assoc

    @staticmethod
    def _filter_out_previously_cut_imgs(
            polygon_name: Union[str,
                                int], src_imgs_containing_polygon: Set[str],
            target_assoc: ImgPolygonAssociator) -> List[str]:
        """Filter out source images from which cutouts containing a polygon
        have already been created.

        Args:
            polygon_name (Union[str, int]): name/id of polygon
            src_imgs_containing_polygon (Set[str]): set of images in source dataset containing the polygon
            target_assoc (ImgPolygonAssociator): associator of target dataset

        Returns:
            List[str]: [description]
        """

        src_imgs_previously_cut_for_this_polygon = set(
            target_assoc._update_from_source_dataset_dict['cut_imgs']
            [polygon_name])
        answer = list(src_imgs_containing_polygon -
                      src_imgs_previously_cut_for_this_polygon)

        return answer


def remove_duplicates(l: list) -> list:
    """Remove duplicates from list."""
    return list(set(l))
