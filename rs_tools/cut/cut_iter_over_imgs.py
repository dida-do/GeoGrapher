"""Mixin that implements a general-purpose higher order function to create or
update datasets of GeoTiffs from existing ones by iterating over images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import json

import pandas as pd
from geopandas import GeoDataFrame
from tqdm.auto import tqdm

from rs_tools.utils.utils import concat_gdfs

if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator

from rs_tools.cut.cut_base import DSCutterBase
from rs_tools.cut.img_filter_predicates import AlwaysTrue as AlwaysTrueImgs
from rs_tools.cut.img_filter_predicates import ImgFilterPredicate
from rs_tools.cut.single_img_cutter_base import SingleImgCutterBase
from rs_tools.global_constants import DATA_DIR_SUBDIRS, IMGS_DF_INDEX_NAME

logger = logging.getLogger(__name__)


class DSCutterIterOverImgs(DSCutterBase):
    """Dataset cutter that iterates over images"""

    img_cutter: SingleImgCutterBase
    img_filter_predicate: ImgFilterPredicate = AlwaysTrueImgs()
    cut_imgs: List[str] = Field(
        default_factory=list,
        description=
        "Names of cut images in source_data_dir. Usually not to be set by hand!"
    )

    def _create(self):
        """Create a new dataset. See create_or_update for more details."""
        return self.create_or_update()

    def _update_from_source(self) -> ImgPolygonAssociator:
        """Update target dataset."""
        return self.create_or_update()

    def create_or_update(self) -> ImgPolygonAssociator:
        """Higher order method to create or update a data set of GeoTiffs by
        iterating over images in the source dataset.

        Create or update a data set of GeoTiffs (images, labels, and associator)
        in target_data_dir from the data set of GeoTiffs in source_data_dir by
        iterating over the images in the source dataset/associator that have
        not been cut to images in the target_data_dir (i.e. all images if the
        target dataset doesn not exist yet), filtering the images
        using the img_filter_predicate, and then cutting using an img_cutter.

        Warning:
            Make sure this does exactly what you want when updating an existing data_dir (e.g. if new polygons have been addded to the source_data_dir that overlap with existing labels in the target_data_dir these labels will not be updated. This should be fixed!). It might be safer to just recut the source_data_dir.

        Args:
            create_or_update (str) : One of 'create' or 'update'.
        """

        # Remember information to determine for which images to generate new labels
        imgs_in_target_dataset_before_update = set(
            self.target_assoc.imgs_df.index)
        added_polygons = []  # updated as we iterate

        # dict to temporarily store information which will be appended to target_assoc's imgs_df after cutting
        new_imgs_dict = {
            index_or_col_name: []
            for index_or_col_name in [IMGS_DF_INDEX_NAME] +
            list(self.source_assoc.imgs_df.columns)
        }

        self.target_assoc.add_to_polygons_df(self.source_assoc.polygons_df)

        # Get names of imgs in src that have not been cut
        imgs_in_src_that_have_been_cut = self.cut_imgs
        mask_imgs_in_src_that_have_not_been_cut = ~self.source_assoc.imgs_df.index.isin(
            imgs_in_src_that_have_been_cut)
        names_of_imgs_in_src_that_have_not_been_cut = self.source_assoc.imgs_df.loc[
            mask_imgs_in_src_that_have_not_been_cut].index

        # Iterate over all images in source dataset that have not been cut.
        for img_name in tqdm(names_of_imgs_in_src_that_have_not_been_cut,
                             desc='Cutting dataset: '):

            # If filter condition is satisfied, (if not, don't do anything) ...
            if self.img_filter_predicate(img_name,
                                         target_assoc=self.target_assoc,
                                         new_img_dict=new_imgs_dict,
                                         source_assoc=self.source_assoc):

                # ... cut the images (and their labels) and remember information to be appended to self.target_assoc imgs_df in return dict
                imgs_from_single_cut_dict = self.img_cutter(
                    img_name=img_name,
                    new_polygons_df=self.target_assoc.polygons_df,
                    new_graph=self.target_assoc._graph)

                # Make sure img_cutter returned dict with same keys as needed by new_imgs_dict.
                assert {
                    IMGS_DF_INDEX_NAME, 'geometry', 'orig_crs_epsg_code'
                } <= set(
                    imgs_from_single_cut_dict.keys()
                ), "dict returned by img_cutter needs the following keys: IMGS_DF_INDEX_NAME, 'geometry', 'orig_crs_epsg_code'."

                # Accumulate information for the new imgs in new_imgs_dict.
                for key in new_imgs_dict.keys():
                    new_imgs_dict[key] += (imgs_from_single_cut_dict[key])

                new_img_names = imgs_from_single_cut_dict[IMGS_DF_INDEX_NAME]
                img_bounding_rectangles = imgs_from_single_cut_dict['geometry']
                for new_img_name, img_bounding_rectangle in zip(
                        new_img_names, img_bounding_rectangles):

                    self.cut_imgs.append(img_name)

                    # Update graph and modify polygons_df in self.target_assoc
                    self.target_assoc._add_img_to_graph_modify_polygons_df(
                        img_name=new_img_name,
                        img_bounding_rectangle=img_bounding_rectangle)

        # Extract accumulated information about the imgs we've created in the target dataset into a dataframe...
        new_imgs_df = GeoDataFrame(new_imgs_dict,
                                   crs=self.target_assoc.imgs_df.crs)
        new_imgs_df.set_index(IMGS_DF_INDEX_NAME, inplace=True)

        # ... and append it to self.imgs_df.
        self.target_assoc.imgs_df = concat_gdfs(
            [self.target_assoc.imgs_df, new_imgs_df])

        # For those images that existed before the update and now intersect with newly added polygons ...
        imgs_w_new_polygons = [
            img_name for polygon_name in added_polygons for img_name in
            self.target_assoc.imgs_intersecting_polygon(polygon_name)
            if img_name in imgs_in_target_dataset_before_update
        ]
        # Delete the old labels (since they won't show the new polygons)...
        for img_name in imgs_w_new_polygons:
            label_path = self.target_assoc.labels_dir / img_name
            label_path.unlink(missing_ok=True)
        # ... and generate new ones.
        self.target_assoc.make_labels(img_names=imgs_w_new_polygons)

        # Finally, save associator and cutter to disk.
        self.target_assoc.save()
        self.save()

        return self.target_assoc
