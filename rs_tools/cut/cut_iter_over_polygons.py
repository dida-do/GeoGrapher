"""
Dataset cutter that iterates over polygons.

Implements a general-purpose higher order function to create or update
datasets of GeoTiffs from existing ones by iterating over polygons.
"""

import logging
from collections import defaultdict
from typing import (Dict, List, Set, Union)

from geopandas import GeoDataFrame
from pydantic import Field
from tqdm.auto import tqdm

from rs_tools import ImgPolygonAssociator
from rs_tools.creator_from_source_dataset_base import DSCreatorFromSourceWithBands
from rs_tools.global_constants import IMGS_DF_INDEX_NAME
from rs_tools.cut.img_selectors import ImgSelector
from rs_tools.cut.polygon_filter_predicates import (AlwaysTrue,
                                                    PolygonFilterPredicate)
from rs_tools.cut.single_img_cutter_base import SingleImgCutter
from rs_tools.utils.utils import concat_gdfs, map_dict_values

logger = logging.getLogger(__name__)


class DSCutterIterOverPolygons(DSCreatorFromSourceWithBands):
    """
    Dataset cutter that iterates over polygons.

    Implements a general-purpose higher order function to create or update
    datasets of GeoTiffs from existing ones by iterating over polygons.
    """

    img_cutter: SingleImgCutter = Field(title="Single image cutter")
    img_selector: ImgSelector = Field(
        title="Image selector",
        description="Selects images from source to cut for a given polygon")
    polygon_filter_predicate: PolygonFilterPredicate = Field(
        default_factory=AlwaysTrue,
        title="Single image cutter",
        description="Filters polygons to be cut")
    cut_imgs: Dict[str, List[str]] = Field(
        default_factory=lambda: defaultdict(list),
        title="Cut images dictionary",
        description="Normally, should not be set by hand! Dict with polygons\
        as keys and lists of images cut for each polygon as values")

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._check_crs_agree()


    def _create(self) -> ImgPolygonAssociator:
        """Create a new dataset. See create_or_update for more details."""
        return self.create_or_update()

    def _update_from_source(self) -> ImgPolygonAssociator:
        """Update target dataset. See create_or_update for more details."""
        return self.create_or_update()

    def create_or_update(self) -> ImgPolygonAssociator:
        """Higher order general purpose method to create or update a dataset of
        GeoTiffs by iterating over polygons.

        Add all polygons in the source dataset to the target dataset and
        iterate over all polygons in the target dataset. For each polygon if
        the polygon_filter_predicate is met use the img_selector to select a
        subset of the images in the source dataset for which no images for
        this polygon have previously been cut from. Cut each image using the
        img_cutter, and add the new images to the target dataset/associator.

        Warning:
            Assumes that the polygons in the target dataset are a subset of the
            polygons in the source dataset. Will break if the assumption is not met.

        Returns:
            ImgPolygonAssociator: associator of newly created or updated dataset
        """

        # Remember information to determine for which images to generate new labels
        imgs_in_target_dataset_before_update = set(
            self.target_assoc.imgs_df.index)
        added_polygons = []  # updated as we iterate

        # dict to temporarily store information which will be
        # appended to self.target_assoc's imgs_df after cutting
        new_imgs_dict = {
            index_or_col_name: []
            for index_or_col_name in [IMGS_DF_INDEX_NAME] +
            list(self.source_assoc.imgs_df.columns)
        }

        # Add all polygons in source dataset to target dataset
        self.target_assoc.add_to_polygons_df(self.source_assoc.polygons_df)

        polygons_to_iterate_over = list(
            filter(
                lambda polygon_name: self.polygon_filter_predicate(
                    polygon_name=polygon_name,
                    target_assoc=self.target_assoc,
                    new_imgs_dict=new_imgs_dict,
                    source_assoc=self.source_assoc,
                ),
                self.target_assoc.polygons_df.index.tolist(),
            ))

        # For each polygon ...
        for polygon_name in tqdm(
                polygons_to_iterate_over,
                desc='Cutting dataset: '):  #!!!!!!!!! all_polygons?????????

            # ... if we want to create new images for it ...
            if self.polygon_filter_predicate(polygon_name=polygon_name,
                                             target_assoc=self.target_assoc,
                                             new_imgs_dict=new_imgs_dict,
                                             source_assoc=self.source_assoc):

                # ... remember it ...
                added_polygons += [polygon_name]

                # ... and then from the images in the source dataset that containing the polygon ...
                potential_source_images = self.source_assoc.imgs_containing_polygon(
                    polygon_name)
                # ... but from which an image for that polygon has not yet been cut ...
                potential_source_images = self._filter_out_previously_cut_imgs(
                    polygon_name=polygon_name,
                    src_imgs_containing_polygon=set(potential_source_images))

                # ... select the images we want to cut from.
                for img_name in self.img_selector(
                        polygon_name=polygon_name,
                        img_names_list=potential_source_images,
                        target_assoc=self.target_assoc,
                        new_imgs_dict=new_imgs_dict,
                        source_assoc=self.source_assoc):

                    # Cut each image (and label) and remember the information to be appended to self.target_assoc imgs_df in return dict
                    imgs_from_single_cut_dict = self.img_cutter(
                        img_name=img_name,
                        polygon_name=polygon_name,
                        source_assoc=self.source_assoc,
                        target_assoc=self.target_assoc,
                        new_imgs_dict=new_imgs_dict,
                        bands=self.bands,
                    )

                    # Make sure img_cutter returned dict with same keys as needed by new_imgs_dict.
                    assert {
                        IMGS_DF_INDEX_NAME, 'geometry', 'orig_crs_epsg_code'
                    } <= set(
                        imgs_from_single_cut_dict.keys()
                    ), "Dict returned by img_cutter needs the following keys: IMGS_DF_INDEX_NAME, 'geometry', 'orig_crs_epsg_code'."

                    # Accumulate information for the new imgs in new_imgs_dict.
                    for key in new_imgs_dict.keys():
                        new_imgs_dict[key] += (imgs_from_single_cut_dict[key])

                    new_img_names = imgs_from_single_cut_dict[
                        IMGS_DF_INDEX_NAME]
                    img_bounding_rectangles = imgs_from_single_cut_dict[
                        'geometry']
                    for new_img_name, img_bounding_rectangle in zip(
                            new_img_names, img_bounding_rectangles):

                        # Update graph and modify polygons_df in self.target_assoc
                        self.target_assoc._add_img_to_graph_modify_polygons_df(
                            img_name=new_img_name,
                            img_bounding_rectangle=img_bounding_rectangle)

                        # Update self.cut_imgs
                        for polygon_name_ in self.target_assoc.polygons_contained_in_img(
                                new_img_name):
                            self.cut_imgs[polygon_name_] += [img_name]

                    # In case the polygon polygon_name is not contained in any of the new_imgs:
                    if img_name not in self.cut_imgs[polygon_name]:
                        self.cut_imgs[polygon_name] += [img_name]

        # Extract accumulated information about the imgs we've created in the target dataset into a dataframe...
        new_imgs_df = GeoDataFrame(new_imgs_dict,
                                   crs=self.target_assoc.imgs_df.crs)
        new_imgs_df.set_index(IMGS_DF_INDEX_NAME, inplace=True)

        # log warning if columns don't agree
        if set(new_imgs_df.columns) - set(
                self.target_assoc.imgs_df.columns) != set() or set(
                    self.target_assoc.imgs_df.columns) - set(
                        new_imgs_df.columns) != set():
            logger.warning("columns of source and target datasets don't agree")

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

        # Remove duplicates from cut_imgs lists
        self.cut_imgs = map_dict_values(remove_duplicates, self.cut_imgs)

        # Finally, save associator to disk.
        self.target_assoc.save()

        return self.target_assoc

    def _filter_out_previously_cut_imgs(
            self, polygon_name: Union[str, int],
            src_imgs_containing_polygon: Set[str]) -> List[str]:
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
            self.cut_imgs[polygon_name])
        answer = list(src_imgs_containing_polygon -
                      src_imgs_previously_cut_for_this_polygon)

        return answer

    def _check_crs_agree(self):
    """Simple safety check: make sure coordinate systems of source and target agree"""
    if self.source_assoc.crs_epsg_code != self.target_assoc.crs_epsg_code:
        raise ValueError(
            "Coordinate systems of source and target associators do not agree"
        )


def remove_duplicates(from_list: list) -> list:
    """Remove duplicates from list."""
    return list(set(from_list))
