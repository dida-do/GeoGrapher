"""Dataset cutter that iterates over vector features.

Implements a general-purpose higher order function to create or update
datasets of GeoTiffs from existing ones by iterating over vector
features.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

from geopandas import GeoDataFrame
from pydantic import Field
from tqdm.auto import tqdm

from geographer.connector import Connector
from geographer.creator_from_source_dataset_base import DSCreatorFromSourceWithBands
from geographer.cutters.feature_filter_predicates import (
    AlwaysTrue,
    FeatureFilterPredicate,
)
from geographer.cutters.img_selectors import ImgSelector
from geographer.cutters.single_img_cutter_base import SingleImgCutter
from geographer.global_constants import RASTER_IMGS_INDEX_NAME
from geographer.label_makers.label_maker_base import LabelMaker
from geographer.utils.utils import concat_gdfs, map_dict_values

logger = logging.getLogger(__name__)


class DSCutterIterOverFeatures(DSCreatorFromSourceWithBands):
    """Dataset cutter that iterates over vector features.

    Implements a general-purpose higher order function to create or
    update datasets of GeoTiffs from existing ones by iterating over
    vector features: Adds all features in the source dataset to the
    target dataset and iterate over all features in the target dataset.
    For each feature if the feature_filter_predicate is met uses the
    img_selector to select a subset of the images in the source dataset
    for which no images for this feature have previously been cut from.
    Each of the images is then cut using the img_cutter, and the new
    images are added to the target dataset/connector.
    """

    feature_filter_predicate: FeatureFilterPredicate = Field(
        default_factory=AlwaysTrue,
        title="Single image cutter",
        description="Filters vector features to be cut",
    )
    img_selector: ImgSelector = Field(
        title="Image selector",
        description="Selects images from source to cut for a given vector feature",
    )
    img_cutter: SingleImgCutter = Field(title="Single image cutter")
    label_maker: Optional[LabelMaker] = Field(
        default=None,
        title="Label maker",
        description="Optional label maker. If given, will be used to recompute labels\
            when necessary. Defaults to None",
    )
    cut_imgs: Dict[str, List[str]] = Field(
        default_factory=lambda: defaultdict(list),
        title="Cut images dictionary",
        description="Normally, should not be set by hand! Dict with vector features\
        as keys and lists of images cut for each vector feature as values",
    )

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._check_crs_agree()

    def cut(self):
        """Cut a dataset.

        Alternate name for the create method. See create_or_update for
        docstring.
        """
        return self.create()

    def _create(self) -> None:
        """Create a new dataset.

        See create_or_update for more details.
        """
        self._create_or_update()

    def _update(self) -> None:
        """Update target dataset.

        See create_or_update for more details.
        """
        self._create_or_update()

    def _after_creating_or_updating(self):
        self.save()

    def create_or_update(self) -> Connector:
        """Create or update target dataset.

        Returns:
            connector of target dataset
        """
        self._create_or_update()
        self.save()
        return self.target_connector

    def _create_or_update(self) -> None:
        """Higher order general purpose method to create or update a dataset of
        GeoTiffs by iterating over vector features.

        Warning:
            Assumes that the vector features in the target dataset are a subset
            of the vector features in the source dataset. Will break if the assumption
            is not met.

        Returns:
            connector of newly created or updated dataset
        """

        # Remember information to determine for which images to generate new labels
        imgs_in_target_dataset_before_update = set(
            self.target_connector.raster_imgs.index
        )
        added_features = []  # updated as we iterate

        # dict to temporarily store information which will be
        # appended to self.target_connector's raster_imgs after cutting
        new_imgs_dict = {
            index_or_col_name: []
            for index_or_col_name in [RASTER_IMGS_INDEX_NAME]
            + list(self.source_connector.raster_imgs.columns)
        }

        # Add vector features in source dataset missing from target dataset
        self._add_missing_vector_features_to_target()

        features_to_iterate_over = list(
            filter(
                lambda feature_name: self.feature_filter_predicate(
                    feature_name=feature_name,
                    target_connector=self.target_connector,
                    new_imgs_dict=new_imgs_dict,
                    source_connector=self.source_connector,
                ),
                self.target_connector.vector_features.index.tolist(),
            )
        )

        # For each feature ...
        for feature_name in tqdm(features_to_iterate_over, desc="Cutting dataset: "):

            # ... if we want to create new images for it ...
            if self.feature_filter_predicate(
                feature_name=feature_name,
                target_connector=self.target_connector,
                new_imgs_dict=new_imgs_dict,
                source_connector=self.source_connector,
            ):

                # ... remember it ...
                added_features += [feature_name]

                # ... and then from the images in the source dataset
                # containing the vector feature ...
                potential_source_images = (
                    self.source_connector.imgs_containing_vector_feature(feature_name)
                )
                # ... but from which an image for that vector feature
                # has not yet been cut ...
                potential_source_images = self._filter_out_previously_cut_imgs(
                    feature_name=feature_name,
                    src_imgs_containing_feature=set(potential_source_images),
                )

                # ... select the images we want to cut from.
                for img_name in self.img_selector(
                    feature_name=feature_name,
                    img_names_list=potential_source_images,
                    target_connector=self.target_connector,
                    new_imgs_dict=new_imgs_dict,
                    source_connector=self.source_connector,
                    cut_imgs=self.cut_imgs,
                ):

                    # Cut each image (and label) and remember the information to be
                    # appended to self.target_connector raster_imgs in return dict
                    imgs_from_single_cut_dict = self.img_cutter(
                        img_name=img_name,
                        feature_name=feature_name,
                        source_connector=self.source_connector,
                        target_connector=self.target_connector,
                        new_imgs_dict=new_imgs_dict,
                        bands=self.bands,
                    )

                    # Make sure img_cutter returned dict with same keys as needed
                    # by new_imgs_dict.
                    assert {
                        RASTER_IMGS_INDEX_NAME,
                        "geometry",
                        "orig_crs_epsg_code",
                    } <= set(
                        imgs_from_single_cut_dict.keys()
                    ), "Dict returned by img_cutter needs the following keys: IMGS_DF_INDEX_NAME, 'geometry', 'orig_crs_epsg_code'."

                    # Accumulate information for the new imgs in new_imgs_dict.
                    for key in new_imgs_dict.keys():
                        new_imgs_dict[key] += imgs_from_single_cut_dict[key]

                    new_img_names = imgs_from_single_cut_dict[RASTER_IMGS_INDEX_NAME]
                    img_bounding_rectangles = imgs_from_single_cut_dict["geometry"]
                    for new_img_name, img_bounding_rectangle in zip(
                        new_img_names, img_bounding_rectangles
                    ):

                        # Update graph and modify vector_features
                        # in self.target_connector
                        self.target_connector._add_img_to_graph_modify_vector_features(
                            img_name=new_img_name,
                            img_bounding_rectangle=img_bounding_rectangle,
                        )

                        # Update self.cut_imgs
                        for (
                            feature_name_
                        ) in self.target_connector.vector_features_contained_in_img(
                            new_img_name
                        ):
                            self.cut_imgs[feature_name_] += [img_name]

                    # In case the vector feature feature_name is not contained in any
                    # of the new_imgs:
                    if img_name not in self.cut_imgs[feature_name]:
                        self.cut_imgs[feature_name] += [img_name]

        # Extract accumulated information about the imgs we've created in the target
        # dataset into a dataframe...
        new_raster_imgs = GeoDataFrame(
            new_imgs_dict, crs=self.target_connector.raster_imgs.crs
        )
        new_raster_imgs.set_index(RASTER_IMGS_INDEX_NAME, inplace=True)

        # log warning if columns don't agree
        if (
            set(new_raster_imgs.columns)
            - set(self.target_connector.raster_imgs.columns)
            != set()
            or set(self.target_connector.raster_imgs.columns)
            - set(new_raster_imgs.columns)
            != set()
        ):
            logger.warning("columns of source and target datasets don't agree")

        # ... and append it to self.raster_imgs.
        self.target_connector.raster_imgs = concat_gdfs(
            [self.target_connector.raster_imgs, new_raster_imgs]
        )

        # For those images that existed before the update and now intersect with newly
        # added vector features ...
        imgs_w_new_features = [
            img_name
            for feature_name in added_features
            for img_name in self.target_connector.imgs_intersecting_vector_feature(
                feature_name
            )
            if img_name in imgs_in_target_dataset_before_update
        ]
        if self.label_maker is not None:
            # ... delete and recompute the labels.
            self.label_maker.recompute_labels(
                connector=self,
                img_names=imgs_w_new_features,
            )

        # Remove duplicates from cut_imgs lists
        self.cut_imgs = map_dict_values(remove_duplicates, self.cut_imgs)

        # Finally, save connector to disk.
        self.target_connector.save()

    def _filter_out_previously_cut_imgs(
        self, feature_name: Union[str, int], src_imgs_containing_feature: Set[str]
    ) -> List[str]:
        """Filter out source images from which cutouts containing a vector
        feature have already been created.

        Args:
            feature_name: name/id of vector feature
            src_imgs_containing_feature: set of images in source dataset
                containing the vector feature
            target_connector: connector of target dataset

        Returns:
            list of filtered rasters
        """

        src_imgs_previously_cut_for_this_feature = set(self.cut_imgs[feature_name])
        answer = list(
            src_imgs_containing_feature - src_imgs_previously_cut_for_this_feature
        )

        return answer

    def _check_crs_agree(self):
        """Simple safety check: make sure coordinate systems of source and
        target agree."""
        if self.source_connector.crs_epsg_code != self.target_connector.crs_epsg_code:
            raise ValueError(
                "Coordinate systems of source and target connectors do not agree"
            )


def remove_duplicates(from_list: list) -> list:
    """Remove duplicates from list."""
    return list(set(from_list))
