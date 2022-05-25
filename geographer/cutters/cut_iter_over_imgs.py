"""
label maker arg

Dataset cutter that iterates over images. Implements a general-purpose higher order function
to create or update datasets of GeoTiffs from existing ones by iterating over images.
"""

import logging
from typing import List, Optional
from pydantic import Field

from geopandas import GeoDataFrame
from tqdm.auto import tqdm
from geographer.creator_from_source_dataset_base import DSCreatorFromSourceWithBands
from geographer.connector import Connector
from geographer.label_makers.label_maker_base import LabelMaker

from geographer.utils.utils import concat_gdfs
from geographer.cutters.img_filter_predicates import AlwaysTrue as AlwaysTrueImgs
from geographer.cutters.img_filter_predicates import ImgFilterPredicate
from geographer.cutters.single_img_cutter_base import SingleImgCutter
from geographer.global_constants import RASTER_IMGS_INDEX_NAME

logger = logging.getLogger(__name__)


class DSCutterIterOverImgs(DSCreatorFromSourceWithBands):
    """
    Dataset cutter that iterates over images. Implements a general-purpose higher order function
    to create or update datasets of GeoTiffs from existing ones by iterating over images.
    """

    img_cutter: SingleImgCutter
    img_filter_predicate: ImgFilterPredicate = AlwaysTrueImgs()
    label_maker: Optional[LabelMaker] = Field(default=None, title="Label maker",
        description="Optional label maker. If given, will be used to recompute labels\
            when necessary. Defaults to None")
    cut_imgs: List[str] = Field(
        default_factory=list,
        description=
        "Names of cut images in source_data_dir. Usually not to be set by hand!"
    )

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._check_crs_agree()

    def cut(self):
        """
        Cut a dataset.

        Alternate name for the create method. See create_or_update for docstring.
        """
        return self.create()

    def _create(self):
        """Create a new dataset. See create_or_update for more details."""
        return self.create_or_update()

    def _update_from_source(self) -> Connector:
        """Update target dataset."""
        return self.create_or_update()

    def create_or_update(self) -> Connector:
        """Higher order method to create or update a data set of GeoTiffs by
        iterating over images in the source dataset.

        Create or update a data set of GeoTiffs (images, labels, and connector)
        in target_data_dir from the data set of GeoTiffs in source_data_dir by
        iterating over the images in the source dataset/connector that have
        not been cut to images in the target_data_dir (i.e. all images if the
        target dataset doesn not exist yet), filtering the images
        using the img_filter_predicate, and then cutting using an img_cutter.

        Warning:
            Make sure this does exactly what you want when updating an existing data_dir
            (e.g. if new vector features have been addded to the source_data_dir that overlap
            with existing labels in the target_data_dir these labels will not be updated.
            This should be fixed!). It might be safer to just recut the source_data_dir.
        """

        # Remember information to determine for which images to generate new labels
        imgs_in_target_dataset_before_update = set(
            self.target_connector.raster_imgs.index)
        added_features = set(self.source_connector.vector_features.index) - set(self.target_connector.vector_features.index)

        # dict to temporarily store information which will be appended
        # to target_connector's raster_imgs after cutting
        new_imgs_dict = {
            index_or_col_name: []
            for index_or_col_name in [RASTER_IMGS_INDEX_NAME] +
            list(self.source_connector.raster_imgs.columns)
        }

        self.target_connector.add_to_vector_features(self.source_connector.vector_features)

        # Iterate over all images in source dataset
        for img_name in tqdm(self.source_connector.raster_imgs.index,
                             desc='Cutting dataset: '):

            # If filter condition is satisfied, (if not, don't do anything) ...
            if self.img_filter_predicate(img_name,
                                         target_connector=self.target_connector,
                                         new_img_dict=new_imgs_dict,
                                         source_connector=self.source_connector,
                                         cut_imgs=self.cut_imgs):

                # ... cut the images (and their labels) and remember information to be appended
                # to self.target_connector raster_imgs in return dict
                imgs_from_single_cut_dict = self.img_cutter(
                    img_name=img_name,
                    source_connector=self.source_connector,
                    target_connector=self.target_connector,
                    new_imgs_dict=new_imgs_dict,
                    bands=self.bands,
                )

                # Make sure img_cutter returned dict with same keys as needed by new_imgs_dict.
                assert {
                    RASTER_IMGS_INDEX_NAME, 'geometry', 'orig_crs_epsg_code'
                } <= set(
                    imgs_from_single_cut_dict.keys()
                ), "dict returned by img_cutter needs the following keys: IMGS_DF_INDEX_NAME, 'geometry', 'orig_crs_epsg_code'."

                # Accumulate information for the new imgs in new_imgs_dict.
                for key in new_imgs_dict.keys():
                    new_imgs_dict[key] += (imgs_from_single_cut_dict[key])

                new_img_names = imgs_from_single_cut_dict[RASTER_IMGS_INDEX_NAME]
                img_bounding_rectangles = imgs_from_single_cut_dict['geometry']
                for new_img_name, img_bounding_rectangle in zip(
                        new_img_names, img_bounding_rectangles):

                    self.cut_imgs.append(img_name)

                    # Update graph and modify vector_features in self.target_connector
                    self.target_connector._add_img_to_graph_modify_vector_features(
                        img_name=new_img_name,
                        img_bounding_rectangle=img_bounding_rectangle)

        # Extract accumulated information about the imgs we've created in the target dataset into a dataframe...
        new_raster_imgs = GeoDataFrame(new_imgs_dict,
                                   crs=self.target_connector.raster_imgs.crs)
        new_raster_imgs.set_index(RASTER_IMGS_INDEX_NAME, inplace=True)

        # ... and append it to self.raster_imgs.
        self.target_connector.raster_imgs = concat_gdfs(
            [self.target_connector.raster_imgs, new_raster_imgs])

        # For those images that existed before the update and now intersect with newly added vector features ...
        imgs_w_new_features = [
            img_name for feature_name in added_features for img_name in
            self.target_connector.imgs_intersecting_vector_feature(feature_name)
            if img_name in imgs_in_target_dataset_before_update
        ]
        if self.label_maker is not None:
            # ... delete and recompute the labels.
            self.label_maker.recompute_labels(
                connector=self,
                img_names=imgs_w_new_features,
            )

        # Finally, save connector and cutter to disk.
        self.target_connector.save()
        self.save()

        return self.target_connector

    def _check_crs_agree(self):
        """Simple safety check: make sure coordinate systems of source and target agree"""
        if self.source_connector.crs_epsg_code != self.target_connector.crs_epsg_code:
            raise ValueError(
                "Coordinate systems of source and target connectors do not agree"
            )
