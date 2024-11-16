"""Dataset cutter that iterates over vector features.

Implements a general-purpose higher order function to create or update
datasets of GeoTiffs from existing ones by iterating over vector
features.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from geopandas import GeoDataFrame
from pydantic import Field
from tqdm.auto import tqdm

from geographer.connector import Connector
from geographer.creator_from_source_dataset_base import DSCreatorFromSourceWithBands
from geographer.cutters.raster_selectors import RasterSelector
from geographer.cutters.single_raster_cutter_base import SingleRasterCutter
from geographer.cutters.vector_filter_predicates import (
    AlwaysTrue,
    VectorFilterPredicate,
)
from geographer.global_constants import RASTER_IMGS_INDEX_NAME
from geographer.label_makers.label_maker_base import LabelMaker
from geographer.utils.utils import concat_gdfs, map_dict_values

logger = logging.getLogger(__name__)


class DSCutterIterOverVectors(DSCreatorFromSourceWithBands):
    """Dataset cutter that iterates over vector features.

    Implements a general-purpose higher order function to create or
    update datasets of GeoTiffs from existing ones by iterating over
    vector features: Adds all features in the source dataset to the
    target dataset and iterate over all features in the target dataset.
    For each vector feature if the vector_filter_predicate is met uses
    the raster_selector to select a subset of the rasters in the source
    dataset for which no rasters for this vector feature have previously
    been cut from. Each of the rasters is then cut using the
    raster_cutter, and the new rasters are added to the target
    dataset/connector.
    """

    vector_filter_predicate: VectorFilterPredicate = Field(
        default_factory=AlwaysTrue,
        title="Single raster cutter",
        description="Filters vector features to be cut",
    )
    raster_selector: RasterSelector = Field(
        title="Raster selector",
        description="Selects rasters from source to cut for a given vector feature",
    )
    raster_cutter: SingleRasterCutter = Field(title="Single raster cutter")
    label_maker: Optional[LabelMaker] = Field(
        default=None,
        title="Label maker",
        description="Optional label maker. If given, will be used to recompute labels\
            when necessary. Defaults to None",
    )
    cut_rasters: dict[str, list[str]] = Field(
        default_factory=lambda: defaultdict(list),
        title="Cut rasters dictionary",
        description="Normally, should not be set by hand! Dict with vector features\
        as keys and lists of rasters cut for each vector feature as values",
    )

    def __init__(self, **data) -> None:
        """Initialize DSCutterIterOverVectors."""
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
        """Create or update a dataset by iterating over vector features.

        Higher order general purpose method to create or update a dataset of
        GeoTiffs by iterating over vector features.

        Warning:
            Assumes that the vector features in the target dataset are a subset
            of the vector features in the source dataset. Will break if the assumption
            is not met.

        Returns:
            connector of newly created or updated dataset
        """
        # Remember information to determine for which rasters to generate new labels
        rasters_in_target_dataset_before_update = set(
            self.target_connector.rasters.index
        )
        added_vectors = []  # updated as we iterate

        # dict to temporarily store information which will be
        # appended to self.target_connector's rasters after cutting
        new_rasters_dict = {
            index_or_col_name: []
            for index_or_col_name in [RASTER_IMGS_INDEX_NAME]
            + list(self.source_connector.rasters.columns)
        }

        # Add vector features in source dataset missing from target dataset
        self._add_missing_vectors_to_target()

        vectors_to_iterate_over = list(
            filter(
                lambda vector_name: self.vector_filter_predicate(
                    vector_name=vector_name,
                    target_connector=self.target_connector,
                    new_rasters_dict=new_rasters_dict,
                    source_connector=self.source_connector,
                ),
                self.target_connector.vectors.index.tolist(),
            )
        )

        # For each vector feature ...
        for vector_name in tqdm(vectors_to_iterate_over, desc="Cutting dataset: "):
            # ... if we want to create new rasters for it ...
            if self.vector_filter_predicate(
                vector_name=vector_name,
                target_connector=self.target_connector,
                new_rasters_dict=new_rasters_dict,
                source_connector=self.source_connector,
            ):
                # ... remember it ...
                added_vectors += [vector_name]

                # ... and then from the rasters in the source dataset
                # containing the vector feature ...
                potential_source_rasters = (
                    self.source_connector.rasters_containing_vector(vector_name)
                )
                # ... but from which a raster for that vector feature
                # has not yet been cut ...
                potential_source_rasters = self._filter_out_previously_cut_rasters(
                    vector_name=vector_name,
                    src_rasters_containing_vector=set(potential_source_rasters),
                )

                # ... select the rasters we want to cut from.
                for raster_name in self.raster_selector(
                    vector_name=vector_name,
                    raster_names_list=potential_source_rasters,
                    target_connector=self.target_connector,
                    new_rasters_dict=new_rasters_dict,
                    source_connector=self.source_connector,
                    cut_rasters=self.cut_rasters,
                ):
                    # Cut each raster (and label) and remember the information to be
                    # appended to self.target_connector rasters in return dict
                    rasters_from_single_cut_dict = self.raster_cutter(
                        raster_name=raster_name,
                        vector_name=vector_name,
                        source_connector=self.source_connector,
                        target_connector=self.target_connector,
                        new_rasters_dict=new_rasters_dict,
                        bands=self.bands,
                    )

                    # Make sure raster_cutter returned dict with same keys as needed
                    # by new_rasters_dict.
                    assert {
                        RASTER_IMGS_INDEX_NAME,
                        "geometry",
                        "orig_crs_epsg_code",
                    } <= set(rasters_from_single_cut_dict.keys()), (
                        "Dict returned by raster_cutter needs the following keys: "
                        "IMGS_DF_INDEX_NAME, 'geometry', 'orig_crs_epsg_code'."
                    )

                    # Accumulate information for the new rasters in new_rasters_dict.
                    for key in new_rasters_dict.keys():
                        new_rasters_dict[key] += rasters_from_single_cut_dict[key]

                    new_raster_names = rasters_from_single_cut_dict[
                        RASTER_IMGS_INDEX_NAME
                    ]
                    raster_bounding_rectangles = rasters_from_single_cut_dict[
                        "geometry"
                    ]
                    for new_raster_name, raster_bounding_rectangle in zip(
                        new_raster_names, raster_bounding_rectangles
                    ):
                        # Update graph and modify vectors
                        # in self.target_connector
                        self.target_connector._add_raster_to_graph_modify_vectors(
                            raster_name=new_raster_name,
                            raster_bounding_rectangle=raster_bounding_rectangle,
                        )

                        # Update self.cut_rasters
                        for (
                            vector_name_
                        ) in self.target_connector.vectors_contained_in_raster(
                            new_raster_name
                        ):
                            self.cut_rasters[vector_name_] += [raster_name]

                    # In case the vector feature vector_name is not contained in any
                    # of the new_rasters:
                    if raster_name not in self.cut_rasters[vector_name]:
                        self.cut_rasters[vector_name] += [raster_name]

        # Extract accumulated information about the rasters we've created in the target
        # dataset into a dataframe...
        new_rasters = GeoDataFrame(
            new_rasters_dict,
            crs=self.target_connector.rasters.crs,
            geometry="geometry",
        )
        new_rasters.set_index(RASTER_IMGS_INDEX_NAME, inplace=True)

        # log warning if columns don't agree
        if (
            set(new_rasters.columns) - set(self.target_connector.rasters.columns)
            != set()
            or set(self.target_connector.rasters.columns) - set(new_rasters.columns)
            != set()
        ):
            logger.warning("columns of source and target datasets don't agree")

        # ... and append it to self.rasters.
        self.target_connector.rasters = concat_gdfs(
            [self.target_connector.rasters, new_rasters]
        )

        # For those rasters that existed before the update and now intersect with newly
        # added vector features ...
        rasters_w_new_vectors = [
            raster_name
            for vector_name in added_vectors
            for raster_name in self.target_connector.rasters_intersecting_vector(
                vector_name
            )
            if raster_name in rasters_in_target_dataset_before_update
        ]
        if self.label_maker is not None:
            # ... delete and recompute the labels.
            self.label_maker.recompute_labels(
                connector=self,
                raster_names=rasters_w_new_vectors,
            )

        # Remove duplicates from cut_rasters lists
        self.cut_rasters = map_dict_values(remove_duplicates, self.cut_rasters)

        # Finally, save connector to disk.
        self.target_connector.save()

    def _filter_out_previously_cut_rasters(
        self, vector_name: str | int, src_rasters_containing_vector: set[str]
    ) -> list[str]:
        """Filter out previously cut rasters.

        Filter out source rasters from which cutouts containing a vector
        feature have already been created.

        Args:
            vector_name: name/id of vector feature
            src_rasters_containing_vector: set of rasters in source dataset
                containing the vector feature
            target_connector: connector of target dataset

        Returns:
            list of filtered rasters
        """
        src_rasters_previously_cut_for_this_vector = set(self.cut_rasters[vector_name])
        answer = list(
            src_rasters_containing_vector - src_rasters_previously_cut_for_this_vector
        )

        return answer

    def _check_crs_agree(self):
        """Check crss agree.

        Check coordinate systems of source and target agree.
        """
        if self.source_connector.crs_epsg_code != self.target_connector.crs_epsg_code:
            raise ValueError(
                "Coordinate systems of source and target connectors do not agree"
            )


def remove_duplicates(from_list: list) -> list:
    """Remove duplicates from list."""
    return list(set(from_list))
