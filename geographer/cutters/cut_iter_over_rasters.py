"""Dataset cutter that iterates over rasters.

Implements a general-purpose higher order function to create or update
datasets of GeoTiffs from existing ones by iterating over rasters.
"""

import logging
from typing import Optional

from geopandas import GeoDataFrame
from pydantic import Field
from tqdm.auto import tqdm

from geographer.connector import Connector
from geographer.creator_from_source_dataset_base import DSCreatorFromSourceWithBands
from geographer.cutters.raster_filter_predicates import AlwaysTrue as AlwaysTrueRasters
from geographer.cutters.raster_filter_predicates import RasterFilterPredicate
from geographer.cutters.single_raster_cutter_base import SingleRasterCutter
from geographer.global_constants import RASTER_IMGS_INDEX_NAME
from geographer.label_makers.label_maker_base import LabelMaker
from geographer.utils.utils import concat_gdfs

logger = logging.getLogger(__name__)


class DSCutterIterOverRasters(DSCreatorFromSourceWithBands):
    """Dataset cutter that iterates over rasters.

    Implements a general-purpose higher order function to create or
    update datasets of GeoTiffs from existing ones by iterating over
    rasters.
    """

    raster_cutter: SingleRasterCutter
    raster_filter_predicate: RasterFilterPredicate = AlwaysTrueRasters()
    label_maker: Optional[LabelMaker] = Field(
        default=None,
        title="Label maker",
        description="Optional label maker. If given, will be used to recompute labels\
            when necessary. Defaults to None",
    )
    cut_rasters: list[str] = Field(
        default_factory=list,
        description=(
            "Names of cut rasters in source_data_dir. Usually not to be set by hand!"
        ),
    )

    def __init__(self, **data) -> None:
        """Initialize DSCutterIterOverRasters."""
        super().__init__(**data)
        self._check_crs_agree()

    def cut(self) -> Connector:
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
        """Update target dataset."""
        self._create_or_update()

    def create_or_update(self) -> Connector:
        """Create or update target dataset.

        Returns:
            connector of target dataset
        """
        self._create_or_update()
        self._after_creating_or_updating()
        self.target_connector.save()
        return self.target_connector

    def _after_creating_or_updating(self):
        self.save()

    def _create_or_update(self) -> None:
        """Create/update dataset by iterating over rasters in source dataset.

        Higher order method to create or update a data set of GeoTiffs by
        iterating over rasters in the source dataset.

        Create or update a data set of GeoTiffs (rasters, labels, and connector)
        in target_data_dir from the data set of GeoTiffs in source_data_dir by
        iterating over the rasters in the source dataset/connector that have
        not been cut to rasters in the target_data_dir (i.e. all rasters if the
        target dataset doesn not exist yet), filtering the rasters
        using the raster_filter_predicate, and then cutting using an raster_cutter.

        Warning:
            Make sure this does exactly what you want when updating an existing
            data_dir (e.g. if new vector features have been addded to the
            source_data_dir that overlap with existing labels in the target_data_dir
            these labels will not be updated. This should be fixed!). It might be safer
            to just recut the source_data_dir.
        """
        # Remember information to determine for which rasters to generate new labels
        rasters_in_target_dataset_before_update = set(
            self.target_connector.rasters.index
        )
        added_vectors = set(self.source_connector.vectors.index) - set(
            self.target_connector.vectors.index
        )

        # dict to temporarily store information which will be appended
        # to target_connector's rasters after cutting
        new_rasters_dict = {
            index_or_col_name: []
            for index_or_col_name in [RASTER_IMGS_INDEX_NAME]
            + list(self.source_connector.rasters.columns)
        }

        # Add vector features in source dataset missing from target dataset
        self._add_missing_vectors_to_target()

        # Iterate over all rasters in source dataset
        for raster_name in tqdm(
            self.source_connector.rasters.index, desc="Cutting dataset: "
        ):
            # If filter condition is satisfied, (if not, don't do anything) ...
            if self.raster_filter_predicate(
                raster_name,
                target_connector=self.target_connector,
                new_raster_dict=new_rasters_dict,
                source_connector=self.source_connector,
                cut_rasters=self.cut_rasters,
            ):
                # ... cut the rasters (and their labels) and remember information
                # to be appended to self.target_connector rasters in return dict
                rasters_from_single_cut_dict = self.raster_cutter(
                    raster_name=raster_name,
                    source_connector=self.source_connector,
                    target_connector=self.target_connector,
                    new_rasters_dict=new_rasters_dict,
                    bands=self.bands,
                )

                # Make sure raster_cutter returned dict with same keys
                # as needed by new_rasters_dict.
                assert {
                    RASTER_IMGS_INDEX_NAME,
                    "geometry",
                    "orig_crs_epsg_code",
                } <= set(rasters_from_single_cut_dict.keys()), (
                    "dict returned by raster_cutter needs the following keys: "
                    "IMGS_DF_INDEX_NAME, 'geometry', 'orig_crs_epsg_code'."
                )

                # Accumulate information for the new rasters in new_rasters_dict.
                for key in new_rasters_dict.keys():
                    new_rasters_dict[key] += rasters_from_single_cut_dict[key]

                new_raster_names = rasters_from_single_cut_dict[RASTER_IMGS_INDEX_NAME]
                raster_bounding_rectangles = rasters_from_single_cut_dict["geometry"]
                for new_raster_name, raster_bounding_rectangle in zip(
                    new_raster_names, raster_bounding_rectangles
                ):
                    self.cut_rasters.append(raster_name)

                    # Update graph and modify vectors in self.target_connector
                    self.target_connector._add_raster_to_graph_modify_vectors(
                        raster_name=new_raster_name,
                        raster_bounding_rectangle=raster_bounding_rectangle,
                    )

        # Extract accumulated information about the rasters we've
        # created in the target dataset into a dataframe...
        new_rasters = GeoDataFrame(
            new_rasters_dict,
            crs=self.target_connector.rasters.crs,
            geometry="geometry",
        )
        new_rasters.set_index(RASTER_IMGS_INDEX_NAME, inplace=True)

        # ... and append it to self.rasters.
        self.target_connector.rasters = concat_gdfs(
            [self.target_connector.rasters, new_rasters]
        )

        # For those rasters that existed before the update
        # and now intersect with newly added vector features ...
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

    def _check_crs_agree(self):
        """Run safety check.

        Make sure coordinate systems of source and target agree.
        """
        if self.source_connector.crs_epsg_code != self.target_connector.crs_epsg_code:
            raise ValueError(
                "Coordinate systems of source and target connectors do not agree"
            )
