"""Mixin that implements adding/dropping rasters."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import pandas as pd
from geopandas import GeoDataFrame

from geographer.utils.connector_utils import _check_df_cols_agree
from geographer.utils.utils import concat_gdfs, deepcopy_gdf

if TYPE_CHECKING:
    from geographer.label_makers.label_maker_base import LabelMaker

log = logging.getLogger(__name__)


class AddDropRastersMixIn:
    """Mix-in that implements methods to add and drop rasters."""

    def add_to_rasters(
        self, new_rasters: GeoDataFrame, label_maker: LabelMaker | None = None
    ):
        """Add rasters to connector's ``rasters`` attribute.

        Adds the new_rasters to the connector's :ref:`rasters` keeping track of
        which (vector) geometries are contained in which rasters.

        Args:
            new_rasters: GeoDataFrame of raster information conforming to the
                connector's rasters format
            label_maker: If given generate labels for new rasters.
        """
        new_rasters = deepcopy_gdf(new_rasters)  # don't want to modify argument

        duplicates = new_rasters[new_rasters.index.duplicated()]
        if len(duplicates) > 0:
            raise ValueError(
                "new_rasters contains rows with duplicate raster_names: "
                f"{duplicates.index.tolist()}"
            )

        rasters_names_in_both = list(set(new_rasters.index) & set(self.rasters.index))
        if rasters_names_in_both:
            raster_names_in_both_str = ", ".join(rasters_names_in_both)
            raise ValueError(
                "conflict: already have entries for rasters "
                f"{raster_names_in_both_str}"
            )

        if new_rasters.geometry.isna().any():
            rasters_with_null_geoms: str = ", ".join(
                new_rasters[new_rasters.geometry.isna()].index
            )
            raise ValueError(
                "new_rasters contains rows with None geometries: "
                f"{rasters_with_null_geoms}"
            )

        self._check_required_df_cols_exist(
            df=new_rasters, df_name="new_rasters", mode="rasters"
        )
        new_rasters = self._get_df_in_crs(
            df=new_rasters,
            df_name="new_rasters",
            crs_epsg_code=self.crs_epsg_code,
        )
        _check_df_cols_agree(
            df=new_rasters,
            df_name="new_rasters",
            self_df=self.rasters,
            self_df_name="self.rasters",
        )

        # go through all new rasters...
        for raster_name in new_rasters.index:
            # add new raster vertex to the graph, add all connections
            # to existing rasters, and modify self.vectors 'raster_count' value
            raster_bounding_rectangle = new_rasters.loc[raster_name, "geometry"]
            self._add_raster_to_graph_modify_vectors(
                raster_name, raster_bounding_rectangle=raster_bounding_rectangle
            )

        # append new_rasters
        self.rasters = concat_gdfs([self.rasters, new_rasters])
        # self.rasters = self.rasters.convert_dtypes()

        if label_maker is not None:
            label_maker.make_labels(
                connector=self, raster_names=new_rasters.index.tolist()
            )

    def drop_rasters(
        self,
        raster_names: Sequence[str],
        remove_rasters_from_disk: bool = True,
        label_maker: LabelMaker | None = None,
    ):
        """Drop rasters from ``rasters`` and from dataset.

        Remove rows from the connector's rasters, delete the corresponding
        vertices in the graph, and delete the raster from disk (unless
        remove_rasters_from_disk is set to False).

        Args:
            raster_names: raster_names/ids of rasters to be dropped.
            remove_rasters_from_disk: If true, delete rasters and labels
                from disk (if they exist). Defaults to True.
            label_maker: If given, will use label_makers
                delete_labels method. Defaults to None.
        """
        # make sure we don't interpret a string as a list of characters
        # in the iteration below:
        if isinstance(raster_names, str):
            raster_names = [raster_names]
        assert pd.api.types.is_list_like(raster_names)

        # drop row from self.rasters
        self.rasters.drop(raster_names, inplace=True)

        # remove all vertices from graph and modify vectors if necessary
        for raster_name in raster_names:
            self._remove_raster_from_graph_modify_vectors(raster_name)

        # remove rasters and labels from disk
        if remove_rasters_from_disk:
            if label_maker is not None:
                label_maker.delete_labels(self, raster_names)
            for dir_ in self.raster_data_dirs:
                for raster_name in raster_names:
                    (dir_ / raster_name).unlink(missing_ok=True)
