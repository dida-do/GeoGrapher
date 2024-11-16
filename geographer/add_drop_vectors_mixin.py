"""Mixin that implements adding/dropping vector features."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import pandas as pd
from geopandas import GeoDataFrame

from geographer.graph.bipartite_graph_mixin import VECTOR_FEATURES_COLOR
from geographer.utils.connector_utils import _check_df_cols_agree
from geographer.utils.utils import concat_gdfs, deepcopy_gdf

if TYPE_CHECKING:
    from geographer.label_makers.label_maker_base import LabelMaker

# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)


class AddDropVectorsMixIn(object):
    """Mix-in that implements adding or dropping vector features."""

    def add_to_vectors(
        self,
        new_vectors: GeoDataFrame,
        label_maker: LabelMaker | None = None,
    ):
        """Add vector features to connector's ``vectors`` attribute.

        Add (or overwrite) vector features in new_vectors and
        update graph encoding intersection/containment relations.

        Args:
            new_vectors: GeoDataFrame of vector features conforming to the
                connector's vectors format
            label_maker: If given generate new labels for rasters containing vector
                features that were added. Defaults to None.
        """
        duplicates = new_vectors[new_vectors.index.duplicated()]
        if len(duplicates) > 0:
            raise ValueError(
                "new_vectors contains rows with duplicate "
                f"vector_names (indices): {duplicates.index.tolist()}"
            )

        vector_names_in_both = list(set(new_vectors.index) & set(self.vectors.index))
        if vector_names_in_both:
            vector_names_in_both_str = ", ".join(vector_names_in_both)
            raise ValueError(
                "conflict: already have entries for vector features "
                f"{vector_names_in_both_str}"
            )

        if len(new_vectors[new_vectors.geometry.isna()]) > 0:
            rows_w_none_geoms = ", ".join(
                new_vectors[new_vectors.geometry.isna()].index
            )
            raise ValueError(
                f"new_vectors contains rows with None vector features: "
                f"{rows_w_none_geoms}"
            )

        new_vectors = self._get_df_in_crs(
            df=new_vectors,
            df_name="new_vectors",
            crs_epsg_code=self.crs_epsg_code,
        )

        new_vectors = deepcopy_gdf(new_vectors)
        new_vectors[self.raster_count_col_name] = 0

        self._check_required_df_cols_exist(
            df=new_vectors,
            df_name="new_vectors",
            mode="vectors",
        )
        _check_df_cols_agree(
            df=new_vectors,
            df_name="new_vectors",
            self_df=self.vectors,
            self_df_name="self.vectors",
        )
        # TODO
        # self._check_classes_in_vectors_contained_in_all_classes(
        #     new_vectors, 'new_vectors')

        # For each new vector feature...
        for vector_name in new_vectors.index:
            # ... add a vertex for the new vector feature to the graph and add all
            # connections to existing rasters.
            self._add_vector_to_graph(vector_name, vectors=new_vectors)

        # Append new_vectors to the connector's (self.)vectors.
        self.vectors = concat_gdfs([self.vectors, new_vectors])
        # self.vectors = self.vectors.convert_dtypes()

        if label_maker is not None:
            # delete labels that need to change and recompute
            rasters_w_new_vectors = [
                raster_name
                for vector_name in new_vectors.index
                for raster_name in self.rasters_intersecting_vector(vector_name)
            ]
            label_maker.recompute_labels(
                connector=self,
                raster_names=rasters_w_new_vectors,
            )

    def drop_vectors(
        self,
        vector_names: Sequence[str | int],
        label_maker: LabelMaker | None = None,
    ):
        """Drop vector features from connector's ``vectors`` attribute.

        Drop vector features from connector's ``vectors`` attribute and
        update graph encoding intersection/containment relations.

        Args:
            vector_names: vector_names/identifiers of vector
                features to be dropped.
            label_maker: If given generate new labels for rasters containing
                vector features that were dropped. Defaults to None.
        """
        # make sure we don't interpret a string as a list of characters
        # in the iteration below:
        if isinstance(vector_names, (str, int)):
            vector_names = [vector_names]
        assert pd.api.types.is_list_like(vector_names)

        names_of_rasters_with_labels_to_recompute = set()

        # remove the vector feature vertices (along with their edges)
        for vector_name in vector_names:
            names_of_rasters_with_labels_to_recompute.update(
                set(self.rasters_intersecting_vector(vector_name))
            )
            self._graph.delete_vertex(
                vector_name, VECTOR_FEATURES_COLOR, force_delete_with_edges=True
            )

        # drop row from self.vectors
        self.vectors.drop(vector_names, inplace=True)

        # recompute labels
        if label_maker is True:
            names_of_rasters_with_labels_to_recompute = list(
                names_of_rasters_with_labels_to_recompute
            )
            label_maker.delete_labels(
                connector=self,
                raster_names=names_of_rasters_with_labels_to_recompute,
            )
            label_maker.make_labels(
                connector=self,
                raster_names=names_of_rasters_with_labels_to_recompute,
            )
