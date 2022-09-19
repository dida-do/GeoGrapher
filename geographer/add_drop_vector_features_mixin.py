"""Mixin that implements adding/dropping vector features."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

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


class AddDropVectorFeaturesMixIn(object):
    """Mix-in that implements adding or dropping vector features."""

    def add_to_vector_features(
        self,
        new_vector_features: GeoDataFrame,
        label_maker: Optional[LabelMaker] = None,
    ):
        """Add vector features to connector's ``vector_features`` attribute.

        Add (or overwrite) vector features in new_vector_features and
        update graph encoding intersection/containment relations.

        Args:
            new_vector_features: GeoDataFrame of vector features conforming to the
                connector's vector_features format
            label_maker: If given generate new labels for images containing vector
                features that were added. Defaults to None.
        """
        duplicates = new_vector_features[new_vector_features.index.duplicated()]
        if len(duplicates) > 0:
            raise ValueError(
                "new_vector_features contains rows with duplicate "
                f"vector_feature_names (indices): {duplicates.index.tolist()}"
            )

        feature_names_in_both = list(
            set(new_vector_features.index) & set(self.vector_features.index)
        )
        if feature_names_in_both:
            feature_names_in_both_str = ", ".join(feature_names_in_both)
            raise ValueError(
                "conflict: already have entries for vector features "
                f"{feature_names_in_both_str}"
            )

        if len(new_vector_features[new_vector_features.geometry.isna()]) > 0:
            rows_w_none_features = ", ".join(
                new_vector_features[new_vector_features.geometry.isna()].index
            )
            raise ValueError(
                f"new_vector_features contains rows with None vector features: "
                f"{rows_w_none_features}"
            )

        new_vector_features = self._get_df_in_crs(
            df=new_vector_features,
            df_name="new_vector_features",
            crs_epsg_code=self.crs_epsg_code,
        )

        new_vector_features = deepcopy_gdf(new_vector_features)
        new_vector_features[self.img_count_col_name] = 0

        self._check_required_df_cols_exist(
            df=new_vector_features,
            df_name="new_vector_features",
            mode="vector_features",
        )
        _check_df_cols_agree(
            df=new_vector_features,
            df_name="new_vector_features",
            self_df=self.vector_features,
            self_df_name="self.vector_features",
        )
        # TODO
        # self._check_classes_in_vector_features_contained_in_all_classes(
        #     new_vector_features, 'new_vector_features')

        # For each new feature...
        for vector_feature_name in new_vector_features.index:

            # ... add a vertex for the new feature to the graph and add all
            # connections to existing images.
            self._add_vector_feature_to_graph(
                vector_feature_name, vector_features=new_vector_features
            )

        # Append new_vector_features to the connector's (self.)vector_features.
        self.vector_features = concat_gdfs([self.vector_features, new_vector_features])
        # self.vector_features = self.vector_features.convert_dtypes()

        if label_maker is not None:
            # delete labels that need to change and recompute
            imgs_w_new_vector_features = [
                img_name
                for vector_feature_name in new_vector_features.index
                for img_name in self.imgs_intersecting_vector_feature(
                    vector_feature_name
                )
            ]
            label_maker.recompute_labels(
                connector=self,
                img_names=imgs_w_new_vector_features,
            )

    def drop_vector_features(
        self,
        vector_feature_names: Sequence[Union[str, int]],
        label_maker: Optional[LabelMaker] = None,
    ):
        """Drop vector features from connector's ``vector_features`` attribute.

        Drop vector features from connector's ``vector_features`` attribute and
        update graph encoding intersection/containment relations.

        Args:
            vector_feature_names: vector_feature_names/identifiers of vector
                features to be dropped.
            label_maker: If given generate new labels for images containing
                vector features that were dropped. Defaults to None.
        """
        # make sure we don't interpret a string as a list of characters
        # in the iteration below:
        if isinstance(vector_feature_names, (str, int)):
            vector_feature_names = [vector_feature_names]
        assert pd.api.types.is_list_like(vector_feature_names)

        names_of_imgs_with_labels_to_recompute = set()

        # remove the feature vertices (along with their edges)
        for vector_feature_name in vector_feature_names:
            names_of_imgs_with_labels_to_recompute.update(
                set(self.imgs_intersecting_vector_feature(vector_feature_name))
            )
            self._graph.delete_vertex(
                vector_feature_name, VECTOR_FEATURES_COLOR, force_delete_with_edges=True
            )

        # drop row from self.vector_features
        self.vector_features.drop(vector_feature_names, inplace=True)

        # recompute labels
        if label_maker is True:
            names_of_imgs_with_labels_to_recompute = list(
                names_of_imgs_with_labels_to_recompute
            )
            label_maker.delete_labels(
                connector=self,
                img_names=names_of_imgs_with_labels_to_recompute,
            )
            label_maker.make_labels(
                connector=self,
                img_names=names_of_imgs_with_labels_to_recompute,
            )
