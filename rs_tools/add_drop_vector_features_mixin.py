import logging
from typing import Literal, Optional, Sequence, Union

import pandas as pd
from geopandas import GeoDataFrame
from rs_tools.label_makers.label_maker_base import LabelMaker
from rs_tools.utils.connector_utils import _check_df_cols_agree

from rs_tools.utils.utils import concat_gdfs, deepcopy_gdf

# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)


class AddDropVectorFeaturesMixIn(object):
    """Mix-in that implements methods to add and drop vector features or images."""

    def add_to_vector_features(
        self,
        new_vector_features: GeoDataFrame,
        label_maker: Optional[LabelMaker] = None,
        force_overwrite: bool = False,
    ):
        """Add (or overwrite) vector features in new_vector_features to the connector
        (i.e. append to the connector's vector_features) keeping track of which
        vector features are contained in which images.

        Args:
            new_vector_features (GeoDataFrame): GeoDataFrame of vector features conforming to the connector's vector_features format
            label_maker (LabelMaker, optional): If given generate new labels for images containing vector features that were added. Defaults to None.
            force_overwrite (bool): whether to overwrite existing rows for vector features, default is False
        """

        new_vector_features = deepcopy_gdf(
            new_vector_features)  #  don't modify argument
        new_vector_features['img_count'] = 0

        duplicates = new_vector_features[
            new_vector_features.index.duplicated()]
        if len(duplicates) > 0:
            raise ValueError(
                f"new_vector_features contains rows with duplicate vector_feature_names: {duplicates.index.tolist()}"
            )

        if len(new_vector_features[new_vector_features.geometry.isna()]) > 0:
            raise ValueError(
                f"new_vector_features contains rows with None vector features: {', '.join(new_vector_features[new_vector_features.geometry.isna()].index)}"
            )

        new_vector_features = self._get_df_in_crs(
            df=new_vector_features,
            df_name='new_vector_features',
            crs_epsg_code=self.crs_epsg_code)
        self._check_required_df_cols_exist(df=new_vector_features,
                                           df_name='new_vector_features',
                                           mode='vector_features')
        _check_df_cols_agree(df=new_vector_features,
                             df_name='new_vector_features',
                             self_df=self.vector_features,
                             self_df_name='self.vector_features')
        self._check_classes_in_vector_features_contained_in_all_classes(
            new_vector_features, 'new_vector_features')

        # For each new feature...
        for vector_feature_name in new_vector_features.index:

            # ... if it already is in the connector ...
            if self._graph.exists_vertex(
                    vector_feature_name, 'vector_features'
            ):  # or: vector_feature_name in self.vector_features.index

                # ... if necessary. ...
                if force_overwrite == True:

                    # ... we overwrite the row in the connector's vector_features ...
                    self.vector_features.loc[
                        vector_feature_name] = new_vector_features.loc[
                            vector_feature_name].copy()

                    # ... and drop the row from new_vector_features, so it won't be in self.vector_features twice after we concatenate vector_features to self.vector_features. ...
                    new_vector_features.drop(vector_feature_name, inplace=True)

                    # Then, we recalculate the connections. ...
                    self._remove_vector_feature_from_graph_modify_vector_features(
                        vector_feature_name)
                    self._add_vector_feature_to_graph(vector_feature_name)

                # Else ...
                else:

                    # ... we drop the row from new_vector_features...
                    new_vector_features.drop(vector_feature_name, inplace=True)

                    log.info(
                        "integrate_new_vector_features: dropping row for %s from input vector_features since is already in the connector! (force_overwrite arg is set to %s)",
                        vector_feature_name, force_overwrite)

            # If it is not in the connector ...
            else:

                # ... add a vertex for the new feature to the graph and add all connections to existing images. ...
                self._add_vector_feature_to_graph(
                    vector_feature_name, vector_features=new_vector_features)

        # Finally, append new_vector_features to the connector's (self.)vector_features.
        self.vector_features = concat_gdfs(
            [self.vector_features, new_vector_features])
        #self.vector_features = self.vector_features.convert_dtypes()

        if label_maker is not None:
            # delete labels that need to change and recompute
            imgs_w_new_vector_features = [
                img_name for vector_feature_name in new_vector_features.index
                for img_name in self.imgs_intersecting_vector_feature(
                    vector_feature_name)
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
        """Drop vector features from connector (i.e. remove rows from the
        connector's vector_features)

        Args:
            vector_feature_names (Sequence[str]): vector_feature_names/identifiers of vector features to be dropped.
            label_maker (LabelMaker, optional): If given generate new labels for images containing vector features that were dropped. Defaults to None.
        """

        # make sure we don't interpret a string as a list of characters in the iteration below:
        if isinstance(vector_feature_names, (str, int)):
            vector_feature_names = [vector_feature_names]
        assert pd.api.types.is_list_like(vector_feature_names)

        names_of_imgs_with_labels_to_recompute = set()

        # remove the feature vertices (along with their edges)
        for vector_feature_name in vector_feature_names:
            names_of_imgs_with_labels_to_recompute.update(
                set(self.imgs_intersecting_vector_feature(
                    vector_feature_name)))
            self._graph.delete_vertex(vector_feature_name,
                                      'vector_features',
                                      force_delete_with_edges=True)

        # drop row from self.vector_features
        self.vector_features.drop(vector_feature_names, inplace=True)

        # recompute labels
        if label_maker is True:
            names_of_imgs_with_labels_to_recompute = list(
                names_of_imgs_with_labels_to_recompute)
            label_maker.delete_labels(
                connector=self,
                img_names=names_of_imgs_with_labels_to_recompute,
            )
            label_maker.make_labels(
                connector=self,
                img_names=names_of_imgs_with_labels_to_recompute,
            )