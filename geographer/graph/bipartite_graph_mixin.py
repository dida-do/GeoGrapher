"""Mix-in that implements the private methods used to manipulate a connector's
bipartite graph."""

import logging
from typing import List, Optional, Union

from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry

from geographer.graph import BipartiteGraph

log = logging.getLogger(__name__)

VECTOR_FEATURES_COLOR = 'vector_features'
RASTER_IMGS_COLOR = 'raster_imgs'


class BipartiteGraphMixIn:
    """Mix-in that implements the public and private interface to the internal
    bipartite graph."""

    def have_img_for_vector_feature(self, feature_name: str) -> bool:
        """Return whether there exists an image fully containing the vector
        feature.

        Args:
            feature_name: Name of vector feature

        Returns:
            `True` if there is an image in the dataset fully containing the vector
            feature, False otherwise.
        """

        return self.vector_features.loc[feature_name, 'img_count'] > 0

    def rectangle_bounding_img(self, img_name: str) -> BaseGeometry:
        """Return shapely geometry bounding an image.

        The geometry is with respect to the connector's (standard) :term:`crs`.

        Args:
            img_name: the img_name/identifier of the image

        Returns:
            shapely geometry giving the bounds of the image in the standard crs
            of the connector
        """

        return self.raster_imgs.loc[img_name, 'geometry']

    def vector_features_intersecting_img(
            self, img_name: Union[str, List[str]]) -> List[str]:
        """Return the vector features which intersect one or (any of) several
        images.

        Args:
            img_name: name/id of image or list names/ids

        Returns:
            list of feature_names/ids of all vector features in connector which
            have non-empty intersection with the raster(s)
        """

        if isinstance(img_name, str):
            img_names = [img_name]
        else:
            img_names = img_name

        feature_names = []

        for img_name in img_names:
            try:
                feature_names += self._graph.vertices_opposite(
                    vertex=img_name, vertex_color=RASTER_IMGS_COLOR)
            except KeyError:
                raise ValueError(f"Unknown image: {img_name}")

        return feature_names

    def imgs_intersecting_vector_feature(self,
                                         feature_name: Union[str, List[str]],
                                         mode: str = 'names') -> List[str]:
        """Return names or paths of all raster images which intersect one or
        (any of) several vector features.

        Args:
            feature_name: name/id (or list) of vector feature(s)
            mode: One of 'names' or 'paths'. In the former case the image names are
            returned in the latter case paths to the images. Defaults to 'names'.

        Returns:
            feature_names/identifiers of all vector features in connector with
            non-empty intersection with the image.
        """

        if isinstance(feature_name, str):
            feature_names = [feature_name]
        else:
            feature_names = feature_name

        img_names = []

        for feature_name in feature_names:
            try:
                img_names += self._graph.vertices_opposite(
                    vertex=feature_name, vertex_color=VECTOR_FEATURES_COLOR)
            except KeyError:
                raise ValueError(f"Unknown vector feature: {feature_name}")

        if mode == 'names':
            answer = img_names
        elif mode == 'paths':
            answer = [self.images_dir / img_name for img_name in img_names]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return answer

    def vector_features_contained_in_img(
            self, img_name: Union[str, List[str]]) -> List[str]:
        """Return vector features fully containing a given image (or any of
        several images).

        Args:
            img_name: name/id of image or list of names/ids of images

        Returns:
            feature_names/identifiers of all vector features in connector
            contained in the image(s).
        """

        if isinstance(img_name, str):
            img_names = [img_name]
        else:
            img_names = img_name

        feature_names = []

        for img_name in img_names:
            try:
                feature_names += self._graph.vertices_opposite(
                    vertex=img_name,
                    vertex_color=RASTER_IMGS_COLOR,
                    edge_data='contains')
            except KeyError:
                raise ValueError(f"Unknown image: {img_name}")

        return feature_names

    def imgs_containing_vector_feature(
        self,
        feature_name: Union[str, List[str]],
        mode: str = 'names',
    ) -> List[str]:
        """Return names or paths of all images in which a given vector feature
        (or any of several) is fully contained.

        Args:
            feature_name: name/id (or list of names) of vector feature(s)
            mode: One of 'names' or 'paths'. In the former case the image names are
            returned in the latter case paths to the images. Defaults to 'names'.

        Returns:
            img_names/identifiers of all images in connector containing
            the vector feature(s)
        """

        if not isinstance(feature_name, list):
            feature_names = [feature_name]
        else:
            feature_names = feature_name

        img_names = []

        for feature_name in feature_names:
            try:
                img_names = self._graph.vertices_opposite(
                    vertex=feature_name,
                    vertex_color=VECTOR_FEATURES_COLOR,
                    edge_data='contains')
            except KeyError:
                raise ValueError(f"Unknown vector feature: {feature_name}")

        if mode == 'names':
            answer = img_names
        elif mode == 'paths':
            answer = [self.images_dir / img_name for img_name in img_names]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return answer

    def does_img_contain_vector_feature(self, img_name: str,
                                        feature_name: str) -> bool:
        """Return whether a raster image fully contains a vector feature.

        Args:
            img_name: Name of image
            feature_name: name of vector feature

        Returns:
            True or False depending on whether the image contains the vector
            feature or not
        """

        return feature_name in self.vector_features_contained_in_img(img_name)

    def is_vector_feature_contained_in_img(self, feature_name: str,
                                           img_name: str) -> bool:
        """Return whether a vector feature is fully contained in a raster
        image.

        Args:
            img_name: Name of image
            feature_name: name of vector feature

        Returns:
            True or False depending on whether the vector feature contains
            the image or not
        """

        return self.does_img_contain_vector_feature(img_name, feature_name)

    def does_img_intersect_vector_feature(self, img_name: str,
                                          feature_name: str) -> bool:
        """Return whether a vector feature intersects a raster image.

        Args:
            img_name: Name of image
            feature_name: name of vector feature

        Returns:
            True or False depending on whether the image intersects the
            vector feature or not
        """

        return feature_name in self.vector_features_intersecting_img(img_name)

    def does_vector_feature_intersect_img(self, feature_name: str,
                                          img_name: str) -> bool:
        """Return whether a vector feature intersects a raster image.

        Args:
            img_name: Name of image
            feature_name: name of vector feature

        Returns:
            True or False depending on whether the vector feature intersects
            the image or not
        """

        return self.does_img_intersect_vector_feature(img_name, feature_name)

    def _connect_img_to_vector_feature(
            self,
            img_name: str,
            feature_name: str,
            contains_or_intersects: Optional[str] = None,
            vector_features: Optional[GeoDataFrame] = None,
            img_bounding_rectangle: Optional[BaseGeometry] = None,
            graph: Optional[BipartiteGraph] = None,
            do_safety_check: bool = True):
        """Connect an image to a vector feature in the graph.

        Remember (i.e. create a connection in the graph) whether the image
        fully contains or just has non-empty intersection with the vector
        feature, i.e. add an edge of the approriate type between the image
        and the vector feature.

        Args:
            img_name: Name of image to connect
            feature_name: Name of vector feature to connect
            contains_or_intersects: Optional connection criteria
            vector_features: Optional vector feature dataframe
            img_bounding_rectangle: vector feature decribing image footprint
            graph: optional bipartied graph
            ignore_safety_check: whether to check contains_or_intersects relation
        """

        if contains_or_intersects not in {'contains', 'intersects', None}:
            raise ValueError(
                f"contains_or_intersects should be one of 'contains' or 'intersects' or None, is {contains_or_intersects}"
            )

        # default vector_features
        if vector_features is None:
            vector_features = self.vector_features

        # default graph
        if graph is None:
            graph = self._graph

        # default img_bounding_rectangle
        if img_bounding_rectangle is None:
            img_bounding_rectangle = self.raster_imgs.loc[img_name, 'geometry']

        # get containment relation if not given
        if contains_or_intersects is None:

            feature_feature = vector_features.loc[feature_name, 'geometry']

            non_empty_intersection = feature_feature.intersects(
                img_bounding_rectangle)
            if not non_empty_intersection:
                log.info(
                    "_connect_img_to_feature: not connecting, since img  %s and vector feature %s do not overlap.",
                    img_name, feature_name)
            else:
                contains_or_intersects = 'contains' if img_bounding_rectangle.contains(
                    feature_feature) else 'intersects'

        elif do_safety_check:
            feature_feature = vector_features.loc[feature_name, 'geometry']
            assert img_bounding_rectangle.intersects(feature_feature)
            assert contains_or_intersects == 'contains' if img_bounding_rectangle.contains(
                feature_feature) else 'intersects'

        graph.add_edge(img_name, RASTER_IMGS_COLOR, feature_name,
                       contains_or_intersects)

        # if the vector feature is fully contained in the image
        # increment the image counter in self.vector_features
        if contains_or_intersects == 'contains':
            vector_features.loc[feature_name, 'img_count'] += 1

    def _add_vector_feature_to_graph(
            self,
            feature_name: str,
            vector_features: Optional[GeoDataFrame] = None):
        """Connects a vector feature to those images in self.raster_imgs with
        which it has non-empty intersection.

        Args:
            feature_name: name/id of vector feature to add
            vector_features: Defaults to None (i.e. self.vector_features).
        """

        # default vector_features
        if vector_features is None:
            vector_features = self.vector_features

        # add vertex if one does not yet exist
        if not self._graph.exists_vertex(feature_name, VECTOR_FEATURES_COLOR):
            self._graph.add_vertex(feature_name, VECTOR_FEATURES_COLOR)

        # raise an exception if the vector feature already has connections
        if list(
                self._graph.vertices_opposite(feature_name,
                                              VECTOR_FEATURES_COLOR)):
            log.warning(
                "_add_feature_to_graph: !!!Warning (connect_feature): vector feature %s already has connections! Probably _add_feature_to_graph is being used wrongly. Check your code!",
                feature_name)

        feature_feature = vector_features.geometry.loc[feature_name]

        # determine intersecting and containing imgs
        intersection_mask = self.raster_imgs.geometry.intersects(
            feature_feature)
        containment_mask = self.raster_imgs.loc[
            intersection_mask].geometry.contains(feature_feature)

        intersecting_imgs = set(self.raster_imgs.loc[intersection_mask].index)
        containing_imgs = set(self.raster_imgs.loc[intersection_mask].
                              loc[containment_mask].index)

        # add edges in graph
        for img_name in containing_imgs:
            self._connect_img_to_vector_feature(
                img_name,
                feature_name,
                'contains',
                vector_features=vector_features,
                do_safety_check=False)
        for img_name in intersecting_imgs - containing_imgs:
            self._connect_img_to_vector_feature(
                img_name,
                feature_name,
                'intersects',
                vector_features=vector_features,
                do_safety_check=False)

    def _add_img_to_graph_modify_vector_features(
            self,
            img_name: str,
            img_bounding_rectangle: Optional[BaseGeometry] = None,
            vector_features: Optional[GeoDataFrame] = None,
            graph: Optional[BipartiteGraph] = None):
        """Create a vertex in the graph for the image if one does not yet exist
        and connect it to all vector features in the graph while modifying the
        img_counts in vector_features where appropriate. The default values
        None for vector_features and graph will be interpreted as
        self.vector_features and self.graph. If img_bounding_rectangle is None,
        we assume we can get it from self. If the image already exists and
        already has connections a warning will be logged.

        Args:
            img_name: Name of image to add
            img_bounding_rectangle: vector feature decribing image footprint
            vector_features: Optional vector features dataframe
            graph: optional bipartied graph
        """

        # default vector_features
        if vector_features is None:
            vector_features = self.vector_features

        # default graph:
        if graph is None:
            graph = self._graph

        # default img_bounding_rectangle
        if img_bounding_rectangle is None:
            img_bounding_rectangle = self.raster_imgs.geometry.loc[img_name]

        # add vertex if it does not yet exist
        if not graph.exists_vertex(img_name, RASTER_IMGS_COLOR):
            graph.add_vertex(img_name, RASTER_IMGS_COLOR)

        # check if img already has connections
        if list(graph.vertices_opposite(img_name, RASTER_IMGS_COLOR)) != []:
            log.warning(
                "!!!Warning (connect_img): image %s already has connections!",
                img_name)

        # # determine intersecting and containing imgs
        intersection_mask = self.vector_features.geometry.intersects(
            img_bounding_rectangle)
        containment_mask = self.vector_features.loc[
            intersection_mask].geometry.within(img_bounding_rectangle)

        intersecting_features = set(
            self.vector_features.loc[intersection_mask].index)
        contained_features = set(self.vector_features.loc[intersection_mask].
                                 loc[containment_mask].index)

        # add edges in graph
        for feature_name in contained_features:
            self._connect_img_to_vector_feature(
                img_name,
                feature_name,
                'contains',
                vector_features=vector_features,
                img_bounding_rectangle=img_bounding_rectangle,
                graph=graph,
                do_safety_check=False)
        for feature_name in intersecting_features - contained_features:
            self._connect_img_to_vector_feature(
                img_name,
                feature_name,
                'intersects',
                vector_features=vector_features,
                img_bounding_rectangle=img_bounding_rectangle,
                graph=graph,
                do_safety_check=False)

    def _remove_vector_feature_from_graph_modify_vector_features(
            self, feature_name: str, set_img_count_to_zero: bool = True):
        """Removes a vector feature from the graph (i.e. removes the vertex and
        all incident edges) and (if set_img_count_to_zero == True) sets the
        vector_features field 'img_count' to 0.

        Args:
            feature_name: vector feature name/id
            set_img_count_to_zero: Whether to set img_count to 0.
        """

        self._graph.delete_vertex(feature_name,
                                  VECTOR_FEATURES_COLOR,
                                  force_delete_with_edges=True)

        if set_img_count_to_zero == True:
            self.vector_features.loc[feature_name, 'img_count'] = 0

    def _remove_img_from_graph_modify_vector_features(self, img_name: str):
        """Removes an img from the graph (i.e. removes the vertex and all
        incident edges) and modifies the vector_features fields 'img_count' for
        the vector features contained in the image.

        Args:
            img_name: name/id of image to remove
        """

        for feature_name in self.vector_features_contained_in_img(img_name):
            self.vector_features.loc[feature_name, 'img_count'] -= 1

        self._graph.delete_vertex(img_name,
                                  RASTER_IMGS_COLOR,
                                  force_delete_with_edges=True)
