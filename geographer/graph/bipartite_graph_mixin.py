"""Mix-in for manipulating a connector's internal graph."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry

from geographer.graph.bipartite_graph_class import BipartiteGraphClass

log = logging.getLogger(__name__)

VECTOR_FEATURES_COLOR = "vectors"
RASTER_IMGS_COLOR = "rasters"


class BipartiteGraphMixIn:
    """Mix-in that interfaces with a connector's internal graph."""

    if TYPE_CHECKING:
        vectors: GeoDataFrame
        rasters: GeoDataFrame
        _graph: BipartiteGraphClass
        rasters_dir: Path

    def have_raster_for_vector(self, vector_name: str) -> bool:
        """Check if there is a raster fully containing the vector feature.

        Args:
            vector_name: Name of vector feature

        Returns:
            `True` if there is a raster in the dataset fully containing the vector
            feature, False otherwise.
        """
        return self.vectors.loc[vector_name, self.raster_count_col_name] > 0

    def rectangle_bounding_raster(self, raster_name: str) -> BaseGeometry:
        """Return shapely geometry bounding a raster.

        The geometry is with respect to the connector's (standard) :term:`crs`.

        Args:
            raster_name: the raster_name/identifier of the raster

        Returns:
            shapely geometry giving the bounds of the raster in the standard crs
            of the connector
        """
        return self.rasters.loc[raster_name, "geometry"]

    def vectors_intersecting_raster(
        self, raster_name: str | list[str]
    ) -> list[str]:
        """Return vector features intersecting one or (any of) several rasters.

        Args:
            raster_name: name/id of raster or list names/ids

        Returns:
            list of vector_names/ids of all vector features in connector which
            have non-empty intersection with the raster(s)
        """
        if isinstance(raster_name, str):
            raster_names = [raster_name]
        else:
            raster_names = raster_name

        vector_names = []

        for raster_name in raster_names:
            try:
                vector_names += self._graph.vertices_opposite(
                    vertex_name=raster_name, vertex_color=RASTER_IMGS_COLOR
                )
            except KeyError:
                raise ValueError(f"Unknown raster: {raster_name}")

        return vector_names

    def rasters_intersecting_vector(
        self,
        vector_name: str | list[str],
        mode: Literal["names", "paths"] = "names",
    ) -> list[str]:
        """Return rasters intersecting several vector feature(s).

        If more than one vector feature is given, return rasters intersecting
        at least one of the vector features.

        Args:
            vector_name: name/id (or list) of vector feature(s)
            mode: One of 'names' or 'paths'. In the former case the raster names are
            returned in the latter case paths to the rasters. Defaults to 'names'.

        Returns:
            vector_names/identifiers of all vector features in connector with
            non-empty intersection with the raster.
        """
        if isinstance(vector_name, str):
            vector_names = [vector_name]
        else:
            vector_names = vector_name

        raster_names = []

        for vector_name in vector_names:
            try:
                raster_names += self._graph.vertices_opposite(
                    vertex_name=vector_name, vertex_color=VECTOR_FEATURES_COLOR
                )
            except KeyError:
                raise ValueError(f"Unknown vector feature: {vector_name}")

        if mode == "names":
            answer = raster_names
        elif mode == "paths":
            answer = [self.rasters_dir / raster_name for raster_name in raster_names]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return answer

    def vectors_contained_in_raster(
        self, raster_name: str | list[str]
    ) -> list[str]:
        """Return vector features fully containing a given raster.

        If several rasters are given return vector features fully containing
        any of the rasters.

        Args:
            raster_name: name/id of raster or list of names/ids of rasters

        Returns:
            vector_names/identifiers of all vector features in connector
            contained in the raster(s).
        """
        if isinstance(raster_name, str):
            raster_names = [raster_name]
        else:
            raster_names = raster_name

        vector_names = []

        for raster_name in raster_names:
            try:
                vector_names += self._graph.vertices_opposite(
                    vertex_name=raster_name,
                    vertex_color=RASTER_IMGS_COLOR,
                    edge_data="contains",
                )
            except KeyError:
                raise ValueError(f"Unknown raster: {raster_name}")

        return vector_names

    def rasters_containing_vector(
        self,
        vector_name: str | list[str],
        mode: Literal["names", "paths"] = "names",
    ) -> list[str]:
        """Return rasters in which a given vector feature is fully contained.

        If multiple vector features are given, return raster which contain
        at least one of the vector features.

        Args:
            vector_name: name/id (or list of names) of vector feature(s)
            mode: One of 'names' or 'paths'. In the former case the raster names are
            returned in the latter case paths to the rasters. Defaults to 'names'.

        Returns:
            raster_names/identifiers of all rasters in connector containing
            the vector feature(s)
        """
        if not isinstance(vector_name, list):
            vector_names = [vector_name]
        else:
            vector_names = vector_name

        raster_names = []

        for vector_name in vector_names:
            try:
                raster_names = self._graph.vertices_opposite(
                    vertex_name=vector_name,
                    vertex_color=VECTOR_FEATURES_COLOR,
                    edge_data="contains",
                )
            except KeyError:
                raise ValueError(f"Unknown vector feature: {vector_name}")

        if mode == "names":
            answer = raster_names
        elif mode == "paths":
            answer = [self.rasters_dir / raster_name for raster_name in raster_names]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return answer

    def does_raster_contain_vector(self, raster_name: str, vector_name: str) -> bool:
        """Return whether a raster fully contains a vector feature.

        Args:
            raster_name: Name of raster
            vector_name: name of vector feature

        Returns:
            True or False depending on whether the raster contains the vector
            feature or not
        """
        return vector_name in self.vectors_contained_in_raster(raster_name)

    def is_vector_contained_in_raster(self, vector_name: str, raster_name: str) -> bool:
        """Return True if a vector feature is fully contained in a raster.

        Args:
            raster_name: Name of raster
            vector_name: name of vector feature

        Returns:
            True or False depending on whether the vector feature contains
            the raster or not
        """
        return self.does_raster_contain_vector(raster_name, vector_name)

    def does_raster_intersect_vector(self, raster_name: str, vector_name: str) -> bool:
        """Return whether a vector feature intersects a raster.

        Args:
            raster_name: Name of raster
            vector_name: name of vector feature

        Returns:
            True or False depending on whether the raster intersects the
            vector feature or not
        """
        return vector_name in self.vectors_intersecting_raster(raster_name)

    def does_vector_intersect_raster(self, vector_name: str, raster_name: str) -> bool:
        """Return whether a vector feature intersects a raster.

        Args:
            raster_name: Name of raster
            vector_name: name of vector feature

        Returns:
            True or False depending on whether the vector feature intersects
            the raster or not
        """
        return self.does_raster_intersect_vector(raster_name, vector_name)

    def _connect_raster_to_vector(
        self,
        raster_name: str,
        vector_name: str,
        contains_or_intersects: str | None = None,
        vectors: GeoDataFrame | None = None,
        raster_bounding_rectangle: BaseGeometry | None = None,
        graph: BipartiteGraphClass | None = None,
        do_safety_check: bool = True,
    ):
        """Connect a raster to a vector feature in the graph.

        Remember (i.e. create a connection in the graph) whether the raster
        fully contains or just has non-empty intersection with the vector
        feature, i.e. add an edge of the approriate type between the raster
        and the vector feature.

        Args:
            raster_name: Name of raster to connect
            vector_name: Name of vector feature to connect
            contains_or_intersects: Optional connection criteria
            vectors: Optional vector feature dataframe
            raster_bounding_rectangle: vector feature decribing raster footprint
            graph: optional bipartied graph
            ignore_safety_check: whether to check contains_or_intersects relation
        """
        if contains_or_intersects not in {"contains", "intersects", None}:
            raise ValueError(
                "contains_or_intersects should be one of 'contains' or 'intersects' "
                f"or None, is {contains_or_intersects}"
            )

        # default vectors
        if vectors is None:
            vectors = self.vectors

        # default graph
        if graph is None:
            graph = self._graph

        # default raster_bounding_rectangle
        if raster_bounding_rectangle is None:
            raster_bounding_rectangle = self.rasters.loc[raster_name, "geometry"]

        # get containment relation if not given
        if contains_or_intersects is None:
            vector_geom = vectors.loc[vector_name, "geometry"]

            non_empty_intersection = vector_geom.intersects(raster_bounding_rectangle)
            if not non_empty_intersection:
                log.info(
                    "_connect_raster_to_vector: not connecting, since "
                    "raster  %s and vector feature %s do not overlap.",
                    raster_name,
                    vector_name,
                )
            else:
                contains_or_intersects = (
                    "contains"
                    if raster_bounding_rectangle.contains(vector_geom)
                    else "intersects"
                )

        elif do_safety_check:
            vector_geom = vectors.loc[vector_name, "geometry"]
            assert raster_bounding_rectangle.intersects(vector_geom)
            assert (
                contains_or_intersects == "contains"
                if raster_bounding_rectangle.contains(vector_geom)
                else "intersects"
            )

        graph.add_edge(
            raster_name, RASTER_IMGS_COLOR, vector_name, contains_or_intersects
        )

        # if the vector feature is fully contained in the raster
        # increment the raster counter in self.vectors
        if contains_or_intersects == "contains":
            vectors.loc[vector_name, self.raster_count_col_name] += 1

    def _add_vector_to_graph(
        self, vector_name: str, vectors: GeoDataFrame | None = None
    ):
        """Connect a vector feature all intersecting rasters.

        Args:
            vector_name: name/id of vector feature to add
            vectors: Defaults to None (i.e. self.vectors).
        """
        # default vectors
        if vectors is None:
            vectors = self.vectors

        # add vertex if one does not yet exist
        if not self._graph.exists_vertex(vector_name, VECTOR_FEATURES_COLOR):
            self._graph.add_vertex(vector_name, VECTOR_FEATURES_COLOR)

        # raise an exception if the vector feature already has connections
        if list(self._graph.vertices_opposite(vector_name, VECTOR_FEATURES_COLOR)):
            log.warning(
                "_add_vector_to_graph: !!!Warning (connect_vector): "
                "vector feature %s already has connections! Probably "
                "_add_vector_to_graph is being used wrongly. Check your code!",
                vector_name,
            )

        vector_geom = vectors.geometry.loc[vector_name]

        # determine intersecting and containing rasters
        intersection_mask = self.rasters.geometry.intersects(vector_geom)
        containment_mask = self.rasters.loc[intersection_mask].geometry.contains(
            vector_geom
        )

        intersecting_rasters = set(self.rasters.loc[intersection_mask].index)
        containing_rasters = set(
            self.rasters.loc[intersection_mask].loc[containment_mask].index
        )

        # add edges in graph
        for raster_name in containing_rasters:
            self._connect_raster_to_vector(
                raster_name,
                vector_name,
                "contains",
                vectors=vectors,
                do_safety_check=False,
            )
        for raster_name in intersecting_rasters - containing_rasters:
            self._connect_raster_to_vector(
                raster_name,
                vector_name,
                "intersects",
                vectors=vectors,
                do_safety_check=False,
            )

    def _add_raster_to_graph_modify_vectors(
        self,
        raster_name: str,
        raster_bounding_rectangle: BaseGeometry | None = None,
        vectors: GeoDataFrame | None = None,
        graph: BipartiteGraphClass | None = None,
    ):
        """Add raster to graph and modify vector features.

        Create a vertex in the graph for the raster if one does not yet exist
        and connect it to all vector features in the graph while modifying the
        raster_counts in vectors where appropriate. The default values
        None for vectors and graph will be interpreted as
        self.vectors and self.graph. If raster_bounding_rectangle is None,
        we assume we can get it from self. If the raster already exists and
        already has connections a warning will be logged.

        Args:
            raster_name: Name of raster to add
            raster_bounding_rectangle: vector feature decribing raster footprint
            vectors: Optional vector features dataframe
            graph: optional bipartied graph
        """
        # default vectors
        if vectors is None:
            vectors = self.vectors

        # default graph:
        if graph is None:
            graph = self._graph

        # default raster_bounding_rectangle
        if raster_bounding_rectangle is None:
            raster_bounding_rectangle = self.rasters.geometry.loc[raster_name]

        # add vertex if it does not yet exist
        if not graph.exists_vertex(raster_name, RASTER_IMGS_COLOR):
            graph.add_vertex(raster_name, RASTER_IMGS_COLOR)

        # check if raster already has connections
        if list(graph.vertices_opposite(raster_name, RASTER_IMGS_COLOR)) != []:
            log.warning(
                "!!!Warning (connect_raster): raster %s already has connections!",
                raster_name,
            )

        # # determine intersecting and containing rasters
        intersection_mask = self.vectors.geometry.intersects(raster_bounding_rectangle)
        containment_mask = self.vectors.loc[intersection_mask].geometry.within(
            raster_bounding_rectangle
        )

        intersecting_vectors = set(self.vectors.loc[intersection_mask].index)
        contained_vectors = set(
            self.vectors.loc[intersection_mask].loc[containment_mask].index
        )

        # add edges in graph
        for vector_name in contained_vectors:
            self._connect_raster_to_vector(
                raster_name,
                vector_name,
                "contains",
                vectors=vectors,
                raster_bounding_rectangle=raster_bounding_rectangle,
                graph=graph,
                do_safety_check=False,
            )
        for vector_name in intersecting_vectors - contained_vectors:
            self._connect_raster_to_vector(
                raster_name,
                vector_name,
                "intersects",
                vectors=vectors,
                raster_bounding_rectangle=raster_bounding_rectangle,
                graph=graph,
                do_safety_check=False,
            )

    def _remove_vector_from_graph_modify_vectors(
        self, vector_name: str, set_raster_count_to_zero: bool = True
    ):
        """Remove vector feature from the graph & modify self.vectors.

        Removes a vector feature from the graph (i.e. removes the vertex and
        all incident edges) and (if set_raster_count_to_zero == True) sets the
        vectors field 'raster_count' to 0.

        Args:
            vector_name: vector feature name/id
            set_raster_count_to_zero: Whether to set raster_count to 0.
        """
        self._graph.delete_vertex(
            vector_name, VECTOR_FEATURES_COLOR, force_delete_with_edges=True
        )

        if set_raster_count_to_zero:
            self.vectors.loc[vector_name, self.raster_count_col_name] = 0

    def _remove_raster_from_graph_modify_vectors(self, raster_name: str):
        """Remove raster from graph & modify self.vectors accordingly.

        Remove an raster from the graph (i.e. remove the vertex and all
        incident edges) and modifiy the vectors fields 'raster_count' for
        the vector features contained in the raster.

        Args:
            raster_name: name/id of raster to remove
        """
        for vector_name in self.vectors_contained_in_raster(raster_name):
            self.vectors.loc[vector_name, self.raster_count_col_name] -= 1

        self._graph.delete_vertex(
            raster_name, RASTER_IMGS_COLOR, force_delete_with_edges=True
        )
