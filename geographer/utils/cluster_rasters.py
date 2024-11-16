"""Cluster rasters.

Given a dataset and an optional list of rasters partition the rasters
into equivalence classes ('clusters') that need to be respected when
generating the train-validation split.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Literal, Tuple

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from networkx import Graph

from geographer import Connector
from geographer.utils.utils import deepcopy_gdf


def get_raster_clusters(
    connector: Connector | Path | str,
    clusters_defined_by: Literal[
        "rasters_that_share_vectors",
        "rasters_that_share_vectors_or_overlap",
    ],
    raster_names: list[str] | None = None,
    preclustering_method: Literal["x then y-axis", "y then x-axis", "x-axis", "y-axis"] | None = "y then x-axis",  # TODO!!!!!!!!!!
) -> list[set[str]]:
    """Return clusters of raster.

    Args:
        connector: connector or path or str to data dir containing connector
        clusters_defined_by: relation between rasters defining clusters
        raster_names: optional list of raster names
        preclustering_method: optional preclustering method to speed
            up clustering

    Returns:
        (names of rasters defining) clusters
    """
    allowed_clusters_defined_by_args = {
        "rasters_that_share_vectors",
        "rasters_that_share_vectors_or_overlap",
    }
    if clusters_defined_by not in allowed_clusters_defined_by_args:
        raise ValueError(f"Unknown clusters_defined_by arg: {clusters_defined_by}")

    if not isinstance(connector, Connector):
        connector = Connector.from_data_dir(connector)
    if raster_names is None:
        raster_names = connector.rasters.index.tolist()

    if preclustering_method is None:
        preclusters = [set(raster_names)]
        singletons, non_singletons = [], preclusters

    elif preclustering_method in {"x-axis", "y-axis"}:
        axis = preclustering_method[0]  # 'x' or 'y'
        geoms = _get_preclustering_geoms(connector=connector, raster_names=raster_names)

        preclusters = _pre_cluster_along_axis(geoms, axis)
        singletons, non_singletons = _separate_non_singletons(preclusters)

    elif preclustering_method in {"x then y-axis", "y then x-axis"}:
        first_axis = preclustering_method[0]
        second_axis = "y" if first_axis == "x" else "x"

        # cluster along first axis
        geoms = _get_preclustering_geoms(connector=connector, raster_names=raster_names)
        preclusters = _pre_cluster_along_axis(geoms, first_axis)

        # cluster along 2nd axis
        singletons, non_singletons = _refine_preclustering_along_second_axis(
            preclusters, second_axis, connector
        )

    else:
        raise ValueError(f"Unknown preclustering_method: {preclustering_method}")

    # build graph
    raster_clusters = singletons
    for non_singleton in non_singletons:
        graph_of_non_singleton = _extract_graph_of_rasters(
            connector=connector,
            clusters_defined_by=clusters_defined_by,
            raster_names=list(non_singleton),
        )

        raster_clusters += list(nx.connected_components(graph_of_non_singleton))

    return raster_clusters


# TODO: rename to refine pre clustering?
def _refine_preclustering_along_second_axis(
    preclusters: list[set[str]], second_axis: Literal["x", "y"], connector: Connector
) -> Tuple[list[set[str]], list[set[str]]]:
    """Refine preclustering along the second axis.

    Args:
        preclusters: preclusters
        second_axis: name of second axis along which to refine pre-clustering

    Returns:
        singleton and non-singleton pre-clusters
    """
    singletons, preclusters_along_2nd_axis = [], []

    for precluster in preclusters:
        if len(precluster) == 1:
            singletons.append(precluster)

        else:
            precluster_geoms = _get_preclustering_geoms(
                connector=connector, raster_names=list(precluster)
            )
            refined_precluster = _pre_cluster_along_axis(precluster_geoms, second_axis)

            preclusters_along_2nd_axis += refined_precluster

    additional_singletons, non_singletons = _separate_non_singletons(
        preclusters_along_2nd_axis
    )
    singletons += additional_singletons

    return singletons, non_singletons


def _get_preclustering_geoms(
    connector: Connector, raster_names: list[str]
) -> GeoDataFrame:
    # raster geoms
    rasters = deepcopy_gdf(connector.rasters[["geometry"]].loc[raster_names])
    rasters["name"] = rasters.index
    rasters["raster_or_polygon"] = "raster"

    raster_names_set = set(raster_names)
    del raster_names

    # determine polygons that overlap w several rasters
    polygons_overlapping_rasters = []
    for polygon_name in connector.vectors.index:
        rasters_intersect_but_dont_contain_polygon = set(
            connector.rasters_intersecting_vector(polygon_name)
        ) - set(connector.rasters_containing_vector(polygon_name))
        if (
            len(rasters_intersect_but_dont_contain_polygon) >= 2
            and raster_names_set & rasters_intersect_but_dont_contain_polygon != set()
        ):
            polygons_overlapping_rasters.append(polygon_name)

    # geoms for those polygons
    vectors = deepcopy_gdf(
        connector.vectors.loc[polygons_overlapping_rasters][["geometry"]]
    )
    vectors["name"] = vectors.index
    vectors["raster_or_polygon"] = "polygon"

    # make sure there are no duplicate names
    assert set(vectors["name"]) & set(rasters["name"]) == set()

    # combine geoms
    geoms = GeoDataFrame(
        pd.concat([rasters, vectors]), crs=rasters.crs, geometry="geometry"
    )

    # don't need ?
    geoms = deepcopy_gdf(geoms)

    # TODO: don't recompute the bounds when we cluster along 2 axes
    if not {"minx", "miny", "maxx", "maxy"} <= set(geoms.columns):
        geoms.drop(
            columns=["minx", "miny", "maxx", "maxy"], errors="ignore", inplace=True
        )
        geoms = GeoDataFrame(
            pd.concat([geoms, geoms.geometry.bounds], axis=1),  # column axis
            crs=geoms.crs,
            geometry="geometry",
        )

    return geoms


def _separate_non_singletons(
    preclusters: list[set[Any]],
) -> tuple[list[set[Any]], list[set[Any]]]:
    singletons, non_singletions = [], []
    for precluster in preclusters:
        if len(precluster) == 1:
            singletons.append(precluster)
        else:
            non_singletions.append(precluster)

    return singletons, non_singletions


# simpler version
def _pre_cluster_along_axis(
    geoms: GeoDataFrame, axis: Literal["x", "y"]
) -> list[set[str]]:
    if axis not in {"x", "y"}:
        raise ValueError("axis arg should be one of 'x', 'y'.")

    mins = geoms[["name", f"min{axis}", "raster_or_polygon"]].copy()
    mins["type"] = "min"
    mins = mins.rename(columns={f"min{axis}": "value"})
    mins = mins.to_dict(orient="records")
    # mins = list(mins.itertuples())

    maxs = geoms[["name", f"max{axis}", "raster_or_polygon"]].copy()
    maxs["type"] = "max"
    maxs = maxs.rename(columns={f"max{axis}": "value"})
    maxs = maxs.to_dict(orient="records")

    interval_endpoints = sorted(
        mins + maxs, key=lambda d: (d["value"], 0 if d["type"] == "min" else 1)
    )  # tuple comparison ensures mins are smaller than maxes for the same values
    # (-> smaller clusters)

    raster_clusters_along_axis = []

    while interval_endpoints != []:
        rightmost_endpoint = interval_endpoints.pop()
        assert rightmost_endpoint["type"] == "max"

        current_cluster = (
            set()
        )  # Should this be {interval_endpoints.pop()}? No, think I'm fine...
        entered_intervals_count = 1
        exited_intervals_count = 0

        while entered_intervals_count - exited_intervals_count > 0:
            next_smaller_endpoint = interval_endpoints.pop()
            if next_smaller_endpoint["raster_or_polygon"] == "raster":
                current_cluster.add(next_smaller_endpoint["name"])

            if next_smaller_endpoint["type"] == "max":
                entered_intervals_count += 1
            elif next_smaller_endpoint["type"] == "min":
                exited_intervals_count += 1
            else:
                raise Exception("something is wrong")

        raster_clusters_along_axis.append(current_cluster)

    return raster_clusters_along_axis


def _extract_graph_of_rasters(
    connector: Connector,
    clusters_defined_by: str,
    raster_names: list[str] = None,
) -> Graph:
    """Extract graph of rasters determined by clusters_defined_by."""
    raster_graph = Graph()
    raster_graph.add_nodes_from(raster_names)

    # add edges to graph
    pairs_of_rasters = itertools.combinations(raster_names, 2)
    are_connected = lambda s: _are_connected_by_an_edge(  # noqa: E731
        *s,
        clusters_defined_by=clusters_defined_by,
        connector=connector,
    )
    pairs_of_connected_rasters = filter(are_connected, pairs_of_rasters)
    raster_graph.add_edges_from(pairs_of_connected_rasters)

    return raster_graph


def _are_connected_by_an_edge(
    raster: str,
    another_raster: str,
    clusters_defined_by: str,
    connector: Connector,
) -> bool:
    """Return True if rasters are connected, else False.

    Return True if there is an edge in the graph of rasters determined
    by the clusters_defined_by relation, else return False.
    """
    raster_bbox = connector.rasters.loc[raster].geometry
    other_raster_bbox = connector.rasters.loc[another_raster].geometry

    if clusters_defined_by == "rasters_that_overlap":
        connected = raster_bbox.intersects(other_raster_bbox)

    elif clusters_defined_by == "rasters_that_share_vectors":
        vectors_in_raster = set(connector.vectors_intersecting_raster(raster))
        vectors_in_other_raster = set(
            connector.vectors_intersecting_raster(another_raster)
        )

        connected = vectors_in_raster & vectors_in_other_raster != set()

    elif clusters_defined_by == "rasters_that_share_vectors_or_overlap":
        connected_bc_rasters_overlap = _are_connected_by_an_edge(
            raster, another_raster, "rasters_that_overlap", connector
        )
        connected_bc_of_shared_polygons = _are_connected_by_an_edge(
            raster, another_raster, "rasters_that_share_vectors", connector
        )

        connected = connected_bc_rasters_overlap or connected_bc_of_shared_polygons

    else:
        raise ValueError(f"Unknown clusters_defined_by arg: {clusters_defined_by}")

    return connected
