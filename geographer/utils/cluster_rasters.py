"""Given a dataset and an optional list of rasters partition the rasters into
equivalence classes ('clusters') that need to be respected when generating the
train-validation split."""

import itertools
from functools import partial
from pathlib import Path
from typing import Any, List, Literal, Optional, Set, Tuple, Union

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from networkx import Graph
from shapely.geometry.polygon import Point, Polygon

from geographer import Connector
from geographer.utils.utils import deepcopy_gdf


def get_raster_clusters(
    connector: Union[Connector, Path, str],
    clusters_defined_by: Literal[
        'rasters_that_share_vector_features',
        'rasters_that_share_vector_features_or_overlap'],
    raster_names: Optional[List[str]] = None,
    preclustering_method: Optional[Literal[
        'x then y-axis', 'y then x-axis', 'x-axis',
        'y-axis']] = 'y then x-axis'  #TODO!!!!!!!!!!
) -> List[Set[str]]:
    """Return clusters of raster.

    Args:
        connector: connector or path or str to data dir containing connector
        clusters_defined_by: relation between rasters defining clusters
        raster_names: optional list of raster names
        preclustering_method (Optional[ str]): optional preclustering method to speed
            up clustering

    Returns:
        (names of rasters defining) clusters
    """

    allowed_clusters_defined_by_args = {
        'rasters_that_share_vector_features',
        'rasters_that_share_vector_features_or_overlap'
    }
    if clusters_defined_by not in allowed_clusters_defined_by_args:
        raise ValueError(
            f"Unknown clusters_defined_by arg: {clusters_defined_by}")

    if not isinstance(connector, Connector):
        connector = Connector.from_data_dir(connector)
    if raster_names is None:
        raster_names = connector.raster_imgs.index.tolist()

    if preclustering_method is None:

        preclusters = [set(raster_names)]
        singletons, non_singletons = [], preclusters

    elif preclustering_method in {'x-axis', 'y-axis'}:

        axis = preclustering_method[0]  # 'x' or 'y'
        geoms = _get_preclustering_geoms(connector=connector,
                                         img_names=raster_names)

        preclusters = _pre_cluster_along_axis(geoms, axis)
        singletons, non_singletons = _separate_non_singletons(preclusters)

    elif preclustering_method in {'x then y-axis', 'y then x-axis'}:

        first_axis = preclustering_method[0]
        second_axis = 'y' if first_axis == 'x' else 'x'

        # cluster along first axis
        geoms = _get_preclustering_geoms(connector=connector,
                                         img_names=raster_names)
        preclusters = _pre_cluster_along_axis(geoms, first_axis)

        # cluster along 2nd axis
        singletons, non_singletons = _refine_preclustering_along_second_axis(
            preclusters, second_axis, connector)

    else:
        raise ValueError(
            f'Unknown preclustering_method: {preclustering_method}')

    # build graph
    img_clusters = singletons
    for non_singleton in non_singletons:
        graph_of_non_singleton = _extract_graph_of_rasters(
            connector=connector,
            clusters_defined_by=clusters_defined_by,
            img_names=list(non_singleton))

        img_clusters += list(nx.connected_components(graph_of_non_singleton))

    return img_clusters


# TODO: rename to refine pre clustering?
def _refine_preclustering_along_second_axis(
        preclusters: List[Set[str]], second_axis: Literal['x', 'y'],
        connector: Connector) -> Tuple[List[Set[str]], List[Set[str]]]:
    """

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
                connector=connector, img_names=list(precluster))
            refined_precluster = _pre_cluster_along_axis(
                precluster_geoms, second_axis)

            preclusters_along_2nd_axis += refined_precluster

    additional_singletons, non_singletons = _separate_non_singletons(
        preclusters_along_2nd_axis)
    singletons += additional_singletons

    return singletons, non_singletons


def _get_preclustering_geoms(connector: Connector,
                             img_names: List[str]) -> GeoDataFrame:

    # image geoms
    raster_imgs = deepcopy_gdf(connector.raster_imgs[['geometry'
                                                      ]].loc[img_names])
    raster_imgs['name'] = raster_imgs.index
    raster_imgs['img_or_polygon'] = 'img'

    img_names_set = set(img_names)
    del img_names

    # determine polygons that overlap w several images
    polygons_overlapping_imgs = []
    for polygon_name in connector.vector_features.index:
        imgs_intersect_but_dont_contain_polygon = set(
            connector.imgs_intersecting_vector_feature(polygon_name)) - set(
                connector.imgs_containing_vector_feature(polygon_name))
        if len(
                imgs_intersect_but_dont_contain_polygon
        ) >= 2 and img_names_set & imgs_intersect_but_dont_contain_polygon != set(
        ):
            polygons_overlapping_imgs.append(polygon_name)

    # geoms for those polygons
    vector_features = deepcopy_gdf(
        connector.vector_features.loc[polygons_overlapping_imgs][['geometry']])
    vector_features['name'] = vector_features.index
    vector_features['img_or_polygon'] = 'polygon'

    # make sure there are no duplicate names
    assert set(vector_features['name']) & set(raster_imgs['name']) == set()

    # combine geoms
    geoms = GeoDataFrame(pd.concat([raster_imgs, vector_features]),
                         crs=raster_imgs.crs)

    # don't need ?
    geoms = deepcopy_gdf(geoms)

    # TODO: don't recompute the bounds when we cluster along 2 axes
    if not {'minx', 'miny', 'maxx', 'maxy'} <= set(geoms.columns):
        geoms.drop(columns=['minx', 'miny', 'maxx', 'maxy'],
                   errors='ignore',
                   inplace=True)
        geoms = GeoDataFrame(
            pd.concat([geoms, geoms.geometry.bounds], axis=1),  # column axis
            crs=geoms.crs)

    return geoms


def _separate_non_singletons(
        preclusters: List[Set[Any]]) -> Tuple[List[Set[Any]], List[Set[Any]]]:

    singletons, non_singletions = [], []
    for precluster in preclusters:
        if len(precluster) == 1:
            singletons.append(precluster)
        else:
            non_singletions.append(precluster)

    return singletons, non_singletions


# simpler version
def _pre_cluster_along_axis(geoms: GeoDataFrame,
                            axis: Literal['x', 'y']) -> List[Set[str]]:

    if axis not in {'x', 'y'}:
        raise ValueError(f"axis arg should be one of 'x', 'y'.")

    mins = geoms[['name', f'min{axis}', 'img_or_polygon']].copy()
    mins['type'] = 'min'
    mins = mins.rename(columns={f'min{axis}': 'value'})
    mins = mins.to_dict(orient='records')
    # mins = list(mins.itertuples())

    maxs = geoms[['name', f'max{axis}', 'img_or_polygon']].copy()
    maxs['type'] = 'max'
    maxs = maxs.rename(columns={f'max{axis}': 'value'})
    maxs = maxs.to_dict(orient='records')

    interval_endpoints = sorted(
        mins + maxs,
        key=lambda d: (d['value'], 0 if d['type'] == 'min' else 1)
    )  # tuple comparison ensures mins are smaller than maxes for the same values
    # (-> smaller clusters)

    img_clusters_along_axis = []

    while interval_endpoints != []:

        rightmost_endpoint = interval_endpoints.pop()
        assert rightmost_endpoint['type'] == 'max'

        current_cluster = set(
        )  # Should this be {interval_endpoints.pop()}? No, think I'm fine...
        entered_intervals_count = 1
        exited_intervals_count = 0

        while entered_intervals_count - exited_intervals_count > 0:
            next_smaller_endpoint = interval_endpoints.pop()
            if next_smaller_endpoint['img_or_polygon'] == 'img':
                current_cluster.add(next_smaller_endpoint['name'])

            if next_smaller_endpoint['type'] == 'max':
                entered_intervals_count += 1
            elif next_smaller_endpoint['type'] == 'min':
                exited_intervals_count += 1
            else:
                raise Exception('something is wrong')

        img_clusters_along_axis.append(current_cluster)

    return img_clusters_along_axis


def _extract_graph_of_rasters(
    connector: Connector,
    clusters_defined_by: str,
    img_names: List[str] = None,
) -> Graph:
    """Extract graph of images with edges determined by the clusters_defined_by
    arg."""

    img_graph = Graph()
    img_graph.add_nodes_from(img_names)

    # add edges to graph
    pairs_of_imgs = itertools.combinations(img_names, 2)
    are_connected = lambda s: partial(_are_connected_by_an_edge,
                                      clusters_defined_by=clusters_defined_by,
                                      connector=connector)(*s)
    pairs_of_connected_imgs = filter(are_connected, pairs_of_imgs)
    img_graph.add_edges_from(pairs_of_connected_imgs)

    return img_graph


def _are_connected_by_an_edge(img: str, another_img: str,
                              clusters_defined_by: str,
                              connector: Connector) -> bool:
    """Return True if there is an edge in the graph of images determined by the
    clusters_defined_by relation."""

    img_bbox = connector.raster_imgs.loc[img].geometry
    other_img_bbox = connector.raster_imgs.loc[another_img].geometry

    if clusters_defined_by == 'imgs_that_overlap':

        connected = img_bbox.intersects(other_img_bbox)

    elif clusters_defined_by == 'rasters_that_share_vector_features':

        vector_features_in_img = set(
            connector.vector_features_intersecting_img(img))
        vector_features_in_other_img = set(
            connector.vector_features_intersecting_img(another_img))

        connected = vector_features_in_img & vector_features_in_other_img != set(
        )

    elif clusters_defined_by == 'rasters_that_share_vector_features_or_overlap':

        connected_bc_imgs_overlap = _are_connected_by_an_edge(
            img, another_img, 'imgs_that_overlap', connector)
        connected_bc_of_shared_polygons = _are_connected_by_an_edge(
            img, another_img, 'rasters_that_share_vector_features', connector)

        connected = connected_bc_imgs_overlap or connected_bc_of_shared_polygons

    else:

        raise ValueError(
            f"Unknown clusters_defined_by arg: {clusters_defined_by}")

    return connected
