"""Test compatibility of raster_imgs, vector_features, and graph."""

import logging
from typing import TYPE_CHECKING

import pandas as pd

from geographer.graph.bipartite_graph_mixin import (
    RASTER_IMGS_COLOR,
    VECTOR_FEATURES_COLOR,
)

if TYPE_CHECKING:
    from geographer.connector import Connector

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def check_graph_vertices_counts(connector: Connector):
    """Test connector invariant.

    Tests whether the set of vertices of the graph corresponds with the
    images and polygons in the connector and whether the number of edges
    leaving the polygon nodes corresponding to imagges fully containing
    the polygon are equal to the values in the vector_features
    'img_count' column.
    """
    img_vertices_not_in_raster_imgs = set(
        connector._graph.vertices(RASTER_IMGS_COLOR)
    ) - set(connector.raster_imgs.index)
    imgs_in_raster_imgs_not_in_graph = set(connector.raster_imgs.index) - set(
        connector._graph.vertices(RASTER_IMGS_COLOR)
    )
    polygon_vertices_not_in_vector_features = set(
        connector._graph.vertices(VECTOR_FEATURES_COLOR)
    ) - set(connector.vector_features.index)
    polygons_in_vector_features_not_in_graph = set(
        connector.vector_features.index
    ) - set(connector._graph.vertices(VECTOR_FEATURES_COLOR))

    set_descriptions_and_differences = zip(
        [
            ("image", "in the connector's graph not in raster_imgs"),
            ("image", "in the connector's raster_imgs not in the graph"),
            ("polygon", "in the connector's graph not in vector_features"),
            ("polygon", "in the connector's vector_features not in graph"),
        ],
        [
            img_vertices_not_in_raster_imgs,
            imgs_in_raster_imgs_not_in_graph,
            polygon_vertices_not_in_vector_features,
            polygons_in_vector_features_not_in_graph,
        ],
    )

    answer = True

    for set_description, set_difference in set_descriptions_and_differences:

        num_elements_in_difference = len(set_difference)

        if num_elements_in_difference != 0:

            answer = False

            are_or_is = "are" if num_elements_in_difference > 1 else "is"

            plural_s = "" if num_elements_in_difference == 1 else "s"

            logger.error(
                f"There {are_or_is} {num_elements_in_difference} "
                f"{set_description[0]}{plural_s} {set_description[1]}: "
                f"{set_difference}"
            )

    # Now, check whether img_count column agrees with results
    # of img_containing_polygon for each polygon
    img_count_edges = connector.vector_features.apply(
        lambda row: len(connector.imgs_containing_vector_feature(row.name)), axis=1
    )
    img_count_edges.rename("img_count_edges", inplace=True)

    counts_correct = connector.vector_features["img_count"] == img_count_edges

    if not counts_correct.all():

        return_df = pd.concat(
            [connector.vector_features["img_count"], img_count_edges], axis=1
        )
        return_df = return_df.loc[~counts_correct]

        logger.error(
            "The img_count doesn't match the number of fully containing rasters"
            "for the following vector features: "
        )
        logger.error(f"{return_df}")

    answer = answer and counts_correct.all()

    return answer
