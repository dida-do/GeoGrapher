"""Test compatibility of rasters, vectors, and graph."""

import logging

import pandas as pd

from geographer.connector import Connector
from geographer.graph.bipartite_graph_mixin import (
    RASTER_IMGS_COLOR,
    VECTOR_FEATURES_COLOR,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def check_graph_vertices_counts(connector: Connector):
    """Test connector invariant.

    Tests whether the set of vertices of the graph corresponds with the
    rasters and polygons in the connector and whether the number of
    edges leaving the polygon nodes corresponding to imagges fully
    containing the polygon are equal to the values in the vectors
    'raster_count' column.
    """
    raster_vertices_not_in_rasters = set(
        connector._graph.vertices(RASTER_IMGS_COLOR)
    ) - set(connector.rasters.index)
    rasters_in_rasters_not_in_graph = set(connector.rasters.index) - set(
        connector._graph.vertices(RASTER_IMGS_COLOR)
    )
    polygon_vertices_not_in_vectors = set(
        connector._graph.vertices(VECTOR_FEATURES_COLOR)
    ) - set(connector.vectors.index)
    polygons_in_vectors_not_in_graph = set(connector.vectors.index) - set(
        connector._graph.vertices(VECTOR_FEATURES_COLOR)
    )

    set_descriptions_and_differences = zip(
        [
            ("raster", "in the connector's graph not in rasters"),
            ("raster", "in the connector's rasters not in the graph"),
            ("polygon", "in the connector's graph not in vectors"),
            ("polygon", "in the connector's vectors not in graph"),
        ],
        [
            raster_vertices_not_in_rasters,
            rasters_in_rasters_not_in_graph,
            polygon_vertices_not_in_vectors,
            polygons_in_vectors_not_in_graph,
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

    # Now, check whether raster_count column agrees with results
    # of raster_containing_polygon for each polygon
    raster_count_edges = connector.vectors.apply(
        lambda row: len(connector.rasters_containing_vector(row.name)), axis=1
    )
    raster_count_edges.rename("raster_count_edges", inplace=True)

    counts_correct = (
        connector.vectors[connector.raster_count_col_name] == raster_count_edges
    )

    if not counts_correct.all():
        return_df = pd.concat(
            [connector.vectors[connector.raster_count_col_name], raster_count_edges],
            axis=1,
        )
        return_df = return_df.loc[~counts_correct]

        logger.error(
            "The raster_count doesn't match the number of fully containing rasters"
            "for the following vector features: "
        )
        logger.error(f"{return_df}")

    answer = answer and counts_correct.all()

    return answer
