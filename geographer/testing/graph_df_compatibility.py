"""Functions useful for testing or debugging."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_graph_vertices_counts(assoc):
    """Test associator invariant.

    Tests whether the set of vertices of the graph corresponds with the
    images and polygons in the associator and whether the number of
    edges leaving the polygon nodes corresponding to imagges fully
    containing the polygon are equal to the values in the polygons_df
    'img_count' column.
    """

    img_vertices_not_in_raster_imgs = set(assoc._graph.vertices('imgs')) - set(
        assoc.raster_imgs.index)
    imgs_in_raster_imgs_not_in_graph = set(assoc.raster_imgs.index) - set(
        assoc._graph.vertices('imgs'))
    polygon_vertices_not_in_polygons_df = set(
        assoc._graph.vertices('polygons')) - set(assoc.polygons_df.index)
    polygons_in_polygons_df_not_in_graph = set(assoc.polygons_df.index) - set(
        assoc._graph.vertices('polygons'))

    set_descriptions_and_differences = \
        zip(
            [('image', 'in the associator\'s graph not in raster_imgs'),
                ('image', 'in the associator\'s raster_imgs not in the graph'),
                ('polygon', 'in the associator\'s graph not in polygons_df'),
                ('polygon', 'in the associator\'s polygons_df not in graph')], \
            [img_vertices_not_in_raster_imgs,
                imgs_in_raster_imgs_not_in_graph,
                polygon_vertices_not_in_polygons_df,
                polygons_in_polygons_df_not_in_graph])

    answer = True

    for set_description, set_difference in set_descriptions_and_differences:

        num_elements_in_difference = len(set_difference)

        if num_elements_in_difference != 0:

            answer = False

            are_or_is = 'are' if num_elements_in_difference > 1 else 'is'

            plural_s = "" if num_elements_in_difference == 1 else "s"

            logger.error(
                f"There {are_or_is} {num_elements_in_difference} {set_description[0]}{plural_s} {set_description[1]}: {set_difference}"
            )

    # Now, check whether img_count column agrees with results of img_containing_polygon for each polygon
    img_count_edges = assoc.polygons_df.apply(
        lambda row: len(assoc.imgs_containing_polygon(row.name)), axis=1)
    img_count_edges.rename("img_count_edges", inplace=True)

    counts_correct = assoc.polygons_df['img_count'] == img_count_edges

    if not counts_correct.all():

        return_df = pd.concat(
            [assoc.polygons_df['img_count'], img_count_edges], axis=1)
        return_df = return_df.loc[~counts_correct]

        logger.error(
            "The img_count doesn't match the number of fully containing images for the following polygons:"
        )
        logger.error(f"{return_df}")

    answer = answer and counts_correct.all()

    return answer
