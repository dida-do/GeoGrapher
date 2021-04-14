"""
Functions useful for testing or debugging.
"""

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def test_graph_vertices(assoc):
    """
    Test associator invariant. EXPLAIN INVARIANT!
    
    Tests whether the set of vertices of the graph corresponds with the images and polygons in the associator.
    """

    img_vertices_not_in_imgs_df = set(assoc.__graph__.vertices('imgs')) - set(assoc.imgs_df.index)
    imgs_in_imgs_df_not_in_graph = set(assoc.imgs_df.index) - set(assoc.__graph__.vertices('imgs'))
    polygon_vertices_not_in_polygons_df = set(assoc.__graph__.vertices('polygons')) - set(assoc.polygons_df.index)
    polygons_in_polygons_df_not_in_graph = set(assoc.polygons_df.index) - set(assoc.__graph__.vertices('polygons')) 
    
    set_descriptions_and_differences = \
        zip(
            [('image', 'in the associator\'s graph not in imgs_df'), 
                ('image', 'in the associator\'s imgs_df not in the graph'), 
                ('polygon', 'in the associator\'s graph not in polygons_df'), 
                ('polygon', 'in the associator\'s polygons_df not in graph')], \
            [img_vertices_not_in_imgs_df, 
                imgs_in_imgs_df_not_in_graph, 
                polygon_vertices_not_in_polygons_df, 
                polygons_in_polygons_df_not_in_graph])

    answer = True

    for set_description, set_difference in set_descriptions_and_differences:

        num_elements_in_difference = len(set_difference)

        if num_elements_in_difference != 0:

            answer = False

            are_or_is = 'are' if num_elements_in_difference > 1 else 'is'

            plural_s = "" if num_elements_in_difference == 1 else "s"

            logger.error(f"There {are_or_is} {num_elements_in_difference} {set_description[0]}{plural_s} {set_description[1]}: {set_difference}")

    return answer

