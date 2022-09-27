"""Utility functions for label making code."""

from geopandas import GeoDataFrame

from geographer.utils.utils import deepcopy_gdf


def convert_vectors_soft_cat_to_cat(
    vectors: GeoDataFrame,
) -> GeoDataFrame:
    """Convert vector features geodataframe from soft to categorical.

    Take a vectors GeoDataFrame in soft-categorical format and return a
    copy converted to categorical format.
    """
    new_vectors = deepcopy_gdf(vectors)

    # make 'type' column
    new_vectors["type"] = (
        new_vectors[
            [col for col in new_vectors.columns if col[:15] == "prob_of_class_"]
        ]
        .idxmax(axis="columns")
        .apply(lambda x: x[15:])
    )

    # drop 'prob_of_class_[seg_class]' cols
    new_vectors.drop(
        [col for col in new_vectors.columns if col[:15] == "prob_of_class_"],
        axis="columns",
        inplace=True,
    )

    return new_vectors
