"""Utility functions for label making code."""

from geopandas import GeoDataFrame

from geographer.utils.utils import deepcopy_gdf


def convert_vector_features_soft_cat_to_cat(
    vector_features: GeoDataFrame,
) -> GeoDataFrame:
    """Convert vector features geodataframe from soft to categorical.

    Take a vector_features GeoDataFrame in soft-categorical format and
    return a copy converted to categorical format.
    """
    new_vector_features = deepcopy_gdf(vector_features)

    # make 'type' column
    new_vector_features["type"] = (
        new_vector_features[
            [col for col in new_vector_features.columns if col[:15] == "prob_of_class_"]
        ]
        .idxmax(axis="columns")
        .apply(lambda x: x[15:])
    )

    # drop 'prob_of_class_[seg_class]' cols
    new_vector_features.drop(
        [col for col in new_vector_features.columns if col[:15] == "prob_of_class_"],
        axis="columns",
        inplace=True,
    )

    return new_vector_features
