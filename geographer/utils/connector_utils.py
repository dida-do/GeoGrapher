"""Utilites used in the Connector class."""

import logging

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries

from geographer.global_constants import STANDARD_CRS_EPSG_CODE
from geographer.graph.bipartite_graph import BipartiteGraph, empty_bipartite_graph
from geographer.graph.bipartite_graph_mixin import (
    RASTER_IMGS_COLOR,
    VECTOR_FEATURES_COLOR,
)

log = logging.getLogger(__name__)


def empty_gdf(
    index_name: str,
    cols_and_types: dict,
    crs_epsg_code: int = STANDARD_CRS_EPSG_CODE,
):
    """Return an empty GeoDataFrame.

    Return an empty GeoDataFrame with specified index and column names and
    types and crs.

    Args:
        index_name: name of the index of the new empty GeoDataFrame
        cols_and_types: dict with keys the names of the index and columns of the
            GeoDataFrame and values the types of the indices/column entries.
        crs_epsg_code: EPSG code of the crs the empty GeoDataFrame should have.

    Returns:
        new_empty_df: the empty vectors_df GeoDataFrame.
    """
    new_empty_gdf_dict = {
        index_name: str,
        "geometry": GeoSeries([]),
        **{
            index_or_col_name: pd.Series([], dtype=index_or_col_type)
            for index_or_col_name, index_or_col_type in cols_and_types.items()
            if index_or_col_name != "geometry"
        },
    }

    new_empty_gdf = GeoDataFrame(
        new_empty_gdf_dict, crs=f"EPSG:{crs_epsg_code}", geometry="geometry"
    )
    new_empty_gdf.set_index(index_name, inplace=True)
    return new_empty_gdf


def empty_gdf_same_format_as(df: GeoDataFrame) -> GeoDataFrame:
    """Create an empty df of the same format as df.

    Create an empty df of the same format (index name, columns, column
    types) as the df argument.

    Args:
        df: input GeoDataFrame.

    Return:
        empty GeoDataFrame of same format as input.
    """
    df_index_name = df.index.name

    df_cols_and_index_types = {df.index.name: df.index.dtype, **df.dtypes.to_dict()}

    crs_epsg_code = df.crs.to_epsg()

    new_empty_df = empty_gdf(
        df_index_name, df_cols_and_index_types, crs_epsg_code=crs_epsg_code
    )

    return new_empty_df


def empty_vectors_same_format_as(vectors: GeoDataFrame) -> GeoDataFrame:
    """Create an empty vectors of the same format.

    Create an empty vectors of the same format (index name,
    columns, column types) as the vectors argument.

    Args:
        vectors: Example polygon dataframe

    Returns:
        New empty dataframe
    """
    return empty_gdf_same_format_as(vectors)


def empty_rasters_same_format_as(rasters: GeoDataFrame) -> GeoDataFrame:
    """Create an empty rasters of the same format.

    Create an empty rasters of the same format (index name, columns,
    column types) as the rasters argument.

    Args:
        rasters: Example rasters dataframe

    Returns:
        New empty rasters datagrame
    """
    return empty_gdf_same_format_as(rasters)


def empty_graph() -> BipartiteGraph:
    """Return an empty bipartite graph to be used by Connector.

    Returns:
        empty graph
    """
    return empty_bipartite_graph(red=VECTOR_FEATURES_COLOR, black=RASTER_IMGS_COLOR)


def _check_df_cols_agree(
    df: GeoDataFrame,
    df_name: str,
    self_df: GeoDataFrame,
    self_df_name: str,
):
    """Log if column names don't agree."""
    if set(df.columns) != set(self_df.columns) and len(self_df) > 0:
        df1_cols_not_in_df2 = set(df.columns) - set(self_df.columns)
        df2_cols_not_in_df1 = set(self_df.columns) - set(df.columns)

        if df1_cols_not_in_df2 != {}:
            log.debug(
                "columns that are in %s but not in %s: %s",
                df_name,
                self_df_name,
                df1_cols_not_in_df2,
            )
        if df2_cols_not_in_df1 != {}:
            log.debug(
                "columns that are in %s but not in %s: %s",
                self_df_name,
                df_name,
                df2_cols_not_in_df1,
            )

        log.debug("columns of %s and %s don't agree.", df_name, df_name)
