"""Utilites used in the Connector class."""

import logging

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries

from geographer.global_constants import STANDARD_CRS_EPSG_CODE
from geographer.graph.bipartite_graph import (BipartiteGraph,
                                              empty_bipartite_graph)
from geographer.graph.bipartite_graph_mixin import (RASTER_IMGS_COLOR,
                                                    VECTOR_FEATURES_COLOR)

log = logging.getLogger(__name__)


def empty_gdf(index_name,
              cols_and_types,
              crs_epsg_code=STANDARD_CRS_EPSG_CODE):
    """Return a empty GeoDataFrame with specified index and column names and
    types and crs.

    Args:
        index_name: name of the index of the new empty GeoDataFrame
        cols_and_types: dict with keys the names of the index and columns of the
            GeoDataFrame and values the types of the indices/column entries.
        crs_epsg_code: EPSG code of the crs the empty GeoDataFrame should have.

    Returns:
        new_empty_df: the empty vector_features_df GeoDataFrame.
    """

    new_empty_gdf_dict = {
        index_name: str,
        'geometry': GeoSeries([]),
        **{
            index_or_col_name: pd.Series([], dtype=index_or_col_type)
            for index_or_col_name, index_or_col_type in cols_and_types.items() if index_or_col_name != 'geometry'
        }
    }

    new_empty_gdf = GeoDataFrame(new_empty_gdf_dict,
                                 crs=f"EPSG:{crs_epsg_code}")
    new_empty_gdf.set_index(index_name, inplace=True)
    return new_empty_gdf


def empty_gdf_same_format_as(df):
    """Creates an empty df of the same format (index name, columns, column
    types) as the df argument."""
    df_index_name = df.index.name

    df_cols_and_index_types = {
        df.index.name: df.index.dtype,
        **df.dtypes.to_dict()
    }

    crs_epsg_code = df.crs.to_epsg()

    new_empty_df = empty_gdf(df_index_name,
                             df_cols_and_index_types,
                             crs_epsg_code=crs_epsg_code)

    return new_empty_df


def empty_vector_features_same_format_as(
        vector_features: GeoDataFrame) -> GeoDataFrame:
    """Creates an empty vector_features of the same format (index name,
    columns, column types) as the vector_features argument.

    Args:
        vector_features: Example polygon dataframe

    Returns:
        New empty dataframe
    """

    return empty_gdf_same_format_as(vector_features)


def empty_raster_imgs_same_format_as(
        raster_imgs: GeoDataFrame) -> GeoDataFrame:
    """Creates an empty raster_imgs of the same format (index name, columns,
    column types) as the raster_imgs argument.

    Args:
        raster_imgs: Example images dataframe

    Returns:
        New empty images datagrame
    """

    return empty_gdf_same_format_as(raster_imgs)


def empty_graph() -> BipartiteGraph:
    """Return an empty bipartite graph to be used by Connector.

    Returns:
        empty graph
    """
    return empty_bipartite_graph(red=VECTOR_FEATURES_COLOR,
                                 black=RASTER_IMGS_COLOR)


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
            log.debug("columns that are in %s but not in %s: %s", df_name,
                      self_df_name, df1_cols_not_in_df2)
        if df2_cols_not_in_df1 != {}:
            log.debug("columns that are in %s but not in %s: %s", self_df_name,
                      df_name, df2_cols_not_in_df1)

        log.debug("columns of %s and %s don't agree.", df_name, df_name)
