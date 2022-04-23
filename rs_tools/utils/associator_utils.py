"""Utilites used in the ImgPolygonAssociator class."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries

from rs_tools.global_constants import STANDARD_CRS_EPSG_CODE

if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator

from rs_tools.graph.bipartite_graph import (BipartiteGraph,
                                            empty_bipartite_graph)


def empty_gdf(index_name,
              cols_and_types,
              crs_epsg_code=STANDARD_CRS_EPSG_CODE):
    """Return a empty GeoDataFrame with specified index and column names and
    types and crs.

    Args:
        - index_name: name of the index of the new empty GeoDataFrame
        - cols_and_types: dict with keys the names of the index and columns of the GeoDataFrame and values the types of the indices/column entries.
        - crs_epsg_code: EPSG code of the crs the empty GeoDataFrame should have.
    Returns:
        - new_empty_df: the empty polygons_df GeoDataFrame.
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


# def empty_gdf(
#         index_name : str,
#         columns : Union[List[str], Dict[str, str]],
#         crs_epsg_code : int=STANDARD_CRS_EPSG_CODE
#         ) -> GeoDataFrame:
#     """Return a empty GeoDataFrame with specified index and column names and crs.

#     :param index_name: name of the index of the new empty GeoDataFrame
#     :param df_cols_and_index_types: dict with keys the names of the index and columns of the GeoDataFrame and values the types of the indices/column entries.
#     :param crs_epsg_code: EPSG code of the crs the empty GeoDataFrame should have.

#     :return: new_empty_df: the empty polygons_df GeoDataFrame.
#     """

#     cols_and_index = [index_name] + list(columns)

#     new_empty_gdf = GeoDataFrame(columns=cols_and_index, crs=f"EPSG:{crs_epsg_code}")
#     new_empty_gdf.set_index(index_name, inplace=True)

#     return new_empty_gdf

# def empty_img_data(
#         img_data_index_name : str,
#         img_data_cols : List[str],
#         crs_epsg_code : int=STANDARD_CRS_EPSG_CODE
#         ) -> GeoDataFrame:
#     """
#     Return a generic empty img_data GeoDataFrame conforming to the ImgPolygonAssociator format.

#     :param img_data_index_name: index name of the new empty img_data
#     :param img_data_cols_and_index_types: dict with keys the names of the index and columns of the new empty img_data and values the types of the index/column entries.
#     :param crs_epsg_code: EPSG code of the crs the empty img_data should have.

#     :return: new_img_data: the empty img_data GeoDataFrame.
#     """

#     return empty_gdf(img_data_index_name, img_data_cols, crs_epsg_code=crs_epsg_code)

# def empty_polygons_df(
#         polygons_df_index_name : str,
#         polygons_df_cols : List[str],
#         crs_epsg_code : int=STANDARD_CRS_EPSG_CODE
#         ) -> GeoDataFrame:
#     """Return a generic empty polygons_df GeoDataFrame conforming to the ImgPolygonAssociator format.

#     Return a generic empty polygons_df GeoDataFrame conforming to the ImgPolygonAssociator format.

#     :param polygons_df_index_name_and_type: name of the index of the new empty polygons_df
#     :param polygons_df_cols_and_index_types: dict with keys the names of the index and columns of the new empty polygons_df and values the types of the indices/column entries.
#     :param crs_epsg_code: EPSG code of the crs the empty polygons_df should have.

#     :return: new_polygons_df: the empty polygons_df GeoDataFrame.
#     """

#     return empty_gdf(polygons_df_index_name, polygons_df_cols, crs_epsg_code=crs_epsg_code)


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


# def empty_gdf_same_format_as(source_df: GeoDataFrame) -> GeoDataFrame:
#     """
#     Creates an empty geodataframe of the same format (index name, columns, column types) as the source_df argument.

#     :param polygons_df: Example polygon dataframe

#     :return: New empty dataframe
#     """
#     df_index_name = source_df.index.name
#     df_cols = source_df.columns
#     crs_epsg_code = source_df.crs.to_epsg()

#     new_empty_df = empty_gdf(df_index_name,
#                              df_cols,
#                              crs_epsg_code=crs_epsg_code)

#     return new_empty_df


def empty_polygons_df_same_format_as(
        polygons_df: GeoDataFrame) -> GeoDataFrame:
    """Creates an empty polygons_df of the same format (index name, columns,
    column types) as the polygons_df argument.

    :param polygons_df: Example polygon dataframe

    :return: New empty dataframe
    """

    return empty_gdf_same_format_as(polygons_df)


def empty_img_data_same_format_as(img_data: GeoDataFrame) -> GeoDataFrame:
    """Creates an empty img_data of the same format (index name, columns, column
    types) as the img_data argument.

    :param img_data: Example images dataframe

    :return: New empty images datagrame
    """

    return empty_gdf_same_format_as(img_data)


def empty_graph() -> BipartiteGraph:
    """Return an empty bipartite graph to be used by ImgPolygonAssociator.

    :returns: empty graph
    """
    return empty_bipartite_graph(red='polygons', black='imgs')


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
