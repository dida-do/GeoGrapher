"""
Utilites used in the ImgPolygonAssociator class.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Union, Optional, TYPE_CHECKING

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries

from rs_tools.global_constants import STANDARD_CRS_EPSG_CODE
if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.graph.bipartite_graph import BipartiteGraph, empty_bipartite_graph


def empty_gdf(
        df_index_name : str,
        df_cols_and_index_types : Dict[str, str],
        crs_epsg_code : int=STANDARD_CRS_EPSG_CODE
        ) -> GeoDataFrame:
    """Return a empty GeoDataFrame with specified index and column names and types and crs.

    :param df_index_name: name of the index of the new empty GeoDataFrame
    :param df_cols_and_index_types: dict with keys the names of the index and columns of the GeoDataFrame and values the types of the indices/column entries.
    :param crs_epsg_code: EPSG code of the crs the empty GeoDataFrame should have.
    
    :return: new_empty_df: the empty polygons_df GeoDataFrame.
    """

    new_empty_gdf_dict = {'geometry': GeoSeries([]),
                                **{index_or_col_name: pd.Series([], dtype=index_or_col_type) 
                                    for index_or_col_name, index_or_col_type in df_cols_and_index_types.items()
                                        if index_or_col_name != 'geometry'}}
    new_empty_gdf = GeoDataFrame(new_empty_gdf_dict, crs=f"EPSG:{crs_epsg_code}")
    new_empty_gdf.set_index(df_index_name, inplace=True)
    
    return new_empty_gdf


def empty_imgs_df(
        imgs_df_index_name : str,
        imgs_df_cols_and_index_types : Dict[str, str],
        crs_epsg_code : int=STANDARD_CRS_EPSG_CODE
        ) -> GeoDataFrame:
    """
    Return a generic empty imgs_df GeoDataFrame conforming to the ImgPolygonAssociator format.
    
    :param imgs_df_index_name: index name of the new empty imgs_df
    :param imgs_df_cols_and_index_types: dict with keys the names of the index and columns of the new empty imgs_df and values the types of the index/column entries.
    :param crs_epsg_code: EPSG code of the crs the empty imgs_df should have.
    
    :return: new_imgs_df: the empty imgs_df GeoDataFrame.
    """

    return empty_gdf(imgs_df_index_name, imgs_df_cols_and_index_types, crs_epsg_code=crs_epsg_code)


def empty_polygons_df(
        polygons_df_index_name : str,
        polygons_df_cols_and_index_types : Dict[str, str],
        crs_epsg_code : int=STANDARD_CRS_EPSG_CODE
        ) -> GeoDataFrame:
    """Return a generic empty polygons_df GeoDataFrame conforming to the ImgPolygonAssociator format.

    Return a generic empty polygons_df GeoDataFrame conforming to the ImgPolygonAssociator format.
    
    :param polygons_df_index_name_and_type: name of the index of the new empty polygons_df
    :param polygons_df_cols_and_index_types: dict with keys the names of the index and columns of the new empty polygons_df and values the types of the indices/column entries.
    :param crs_epsg_code: EPSG code of the crs the empty polygons_df should have.
    
    :return: new_polygons_df: the empty polygons_df GeoDataFrame.
    """

    return empty_gdf(polygons_df_index_name, polygons_df_cols_and_index_types, crs_epsg_code=crs_epsg_code)


def empty_gdf_same_format_as(source_df: GeoDataFrame) -> GeoDataFrame:
    """
    Creates an empty geodataframe of the same format (index name, columns, column types) as the source_df argument.

    :param polygons_df: Example polygon dataframe

    :return: New empty dataframe
    """
    df_index_name = source_df.index.name

    df_cols_and_index_types = {source_df.index.name: source_df.index.dtype, 
                               **source_df.dtypes.to_dict()}

    crs_epsg_code = source_df.crs.to_epsg()

    new_empty_df = empty_gdf(df_index_name,
                             df_cols_and_index_types, 
                             crs_epsg_code=crs_epsg_code)

    return new_empty_df


def empty_polygons_df_same_format_as(polygons_df: GeoDataFrame) -> GeoDataFrame:
    """
    Creates an empty polygons_df of the same format (index name, columns, column types) as the polygons_df argument.

    :param polygons_df: Example polygon dataframe

    :return: New empty dataframe
    """

    return empty_gdf_same_format_as(polygons_df)


def empty_imgs_df_same_format_as(imgs_df: GeoDataFrame) -> GeoDataFrame:
    """
    Creates an empty imgs_df of the same format (index name, columns, column types) as the imgs_df argument.

    :param imgs_df: Example images dataframe

    :return: New empty images datagrame
    """

    return empty_gdf_same_format_as(imgs_df)
    

def empty_graph() -> BipartiteGraph:
    """
    Return an empty bipartite graph to be used by ImgPolygonAssociator.
    
    :returns: empty graph
    """
    return empty_bipartite_graph(red='polygons', black='imgs')
