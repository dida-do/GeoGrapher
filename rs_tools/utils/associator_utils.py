"""
Utilites used in the ImgPolygonAssociator class.
"""

from pathlib import Path
from typing import Dict, Union, Optional

import pandas as pd
import geopandas as gpd

from rs_tools.img_polygon_associator import ImgPolygonAssociator, STANDARD_CRS_EPSG_CODE
from rs_tools.graph.bipartite_graph import BipartiteGraph, empty_bipartite_graph


def empty_gdf(df_index_name: str,
                df_cols_and_index_types: Dict[str, str],
                crs_epsg_code: int=STANDARD_CRS_EPSG_CODE) -> gpd.GeoDataFrame:
    """Return a empty GeoDataFrame with specified index and column names and types and crs.

    :param df_index_name: name of the index of the new empty GeoDataFrame
    :param df_cols_and_index_types: dict with keys the names of the index and columns of the GeoDataFrame and values the types of the indices/column entries.
    :param crs_epsg_code: EPSG code of the crs the empty GeoDataFrame should have.
    
    :return: new_empty_df: the empty polygons_df GeoDataFrame.
    """

    new_empty_gdf_dict = {'geometry': gpd.GeoSeries([]),
                                **{index_or_col_name: pd.Series([], dtype=index_or_col_type) 
                                    for index_or_col_name, index_or_col_type in df_cols_and_index_types.items()
                                        if index_or_col_name != 'geometry'}}
    new_empty_gdf = gpd.GeoDataFrame(new_empty_gdf_dict, crs=f"EPSG:{crs_epsg_code}")
    new_empty_gdf.set_index(df_index_name, inplace=True)
    return new_empty_gdf


def empty_imgs_df(imgs_df_index_name: str,
                    imgs_df_cols_and_index_types: Dict[str, str],
                    crs_epsg_code: int=STANDARD_CRS_EPSG_CODE) -> gpd.GeoDataFrame:
    """
    Return a generic empty imgs_df GeoDataFrame conforming to the ImgPolygonAssociator format.
    
    :param imgs_df_index_name: index name of the new empty imgs_df
    :param imgs_df_cols_and_index_types: dict with keys the names of the index and columns of the new empty imgs_df and values the types of the index/column entries.
    :param crs_epsg_code: EPSG code of the crs the empty imgs_df should have.
    
    :return: new_imgs_df: the empty imgs_df GeoDataFrame.
    """

    return empty_gdf(imgs_df_index_name, imgs_df_cols_and_index_types, crs_epsg_code=crs_epsg_code)


def empty_polygons_df(polygons_df_index_name: str,
                        polygons_df_cols_and_index_types: Dict[str, str],
                        crs_epsg_code: int=STANDARD_CRS_EPSG_CODE) -> gpd.GeoDataFrame:
    """Return a generic empty polygons_df GeoDataFrame conforming to the ImgPolygonAssociator format.

    Return a generic empty polygons_df GeoDataFrame conforming to the ImgPolygonAssociator format.
    
    :param polygons_df_index_name_and_type: name of the index of the new empty polygons_df
    :param polygons_df_cols_and_index_types: dict with keys the names of the index and columns of the new empty polygons_df and values the types of the indices/column entries.
    :param crs_epsg_code: EPSG code of the crs the empty polygons_df should have.
    
    :return: new_polygons_df: the empty polygons_df GeoDataFrame.
    """

    return empty_gdf(polygons_df_index_name, polygons_df_cols_and_index_types, crs_epsg_code=crs_epsg_code)


def empty_polygons_df_same_format_as(polygons_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Creates an empty polygons_df of the same format (index name, columns, column types) as the polygons_df argument.

    :param polygons_df: Example polygon dataframe

    :return: New empty dataframe
    """
    polygons_df_index_name = polygons_df.index.name

    polygons_df_cols_and_index_types = {polygons_df.index.name: polygons_df.index.dtype, 
                                        **polygons_df.dtypes.to_dict()}

    crs_epsg_code = polygons_df.crs.to_epsg()

    new_empty_polygons_df = empty_polygons_df(polygons_df_index_name, 
                                                polygons_df_cols_and_index_types, 
                                                crs_epsg_code=crs_epsg_code)

    return new_empty_polygons_df


def empty_imgs_df_same_format_as(imgs_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Creates an empty imgs_df of the same format (index name, columns, column types) as the imgs_df argument.

    :param imgs_df: Example images dataframe

    :return: New empty images datagrame
    """
    imgs_df_index_name = imgs_df.index.name

    imgs_df_cols_and_index_types = {imgs_df.index.name: imgs_df.index.dtype, 
                                        **imgs_df.dtypes.to_dict()}

    crs_epsg_code = imgs_df.crs.to_epsg()

    new_empty_imgs_df = empty_imgs_df(imgs_df_index_name, 
                                                imgs_df_cols_and_index_types, 
                                                crs_epsg_code=crs_epsg_code)

    return new_empty_imgs_df


def empty_assoc_same_format_as(target_data_dir: Union[str, Path],
                                source_data_dir: Optional[Union[str, Path]]=None,
                                source_assoc: Optional[ImgPolygonAssociator]=None) -> ImgPolygonAssociator:
    """
    Creates an empty associator with data_dir target_data_dir of the same format as an existing one in source_data_dir or one given as source_assoc (same polygons_df and imgs_df columns and index names and paramaters).

    :param target_data_dir: New associator data directory
    :param source_data_dir: Source associator data directory
    :param source_assoc: optional source associator

    :returns: new associator
    """

    # exactly one of source_data_dir or source_assoc should be given
    assert (source_data_dir != None) ^ (source_assoc != None)

    if source_assoc == None:
        source_assoc = ImgPolygonAssociator(source_data_dir)

    # new empty polygons_df
    new_empty_polygons_df = empty_polygons_df_same_format_as(source_assoc.polygons_df)                                        
    
    # new empty imgs_df
    new_empty_imgs_df = empty_imgs_df_same_format_as(source_assoc.imgs_df) 

    new_empty_assoc = ImgPolygonAssociator(data_dir=Path(target_data_dir), 
                        imgs_df=new_empty_imgs_df, 
                        polygons_df=new_empty_polygons_df, 
                        **source_assoc._params_dict)

    return new_empty_assoc
    

def empty_graph() -> BipartiteGraph:
    """
    Return an empty bipartite graph to be used by ImgPolygonAssociator.
    
    :returns: empty graph
    """
    return empty_bipartite_graph(red='polygons', black='imgs')
