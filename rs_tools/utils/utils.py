"""
Utility functions.


transform_shapely_geometry(geometry, from_epsg, to_epsg): Transforms a shapely geometry from one crs to another.

round_shapely_geometry(geometry, ndigits=1): Rounds the coordinates of a shapely vector geometry. Useful in some cases for testing the coordinate conversion of image bounding rectangles.
"""

from typing import Union, Callable, List, Any
import copy
import numpy as np
import rasterio.mask
import rasterio as rio
import os
import geopandas as gpd
from geopandas import GeoDataFrame
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon, MultiLineString, LinearRing, LineString, GeometryCollection
from shapely.ops import transform
import pyproj

GEOMS_UNION = Union[Point, Polygon, MultiPoint, MultiPolygon, MultiLineString, LinearRing, LineString, GeometryCollection]


def transform_shapely_geometry(geometry: GEOMS_UNION,
                                from_epsg: int,
                                to_epsg: int) -> GEOMS_UNION:
    """
    Transform a shapely geometry (e.g. Polygon or Point) from one crs to another.

    Args:
        geometry (GEOMS_UNION): shapely geometry to be transformed.
        from_epsg (int): EPSG code of crs to be transformed from.
        to_epsg (int): EPSG code of crs to be transformed to.

    Returns:
        GEOMS_UNION: transformed shapely geometry
    """

    # define the coordinate transform ...
    project = pyproj.Transformer.from_crs(f"epsg:{from_epsg}", f"epsg:{to_epsg}", always_xy=True)

    # ... and apply it:
    transformed_geometry = transform(project.transform, geometry)

    # make sure northeasting behavior agrees for both crs
    from_crs = rio.crs.CRS.from_epsg(from_epsg)
    to_crs = rio.crs.CRS.from_epsg(to_epsg)
    assert rio.crs.epsg_treats_as_northingeasting(from_crs) == rio.crs.epsg_treats_as_northingeasting(to_crs), f"safety check that both crs treat as northeasting failed!"

    return transformed_geometry


def round_shapely_geometry(geometry: GEOMS_UNION, ndigits=1) -> Union[Polygon, Point]:
    """
    Round the coordinates of a shapely geometry (e.g. Polygon or Point). Useful in some cases for testing the coordinate conversion of image bounding rectangles.

    Args:
        geometry (GEOMS_UNION): shapely geometry to be rounded
        ndigits (int, optional): number of significant digits to round to. Defaults to 1.

    Returns:
        GEOMS_UNION: geometry with all coordinates rounded to ndigits number of significant digits.
    """

    return transform(lambda x,y: (round(x, ndigits), round(y, ndigits)), geometry)


def deepcopy_gdf(gdf: GeoDataFrame) -> GeoDataFrame:

    gdf_copy = GeoDataFrame(columns=gdf.columns,
                        data=copy.deepcopy(gdf.values),
                        crs=gdf.crs)
    gdf_copy = gdf_copy.astype(gdf.dtypes)
    gdf_copy.set_index(gdf.index, inplace=True)

    return gdf_copy


# def concat_gdfs(
#         objs: List[Union[GeoDataFrame],
#         **kwargs: Any
#     ) -> GeoDataFrame:

#     for obj in objs:
#         if isinstance(obj, GeoDataFrame) and not obj.crs == objs[0].crs:
#             raise ValueError('all geodataframes should have the same crs')
#         elif not isinstance(pd.DataFrame):
#             raise ValueError('all geodataframes should have the same crs')

#     return GeoDataFrame(pd.concat(objs, **kwargs), crs=objs[0].crs)


def map_dict_values(
        fun : Callable,
        dict_arg : dict
        ) -> dict:
        return {key : fun(val) for key, val in dict_arg.items()}

