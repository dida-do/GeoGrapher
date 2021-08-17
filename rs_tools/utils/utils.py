"""
Utility functions. 


transform_shapely_geometry(geometry, from_epsg, to_epsg): Transforms a shapely geometry from one crs to another.

round_shapely_geometry(geometry, ndigits=1): Rounds the coordinates of a shapely vector geometry. Useful in some cases for testing the coordinate conversion of image bounding rectangles. 
"""

import numpy as np
import rasterio.mask
import rasterio as rio
import os
import geopandas as gpd
import shapely
from shapely.geometry import box
from shapely.ops import transform
import pyproj
    


def transform_shapely_geometry(geometry, from_epsg, to_epsg):
    """
    Transforms a shapely geometry from one crs to another.

    Args:
        - geometry (shapely geometry): shapely geometry to be transformed.
        - from_epsg: epsg code of crs to be transformed from.
        - to_epsg: epsg code of crs to be transformed to.

    Returns:
        - transformed shapely geometry.
    """

    # define the coordinate transform ...
    project = pyproj.Transformer.from_crs(f"epsg:{from_epsg}", f"epsg:{to_epsg}", always_xy=True)

    # ... and apply it:
    transformed_geometry = transform(project.transform, geometry)

    # the coordinate axes might be out of order. Let's check if we have to switch them:
    from_crs = rio.crs.CRS.from_epsg(from_epsg)
    to_crs = rio.crs.CRS.from_epsg(to_epsg)

    """if rio.crs.epsg_treats_as_latlong(from_crs) != rio.crs.epsg_treats_as_latlong(to_crs):
        transformed_geometry = transform(lambda x, y: (y,x), transformed_geometry) """

    # make sure directions agree
    assert rio.crs.epsg_treats_as_northingeasting(from_crs) == rio.crs.epsg_treats_as_northingeasting(to_crs)

    return transformed_geometry


def round_shapely_geometry(geometry, ndigits=1):
    """Rounds the coordinates of a shapely vector geometry. Useful in some cases for testing the coordinate conversion of image bounding rectangles. 
    
    Args:
        - geometry: (shapely geometry) shapely geometry to be rounded.
        - ndigits: (int) number of significant digits to round to.
        
    Returns:
        - shapely geometry with all coordinates rounded to ndigits number of significant digits.
    """
    
    return transform(lambda x,y: (round(x, ndigits), round(y, ndigits)), geometry)


def deepcopy_gdf(gdf: GeoDataFrame) -> GeoDataFrame:
    
    gdf_copy = GeoDataFrame(columns=gdf.columns, 
                        data=copy.deepcopy(gdf.values), 
                        crs=gdf.crs)
    gdf_copy = gdf_copy.astype(gdf.dtypes)    
    gdf_copy.set_index(gdf.index, inplace=True)
    
    return gdf_copy

