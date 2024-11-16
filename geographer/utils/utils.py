"""Utility functions.

transform_shapely_geometry(geometry, from_epsg, to_epsg): Transforms a
shapely geometry from one crs to another.

round_shapely_geometry(geometry, ndigits=1): Rounds the coordinates of a
shapely vector geometry. Useful in some cases for testing the coordinate
conversion of raster bounding rectangles.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Union

import fiona
import geopandas as gpd
import pandas as pd
import pyproj
import rasterio as rio
from fiona.drvsupport import supported_drivers
from geopandas import GeoDataFrame
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import transform

from geographer.global_constants import (
    RASTER_IMGS_INDEX_NAME,
    VECTOR_FEATURES_INDEX_NAME,
)

supported_drivers["KML"] = "rw"

GEOMS_UNION = Union[
    Point,
    Polygon,
    MultiPoint,
    MultiPolygon,
    MultiLineString,
    LinearRing,
    LineString,
    GeometryCollection,
]


def create_logger(app_name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger.

    Serves as a unified way to instantiate a new logger. Will create a new
    logging instance with the name app_name. The logging output is sent to the
    console via a logging.StreamHandler() instance. The output will be
    formatted using the logging time, the logger name, the level at which the
    logger was called and the logging message. As the root logger threshold is
    set to WARNING, the instantiation via logging.getLogger(__name__) results
    in a logger instance, which console handel also has the threshold set to
    WARNING. One needs to additionally set the console handler level to the
    desired level, which is done by this function.

    .. note::

        Function might be adapted for more specialized usage in the future

    Args:
        app_name: Name of the logger. Will appear in the console output
        level: threshold level for the new logger.

    Returns:
        logging.Logger: new logging instance

    Examples::

    >>> import logging
    >>> logger=create_logger(__name__,logging.DEBUG)
    """
    # create new up logger
    logger = logging.getLogger(app_name)
    logger.setLevel(level)

    # create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to the console handler
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def transform_shapely_geometry(
    geometry: GEOMS_UNION, from_epsg: int, to_epsg: int
) -> GEOMS_UNION:
    """Transform a shapely geometry from one crs to another.

    Args:
        geometry: shapely geometry to be transformed.
        from_epsg: EPSG code of crs to be transformed from.
        to_epsg: EPSG code of crs to be transformed to.

    Returns:
        transformed shapely geometry
    """
    # define the coordinate transform ...
    project = pyproj.Transformer.from_crs(
        f"epsg:{from_epsg}", f"epsg:{to_epsg}", always_xy=True
    )

    # ... and apply it:
    transformed_geometry = transform(project.transform, geometry)

    # make sure northeasting behavior agrees for both crs
    from_crs = rio.crs.CRS.from_epsg(from_epsg)
    to_crs = rio.crs.CRS.from_epsg(to_epsg)
    assert rio.crs.epsg_treats_as_northingeasting(
        from_crs
    ) == rio.crs.epsg_treats_as_northingeasting(
        to_crs
    ), "safety check that both crs treat as northeasting failed!"

    return transformed_geometry


def round_shapely_geometry(geometry: GEOMS_UNION, ndigits=1) -> Polygon| Point:
    """Round the coordinates of a shapely geometry.

    Round the coordinates of a shapely geometry (e.g. Polygon or Point).
    Useful in some cases for testing the coordinate conversion of raster
    bounding rectangles.

    Args:
        geometry: shapely geometry to be rounded
        ndigits: number of significant digits to round to. Defaults to 1.

    Returns:
        geometry with all coordinates rounded to ndigits number of significant digits.
    """
    return transform(lambda x, y: (round(x, ndigits), round(y, ndigits)), geometry)


def deepcopy_gdf(gdf: GeoDataFrame) -> GeoDataFrame:
    """Return deepcopy of GeoDataFrame."""
    gdf_copy = GeoDataFrame(
        columns=gdf.columns,
        data=copy.deepcopy(gdf.values),
        crs=gdf.crs,
        geometry=gdf.geometry.name,
    )
    gdf_copy = gdf_copy.astype(gdf.dtypes)
    gdf_copy.set_index(gdf.index, inplace=True)

    return gdf_copy


def concat_gdfs(objs: list[GeoDataFrame], **kwargs: Any) -> GeoDataFrame:
    """Return concatentation of a list of GeoDataFrames.

    The crs and index name of the returned concatenated GeoDataFrames
    will be the crs and index name of the first GeoDataFrame in the
    list.
    """
    for obj in objs:
        if isinstance(obj, GeoDataFrame):
            if obj.crs != objs[0].crs:
                raise ValueError("All geodataframes should have the same CRS")
            if obj.geometry.name != objs[0].geometry.name:
                raise ValueError(
                    "All geodataframes should have the same geometry column!"
                )
        elif not isinstance(obj, GeoDataFrame):
            raise ValueError("all objs should be GeoDataFrames")

    concatenated_gdf = GeoDataFrame(
        pd.concat(objs, **kwargs), crs=objs[0].crs, geometry=objs[0].geometry.name
    )
    concatenated_gdf.index.name = objs[0].index.name

    return concatenated_gdf


def map_dict_values(fun: Callable, dict_arg: dict) -> dict:
    """Apply function to all values of a dict."""
    return {key: fun(val) for key, val in dict_arg.items()}


def create_kml_all_geodataframes(
    data_dir: Path | str, out_path: Path | str
) -> None:
    """Create KML file from a dataset's rasters and vectors.

    Can be used to visualize data in Google Earth Pro.
    """
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    if out_path.suffix not in {".kml", ".KML"}:
        raise ValueError("out_path should have .kml suffix")

    rasters_path = data_dir / "connector/rasters.geojson"
    vectors_path = data_dir / "connector/vectors.geojson"

    rasters = gpd.read_file(rasters_path)[["geometry", RASTER_IMGS_INDEX_NAME]]
    vectors = gpd.read_file(vectors_path)[["geometry", VECTOR_FEATURES_INDEX_NAME]]

    rasters["Description"] = "raster"
    rasters["Name"] = rasters[RASTER_IMGS_INDEX_NAME]
    vectors["Description"] = "vector feature"
    vectors["Name"] = vectors[VECTOR_FEATURES_INDEX_NAME]

    combined = concat_gdfs([rasters, vectors])

    with fiona.drivers():
        combined.to_file(out_path, driver="KML")


def removeprefix(input_str: str, prefix: str) -> str:
    """Remove prefix from string."""
    if prefix and input_str.startswith(prefix):
        return input_str[len(prefix) :]  # noqa: E203
    return input_str
