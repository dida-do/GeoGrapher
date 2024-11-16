"""Create associator rasters from a directory containing GeoTiff rasters."""

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Callable

import rasterio as rio
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, box
from tqdm.auto import tqdm

from geographer.utils.utils import GEOMS_UNION, transform_shapely_geometry


def default_read_in_raster_for_raster_df_function(
    raster_path: Path,
) -> tuple[int, Polygon]:
    """Read in crs and bbox defining a GeoTIFF raster.

    Args:
        raster_path: location of the raster

    Returns:
        tuple: crs code of the raster, bounding rectangle of the raster
    """
    if raster_path.suffix in [".tif", ".tiff"]:
        # ... open them in rasterio ...
        with rio.open(raster_path, "r") as src:
            # ... extract information ...

            orig_crs_epsg_code = src.crs.to_epsg()

            raster_bounding_rectangle_orig_crs = box(*src.bounds)

    else:
        orig_crs_epsg_code, raster_bounding_rectangle_orig_crs = None, None

    return orig_crs_epsg_code, raster_bounding_rectangle_orig_crs


def rasters_from_rasters_dir(
    rasters_dir: pathlib.Path | str,
    rasters_crs_epsg_code: int | None = None,
    raster_names: list[str] | None = None,
    rasters_datatype: str = "tif",
    read_in_raster_for_raster_df_function: Callable[
        [Path], tuple[int, Polygon]
    ] = default_read_in_raster_for_raster_df_function,
) -> GeoDataFrame:
    """Return rasters from a directory of GeoTiffs.

    Build and return an associator rasters from a directory of rasters
    (or from a data directory). Only the index (rasters_index_name,
    defaults to raster_name), geometry column (coordinates of the
    raster_bounding_rectangle, and orig_crs_epsg_code (epsg code of crs the raster
    is in) columns will be populated, custom columns will have to be populated
    by a custom written function.

    Args:
        rasters_dir: path of the directory that the rasters are in (assumes the dir
            has no rasters subdir), or path to a data_dir with a rasters subdir.
        rasters_crs_epsg_code: epsg code of rasters crs to be returned.
        raster_names: optional list of raster names. Defaults to None,
            i.e. all rasters in rasters_dir.
        rasters_datatype: datatype suffix of the rasters
        read_in_raster_for_raster_df_function: function that reads in the crs code
            and the bounding rectangle for the rasters

    Returns:
        rasters conforming to the associator rasters format with index
        rasters_index_name and columns geometry and orig_crs_epsg_code
    """
    # stupid hack to avoid (not really) circular importing python can't deal with.
    from geographer.global_constants import STANDARD_CRS_EPSG_CODE

    rasters_dir = Path(rasters_dir)

    if rasters_crs_epsg_code is None:
        rasters_crs_epsg_code = STANDARD_CRS_EPSG_CODE

    if raster_names is None:
        raster_paths = rasters_dir.glob(f"*.{rasters_datatype}")
    else:
        raster_paths = (rasters_dir / raster_name for raster_name in raster_names)

    # dict to keep track of information about the rasters that
    # we will make the rasters from.
    new_rasters_dict: dict[str, str | GEOMS_UNION | int] = {
        index_or_col_name: []
        for index_or_col_name in {"raster_name", "geometry", "orig_crs_epsg_code"}
    }

    # for all rasters in dir ...
    for raster_path in tqdm(raster_paths, desc="building rasters"):
        (
            orig_crs_epsg_code,
            raster_bounding_rectangle_orig_crs,
        ) = read_in_raster_for_raster_df_function(raster_path)

        if orig_crs_epsg_code is None or raster_bounding_rectangle_orig_crs is None:
            continue

        raster_bounding_rectangle_rasters_crs = transform_shapely_geometry(
            raster_bounding_rectangle_orig_crs,
            orig_crs_epsg_code,
            rasters_crs_epsg_code,
        )

        # and put the information into a dict.
        raster_info_dict = {
            "raster_name": raster_path.name,
            "geometry": raster_bounding_rectangle_rasters_crs,
            "orig_crs_epsg_code": int(orig_crs_epsg_code),
        }

        #  Add information about the raster to new_rasters_dict ...
        for key in new_rasters_dict.keys():
            new_rasters_dict[key].append(raster_info_dict[key])

    # ... and create a rasters GeoDatFrame from new_rasters_dict:
    new_rasters = GeoDataFrame(new_rasters_dict, geometry="geometry")
    new_rasters.set_crs(epsg=rasters_crs_epsg_code, inplace=True)
    new_rasters.set_index("raster_name", inplace=True)

    return new_rasters
