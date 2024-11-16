"""Test get_raster_clusters.

Test get_raster_clusters from geographer.utils.cluster_rasters.
"""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, box

from geographer import Connector
from geographer.global_constants import (
    RASTER_IMGS_INDEX_NAME,
    STANDARD_CRS_EPSG_CODE,
    VECTOR_FEATURES_INDEX_NAME,
)
from geographer.utils.cluster_rasters import get_raster_clusters


def test_cluster_rasters():
    """Test get_raster_clusters.

    Test get_raster_clusters from geographer.utils.cluster_rasters.
    """
    # Create empty connector
    data_dir = Path("/whatever/")
    connector = Connector.from_scratch(data_dir=data_dir)
    """
    Create vectors
    """
    new_vectors = gpd.GeoDataFrame(geometry=gpd.GeoSeries([]))
    new_vectors.rename_axis(VECTOR_FEATURES_INDEX_NAME, inplace=True)

    # polygon names and geometries
    polygon1 = Polygon(
        [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)]
    )  # intersects both raster1 and raster2 below
    polygon2 = box(3, 3, 3.5, 3.5)  # contained in raster1 and raster3
    # polygon3 = box(-2, -2, -1, -1)

    # add the polygon names and geometries to the geodataframe
    for p_name, p_geom in zip(["polygon1", "polygon2"], [polygon1, polygon2]):
        new_vectors.loc[p_name, "geometry"] = p_geom

    new_vectors = new_vectors.set_crs(epsg=STANDARD_CRS_EPSG_CODE)
    connector.add_to_vectors(new_vectors)
    """
    Create rasters
    """
    new_rasters = gpd.GeoDataFrame(geometry=gpd.GeoSeries([]))
    new_rasters.rename_axis(RASTER_IMGS_INDEX_NAME, inplace=True)

    # geometries (raster bounding rectangles)
    # rasters 1 and 2 dont intersect but are close.
    # both intersect polygon1
    # raster 2 contains polygon2
    bounding_rectangle1 = box(-4, 0, 4, 8)
    bounding_rectangle2 = box(-4, -0.5, 4, -8.5)

    # intersects raster 2, contains polygon 2
    bounding_rectangle3 = box(2, 2, 10, 10)

    # 4, 5, 6 form a chain:
    # 5 intersects both 4 and 6
    # 4 and 6 do not intersect
    bounding_rectangle4 = box(100, 100, 108, 108)
    bounding_rectangle5 = box(106, 106, 114, 114)
    bounding_rectangle6 = box(112, 112, 120, 120)

    # add to new_rasters
    for raster_name, bounding_rectangle in zip(
        ["raster1", "raster2", "raster3", "raster4", "raster5", "raster6"],
        [
            bounding_rectangle1,
            bounding_rectangle2,
            bounding_rectangle3,
            bounding_rectangle4,
            bounding_rectangle5,
            bounding_rectangle6,
        ],
    ):
        new_rasters.loc[raster_name, "geometry"] = bounding_rectangle

    new_rasters = new_rasters.set_crs(epsg=STANDARD_CRS_EPSG_CODE)
    connector.add_to_rasters(new_rasters)
    """
    Test clusters defined by 'rasters_that_share_vectors_or_overlap'
    """
    clusters = get_raster_clusters(
        connector=connector,
        clusters_defined_by="rasters_that_share_vectors_or_overlap",
        preclustering_method="y then x-axis",
    )
    assert set(map(frozenset, clusters)) == {
        frozenset({"raster3", "raster1", "raster2"}),
        frozenset({"raster4", "raster6", "raster5"}),
    }
    """
    Test clusters defined by 'rasters_that_share_vectors'
    """
    clusters = get_raster_clusters(
        connector=connector,
        clusters_defined_by="rasters_that_share_vectors",
        preclustering_method="y then x-axis",
    )
    assert set(map(frozenset, clusters)) == {
        frozenset({"raster3", "raster1", "raster2"}),
        frozenset({"raster4"}),
        frozenset({"raster5"}),
        frozenset({"raster6"}),
    }


if __name__ == "__main__":
    test_cluster_rasters()
