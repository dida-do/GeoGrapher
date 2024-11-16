"""Test adding/dropping vector features and rasters from Connector.

Simple pytest test suite for the Connector using dummy vectors
and rasters dataframes.

See connector_test.png for a visualization of the test data
(polygons as vectors and rasters).

TODO: Test save/from_data_dir with clean_up
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box

from geographer.connector import Connector
from geographer.global_constants import (
    RASTER_IMGS_INDEX_NAME,
    STANDARD_CRS_EPSG_CODE,
    VECTOR_FEATURES_INDEX_NAME,
)
from geographer.graph.bipartite_graph_mixin import (
    RASTER_IMGS_COLOR,
    VECTOR_FEATURES_COLOR,
)
from geographer.testing.graph_df_compatibility import check_graph_vertices_counts

TASK_FEATURE_CLASSES = ["class1", "class2"]


def test_connector():
    """Test adding/dropping vector features and rasters from Connector."""
    # Create empty connector
    data_dir = Path("/whatever/")
    connector = Connector.from_scratch(
        data_dir=data_dir, task_vector_classes=TASK_FEATURE_CLASSES
    )
    """
    Toy vectors
    """
    # create empty GeoDataFrame with the right index name
    new_vectors = gpd.GeoDataFrame(geometry=gpd.GeoSeries([]))
    new_vectors.rename_axis(VECTOR_FEATURES_INDEX_NAME, inplace=True)

    # polygon names and geometries
    polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    polygon2 = box(4, 4, 5, 5)
    polygon3 = box(-2, -2, -1, -1)

    # add the polygon names and geometries to the geodataframe
    for p_name, p_geom in zip(
        ["polygon1", "polygon2", "polygon3"], [polygon1, polygon2, polygon3]
    ):
        new_vectors.loc[p_name, "geometry"] = p_geom

    # create the other columns:
    new_vectors["some_vector_attribute"] = "foo"
    new_vectors["type"] = "class1"

    # set crs
    new_vectors = new_vectors.set_crs(epsg=STANDARD_CRS_EPSG_CODE)
    """Test add_to_vectors."""
    # add vectors
    connector.add_to_vectors(new_vectors)

    connector_vectors_no_raster_count = connector.vectors[
        [col for col in connector.vectors.columns if col != "raster_count"]
    ]
    pd.testing.assert_frame_equal(
        connector_vectors_no_raster_count,
        new_vectors,
    )
    assert check_graph_vertices_counts(connector)
    """Toy rasters."""

    # empty GeoDataFrame with right index name
    new_rasters = gpd.GeoDataFrame(geometry=gpd.GeoSeries([]))
    new_rasters.rename_axis(RASTER_IMGS_INDEX_NAME, inplace=True)

    # geometries (raster bounding rectangles)
    bounding_rectangle1 = box(
        -0.5, -0.5, 6, 6
    )  # contains both p1 and p2 (to be defined later, see below), doesn't intersect p3
    bounding_rectangle2 = box(-1.5, -1.5, 0.5, 0.5)
    # bounding_rectangle2 has non-empty intersection with p1 and p3,
    # but does not contain either, no intersection with p2

    # add to new_rasters
    for raster_name, bounding_rectangle in zip(
        ["raster1", "raster2"], [bounding_rectangle1, bounding_rectangle2]
    ):
        new_rasters.loc[raster_name, "geometry"] = bounding_rectangle

    new_rasters["some_raster_attribute"] = "bar"

    # set crs
    new_rasters = new_rasters.set_crs(epsg=STANDARD_CRS_EPSG_CODE)
    """Test add_to_rasters."""
    connector.add_to_rasters(new_rasters)

    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon1": {"raster1": "contains", "raster2": "intersects"},
            "polygon2": {"raster1": "contains"},
            "polygon3": {"raster2": "intersects"},
        },
        RASTER_IMGS_COLOR: {
            "raster1": {"polygon1": "contains", "polygon2": "contains"},
            "raster2": {"polygon1": "intersects", "polygon3": "intersects"},
        },
    }
    assert check_graph_vertices_counts(connector)
    """Test have_raster_for_vector, rectangle_bounding_raster,
    polygons_intersecting_raster, polygons_contained_in_raster,
    rasters_containing_vector, values of 'have_raster?' column in
    connector.vectors."""
    assert (connector.rectangle_bounding_raster("raster1")).equals(
        box(-0.5, -0.5, 6, 6)
    )
    assert list(connector.rasters_containing_vector("polygon1")) == ["raster1"]
    assert list(connector.vectors_contained_in_raster("raster2")) == []
    assert set(connector.vectors_intersecting_raster("raster2")) == {
        "polygon1",
        "polygon3",
    }
    assert set(connector.vectors_intersecting_raster("raster1")) == {
        "polygon1",
        "polygon2",
    }
    """Add more rasters."""
    # empty GeoDataFrame with right index name
    new_rasters2 = gpd.GeoDataFrame(geometry=gpd.GeoSeries([]))
    new_rasters2.rename_axis(RASTER_IMGS_INDEX_NAME, inplace=True)

    # the new_rasters2 geometries will be the raster bounding rectangles here:
    bounding_rectangle3 = box(
        -3, -3, 7, 7
    )  # contains all of p1, p2, p3, p4 (to be defined below)
    bounding_rectangle4 = box(-1.5, -1.5, 2, 2)
    # bounding_rectangle4 contains p1 and p4 (to be defined below), has
    # non-empty intersection with p3, but does not intersect p2

    # add them to new_rasters2
    for raster_name, bounding_rectangle in zip(
        ["raster3", "raster4"], [bounding_rectangle3, bounding_rectangle4]
    ):
        new_rasters2.loc[raster_name, "geometry"] = bounding_rectangle

    new_rasters["some_raster_attribute"] = "foobar"

    # set crs
    new_rasters2 = new_rasters2.set_crs(epsg=STANDARD_CRS_EPSG_CODE)

    # integrate new_rasters2
    connector.add_to_rasters(new_rasters2)

    # test containment/intersection relations, i.e. graph structure
    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon1": {
                "raster1": "contains",
                "raster2": "intersects",
                "raster3": "contains",
                "raster4": "contains",
            },
            "polygon2": {"raster1": "contains", "raster3": "contains"},
            "polygon3": {
                "raster2": "intersects",
                "raster3": "contains",
                "raster4": "intersects",
            },
        },
        RASTER_IMGS_COLOR: {
            "raster1": {"polygon1": "contains", "polygon2": "contains"},
            "raster2": {"polygon1": "intersects", "polygon3": "intersects"},
            "raster3": {
                "polygon1": "contains",
                "polygon2": "contains",
                "polygon3": "contains",
            },
            "raster4": {"polygon1": "contains", "polygon3": "intersects"},
        },
    }
    assert check_graph_vertices_counts(connector)
    """Drop vector feature."""
    connector.drop_vectors("polygon3")

    # test containment/intersection relations, i.e. graph structure
    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon1": {
                "raster1": "contains",
                "raster2": "intersects",
                "raster3": "contains",
                "raster4": "contains",
            },
            "polygon2": {"raster1": "contains", "raster3": "contains"},
        },
        RASTER_IMGS_COLOR: {
            "raster1": {"polygon1": "contains", "polygon2": "contains"},
            "raster2": {
                "polygon1": "intersects",
            },
            "raster3": {
                "polygon1": "contains",
                "polygon2": "contains",
            },
            "raster4": {
                "polygon1": "contains",
            },
        },
    }
    assert check_graph_vertices_counts(connector)
    """Add more vector features."""

    # create empty GeoDataFrame with the right index name
    new_vectors2 = gpd.GeoDataFrame(geometry=gpd.GeoSeries([]))
    new_vectors2.rename_axis(VECTOR_FEATURES_INDEX_NAME, inplace=True)

    # polygon names and geometries
    polygon4 = box(-1, -1, 0, 0)  # genuinely new entry
    polygon3 = box(-100, -100, -99, -99)

    # add the polygon names and geometries to the geodataframe
    for p_name, p_geom in zip(["polygon3", "polygon4"], [polygon3, polygon4]):
        new_vectors2.loc[p_name, "geometry"] = p_geom

    new_vectors2["some_vector_attribute"] = "What ho, Jeeves!"

    # set crs
    new_vectors2 = new_vectors2.set_crs(epsg=STANDARD_CRS_EPSG_CODE)

    # integrate new_vectors with force_overwrite=True:
    connector.add_to_vectors(new_vectors2)

    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon1": {
                "raster1": "contains",
                "raster2": "intersects",
                "raster3": "contains",
                "raster4": "contains",
            },
            "polygon2": {"raster1": "contains", "raster3": "contains"},
            "polygon3": {},
            "polygon4": {
                "raster1": "intersects",
                "raster2": "contains",
                "raster3": "contains",
                "raster4": "contains",
            },
        },
        RASTER_IMGS_COLOR: {
            "raster1": {
                "polygon1": "contains",
                "polygon2": "contains",
                "polygon4": "intersects",
            },
            "raster2": {"polygon1": "intersects", "polygon4": "contains"},
            "raster3": {
                "polygon1": "contains",
                "polygon2": "contains",
                "polygon4": "contains",
            },
            "raster4": {"polygon1": "contains", "polygon4": "contains"},
        },
    }

    assert check_graph_vertices_counts(connector)

    # assert we have no duplicate entries
    assert len(connector.rasters) == 4
    assert len(connector.vectors) == 4
    """Test drop_rasters."""
    connector.drop_rasters(["raster2", "raster3"])

    assert len(connector.rasters) == 2

    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon1": {"raster1": "contains", "raster4": "contains"},
            "polygon2": {"raster1": "contains"},
            "polygon3": {},
            "polygon4": {"raster1": "intersects", "raster4": "contains"},
        },
        RASTER_IMGS_COLOR: {
            "raster1": {
                "polygon1": "contains",
                "polygon2": "contains",
                "polygon4": "intersects",
            },
            "raster4": {"polygon1": "contains", "polygon4": "contains"},
        },
    }

    assert check_graph_vertices_counts(connector)
    """Test drop_vectors."""
    connector.drop_vectors(["polygon1", "polygon3"])

    assert len(connector.vectors) == 2

    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon2": {"raster1": "contains"},
            "polygon4": {"raster1": "intersects", "raster4": "contains"},
        },
        RASTER_IMGS_COLOR: {
            "raster1": {"polygon2": "contains", "polygon4": "intersects"},
            "raster4": {"polygon4": "contains"},
        },
    }

    connector.drop_rasters("raster4")
    assert len(connector.rasters) == 1
    assert check_graph_vertices_counts(connector)


if __name__ == "__main__":
    test_connector()
