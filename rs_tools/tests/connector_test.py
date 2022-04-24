"""
TODO: Doesn't work at the moment. Update to work with rs_tools connector.

Simple pytest test suite for the Connector using dummy vector_features and raster_imgs dataframes.

# TODO: change name of img_polygon_connector_artificial_data_test
See img_polygon_connector_artificial_data_test for a visualization of the test data (polygons and images).
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
from rs_tools.global_constants import RASTER_IMGS_INDEX_NAME, STANDARD_CRS_EPSG_CODE, VECTOR_FEATURES_INDEX_NAME
from rs_tools.graph.bipartite_graph_mixin import RASTER_IMGS_COLOR, VECTOR_FEATURES_COLOR
import pytest
import shapely as shp
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Polygon, box
from rs_tools import Connector

TASK_FEATURE_CLASSES = ["h", "t"]


def test_connector():

    # Create empty connector
    data_dir = Path("/whatever/")
    connector = Connector.from_scratch(
        data_dir=data_dir, task_feature_classes=TASK_FEATURE_CLASSES)
    """
    Toy vector_features
    """
    # create empty GeoDataFrame with the right index name
    new_vector_features = gpd.GeoDataFrame()
    new_vector_features.rename_axis(VECTOR_FEATURES_INDEX_NAME, inplace=True)

    # polygon names and geometries
    polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    polygon2 = box(4, 4, 5, 5)
    polygon3 = box(-2, -2, -1, -1)

    # add the polygon names and geometries to the geodataframe
    for p_name, p_geom in zip(['polygon1', 'polygon2', 'polygon3'],
                              [polygon1, polygon2, polygon3]):
        new_vector_features.loc[p_name, 'geometry'] = p_geom

    # create the other columns:
    new_vector_features["some_feature_attribute"] = "foo"
    new_vector_features["type"] = "t"

    # set crs
    new_vector_features = new_vector_features.set_crs(
        epsg=STANDARD_CRS_EPSG_CODE)
    """
    Test add_to_vector_features
    """
    # integrate vector_features
    connector.add_to_vector_features(new_vector_features)

    connector_vector_features_no_img_count = connector.vector_features[[
        col for col in connector.vector_features.columns if col != 'img_count'
    ]]
    # WHY DOES THIS NOT WORK?
    # assert assert_geodataframe_equal(connector_vector_features_no_img_count, new_vector_features)
    # This does not work either
    #assert connector_vector_features_no_img_count == new_vector_features
    pd.testing.assert_frame_equal(
        connector_vector_features_no_img_count,
        new_vector_features,
    )
    """
    Toy raster_imgs
    """

    # empty GeoDataFrame with right index name
    new_raster_imgs = gpd.GeoDataFrame()
    new_raster_imgs.rename_axis(RASTER_IMGS_INDEX_NAME, inplace=True)

    # geometries (img bounding rectangles)
    bounding_rectangle1 = box(
        -0.5, -0.5, 6, 6
    )  # contains both p1 and p2 (to be defined later, see below), doesn't intersect p3
    bounding_rectangle2 = box(
        -1.5, -1.5, 0.5, 0.5
    )  # has non-empty intersection with p1 and p3, but does not contain either, no intersection with p2

    # add to new_raster_imgs
    for img_name, bounding_rectangle in zip(
        ['img1', 'img2'], [bounding_rectangle1, bounding_rectangle2]):
        new_raster_imgs.loc[img_name, 'geometry'] = bounding_rectangle

    new_raster_imgs["some_img_attribute"] = "bar"

    # set crs
    new_raster_imgs = new_raster_imgs.set_crs(epsg=STANDARD_CRS_EPSG_CODE)
    """
    Test add_to_raster_imgs
    """
    connector.add_to_raster_imgs(new_raster_imgs)

    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon1": {
                "img1": "contains",
                "img2": "intersects"
            },
            "polygon2": {
                "img1": "contains"
            },
            "polygon3": {
                "img2": "intersects"
            }
        },
        RASTER_IMGS_COLOR: {
            "img1": {
                "polygon1": "contains",
                "polygon2": "contains"
            },
            "img2": {
                "polygon1": "intersects",
                "polygon3": "intersects"
            }
        }
    }
    """
    Test have_img_for_feature, rectangle_bounding_img, polygons_intersecting_img, polygons_contained_in_img, imgs_containing_feature, values of 'have_img?' column in connector.vector_features.
    """
    assert (connector.rectangle_bounding_img('img1')).equals(
        box(-0.5, -0.5, 6, 6))
    assert list(
        connector.imgs_containing_vector_feature('polygon1')) == ['img1']
    assert list(connector.vector_features_contained_in_img('img2')) == []
    assert set(connector.vector_features_intersecting_img('img2')) == {
        'polygon1', 'polygon3'
    }
    assert set(connector.vector_features_intersecting_img('img1')) == {
        'polygon1', 'polygon2'
    }
    """
    Integrate another img_df with a row that already exists in connector.raster_imgs
    """
    # empty GeoDataFrame with right index name
    new_raster_imgs2 = gpd.GeoDataFrame()
    new_raster_imgs2.rename_axis(RASTER_IMGS_INDEX_NAME, inplace=True)

    # the new_raster_imgs2 geometries will be the img bounding rectangles here:
    bounding_rectangle1 = box(
        -10, -10, 10, 10
    )  # contains all polygons, but an polygon with that name already exists in connector, so should be ignored by add_to_vector_features
    bounding_rectangle3 = box(
        -3, -3, 7, 7)  # contains all of p1, p2, p3, p4 (to be defined below)
    bounding_rectangle4 = box(
        -1.5, -1.5, 2, 2
    )  # contains p1 and p4 (to be defined below), has non-empty intersection with p3, but does not intersect p2

    # add them to new_raster_imgs2
    for img_name, bounding_rectangle in zip(
        ['img1', 'img3', 'img4'],
        [bounding_rectangle1, bounding_rectangle3, bounding_rectangle4]):
        new_raster_imgs2.loc[img_name, 'geometry'] = bounding_rectangle

    new_raster_imgs["some_img_attribute"] = "foobar"

    # set crs
    new_raster_imgs2 = new_raster_imgs2.set_crs(epsg=STANDARD_CRS_EPSG_CODE)

    # integrate new_raster_imgs2
    connector.add_to_raster_imgs(new_raster_imgs2)

    # test containment/intersection relations, i.e. graph structure
    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon1": {
                "img1": "contains",
                "img2": "intersects",
                "img3": "contains",
                "img4": "contains"
            },
            "polygon2": {
                "img1": "contains",
                "img3": "contains"
            },
            "polygon3": {
                "img2": "intersects",
                "img3": "contains",
                "img4": "intersects"
            }
        },
        RASTER_IMGS_COLOR: {
            "img1": {
                "polygon1": "contains",
                "polygon2": "contains"
            },
            "img2": {
                "polygon1": "intersects",
                "polygon3": "intersects"
            },
            "img3": {
                "polygon1": "contains",
                "polygon2": "contains",
                "polygon3": "contains"
            },
            "img4": {
                "polygon1": "contains",
                "polygon3": "intersects"
            }
        }
    }
    """
    Integrate another vector_features with a row that already exists in the connector's vector_features
    """

    # create empty GeoDataFrame with the right index name
    new_vector_features2 = gpd.GeoDataFrame()
    new_vector_features2.rename_axis(VECTOR_FEATURES_INDEX_NAME, inplace=True)

    # polygon names and geometries
    polygon4 = box(-1, -1, 0, 0)  # genuinely new entry
    polygon3 = box(
        -100, -100, -99,
        -99)  # entry for polygon3 already exists in connector.vector_features

    # add the polygon names and geometries to the geodataframe
    for p_name, p_geom in zip(['polygon3', 'polygon4'], [polygon3, polygon4]):
        new_vector_features2.loc[p_name, 'geometry'] = p_geom

    new_vector_features2["some_feature_attribute"] = "What ho, Jeeves!"

    # set crs
    new_vector_features2 = new_vector_features2.set_crs(
        epsg=STANDARD_CRS_EPSG_CODE)

    # integrate new_vector_features with force_overwrite=True:
    connector.add_to_vector_features(new_vector_features2,
                                     force_overwrite=True)

    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon1": {
                "img1": "contains",
                "img2": "intersects",
                "img3": "contains",
                "img4": "contains"
            },
            "polygon2": {
                "img1": "contains",
                "img3": "contains"
            },
            "polygon3": {},
            "polygon4": {
                "img1": "intersects",
                "img2": "contains",
                "img3": "contains",
                "img4": "contains"
            }
        },
        RASTER_IMGS_COLOR: {
            "img1": {
                "polygon1": "contains",
                "polygon2": "contains",
                "polygon4": "intersects"
            },
            "img2": {
                "polygon1": "intersects",
                "polygon4": "contains"
            },
            "img3": {
                "polygon1": "contains",
                "polygon2": "contains",
                "polygon4": "contains"
            },
            "img4": {
                "polygon1": "contains",
                "polygon4": "contains"
            }
        }
    }

    # assert we have no duplicate entries
    assert len(connector.raster_imgs) == 4
    assert len(connector.vector_features) == 4
    """
    Test drop_raster_imgs
    """
    connector.drop_raster_imgs(['img2', 'img3'])

    assert len(connector.raster_imgs) == 2

    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon1": {
                "img1": "contains",
                "img4": "contains"
            },
            "polygon2": {
                "img1": "contains"
            },
            "polygon3": {},
            "polygon4": {
                "img1": "intersects",
                "img4": "contains"
            }
        },
        RASTER_IMGS_COLOR: {
            "img1": {
                "polygon1": "contains",
                "polygon2": "contains",
                "polygon4": "intersects"
            },
            "img4": {
                "polygon1": "contains",
                "polygon4": "contains"
            }
        }
    }
    """
    Test drop_vector_features
    """
    connector.drop_vector_features(['polygon1', 'polygon3'])

    assert len(connector.vector_features) == 2

    assert connector._graph._graph_dict == {
        VECTOR_FEATURES_COLOR: {
            "polygon2": {
                "img1": "contains"
            },
            "polygon4": {
                "img1": "intersects",
                "img4": "contains"
            }
        },
        RASTER_IMGS_COLOR: {
            "img1": {
                "polygon2": "contains",
                "polygon4": "intersects"
            },
            "img4": {
                "polygon4": "contains"
            }
        }
    }

    connector.drop_raster_imgs("img4")
    assert len(connector.raster_imgs) == 1


if __name__ == "__main__":
    test_connector()
