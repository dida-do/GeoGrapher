"""
TODO: Doesn't work at the moment. Update to work with rs_tools connector. 

Simple pytest test suite for the Connector using dummy vector_features and raster_imgs dataframes.

# TODO: change name of img_polygon_connector_artificial_data_test
See img_polygon_connector_artificial_data_test for a visualization of the test data (polygons and images). 
"""

import pathlib
from pathlib import Path

import geopandas as gpd
import pandas as pd
from global_constants import RASTER_IMGS_INDEX_NAME, STANDARD_CRS_EPSG_CODE, VECTOR_FEATURES_INDEX_NAME
from graph.bipartite_graph_mixin import RASTER_IMGS_COLOR, VECTOR_FEATURES_COLOR
import pytest
import shapely as shp
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Polygon, box

from rs_tools import Connector

# TODO: clean up!
# column and index names for the internal vector_features geodataframe of the connector
IMGS_DF_INDEX_NAME_AND_TYPE = {'img_name': str}
IMGS_DF_INDEX_NAME = list(IMGS_DF_INDEX_NAME_AND_TYPE.keys())[0]
IMGS_DF_COLS_AND_TYPES = {
    'geometry': shp.geometry,
    'orig_crs_epsg_code': int,
    'img_processed?': bool,
    'timestamp': str
}  # needs to be str, since geopandas can't save datetimes
IMGS_DF_COLS_AND_INDEX_TYPES = {
    **IMGS_DF_INDEX_NAME_AND_TYPE,
    **IMGS_DF_COLS_AND_TYPES
}

# column and index names for the internal img_data geodataframe of the connector
POLYGONS_DF_INDEX_NAME_AND_TYPE = {VECTOR_FEATURES_INDEX_NAME: str}
POLYGONS_DF_INDEX_NAME = list(POLYGONS_DF_INDEX_NAME_AND_TYPE)[0]
POLYGONS_DF_COLS_AND_TYPES = {
    'geometry': shp.geometry,
    'have_img?':
    bool,  # careful: if img hasn't been converted and added might be different from whether there is a connection to that img in the graph.)
    'have_img_downloaded?': bool,
    'download_exception': str,
    'type': str,  # object type
    'visibility': int,
    'timestamp': str,
    'revisit_later?': bool
}
POLYGONS_DF_COLS_AND_INDEX_TYPES = {
    **POLYGONS_DF_INDEX_NAME_AND_TYPE,
    **POLYGONS_DF_COLS_AND_TYPES
}

TASK_FEATURE_CLASSES = ["h", "t"]


def test_connector():

    # Create empty connector
    data_dir = Path(
        "/whatever/"
    )
    connector = Connector.from_scratch(data_dir=data_dir, task_feature_classes=TASK_FEATURE_CLASSES)

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
    # new_vector_features["have_img?"] = False
    # new_vector_features["have_img_downloaded?"] = False
    # new_vector_features["download_exception"] = str(None)
    new_vector_features["some_feature_attribute"] = "foo"
    new_vector_features["type"] = "t"
    # new_vector_features["visibility"] = "1"
    # new_vector_features["timestamp"] = "08/15/1948"
    # new_vector_features["revisit_later?"] = False

    # set crs
    new_vector_features = new_vector_features.set_crs(epsg=STANDARD_CRS_EPSG_CODE)
    """
    Test add_to_vector_features
    """
    # integrate vector_features
    connector.add_to_vector_features(new_vector_features)

    # WHY DOES THIS NOT WORK?
    # assert assert_geodataframe_equal(connector.vector_features, new_vector_features)
    # This does not work either
    #assert connector.vector_features == new_vector_features
    pd.testing.assert_frame_equal(connector.vector_features, new_vector_features)
    """
    Toy img_data
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

    # TODO: clean up
    # # add values for the missing columns 'img_processed?' (bool), and 'orig_crs_epsg_code' (int):
    # new_raster_imgs['img_processed?'] = True
    # new_raster_imgs['orig_crs_epsg_code'] = 4326
    # new_raster_imgs['timestamp'] = "some timestamp, whatever..."
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
    assert connector.have_img_for_feature('polygon1') == True
    assert connector.have_img_for_feature('polygon3') == False
    assert (connector.rectangle_bounding_img('img1')).equals(box(-0.5, -0.5, 6, 6))
    assert list(connector.imgs_containing_feature('polygon1')) == ['img1']
    assert list(connector.features_contained_in_img('img2')) == []
    assert list(
        connector.features_intersecting_img('img2')) == ['polygon1', 'polygon3']
    assert list(
        connector.features_intersecting_img('img1')) == ['polygon1', 'polygon2']

    assert connector.vector_features.loc['polygon1', 'have_img?'] == True
    assert connector.vector_features.loc['polygon2', 'have_img?'] == True
    assert connector.vector_features.loc['polygon3', 'have_img?'] == False
    """
    Integrate another img_df with a row that already exists in connector.img_data 
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

    # # add values for the missing columns 'img_processed?' (bool), and 'orig_crs_epsg_code' (int):
    # new_raster_imgs2['img_processed?'] = True
    # new_raster_imgs2['orig_crs_epsg_code'] = 4326
    # new_raster_imgs2['timestamp'] = "some timestamp, whatever..."
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

    # TODO: remove?
    # # test 'have_img?' column in connector.vector_features
    # assert connector.vector_features.loc['polygon1', 'have_img?'] == True
    # assert connector.vector_features.loc['polygon2', 'have_img?'] == True
    # assert connector.vector_features.loc['polygon3', 'have_img?'] == True
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

    # TODO: clean up!
    # # create the other columns:
    # new_vector_features2["have_img?"] = False
    # new_vector_features2["have_img_downloaded?"] = False
    # new_vector_features2["download_exception"] = str(None)
    # new_vector_features2["type"] = "t"
    # new_vector_features2["visibility"] = "1"
    # new_vector_features2["timestamp"] = "08/15/1948"
    # new_vector_features2["revisit_later?"] = False
    new_vector_features2["some_feature_attribute"] = "What ho, Jeeves!"

    # set crs
    new_vector_features2 = new_vector_features2.set_crs(epsg=STANDARD_CRS_EPSG_CODE)

    # integrate new_vector_features with force_overwrite=True:
    connector.add_to_vector_features(new_vector_features2, force_overwrite=True)

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
    assert len(connector.img_data) == 4
    assert len(connector.vector_features) == 4

    # TODO: remove?
    # # test 'have_img?' column in connector.vector_features
    # assert connector.vector_features.loc['polygon1', 'have_img?'] == True
    # assert connector.vector_features.loc['polygon2', 'have_img?'] == True
    # assert connector.vector_features.loc['polygon3', 'have_img?'] == False
    # assert connector.vector_features.loc['polygon4', 'have_img?'] == True
    """
    Test drop_imgs
    """
    connector.drop_imgs(['img2', 'img3'])

    assert len(connector.img_data) == 2

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
    Test drop_features
    """
    connector.drop_features(['polygon1', 'polygon3'])

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

    """
    Test that drop_imgs modifies the 'have_img?' column in connector.vector_features
    """
    connector.drop_imgs("img4")

    assert len(connector.img_data) == 1

    # TODO: clean up, rewrite comment above?
    # assert connector.vector_features.loc['polygon4', 'have_img?'] == False
