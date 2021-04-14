"""
Simple pytest test suite of rstoolsImgPolygonAssociator using artificial data (i.e. we write a polygons_df and an imgs_df by hand and don't test any methods (e.g. add/download missing images, create labels) that deal with the actual images.)

See img_polygon_associator_artificial_data_test for a visualization of the test data (polygons and images). 
"""

import pytest
import pathlib
from pathlib import Path
import pandas as pd
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
import shapely as shp
from shapely.geometry import Polygon, box



#import img_polygon_associator_rstools as ipa
import rs_tools.img_polygon_associator as ipa



# column and index names for the internal polygons_df geodataframe of the associator
IMGS_DF_INDEX_NAME_AND_TYPE = {'img_name': str}
IMGS_DF_INDEX_NAME = list(IMGS_DF_INDEX_NAME_AND_TYPE.keys())[0]
IMGS_DF_COLS_AND_TYPES = {'geometry': shp.geometry, 
                                    'orig_crs_epsg_code': int, 
                                    'img_processed?': bool, 
                                    'timestamp': str} # needs to be str, since geopandas can't save datetimes
IMGS_DF_COLS_AND_INDEX_TYPES = {**IMGS_DF_INDEX_NAME_AND_TYPE, **IMGS_DF_COLS_AND_TYPES}

# column and index names for the internal imgs_df geodataframe of the associator
POLYGONS_DF_INDEX_NAME_AND_TYPE = {'polygon_name': str}
POLYGONS_DF_INDEX_NAME = list(POLYGONS_DF_INDEX_NAME_AND_TYPE)[0]
POLYGONS_DF_COLS_AND_TYPES = {'geometry': shp.geometry, 
                                        'have_img?': bool, # careful: if img hasn't been converted and added might be different from whether there is a connection to that img in the graph.)
                                        'have_img_downloaded?': bool,
                                        'download_exception': str,
                                        'type': str, # object type
                                        'visibility': int,
                                        'timestamp': str,
                                        'revisit_later?': bool}
POLYGONS_DF_COLS_AND_INDEX_TYPES = {**POLYGONS_DF_INDEX_NAME_AND_TYPE, **POLYGONS_DF_COLS_AND_TYPES}

SEGMENTATION_CLASSES=["h", "t"]




def test_img_polygon_associator_rstools():

    # Create associator from empty polygons_df, imgs_df, graph:
    empty_polygons_df = ipa.empty_polygons_df(polygons_df_index_name=POLYGONS_DF_INDEX_NAME, 
                                    polygons_df_cols_and_index_types=POLYGONS_DF_COLS_AND_INDEX_TYPES)
    empty_imgs_df = ipa.empty_imgs_df(imgs_df_index_name=IMGS_DF_INDEX_NAME,
                            imgs_df_cols_and_index_types=IMGS_DF_COLS_AND_INDEX_TYPES)
    data_dir = Path("/whatever/") # We need a data_dir argument to create the associator, but the actual value is irrelevant since we're dealing with artificial data and writing or reading anythin from disk. 

    assoc = ipa.ImgPolygonAssociator(data_dir=data_dir, 
                                            polygons_df=empty_polygons_df, 
                                            imgs_df=empty_imgs_df, 
                                            segmentation_classes=SEGMENTATION_CLASSES)


    """
    Toy polygons_df
    """
    # create empty GeoDataFrame with the right index name
    new_polygons_df = gpd.GeoDataFrame()
    new_polygons_df.rename_axis('polygon_name', inplace=True)

    # polygon names and geometries
    polygon1 = Polygon([(0,0),(0,1),(1,1),(1,0),(0,0)])
    polygon2 = box(4,4,5,5)
    polygon3 = box(-2,-2,-1,-1)

    # add the polygon names and geometries to the geodataframe
    for p_name, p_geom in zip(['polygon1', 'polygon2', 'polygon3'], [polygon1, polygon2, polygon3]):
        new_polygons_df.loc[p_name, 'geometry'] = p_geom

    # create the other columns:
    new_polygons_df["have_img?"] = False
    new_polygons_df["have_img_downloaded?"] = False
    new_polygons_df["download_exception"] = str(None)
    new_polygons_df["type"] = "t"
    new_polygons_df["visibility"] = "1"
    new_polygons_df["timestamp"] = "08/15/1948"
    new_polygons_df["revisit_later?"] = False

    # set crs
    new_polygons_df = new_polygons_df.set_crs(epsg=4326)


    """
    Test integrate_new_polygons_df
    """
    # integrate polygons_df
    assoc.integrate_new_polygons_df(new_polygons_df)

    # WHY DOES THIS NOT WORK?
    # assert assert_geodataframe_equal(assoc.polygons_df, new_polygons_df)
    # This does not work either
    #assert assoc.polygons_df == new_polygons_df
    pd.testing.assert_frame_equal(assoc.polygons_df, new_polygons_df)  
  

    """
    Toy imgs_df
    """

    # empty GeoDataFrame with right index name
    new_imgs_df = gpd.GeoDataFrame()
    new_imgs_df.rename_axis('img_name', inplace=True)

    # the geometries will be the img bounding rectangles

    bounding_rectangle1 = box(-0.5, -0.5, 6, 6) # contains both p1 and p2 (to be defined later, see below), doesn't intersect p3
    bounding_rectangle2 = box(-1.5, -1.5, 0.5, 0.5) # has non-empty intersection with p1 and p3, but does not contain either, no intersection with p2

    # add them to new_imgs_df
    for img_name, bounding_rectangle in zip(['img1', 'img2'], [bounding_rectangle1, bounding_rectangle2]): 
        new_imgs_df.loc[img_name, 'geometry'] = bounding_rectangle

    # add values for the missing columns 'img_processed?' (bool), and 'orig_crs_epsg_code' (int):
    new_imgs_df['img_processed?'] = True
    new_imgs_df['orig_crs_epsg_code'] = 4326
    new_imgs_df['timestamp'] = "some timestamp, whatever..."

    # set crs
    new_imgs_df = new_imgs_df.set_crs(epsg=4326)


    """
    Test integrate_new_imgs_df
    """
    assoc.integrate_new_imgs_df(new_imgs_df)

    assert assoc._graph._graph_dict == {"polygons": {
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
                                            "imgs": {
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
    Test have_img_for_polygon, rectangle_bounding_img, polygons_intersecting_img, polygons_contained_in_img, imgs_containing_polygon, values of 'have_img?' column in assoc.polygons_df.
    """
    assert assoc.have_img_for_polygon('polygon1') == True
    assert assoc.have_img_for_polygon('polygon3') == False
    assert (assoc.rectangle_bounding_img('img1')).equals(box(-0.5, -0.5, 6, 6))
    assert list(assoc.imgs_containing_polygon('polygon1')) == ['img1']
    assert list(assoc.polygons_contained_in_img('img2')) == []
    assert list(assoc.polygons_intersecting_img('img2')) == ['polygon1', 'polygon3']
    assert list(assoc.polygons_intersecting_img('img1')) == ['polygon1', 'polygon2']
    
    assert assoc.polygons_df.loc['polygon1', 'have_img?'] == True
    assert assoc.polygons_df.loc['polygon2', 'have_img?'] == True
    assert assoc.polygons_df.loc['polygon3', 'have_img?'] == False


    """
    Integrate another img_df with a row that already exists in assoc.imgs_df 
    """
    # empty GeoDataFrame with right index name
    new_imgs_df2 = gpd.GeoDataFrame()
    new_imgs_df2.rename_axis('img_name', inplace=True)

    # the new_imgs_df2 geometries will be the img bounding rectangles here:
    bounding_rectangle1 = box(-10, -10, 10, 10) # contains all polygons, but an polygon with that name already exists in associator, so should be ignored by integrate_new_polygons_df
    bounding_rectangle3 = box(-3, -3, 7, 7) # contains all of p1, p2, p3, p4 (to be defined below)
    bounding_rectangle4 = box(-1.5, -1.5, 2, 2) # contains p1 and p4 (to be defined below), has non-empty intersection with p3, but does not intersect p2

    # add them to new_imgs_df2
    for img_name, bounding_rectangle in zip(['img1', 'img3', 'img4'], [bounding_rectangle1, bounding_rectangle3, bounding_rectangle4]): 
        new_imgs_df2.loc[img_name, 'geometry'] = bounding_rectangle

    # add values for the missing columns 'img_processed?' (bool), and 'orig_crs_epsg_code' (int):
    new_imgs_df2['img_processed?'] = True
    new_imgs_df2['orig_crs_epsg_code'] = 4326
    new_imgs_df2['timestamp'] = "some timestamp, whatever..."

    # set crs
    new_imgs_df2 = new_imgs_df2.set_crs(epsg=4326)

    # integrate new_imgs_df2
    assoc.integrate_new_imgs_df(new_imgs_df2)

    # test containment/intersection relations, i.e. graph structure
    assert assoc._graph._graph_dict == {"polygons": {
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
                                                "imgs": {
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

    # test 'have_img?' column in assoc.polygons_df
    assert assoc.polygons_df.loc['polygon1', 'have_img?'] == True
    assert assoc.polygons_df.loc['polygon2', 'have_img?'] == True
    assert assoc.polygons_df.loc['polygon3', 'have_img?'] == True


    """
    Integrate another polygons_df with a row that already exists in the associator's polygons_df
    """

    # create empty GeoDataFrame with the right index name
    new_polygons_df2 = gpd.GeoDataFrame()
    new_polygons_df2.rename_axis('polygon_name', inplace=True)

    # polygon names and geometries
    polygon4 = box(-1, -1, 0, 0) # genuinely new entry
    polygon3 = box(-100,-100,-99,-99) # entry for polygon3 already exists in assoc.polygons_df 

    # add the polygon names and geometries to the geodataframe
    for p_name, p_geom in zip(['polygon3', 'polygon4'], [polygon3, polygon4]):
        new_polygons_df2.loc[p_name, 'geometry'] = p_geom

    # create the other columns:
    new_polygons_df2["have_img?"] = False
    new_polygons_df2["have_img_downloaded?"] = False
    new_polygons_df2["download_exception"] = str(None)
    new_polygons_df2["type"] = "t"
    new_polygons_df2["visibility"] = "1"
    new_polygons_df2["timestamp"] = "08/15/1948"
    new_polygons_df2["revisit_later?"] = False

    # set crs
    new_polygons_df2 = new_polygons_df2.set_crs(epsg=4326)

    # integrate new_polygons_df with force_overwrite=True:
    assoc.integrate_new_polygons_df(new_polygons_df2, force_overwrite=True)

    assert assoc._graph._graph_dict == {"polygons": {
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
                                            "imgs": {
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
    assert len(assoc.imgs_df) == 4
    assert len(assoc.polygons_df) == 4

    # test 'have_img?' column in assoc.polygons_df
    assert assoc.polygons_df.loc['polygon1', 'have_img?'] == True
    assert assoc.polygons_df.loc['polygon2', 'have_img?'] == True
    assert assoc.polygons_df.loc['polygon3', 'have_img?'] == False
    assert assoc.polygons_df.loc['polygon4', 'have_img?'] == True


    """
    Test drop_imgs
    """
    assoc.drop_imgs(['img2', 'img3'])

    assert len(assoc.imgs_df) == 2

    assert assoc._graph._graph_dict == {"polygons": {
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
                                                "imgs": {
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
    Test drop_polygons
    """
    assoc.drop_polygons(['polygon1', 'polygon3'])

    assert len(assoc.polygons_df) == 2

    assert assoc._graph._graph_dict == {"polygons": {
                                                "polygon2": {
                                                    "img1": "contains"
                                                },
                                                "polygon4": {
                                                    "img1": "intersects",
                                                    "img4": "contains"
                                                }
                                            },
                                            "imgs": {
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
    Test that drop_imgs modifies the 'have_img?' column in assoc.polygons_df
    """
    assoc.drop_imgs("img4")

    assert len(assoc.imgs_df) == 1

    assert assoc.polygons_df.loc['polygon4', 'have_img?'] == False
