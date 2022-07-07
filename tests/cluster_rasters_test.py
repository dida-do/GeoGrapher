from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, box
from geographer import Connector
from geographer.utils.cluster_rasters import get_raster_clusters
from geographer.global_constants import RASTER_IMGS_INDEX_NAME, STANDARD_CRS_EPSG_CODE, VECTOR_FEATURES_INDEX_NAME
from geographer.graph.bipartite_graph_mixin import RASTER_IMGS_COLOR, VECTOR_FEATURES_COLOR
from tests.mock_download_test import MOCK_DOWNLOAD_SOURCE_DATA_DIR
from tests.utils import get_test_dir


def test_cluster_rasters():

    # Create empty connector
    data_dir = Path("/whatever/")
    connector = Connector.from_scratch(
        data_dir=data_dir) #, task_feature_classes=['class1', 'class2'])

    """
    Create vector_features
    """
    new_vector_features = gpd.GeoDataFrame()
    new_vector_features.rename_axis(VECTOR_FEATURES_INDEX_NAME, inplace=True)

    # polygon names and geometries
    polygon1 = Polygon([(1, 1), (1, -1), (-1, -1), (-1, 1), (1,1)]) # intersects both img1 and img2 below
    polygon2 = box(3, 3, 3.5, 3.5) # contained in img1 and img3
    # polygon3 = box(-2, -2, -1, -1)

    # add the polygon names and geometries to the geodataframe
    for p_name, p_geom in zip(['polygon1', 'polygon2'],
                              [polygon1, polygon2]):
        new_vector_features.loc[p_name, 'geometry'] = p_geom

    new_vector_features = new_vector_features.set_crs(
        epsg=STANDARD_CRS_EPSG_CODE)
    connector.add_to_vector_features(new_vector_features)

    """
    Create raster_imgs
    """
    new_raster_imgs = gpd.GeoDataFrame()
    new_raster_imgs.rename_axis(RASTER_IMGS_INDEX_NAME, inplace=True)

    # geometries (img bounding rectangles)
    # images 1 and 2 dont intersect but are close.
    # both intersect polygon1
    # image 2 contains polygon2
    bounding_rectangle1 = box(
        -4, 0, 4, 8
    )
    bounding_rectangle2 = box(
        -4, -0.5, 4, -8.5
    )

    # intersects image 2, contains polygon 2
    bounding_rectangle3 = box(
        2, 2, 10, 10
    )

    # 4, 5, 6 form a chain:
    # 5 intersects both 4 and 6
    # 4 and 6 do not intersect
    bounding_rectangle4 = box(
        100, 100, 108, 108
    )
    bounding_rectangle5 = box(
        106, 106, 114, 114
    )
    bounding_rectangle6 = box(
        112, 112, 120, 120
    )

    # add to new_raster_imgs
    for img_name, bounding_rectangle in zip(
            ['img1', 'img2', 'img3', 'img4', 'img5', 'img6'],
            [bounding_rectangle1, bounding_rectangle2, bounding_rectangle2,\
             bounding_rectangle4, bounding_rectangle5, bounding_rectangle6]
        ):
        new_raster_imgs.loc[img_name, 'geometry'] = bounding_rectangle

    new_raster_imgs = new_raster_imgs.set_crs(epsg=STANDARD_CRS_EPSG_CODE)
    connector.add_to_raster_imgs(new_raster_imgs)

    """
    Test clusters defined by 'rasters_that_share_vector_features_or_overlap'
    """
    clusters = get_raster_clusters(
        connector=connector,
        clusters_defined_by='rasters_that_share_vector_features_or_overlap',
        preclustering_method='y then x-axis')
    assert set(map(frozenset, clusters)) == {
        frozenset({'img3', 'img1', 'img2'}),
        frozenset({'img4', 'img6', 'img5'})
    }

    """
    Test clusters defined by 'rasters_that_share_vector_features'
    """
    clusters = get_raster_clusters(
        connector=connector,
        clusters_defined_by='rasters_that_share_vector_features',
        preclustering_method='y then x-axis')
    assert set(map(frozenset, clusters)) == {
        frozenset({'img3', 'img1', 'img2'}),
        frozenset({'img4'}),
        frozenset({'img5'}),
        frozenset({'img6'}),
    }

if __name__ == "__main__":
    test_cluster_rasters()
