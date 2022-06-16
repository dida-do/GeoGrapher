"""
TODO: write more assert statements!
- picture resulting grid, check graph is correct, write corresponding assert statement for the graph
- test update
DONE - check union / intersections
- bands
- tempelhof contained in union of imgs intersecting it

"""

from pathlib import Path
import shutil
from shapely.ops import unary_union
from geographer.connector import Connector
from geographer.creator_from_source_dataset_base import DSCreatorFromSource
from geographer.cutters.cut_every_img_to_grid import get_cutter_every_img_to_grid
from geographer.cutters.cut_iter_over_imgs import DSCutterIterOverImgs
from geographer.cutters.img_filter_predicates import ImgsNotPreviouslyCutOnly
from geographer.cutters.single_img_cutter_grid import SingleImgCutterToGrid
from geographer.testing.graph_df_compatibility import check_graph_vertices_counts
from geographer.utils.utils import create_kml_all_geodataframes, deepcopy_gdf
from tests.utils import get_test_dir

CUT_INTO = 60 # 36 or 30?
CUT_SOURCE_DATA_DIR_NAME = 'cut_source'

def test_cut_every_img_to_grid():

    assert 10980 % CUT_INTO == 0

    source_data_dir=get_test_dir() / CUT_SOURCE_DATA_DIR_NAME
    target_data_dir=get_test_dir() / 'temp/cut_every_img_to_grid'
    shutil.rmtree(target_data_dir, ignore_errors=True)

    cutter_name='every_img_to_grid_cutter'
    cutter = get_cutter_every_img_to_grid(
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        name=cutter_name,
        new_img_size= 10980 // CUT_INTO
    )
    cutter.cut()

    source_connector = Connector.from_data_dir(source_data_dir)
    target_connector = Connector.from_data_dir(target_data_dir)

    assert check_graph_vertices_counts(target_connector)
    # count number of small images
    assert len(target_connector.raster_imgs) == CUT_INTO*CUT_INTO

    # check union of small images is large image
    large_img_geom = source_connector.raster_imgs.iloc[0].geometry
    union_small_img_geoms = unary_union(target_connector.raster_imgs.geometry.tolist())
    intersection = union_small_img_geoms & large_img_geom
    union = union_small_img_geoms | large_img_geom
    assert abs(intersection.area / union.area - 1) < 0.01

    # assert tempelhofer feld contained in union of images intersecting it
    tempelhofer_feld_geom = target_connector.vector_features.loc["berlin_tempelhofer_feld"].geometry
    intersecting_geoms = target_connector.raster_imgs.loc[target_connector.imgs_intersecting_vector_feature("berlin_tempelhofer_feld")].geometry.tolist()
    assert tempelhofer_feld_geom.within(unary_union(intersecting_geoms))

    # load and recut (shouldn't do anything)
    raster_imgs_before_cutting = deepcopy_gdf(target_connector.raster_imgs)
    cutter = DSCutterIterOverImgs.from_json_file(
        target_connector.connector_dir / f"{cutter_name}.json")
    cutter.cut()

    assert (target_connector.raster_imgs == raster_imgs_before_cutting).all().all()

if __name__ == "__main__":
    test_cut_every_img_to_grid()