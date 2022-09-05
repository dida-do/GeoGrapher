"""
TODO: rename source dataset or create separate source dataset
TODO: write more assert statements!
- test update
- test with target_img_count > 1
- check union / intersections
- bands: unit test
"""

from pathlib import Path
import shutil
from typing import List
from shapely.geometry import Polygon
from shapely.ops import unary_union
from geographer.connector import Connector
from geographer.cutters.cut_imgs_around_every_feature import get_cutter_imgs_around_every_feature #, DSCutterImgsAroundEveryFeature, 
from geographer.cutters.cut_iter_over_features import DSCutterIterOverFeatures
from geographer.cutters.feature_filter_predicates import IsFeatureMissingImgs
from geographer.cutters.img_selectors import RandomImgSelector
from geographer.cutters.single_img_cutter_around_feature import SingleImgCutterAroundFeature
from geographer.testing.graph_df_compatibility import check_graph_vertices_counts
from cut_every_img_to_grid_test import CUT_SOURCE_DATA_DIR_NAME
from utils import get_test_dir

IMG_SIZE = 128

def test_imgs_around_every_feature():

    source_data_dir=get_test_dir() / CUT_SOURCE_DATA_DIR_NAME
    target_data_dir=get_test_dir() / 'temp/imgs_around_every_feature'
    shutil.rmtree(target_data_dir, ignore_errors=True)

    # TODO: remove
    shutil.rmtree(target_data_dir, ignore_errors=True)

    cutter = get_cutter_imgs_around_every_feature(
        target_img_count=1,
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        name='every_img_to_grid_cutter',
        new_img_size=IMG_SIZE,
    )
    # cutter = DSCutterImgsAroundEveryFeature(
    #     target_img_count=1,
    #     source_data_dir=source_data_dir,
    #     target_data_dir=target_data_dir,
    #     name='every_img_to_grid_cutter',
    #     new_img_size=IMG_SIZE,
    # )
    cutter.cut()

    # target_img_count=2 (but only 1 images in source)
    json_path = cutter.target_connector.connector_dir / f"{cutter.name}.json"
    cutter = DSCutterIterOverFeatures.from_json_file(
        json_path,
        constructor_symbol_table={
            "DSCutterIterOverFeatures": DSCutterIterOverFeatures,
            "IsFeatureMissingImgs": IsFeatureMissingImgs,
            "RandomImgSelector": RandomImgSelector,
            "SingleImgCutterAroundFeature": SingleImgCutterAroundFeature,
        })
    cutter.img_selector = RandomImgSelector(target_img_count=2)
    cutter.cut()

    source_connector = Connector.from_data_dir(source_data_dir)
    target_connector = Connector.from_data_dir(target_data_dir)

    assert check_graph_vertices_counts(target_connector)

    assert target_connector._graph._graph_dict == {
        "vector_features": {
            "berlin_brandenburg_gate": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_brandenburg_gate.tif": "contains"
            },
            "berlin_reichstag": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_brandenburg_gate.tif": "contains"
            },
            "lisbon_praca_do_comercio": {},
            "lisbon_castelo_de_sao_jorge": {},
            "berlin_platz_des_18_maerz": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_brandenburg_gate.tif": "contains"
            },
            "berlin_tiergarten": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tiergarten.tif": "contains"
            },
            "berlin_brandenburg_gate_square": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_brandenburg_gate.tif": "contains"
            },
            "berlin_tempelhofer_feld": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_0_0.tif": "intersects",
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_0_1.tif": "intersects",
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_1_0.tif": "intersects",
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_1_1.tif": "intersects"
            }
        },
        "raster_imgs": {
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_brandenburg_gate.tif": {
                "berlin_brandenburg_gate_square": "contains",
                "berlin_brandenburg_gate": "contains",
                "berlin_reichstag": "contains",
                "berlin_platz_des_18_maerz": "contains"
            },
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tiergarten.tif": {
                "berlin_tiergarten": "contains"
            },
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_0_0.tif": {
                "berlin_tempelhofer_feld": "intersects"
            },
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_0_1.tif": {
                "berlin_tempelhofer_feld": "intersects"
            },
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_1_0.tif": {
                "berlin_tempelhofer_feld": "intersects"
            },
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_1_1.tif": {
                "berlin_tempelhofer_feld": "intersects"
            }
        }
    }

    # make sure tempelhofer feld is contained in union of images intersecting it
    tempelhofer_feld: Polygon = target_connector.vector_features.loc["berlin_tempelhofer_feld"].geometry
    imgs_intersecting_tempelhofer_feld: List[str] = target_connector.imgs_intersecting_vector_feature("berlin_tempelhofer_feld")
    bboxes = target_connector.raster_imgs.geometry.loc[imgs_intersecting_tempelhofer_feld].tolist()
    union_imgs: Polygon = unary_union(bboxes)
    assert tempelhofer_feld.within (union_imgs)

if __name__ == "__main__":
    test_imgs_around_every_feature()