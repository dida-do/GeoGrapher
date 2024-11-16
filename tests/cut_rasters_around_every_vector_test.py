"""Test get_cutter_rasters_around_every_vector.

Test get_cutter_rasters_around_every_vector from
geographer.cutters.cut_rasters_around_every_vector.

TODO: rename source dataset or create separate source dataset
TODO: write more assert statements!
- test update
- test with target_raster_count > 1
- check union / intersections
- bands: unit test
"""

import shutil

from shapely.geometry import Polygon
from shapely.ops import unary_union
from utils import get_test_dir

from geographer.connector import Connector
from geographer.cutters.cut_iter_over_vectors import DSCutterIterOverVectors
from geographer.cutters.cut_rasters_around_every_vector import (
    get_cutter_rasters_around_every_vector,
)
from geographer.cutters.raster_selectors import RandomRasterSelector
from geographer.cutters.single_raster_cutter_around_vector import (
    SingleRasterCutterAroundVector,
)
from geographer.cutters.vector_filter_predicates import IsVectorMissingRasters
from geographer.testing.graph_df_compatibility import check_graph_vertices_counts

IMG_SIZE = 128


def test_rasters_around_every_vector(dummy_cut_source_data_dir):
    """Test get_cutter_rasters_around_every_vector."""
    source_data_dir = dummy_cut_source_data_dir
    target_data_dir = get_test_dir() / "temp/rasters_around_every_vector"
    shutil.rmtree(target_data_dir, ignore_errors=True)

    cutter = get_cutter_rasters_around_every_vector(
        target_raster_count=1,
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        name="every_raster_to_grid_cutter",
        new_raster_size=IMG_SIZE,
    )
    cutter.cut()

    # target_raster_count=2 (but only 1 rasters in source)
    json_path = cutter.target_connector.connector_dir / f"{cutter.name}.json"
    cutter = DSCutterIterOverVectors.from_json_file(
        json_path,
        constructor_symbol_table={
            "DSCutterIterOverVectors": DSCutterIterOverVectors,
            "IsVectorMissingRasters": IsVectorMissingRasters,
            "RandomRasterSelector": RandomRasterSelector,
            "SingleRasterCutterAroundVector": SingleRasterCutterAroundVector,
        },
    )
    cutter.raster_selector = RandomRasterSelector(target_raster_count=2)
    cutter.cut()

    target_connector = Connector.from_data_dir(target_data_dir)

    assert check_graph_vertices_counts(target_connector)

    assert target_connector._graph._graph_dict == {
        "vectors": {
            "berlin_brandenburg_gate": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_brandenburg_gate.tif": "contains"  # noqa: E501
            },
            "berlin_reichstag": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_brandenburg_gate.tif": "contains"  # noqa: E501
            },
            "lisbon_praca_do_comercio": {},
            "lisbon_castelo_de_sao_jorge": {},
            "berlin_platz_des_18_maerz": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_brandenburg_gate.tif": "contains"  # noqa: E501
            },
            "berlin_tiergarten": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tiergarten.tif": "contains"  # noqa: E501
            },
            "berlin_brandenburg_gate_square": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_brandenburg_gate.tif": "contains"  # noqa: E501
            },
            "berlin_tempelhofer_feld": {
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_0_0.tif": "intersects",  # noqa: E501
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_0_1.tif": "intersects",  # noqa: E501
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_1_0.tif": "intersects",  # noqa: E501
                "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_1_1.tif": "intersects",  # noqa: E501
            },
        },
        "rasters": {
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_brandenburg_gate.tif": {  # noqa: E501
                "berlin_brandenburg_gate_square": "contains",
                "berlin_brandenburg_gate": "contains",
                "berlin_reichstag": "contains",
                "berlin_platz_des_18_maerz": "contains",
            },
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tiergarten.tif": {  # noqa: E501
                "berlin_tiergarten": "contains"
            },
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_0_0.tif": {  # noqa: E501
                "berlin_tempelhofer_feld": "intersects"
            },
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_0_1.tif": {  # noqa: E501
                "berlin_tempelhofer_feld": "intersects"
            },
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_1_0.tif": {  # noqa: E501
                "berlin_tempelhofer_feld": "intersects"
            },
            "S2A_MSIL2A_20220309T100841_N0400_R022_T32UQD_20220309T121849_berlin_tempelhofer_feld_1_1.tif": {  # noqa: E501
                "berlin_tempelhofer_feld": "intersects"
            },
        },
    }

    # make sure tempelhofer feld is contained in union of rasters intersecting it
    tempelhofer_feld: Polygon = target_connector.vectors.loc[
        "berlin_tempelhofer_feld"
    ].geometry
    rasters_intersecting_tempelhofer_feld: list[str] = (
        target_connector.rasters_intersecting_vector("berlin_tempelhofer_feld")
    )
    bboxes = target_connector.rasters.geometry.loc[
        rasters_intersecting_tempelhofer_feld
    ].tolist()
    union_rasters: Polygon = unary_union(bboxes)
    assert tempelhofer_feld.within(union_rasters)
