"""Test get_cutter_every_raster_to_grid.

Test get_cutter_every_raster_to_grid from geographer.cutters.cut_every_raster_to_grid.

TODO: write more assert statements!
- picture resulting grid, check graph is correct,
    write corresponding assert statement for the graph
- test update
- bands
- AOI - 1 is slightly off: CUT_INTO=2: 0.05, CUT_INTO=4: 0.026, CUT_INTO=8: ...
    this is due to funny rounding when bboxes are computed? there is some overlap...
    not sure how to fix this.
"""

import shutil

from shapely.ops import unary_union
from utils import get_test_dir

from geographer.connector import Connector
from geographer.cutters.cut_every_raster_to_grid import get_cutter_every_raster_to_grid
from geographer.cutters.cut_iter_over_rasters import DSCutterIterOverRasters
from geographer.testing.graph_df_compatibility import check_graph_vertices_counts
from geographer.utils.utils import deepcopy_gdf

CUT_INTO = 60  # number of intervals to cut side lengths into


def test_cut_every_raster_to_grid(dummy_cut_source_data_dir):
    """Test get_cutter_every_raster_to_grid."""
    assert 10980 % CUT_INTO == 0

    source_data_dir = dummy_cut_source_data_dir
    target_data_dir = get_test_dir() / "temp/cut_every_raster_to_grid"
    shutil.rmtree(target_data_dir, ignore_errors=True)

    cutter_name = "every_raster_to_grid_cutter"
    cutter = get_cutter_every_raster_to_grid(
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        name=cutter_name,
        new_raster_size=10980 // CUT_INTO,
    )
    cutter.cut()

    source_connector = Connector.from_data_dir(source_data_dir)
    target_connector = Connector.from_data_dir(target_data_dir)

    assert check_graph_vertices_counts(target_connector)
    # count number of small rasters
    assert len(target_connector.rasters) == CUT_INTO * CUT_INTO

    # check union of small rasters is large raster
    large_raster_geom = source_connector.rasters.iloc[0].geometry
    union_small_raster_geoms = unary_union(target_connector.rasters.geometry.tolist())
    intersection = union_small_raster_geoms & large_raster_geom
    union = union_small_raster_geoms | large_raster_geom
    assert abs(intersection.area / union.area - 1) < 0.01

    # assert tempelhofer feld contained in union of rasters intersecting it
    tempelhofer_feld_geom = target_connector.vectors.loc[
        "berlin_tempelhofer_feld"
    ].geometry
    intersecting_geoms = target_connector.rasters.loc[
        target_connector.rasters_intersecting_vector("berlin_tempelhofer_feld")
    ].geometry.tolist()
    assert tempelhofer_feld_geom.within(unary_union(intersecting_geoms))

    # load and recut (shouldn't do anything)
    rasters_before_cutting = deepcopy_gdf(target_connector.rasters)
    cutter = DSCutterIterOverRasters.from_json_file(
        target_connector.connector_dir / f"{cutter_name}.json"
    )
    cutter.cut()

    assert (target_connector.rasters == rasters_before_cutting).all().all()
