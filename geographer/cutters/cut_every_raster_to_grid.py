"""Cut rasters in the source dataset to a grid of rasters.

Dataset cutter that cuts rasters in the source dataset to a grid of
rasters.
"""

import logging
from pathlib import Path
from typing import Optional

from geographer.cutters.cut_iter_over_rasters import DSCutterIterOverRasters
from geographer.cutters.raster_filter_predicates import (
    RasterFilterPredicate,
    RastersNotPreviouslyCutOnly,
)
from geographer.cutters.single_raster_cutter_grid import SingleRasterCutterToGrid
from geographer.cutters.type_aliases import RasterSize

logger = logging.getLogger(__name__)


def get_cutter_every_raster_to_grid(
    source_data_dir: Path,
    target_data_dir: Path,
    name: str,
    new_raster_size: RasterSize = 512,
    raster_filter_predicate: Optional[RasterFilterPredicate] = None,
) -> DSCutterIterOverRasters:
    """Return dataset cutter that cuts every raster to a grid.

    Return dataset cutter that cuts every raster in the source dataset
    to a grid of rasters.

    Args:
        source_data_dir: source data dir
        target_data_dir: target data dir
        name: name of cutter, used when saving the cutter
        new_raster_size: size of new rasters. Defaults to 512.
        raster_filter_predicate: raster filter predicate to select
            rasters. Defaults to None (i.e. cut all
            rasters that have not been previously cut).

    Returns:
        DSCutterIterOverRasters: dataset cutter
    """
    if raster_filter_predicate is None:
        raster_filter_predicate = RastersNotPreviouslyCutOnly()
    raster_cutter = SingleRasterCutterToGrid(new_raster_size=new_raster_size)

    return DSCutterIterOverRasters(
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        name=name,
        raster_cutter=raster_cutter,
        raster_filter_predicate=raster_filter_predicate,
        cut_rasters=[],
    )
