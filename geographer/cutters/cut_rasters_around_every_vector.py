"""Dataset cutter that cuts out rasters around vector features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from geographer.cutters.cut_iter_over_vectors import DSCutterIterOverVectors
from geographer.cutters.raster_selectors import RandomRasterSelector, RasterSelector
from geographer.cutters.single_raster_cutter_around_vector import (
    SingleRasterCutterAroundVector,
)
from geographer.cutters.type_aliases import RasterSize
from geographer.cutters.vector_filter_predicates import (
    IsVectorMissingRasters,
    VectorFilterPredicate,
)

logger = logging.getLogger(__name__)


def get_cutter_rasters_around_every_vector(
    source_data_dir: Path | str,
    target_data_dir: Path | str,
    name: str,
    mode: Literal["random", "centered", "variable"] = "random",
    new_raster_size: RasterSize | None = 512,
    min_new_raster_size: RasterSize | None = None,
    scaling_factor: float | None = None,
    target_raster_count: int = 1,
    bands: dict | None = None,
    random_seed: int = 10,
) -> DSCutterIterOverVectors:
    """Return dataset cutter that creates cutouts around vector features.

    In "random" or "centered" all cutouts will be of the same size.
    In "variable" mode the size of a cutout is the maximum of
    scaling_factor*size_of_bounding_box or min_new_raster_size

    Args:
        source_data_dir: source data dir
        target_data_dir: target data dir
        name: name
        new_raster_size: new raster size for
            "random" and "centered" modes. Defaults to 512.
        min_new_raster_size: lower bound on raster
            size for "variable" mode. Defaults to None.
        scaling_factor: scaling factor for
            "variable" mode. Defaults to None.
        target_raster_count: targeted number of rasters per vector
            feature to create. Defaults to 1.
        mode: On. Defaults to "random".
        bands: bands dict. Defaults to None.
        random_seed: random seed. Defaults to 10.

    Returns:
        DSCutterIterOverVectors: dataset cutter
    """
    is_vector_missing_rasters: VectorFilterPredicate = IsVectorMissingRasters(
        target_raster_count=target_raster_count
    )
    random_raster_selector: RasterSelector = RandomRasterSelector(
        target_raster_count=target_raster_count
    )
    small_rasters_around_vectors_cutter = SingleRasterCutterAroundVector(
        mode=mode,
        new_raster_size=new_raster_size,
        min_new_raster_size=min_new_raster_size,
        scaling_factor=scaling_factor,
        random_seed=random_seed,
    )

    return DSCutterIterOverVectors(
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        name=name,
        vector_filter_predicate=is_vector_missing_rasters,
        raster_selector=random_raster_selector,
        raster_cutter=small_rasters_around_vectors_cutter,
        bands=bands,
    )
