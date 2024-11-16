"""SingleRasterCutter that extracts pre defined bboxes from a raster."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio as rio
from geopandas import GeoDataFrame
from pydantic import PrivateAttr, field_validator
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from geographer.connector import Connector
from geographer.cutters.single_raster_cutter_base import SingleRasterCutter
from geographer.cutters.type_aliases import RasterSize

logger = logging.getLogger(__name__)


def _correct_window_offset(
    offset: int | float, size: int | float, new_size: int
) -> int:
    center = offset + size / 2
    return int(center - new_size / 2)


class SingleRasterCutterFromBBoxes(SingleRasterCutter):
    """SingleRasterCutter that extracts pre defined bboxes from a raster.

    The new size of the rasters must be specified as it is used to
    ensure a standardised output.
    """

    new_raster_size: RasterSize
    bbox_geojson_path: Path

    _bboxes_df: GeoDataFrame = PrivateAttr()

    def __init__(self, **data) -> None:
        """Initialize a SingleRasterCutterFromBBoxes.

        Args:
            new_raster_size: size of new raster
            bbox_geojson_path: path to geojson file containing the bboxes
        """
        super().__init__(**data)
        self._bboxes_df = gpd.read_file(self.bbox_geojson_path)

    @field_validator("bbox_geojson_path")
    def path_points_to_geojson(cls, value: Path):
        """Validate path exists and points to geojson."""
        if value.suffix != ".geojson":
            raise ValueError("Path should point to .geojson file")
        if not value.is_file():
            raise FileNotFoundError(f".geojson file does not exist: {value}")
        return value

    @field_validator("new_raster_size")
    def new_raster_size_type_correctness(cls, value: RasterSize) -> RasterSize:
        """Validate new_raster_size has correct type."""
        is_int: bool = isinstance(value, int)
        is_pair_of_ints: bool = (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(entry, int) for entry in value)
        )
        if not (is_int or is_pair_of_ints):
            raise TypeError(
                "new_raster_size needs to be an integer or a pair of integers!"
            )
        return value

    @field_validator("new_raster_size")
    def new_raster_size_side_lengths_must_be_positive(
        cls, value: RasterSize
    ) -> RasterSize:
        """Validate new_raster_size side lengths are positive."""
        if isinstance(value, tuple) and not all(val > 0 for val in value):
            logger.error("new_raster_size: need positive side length(s)")
            raise ValueError("new_raster_size: need positive side length(s)")
        elif isinstance(value, int) and value <= 0:
            logger.error("new_raster_size: need positive side length(s)")
            raise ValueError("new_raster_size: need positive side length(s)")
        return value

    @property
    def new_raster_size_rows(self) -> int:
        """Return number of rows of new raster size."""
        if isinstance(self.new_raster_size, tuple):
            return self.new_raster_size[0]
        else:
            return self.new_raster_size

    @property
    def new_raster_size_cols(self) -> int:
        """Return number of columns of new raster size."""
        if isinstance(self.new_raster_size, tuple):
            return self.new_raster_size[1]
        else:
            return self.new_raster_size

    def _get_windows_transforms_raster_names(
        self,
        source_raster_name: str,
        source_connector: Connector,
        target_connector: Connector | None = None,
        new_rasters_dict: dict | None = None,
        **kwargs: Any,
    ) -> list[str]:
        source_raster_path = source_connector.rasters_dir / source_raster_name

        with rio.open(source_raster_path) as src:
            raster_bounds = box(*src.bounds)
            bounding_boxes = self.bounding_boxes.to_crs(src.crs)
            bounding_boxes = bounding_boxes.loc[
                bounding_boxes.geometry.within(raster_bounds)
            ]
            windows_transforms_raster_names = []
            for i, geometry in enumerate(bounding_boxes.geometry):
                initial_window = from_bounds(*geometry.bounds, src.transform)

                new_col_off = _correct_window_offset(
                    initial_window.col_off,
                    initial_window.width,
                    self.new_raster_size_cols,
                )

                new_row_off = _correct_window_offset(
                    initial_window.row_off,
                    initial_window.height,
                    self.new_raster_size_rows,
                )

                window = Window(
                    new_col_off,
                    new_row_off,
                    self.new_raster_size_cols,
                    self.new_raster_size_rows,
                )

                window_transform = src.window_transform(window)
                new_raster_name = f"{Path(source_raster_name).stem}_{i}.tif"

                windows_transforms_raster_names.append(
                    (window, window_transform, new_raster_name)
                )

        return windows_transforms_raster_names
