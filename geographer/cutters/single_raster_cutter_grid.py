"""SingleRasterCutter that cuts a raster to a grid of rasters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import rasterio as rio
from affine import Affine
from pydantic import field_validator
from rasterio.windows import Window

from geographer.connector import Connector
from geographer.cutters.single_raster_cutter_base import SingleRasterCutter
from geographer.cutters.type_aliases import RasterSize

logger = logging.getLogger(__name__)


class SingleRasterCutterToGrid(SingleRasterCutter):
    """SingleRasterCutter that cuts a raster into a grid of rasters."""

    new_raster_size: RasterSize

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
    ) -> list[tuple[Window, Affine, str]]:
        source_raster_path = source_connector.rasters_dir / source_raster_name

        with rio.open(source_raster_path) as src:
            if not src.height % self.new_raster_size_rows == 0:
                logger.warning(
                    "number of rows in source raster not divisible by "
                    "number of rows in new rasters"
                )
            if not src.width % self.new_raster_size_cols == 0:
                logger.warning(
                    "number of columns in source raster not divisible \
                        by number of columns in new rasters"
                )

        windows_transforms_raster_names = []

        # Iterate through grid ...
        for i in range(src.width // self.new_raster_size_cols):
            for j in range(src.height // self.new_raster_size_rows):
                # ... remember windows, ...
                window = Window(
                    i * self.new_raster_size_cols,
                    j * self.new_raster_size_rows,
                    width=self.new_raster_size_cols,
                    height=self.new_raster_size_rows,
                )

                # ... transforms ...
                window_transform = src.window_transform(window)

                # ... and raster names.
                new_raster_name = f"{Path(source_raster_name).stem}_{j}_{i}.tif"

                windows_transforms_raster_names.append(
                    (window, window_transform, new_raster_name)
                )

        return windows_transforms_raster_names
