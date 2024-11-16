"""SingleRasterCutter that creates a cutout around a vector feature."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any, Literal, Optional

import rasterio as rio
from affine import Affine
from pydantic import PrivateAttr
from rasterio.io import DatasetReader
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from geographer.connector import Connector
from geographer.cutters.single_raster_cutter_base import SingleRasterCutter
from geographer.cutters.type_aliases import RasterSize
from geographer.utils.utils import transform_shapely_geometry

logger = logging.getLogger(__name__)


class SingleRasterCutterAroundVector(SingleRasterCutter):
    """SingleRasterCutter that creates a cutout around a vector feature.

    SingleRasterCutter that cuts a small raster (or several contiguous
    such rasters if the vector feature does not fit into a single one)
    around each vector feature in the raster accepted by the feature
    filter predicate.
    """

    mode: Literal["random", "centered", "variable"]
    new_raster_size: Optional[RasterSize] = None
    scaling_factor: Optional[float] = 1.2
    min_new_raster_size: Optional[RasterSize] = None
    random_seed: int = 42

    _rows: Optional[int] = PrivateAttr(default=None)  # Do not set by hand.
    _cols: Optional[int] = PrivateAttr(default=None)  # Do not set by hand.

    def __init__(
        self,
        mode: str,
        new_raster_size: RasterSize | None = None,
        scaling_factor: float | None = 1.2,
        min_new_raster_size: RasterSize | None = None,
        random_seed: int = 42,
        **kwargs,
    ) -> None:
        """Initialize SingleRasterCutterAroundVector.

        Args:
            mode: One of 'random', 'centered', 'variable'.
                If 'random' rasters (or minimal raster grids) will be randomly chosen
                subject to constraint that they fully contain the vector features, if
                'centered' will be centered on the vector features. If 'variable' the
                raster size will be the max of some minimum size and a multiple of the
                bounding rectangle of the vector feature. Defaults to 'random'.
            new_raster_size: size (side length of square or rows, cols).
                Only needed if mode is 'centered' or 'random'.
            scaling factor: factor to scale the bounding rectangle
                of the vector feature by to get the raster size.
            min_new_raster_size: minimum size of new raster in 'variable' mode.
            random_seed: random seed. Defaults to 42.

        Raises:
            ValueError: If the mode is unknown.
        """
        super().__init__(
            mode=mode,
            new_raster_size=new_raster_size,
            scaling_factor=scaling_factor,
            min_new_raster_size=min_new_raster_size,
            random_seed=random_seed,
            **kwargs,
        )

        if mode in {"random", "centered"}:
            if new_raster_size is None:
                raise ValueError(
                    f"if mode is {mode}, new_raster_size should not be None"
                )
            self._check_raster_size_type_and_value(new_raster_size)
            self._rows, self._cols = self._get_size_rows_cols(new_raster_size)
        elif mode in {"variable"}:
            if scaling_factor is None:
                raise ValueError(
                    f"if mode is {mode}, scaling_factor should not be None"
                )
            if min_new_raster_size is None:
                raise ValueError(
                    f"if mode is {mode}, min_new_raster_size should not be None"
                )
            self._check_raster_size_type_and_value(min_new_raster_size)
            self._rows, self._cols = self._get_size_rows_cols(min_new_raster_size)
        else:
            logger.error("unknown mode: %s", mode)
            raise ValueError(f"unknown mode: {mode}")

        random.seed(random_seed)

    def _check_raster_size_type_and_value(self, raster_size: RasterSize):
        """Check type and value of arg."""
        if not isinstance(raster_size, int) or (
            isinstance(raster_size, tuple)
            and len(raster_size) == 2
            and all(isinstance(entry, int) for entry in raster_size)
        ):
            raise TypeError(
                "new_raster_size needs to be an integer or a pair of integers!"
            )

        raster_size_rows, raster_size_cols = self._get_size_rows_cols(raster_size)

        if not raster_size_rows > 0:
            logger.error(
                "new_raster_size is %s, need to have positive side length(s)",
                raster_size,
            )
            raise ValueError("{raster_size} needs to have positive side length(s)")
        if not raster_size_cols > 0:
            logger.error(
                "new_raster_size is %s, need to have positive side length(s)",
                raster_size,
            )
            raise ValueError("{raster_size} needs to have positive side length(s)")

    @staticmethod
    def _get_size_rows_cols(
        raster_size: int | tuple[int, int],
    ) -> tuple[int, int]:
        if isinstance(raster_size, tuple):
            new_raster_size_rows = raster_size[0]
            new_raster_size_cols = raster_size[1]
        else:
            new_raster_size_rows = raster_size
            new_raster_size_cols = raster_size

        return new_raster_size_rows, new_raster_size_cols

    def _get_windows_transforms_raster_names(
        self,
        source_raster_name: str,
        source_connector: Connector,
        target_connector: Connector,
        new_rasters_dict: dict | None = None,
        **kwargs: Any,
    ) -> list[tuple[Window, Affine, str]]:
        """Return windwos, transforms, and names of new rasters.

        Given a vector feature and a GeoTiff raster fully containing it
        return a list of windows, window transforms, and new raster_names defining
        a minimal rectangular grid in the raster covering the vector feature.

        Args:
            vector_name: feature identifier
            source_raster_name: name of source raster
            target_connector: connector of target dataset
            new_rasters_dict: dict with keys index or column names of
                target_connector.rasters and values lists of entries correspondong
                to rasters containing information about cut rasters not yet appended
                to target_connector.
            vector_crs_epsg_code: EPSG code of the vector feature crs
            **kwargs: keyword arguments

        Returns:
            list of windows, window_transformations, and new raster names
        """
        if "vector_name" not in kwargs:
            raise ValueError("Need vector feature name")
        vector_name = kwargs["vector_name"]

        source_raster_path = source_connector.rasters_dir / source_raster_name

        vector_geom = target_connector.vectors.loc[vector_name, "geometry"]

        with rio.open(source_raster_path) as src:
            # transform vector feature from connector's crs to raster source crs
            transformed_vector_geom = transform_shapely_geometry(
                vector_geom,
                from_epsg=source_connector.vectors.crs.to_epsg(),
                to_epsg=src.crs.to_epsg(),
            )

            # FOR DEBUGGING:
            raster_bbox = box(*src.bounds)
            assert source_connector.vectors.loc[vector_name].geometry.within(
                source_connector.rasters.loc[source_raster_name].geometry
            )
            if not transformed_vector_geom.within(raster_bbox):
                logger.debug(
                    "raster %s doesn't contain vector feature %s in raster crs",
                    source_raster_name,
                    vector_name,
                )
                transformed_vector_geom = raster_bbox.intersection(
                    transformed_vector_geom
                )

            min_row, max_row, min_col, max_col = self._get_min_max_row_col(
                raster=src, transformed_vector_geom=transformed_vector_geom
            )

            assert min(min_row, max_row, min_col, max_col) >= 0, (
                "nonsensical negative max/min row/col values. "
                "sth went wrong cutting {source_raster_name} for {vector_name}"
            )

            if self.mode in {"centered", "random"}:
                new_raster_size_rows = self._rows
                new_raster_size_cols = self._cols
            elif self.mode == "variable":
                new_raster_size_rows = max(
                    self.scaling_factor * (max_row - min_row),  # type: ignore
                    self._rows,
                )
                new_raster_size_cols = max(
                    self.scaling_factor * (max_col - min_col),  # type: ignore
                    self._cols,
                )

            (
                row_off,
                col_off,
                num_small_rasters_in_row_direction,
                num_small_rasters_in_col_direction,
            ) = self._get_grid_row_col_offsets_num_windows_row_col_direction(
                raster=src,
                new_raster_size_rows=new_raster_size_rows,
                new_raster_size_cols=new_raster_size_cols,
                min_row=min_row,
                max_row=max_row,
                min_col=min_col,
                max_col=max_col,
                transformed_vector_geom=transformed_vector_geom,
            )

            # The row and col offs and number of rasters in row and col direction define
            # a grid. Iterate through the grid and accumulate windows, transforms,
            # and raster_names in a list:

            windows_transforms_raster_names_single_geom = []

            for raster_row in range(num_small_rasters_in_row_direction):
                for raster_col in range(num_small_rasters_in_col_direction):
                    # Define the square window with the calculated offsets.
                    window = rio.windows.Window(
                        col_off=col_off + new_raster_size_cols * raster_col,
                        row_off=row_off + new_raster_size_rows * raster_row,
                        width=new_raster_size_cols,
                        height=new_raster_size_rows,
                    )

                    # Remember the transform for the new geotiff.
                    window_transform = src.window_transform(window)

                    # Generate new raster name.
                    raster_name_no_extension = Path(source_raster_name).stem

                    # (if there is only one window in the grid)
                    if (
                        num_small_rasters_in_row_direction == 1
                        and num_small_rasters_in_col_direction == 1
                    ):
                        new_raster_name = (
                            f"{raster_name_no_extension}_{vector_name}.tif"
                        )
                    else:
                        new_raster_name = (
                            f"{raster_name_no_extension}_{vector_name}_"
                            f"{raster_row}_{raster_col}.tif"
                        )

                    window_bounding_rectangle = box(
                        *rio.windows.bounds(window, src.transform)
                    )

                    # append window if it intersects the vector feature
                    if window_bounding_rectangle.intersects(transformed_vector_geom):
                        windows_transforms_raster_names_single_geom.append(
                            (window, window_transform, new_raster_name)
                        )

            return windows_transforms_raster_names_single_geom

    def _get_min_max_row_col(
        self, raster: DatasetReader, transformed_vector_geom: BaseGeometry
    ) -> tuple[int, int, int, int]:
        """Return bounds of enveloping rectangle of vector feature.

        Bounds returned are min_row, max_row, min_col, max_col.
        """
        # Find min and max row of rectangular envelope of vector feature
        list_of_rectangle_corner_coords = list(
            transformed_vector_geom.envelope.exterior.coords
        )[:5]
        list_of_enveloping_rectangle_row_col_pairs = list(
            map(lambda pair: raster.index(*pair), list_of_rectangle_corner_coords)
        )

        tuple_of_rows_of_rectangle_corners = tuple(
            zip(*list_of_enveloping_rectangle_row_col_pairs)
        )[0]
        tuple_of_cols_of_rectangle_corners = tuple(
            zip(*list_of_enveloping_rectangle_row_col_pairs)
        )[1]

        min_row = min(*tuple_of_rows_of_rectangle_corners)
        max_row = max(*tuple_of_rows_of_rectangle_corners)
        min_col = min(*tuple_of_cols_of_rectangle_corners)
        max_col = max(*tuple_of_cols_of_rectangle_corners)

        return min_row, max_row, min_col, max_col

    def _get_grid_row_col_offsets_num_windows_row_col_direction(
        self,
        raster: DatasetReader,
        transformed_vector_geom: BaseGeometry,
        new_raster_size_rows: int,
        new_raster_size_cols: int,
        min_row: int,
        max_row: int,
        min_col: int,
        max_col: int,
    ) -> tuple[int, int, int, int]:
        """Return row and col offsets and number of windows.

        Return row and col offsets and number of windows in row and in
        col direction such that the resulting grid is minimal grid fully
        covering the vector feature.
        """
        num_small_rasters_in_row_direction = math.ceil(
            float(max_row - min_row) / new_raster_size_rows
        )

        num_small_rasters_in_col_direction = math.ceil(
            float(max_col - min_col) / new_raster_size_cols
        )

        # Choose row and col offset
        if self.mode == "random":
            # ... choose row and col offsets randomly subject to constraint that the
            # grid of raster windows contains rectangular envelope of vector feature.
            row_off = random.randint(
                max(
                    0,
                    max_row - new_raster_size_rows * num_small_rasters_in_row_direction,
                ),
                min(
                    min_row,
                    raster.height
                    - new_raster_size_rows * num_small_rasters_in_row_direction,
                ),
            )

            col_off = random.randint(
                max(
                    0,
                    max_col - new_raster_size_cols * num_small_rasters_in_col_direction,
                ),
                min(
                    min_col,
                    raster.width
                    - new_raster_size_cols * num_small_rasters_in_col_direction,
                ),
            )

        elif self.mode in {"centered", "variable"}:
            # ... to find the row, col offsets to center the vector feature ...

            # ...we first find the centroid of the vector feature in the raster crs ...
            vector_centroid_coords = transformed_vector_geom.envelope.centroid.coords[0]

            # ... extract the row, col of the centroid ...
            centroid_row, centroid_col = raster.index(*vector_centroid_coords)

            # and then choose offsets to center the vector feature.
            row_off = centroid_row - (
                new_raster_size_rows * num_small_rasters_in_row_direction // 2
            )
            col_off = centroid_col - (
                new_raster_size_cols * num_small_rasters_in_col_direction // 2
            )

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return (
            row_off,
            col_off,
            num_small_rasters_in_row_direction,
            num_small_rasters_in_col_direction,
        )
