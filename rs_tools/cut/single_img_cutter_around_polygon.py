"""
ImgsAroundPolygonCutter - SingleImgCutter that creates a cutout around a
given polygon from a source image.
"""

import logging
import math
import random
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union
from pydantic import Field

from affine import Affine
import rasterio as rio
from rasterio.io import DatasetReader
from rasterio.windows import Window
from shapely.geometry import Polygon, box

from rs_tools.cut.type_aliases import ImgSize
from rs_tools.cut.single_img_cutter_base import SingleImgCutterBase
from rs_tools.utils.utils import transform_shapely_geometry
from rs_tools.img_polygon_associator import ImgPolygonAssociator

logger = logging.getLogger(__name__)


class ImgsAroundPolygonCutter(SingleImgCutterBase):
    """SingleImgCutter that cuts a small image (or several contiguous such
    images if the polygon does not fit into a single one) around each polygon
    in the image accepted by the polygon filter predicate."""

    mode: Literal["random", "centered", "variable"]
    new_img_size: Optional[ImgSize] = None
    scaling_factor: Optional[float] = 1.2
    min_new_img_size: Optional[ImgSize] = None
    img_bands: Optional[List[int]] = None
    label_bands: Optional[List[int]] = None
    random_seed: int = 42

    _rows: Optional[int] = Field(
        None, description="Do not set by hand. If None, will be inferred.")
    _cols: Optional[int] = Field(
        None, description="Do not set by hand. If None, will be inferred.")

    def __init__(
        self,
        mode: str,
        new_img_size: Optional[ImgSize] = None,
        scaling_factor: Optional[float] = 1.2,
        min_new_img_size: Optional[ImgSize] = None,
        img_bands: Optional[List[int]] = None,
        label_bands: Optional[List[int]] = None,
        random_seed: int = 42,
        **kwargs,
    ) -> None:
        """
        Args:
            mode (str, optional): One of 'random', 'centered', 'variable'. If 'random' images (or minimal image grids) will be randomly chosen subject to constraint that they fully contain the polygons, if 'centered' will be centered on the polygons. If 'variable' the image size will be the max of some minimum size and a multiple of the bounding rectangle of the polygon. Defaults to 'random'.
            new_img_size (Optional[ImgSize]): size (side length of square or rows, cols). Only needed if mode is 'centered' or 'random'.
            scaling factor (Optional[float]): factor to scale the bounding rectangle of the polygon by to get the image size.
            min_new_img_size (Optional[ImgSize]): minimum size of new image in 'variable' mode.
            img_bands (Optional[List[int]], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
            label_bands (Optional[List[int]], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
            random_seed (int, optional). random seed. Defaults to 42.

        Raises:
            ValueError: If the mode is unknown.
        """

        if mode in {'random', 'centered'}:
            if new_img_size is None:
                raise ValueError(
                    f"if mode is {mode}, new_img_size should not be None")
            self._check_img_size_type_and_value(new_img_size)
            self._rows, self._cols = self._get_size_rows_cols(new_img_size)
        elif mode in {'variable'}:
            if scaling_factor is None:
                raise ValueError(
                    f"if mode is {mode}, scaling_factor should not be None")
            if min_new_img_size is None:
                raise ValueError(
                    f"if mode is {mode}, min_new_img_size should not be None")
            self._check_img_size_type_and_value(min_new_img_size)
            self._rows, self._cols = self._get_size_rows_cols(min_new_img_size)
        else:
            logger.error("unknown mode: %s", mode)
            raise ValueError(f"unknown mode: {mode}")

        random.seed(random_seed)

        super().__init__(
            mode=mode,
            new_img_size=new_img_size,
            scaling_factor=scaling_factor,
            min_new_img_size=min_new_img_size,
            img_bands=img_bands,
            label_bands=label_bands,
            random_seed=random_seed,
            **kwargs,
        )

    def _check_img_size_type_and_value(self, img_size: ImgSize):
        """Check type and value of arg"""
        if not isinstance(img_size, int) or (isinstance(img_size, tuple)
                                             and len(img_size) == 2 and all(
                                                 isinstance(entry, int)
                                                 for entry in img_size)):
            raise TypeError(
                "new_img_size needs to be an integer or a pair of integers!")

        img_size_rows, img_size_cols = self._get_size_rows_cols(img_size)

        if not img_size_rows > 0:
            logger.error("%s need to have positive side length(s)", arg_name)
            raise ValueError(
                "{arg_name} needs to have positive side length(s)")
        if not img_size_cols > 0:
            logger.error("%s needs to have positive side length(s)", arg_name)
            raise ValueError(
                "{arg_name} needs to have positive side length(s)")

    @staticmethod
    def _get_size_rows_cols(
            img_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:

        if isinstance(img_size, tuple):
            new_img_size_rows = img_size[0]
            new_img_size_cols = img_size[1]
        else:
            new_img_size_rows = img_size
            new_img_size_cols = img_size

        return new_img_size_rows, new_img_size_cols

    def _get_windows_transforms_img_names(
            self,
            polygon_name: Union[str, int],
            source_img_name: str,
            source_assoc: ImgPolygonAssociator,
            target_assoc: ImgPolygonAssociator,
            new_imgs_dict: Optional[dict] = None,
            **kwargs: Any) -> List[Tuple[Window, Affine, str]]:
        """Given a polygon and a GeoTiff image fully containing it return a
        list of windows, window transforms, and new img_names defining a
        minimal rectangular grid in the image covering the polygon.

        Args:
            polygon_name (Union[str, int]): polygon identifier
            source_img_name (str): name of source image
            target_assoc (ImgPolygonAssociator): associator of target dataset
            new_imgs_dict (dict): dict with keys index or column names of target_assoc.imgs_df and values lists of entries correspondong to images containing information about cut images not yet appended to target_assoc.            polygon_crs_epsg_code (int): EPSG code of the polygon crs
            **kwargs (Any): keyword arguments

        Returns:
            List[Tuple[Window, Affine, str]]: list of windows, window_transformations, and new image names
        """

        source_img_path = source_assoc.images_dir / source_img_name

        polygon_geometry = target_assoc.polygons_df.loc[polygon_name,
                                                        "geometry"]

        with rio.open(source_img_path) as src:

            # transform polygon from assoc's crs to image source crs
            transformed_polygon_geometry = transform_shapely_geometry(
                polygon_geometry,
                from_epsg=source_assoc.polygons_df.crs.to_epsg(),
                to_epsg=src.crs.to_epsg())

            # FOR DEBUGGING:
            img_bbox = box(*src.bounds)
            assert source_assoc.polygons_df.loc[polygon_name].geometry.within(
                source_assoc.imgs_df.loc[source_img_name].geometry)
            if not transformed_polygon_geometry.within(img_bbox):
                logger.debug("img %s doesn't contain polygon %s in img crs",
                             source_img_name, polygon_name)
                transformed_polygon_geometry = img_bbox.intersection(
                    transformed_polygon_geometry)

            min_row, max_row, min_col, max_col = self._get_min_max_row_col(
                img=src,
                transformed_polygon_geometry=transformed_polygon_geometry)

            assert min(
                min_row, max_row, min_col, max_col
            ) >= 0, f'nonsensical negative max/min row/col values. sth went wrong cutting {source_img_name} for {polygon_name}'

            if self.mode in {'centered', 'random'}:
                new_img_size_rows = self._rows
                new_img_size_cols = self._cols
            elif self.mode == 'variable':
                new_img_size_rows = max(
                    self.scaling_factor * (max_row - min_row), self._rows)
                new_img_size_cols = max(
                    self.scaling_factor * (max_col - min_col), self._cols)

            row_off, col_off, num_small_imgs_in_row_direction, num_small_imgs_in_col_direction = \
                self._get_grid_row_col_offsets_num_windows_row_col_direction(
                    img = src,
                    new_img_size_rows=new_img_size_rows,
                    new_img_size_cols=new_img_size_cols,
                    min_row=min_row,
                    max_row=max_row,
                    min_col=min_col,
                    max_col=max_col,
                    transformed_polygon_geometry=transformed_polygon_geometry)

            # The row and col offs and number of images in row and col direction define a grid. Iterate through the grid and accumulate windows, transforms, and img_names in a list:

            windows_transforms_img_names_single_polygon = []

            for img_row in range(num_small_imgs_in_row_direction):
                for img_col in range(num_small_imgs_in_col_direction):

                    # Define the square window with the calculated offsets.
                    window = rio.windows.Window(
                        col_off=col_off + new_img_size_cols * img_col,
                        row_off=row_off + new_img_size_rows * img_row,
                        width=new_img_size_cols,
                        height=new_img_size_rows)

                    # Remember the transform for the new geotiff.
                    window_transform = src.window_transform(window)

                    # Generate new img name.
                    img_name_no_extension = Path(source_img_name).stem

                    # (if there is only one window in the grid)
                    if num_small_imgs_in_row_direction == 1 and num_small_imgs_in_col_direction == 1:
                        new_img_name = f"{img_name_no_extension}_{polygon_name}.tif"
                    else:
                        new_img_name = f"{img_name_no_extension}_{polygon_name}_{img_row}_{img_col}.tif"

                    window_bounding_rectangle = box(
                        *rio.windows.bounds(window, src.transform))

                    # append window if it intersects the polygon
                    if window_bounding_rectangle.intersects(
                            transformed_polygon_geometry):

                        windows_transforms_img_names_single_polygon.append(
                            (window, window_transform, new_img_name))

            return windows_transforms_img_names_single_polygon

    def _get_min_max_row_col(
            self, img: DatasetReader, transformed_polygon_geometry: Polygon
    ) -> Tuple[int, int, int, int]:
        """Return min_row, max_row, min_col, max_col of enveloping rectangle of
        polygon."""

        # Find min and max row of rectangular envelope of polygon
        list_of_rectangle_corner_coords = list(
            transformed_polygon_geometry.envelope.exterior.coords)[:5]
        list_of_enveloping_rectangle_row_col_pairs = list(
            map(lambda pair: img.index(*pair),
                list_of_rectangle_corner_coords))

        tuple_of_rows_of_rectangle_corners = tuple(
            zip(*list_of_enveloping_rectangle_row_col_pairs))[0]
        tuple_of_cols_of_rectangle_corners = tuple(
            zip(*list_of_enveloping_rectangle_row_col_pairs))[1]

        min_row = min(*tuple_of_rows_of_rectangle_corners)
        max_row = max(*tuple_of_rows_of_rectangle_corners)
        min_col = min(*tuple_of_cols_of_rectangle_corners)
        max_col = max(*tuple_of_cols_of_rectangle_corners)

        return min_row, max_row, min_col, max_col

    def _get_grid_row_col_offsets_num_windows_row_col_direction(
            self, img: DatasetReader, transformed_polygon_geometry: Polygon,
            new_img_size_rows: int, new_img_size_cols: int, min_row: int,
            max_row: int, min_col: int,
            max_col: int) -> Tuple[int, int, int, int]:
        """Return row and col offsets and number of windows in row and in col
        direction such that the resulting grid is minimal grid fully covering
        the polygon."""

        num_small_imgs_in_row_direction = math.ceil(
            float(max_row - min_row) / new_img_size_rows)

        num_small_imgs_in_col_direction = math.ceil(
            float(max_col - min_col) / new_img_size_cols)

        # Choose row and col offset
        if self.mode == 'random':

            # ... choose row and col offsets randomly subject to constraint that the grid of image windows contains rectangular envelope of polygon.
            row_off = random.randint(
                max(
                    0, max_row -
                    new_img_size_rows * num_small_imgs_in_row_direction),
                min(
                    min_row, img.height -
                    new_img_size_rows * num_small_imgs_in_row_direction))

            col_off = random.randint(
                max(
                    0, max_col -
                    new_img_size_cols * num_small_imgs_in_col_direction),
                min(
                    min_col, img.width -
                    new_img_size_cols * num_small_imgs_in_col_direction))

        elif self.mode in {'centered', 'variable'}:

            # ... to find the row, col offsets to center the polygon ...

            # ...we first find the centroid of the polygon in the img crs ...
            polygon_centroid_coords = transformed_polygon_geometry.envelope.centroid.coords[
                0]

            # ... extract the row, col of the centroid ...
            centroid_row, centroid_col = img.index(*polygon_centroid_coords)

            # and then choose offsets to center the polygon.
            row_off = centroid_row - (new_img_size_rows *
                                      num_small_imgs_in_row_direction // 2)
            col_off = centroid_col - (new_img_size_cols *
                                      num_small_imgs_in_col_direction // 2)

        else:

            raise ValueError(f"Unknown mode: {self.mode}")

        return row_off, col_off, num_small_imgs_in_row_direction, num_small_imgs_in_col_direction
