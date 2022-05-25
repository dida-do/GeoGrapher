"""
A SingleImgCutter to extract one or several pre defined bounding box from an image.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import geopandas as gpd
import rasterio as rio
from geopandas import GeoDataFrame
from pydantic import PrivateAttr, validator
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from geographer.cutters.single_img_cutter_base import SingleImgCutter
from geographer.cutters.type_aliases import ImgSize
from geographer.connector import Connector

logger = logging.getLogger(__name__)


def _correct_window_offset(offset: Union[int, float], size: Union[int, float],
                           new_size: int) -> int:
    center = offset + size / 2
    return int(center - new_size / 2)


class SingleImgCutterFromBBoxes(SingleImgCutter):
    """
    A SingleImgCutter to extract one or several pre defined bounding box from an image.
    The new size of the images must be specified as it is used to ensure a
    standardised output.
    """

    new_img_size: ImgSize
    bbox_geojson_path: Path

    _bboxes_df: GeoDataFrame = PrivateAttr()

    def __init__(self, **data) -> None:
        """
        Args:
            new_img_size (ImgSize): size of new image
            bbox_geojson_path (Path): path to geojson file containing the bboxes
        """

        super().__init__(**data)
        self._bboxes_df = gpd.read_file(self.bbox_geojson_path,
                                        driver="GeoJSON")

    @validator('bbox_geojson_path')
    def path_points_to_geojson(cls, value: Path):
        """Validator: Make sure path exists and points to geojson"""
        if value.suffix != ".geojson":
            raise ValueError("Path should point to .geojson file")
        if not value.is_file():
            raise FileNotFoundError(f".geojson file does not exist: {value}")
        return value

    @validator("new_img_size")
    def new_img_size_type_correctness(cls, value: ImgSize) -> ImgSize:
        """Validator: make sure new_img_size has correct type"""
        is_int: bool = isinstance(value, int)
        is_pair_of_ints: bool = isinstance(
            value, tuple) and len(value) == 2 and all(
                isinstance(entry, int) for entry in value)
        if not (is_int or is_pair_of_ints):
            raise TypeError(
                "new_img_size needs to be an integer or a pair of integers!")
        return value

    @validator("new_img_size")
    def new_img_size_side_lengths_must_be_positive(cls,
                                                   value: ImgSize) -> ImgSize:
        """Validate new_img_size side lengths are positive"""
        if isinstance(value, tuple) and not all(val > 0 for val in value):
            logger.error("new_img_size: need positive side length(s)")
            raise ValueError("new_img_size: need positive side length(s)")
        elif isinstance(value, int) and value <= 0:
            logger.error("new_img_size: need positive side length(s)")
            raise ValueError("new_img_size: need positive side length(s)")
        return value

    @property
    def new_img_size_rows(self) -> int:
        """Return number of rows of new image size"""
        if isinstance(self.new_img_size, tuple):
            return self.new_img_size[0]
        else:
            return self.new_img_size

    @property
    def new_img_size_cols(self) -> int:
        """Return number of columns of new image size"""
        if isinstance(self.new_img_size, tuple):
            return self.new_img_size[1]
        else:
            return self.new_img_size

    def _get_windows_transforms_img_names(
        self,
        source_img_name: str,
        source_connector: Connector,
        target_connector: Optional[Connector] = None,
        new_imgs_dict: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[str]:

        source_img_path = source_connector.images_dir / source_img_name

        with rio.open(source_img_path) as src:
            img_bounds = box(*src.bounds)
            bounding_boxes = self.bounding_boxes.to_crs(src.crs)
            bounding_boxes = bounding_boxes.loc[bounding_boxes.geometry.within(
                img_bounds)]
            windows_transforms_img_names = []
            for i, geometry in enumerate(bounding_boxes.geometry):
                initial_window = from_bounds(*geometry.bounds, src.transform)

                new_col_off = _correct_window_offset(initial_window.col_off,
                                                     initial_window.width,
                                                     self.new_img_size_cols)

                new_row_off = _correct_window_offset(initial_window.row_off,
                                                     initial_window.height,
                                                     self.new_img_size_rows)

                window = Window(new_col_off, new_row_off,
                                self.new_img_size_cols, self.new_img_size_rows)

                window_transform = src.window_transform(window)
                new_img_name = f"{Path(source_img_name).stem}_{i}.tif"

                windows_transforms_img_names.append(
                    (window, window_transform, new_img_name))

        return windows_transforms_img_names
