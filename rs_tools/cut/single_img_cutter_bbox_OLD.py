"""SingleImgCutter that cuts a small image from a predefined bounding box on
the image accepted by the polygon filter predicate."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import rasterio as rio
from geopandas import GeoDataFrame
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon, box

from rs_tools.cut.type_aliases import ImgSize

if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator

from rs_tools.cut.single_img_cutter_base import SingleImgCutterBase

logger = logging.getLogger(__name__)


def _correct_window_offset(offset: Union[int, float], size: Union[int, float],
                           new_size: int) -> int:
    center = offset + size / 2
    return int(center - new_size / 2)


class ToImgBBoxCutter(SingleImgCutterBase):
    """SingleImgCutter that cuts an image using a predefined GeoDataFrame of
    bounding boxes."""

    def __init__(self,
                 source_assoc: ImgPolygonAssociator,
                 target_images_dir: Union[Path, str],
                 target_labels_dir: Union[Path, str],
                 new_img_size: Optional[ImgSize],
                 bounding_boxes: GeoDataFrame,
                 img_bands: Optional[List[int]] = None,
                 label_bands: Optional[List[int]] = None) -> None:
        """An image cutter to extract a pre defined bounding box from an image.
        The new size of the images must be specified as it is used to ensure a
        standardised output.

        :param source_assoc: associator of dataset images are to be cut from.
        :type source_assoc: ImgPolygonAssociator
        :param target_images_dir: images directory of target dataset
        :type target_images_dir: Union[Path, str]
        :param target_labels_dir: labels directory of target dataset
        :type target_labels_dir: Union[Path, str]
        :param new_img_size: size (side length of square or rows, cols)
        :type new_img_size: Union[int, Tuple[int, int]]
        :param bounding_boxes: Bounding boxes to cut from imahe
        :type bounding_boxes: GeoDataFrame
        :param img_bands: list of bands to extract from source images. Defaults to None (i.e. all bands).
        :type img_bands: Optional[List[int]]
        :param label_bands: list of bands to extract from source labels. Defaults to None (i.e. all bands).
        :type label_bands: Optional[List[int]]
        """

        super().__init__(source_assoc=source_assoc,
                         target_images_dir=target_images_dir,
                         target_labels_dir=target_labels_dir,
                         img_bands=img_bands,
                         label_bands=label_bands)

        # Save bounding boxes
        self.bounding_boxes = bounding_boxes

        # Check new_img_size arg type
        if not isinstance(new_img_size, int) or (
                isinstance(new_img_size, tuple) and len(new_img_size) == 2
                and all(isinstance(entry, int) for entry in new_img_size)):
            raise TypeError(
                "new_img_size needs to be an integer or a pair of integers!")

        if isinstance(new_img_size, tuple):
            self.new_img_size_rows = new_img_size[0]
            self.new_img_size_cols = new_img_size[1]
        else:
            self.new_img_size_rows = new_img_size
            self.new_img_size_cols = new_img_size

        if not self.new_img_size_rows > 0:
            logger.error("new_img_size needs to have positive side length(s)")
            raise ValueError(
                "new_img_size needs to have positive side length(s)")
        if not self.new_img_size_cols > 0:
            logger.error("new_img_size needs to have positive side length(s)")
            raise ValueError(
                "new_img_size needs to have positive side length(s)")

    def _get_windows_transforms_img_names(
            self,
            source_img_name: str,
            target_assoc: Optional[ImgPolygonAssociator] = None,
            new_imgs_dict: Optional[dict] = None,
            **kwargs: Any) -> List[str]:

        source_img_path = self.source_assoc._images_dir / source_img_name

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
