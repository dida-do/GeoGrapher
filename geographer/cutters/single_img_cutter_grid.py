"""SingleImgCutter that cuts an image to a grid of images."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from pydantic import validator, BaseModel
import rasterio as rio
from affine import Affine
from rasterio.windows import Window

from geographer.connector import Connector
from geographer.cutters.type_aliases import ImgSize
from geographer.cutters.single_img_cutter_base import SingleImgCutter

logger = logging.getLogger(__name__)


class SingleImgCutterToGrid(SingleImgCutter):
    """SingleImgCutter that cuts an image into a grid of images."""

    new_img_size: ImgSize

    @validator("new_img_size")
    def new_img_size_type_correctness(cls, value: ImgSize) -> ImgSize:
        """Validate new_img_size has correct type"""
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
            **kwargs: Any) -> List[Tuple[Window, Affine, str]]:

        source_img_path = source_connector.images_dir / source_img_name

        with rio.open(source_img_path) as src:

            if not src.height % self.new_img_size_rows == 0:
                logger.warning(
                    "number of rows in source image not divisible by number of rows in new images"
                )
            if not src.width % self.new_img_size_cols == 0:
                logger.warning(
                    "number of columns in source image not divisible \
                        by number of columns in new images")

        windows_transforms_img_names = []

        # Iterate through grid ...
        for i in range(src.width // self.new_img_size_cols):
            for j in range(src.height // self.new_img_size_rows):

                # ... remember windows, ...
                window = Window(i * self.new_img_size_cols,
                                j * self.new_img_size_rows,
                                width=self.new_img_size_cols,
                                height=self.new_img_size_rows)

                # ... transforms ...
                window_transform = src.window_transform(window)

                # ... and image names.
                new_img_name = f"{Path(source_img_name).stem}_{j}_{i}.tif"

                windows_transforms_img_names.append(
                    (window, window_transform, new_img_name))

        return windows_transforms_img_names
