""" 
SingleImgCutter that cuts an image to a grid of images. 
"""

from typing import Any, Union, List, Optional, Tuple
from rs_tools.cut.type_aliases import ImgSize
import logging
from pathlib import Path
import rasterio as rio
from rasterio.windows import Window
from affine import Affine

from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.cut.single_img_cutter_base import SingleImgCutter

logger = logging.getLogger(__name__)


class ImgToGridCutter(SingleImgCutter):
    """
    SingleImgCutter that cuts an image into a grid of images.
    """

    def __init__(self, 
            source_assoc : ImgPolygonAssociator, 
            target_images_dir : Union[Path, str], 
            target_labels_dir : Union[Path, str], 
            new_img_size : ImgSize, 
            img_bands : Optional[List[int]], 
            label_bands : Optional[List[int]]
            ) -> None:
        """
        Args:
            source_assoc (ImgPolygonAssociator): associator of dataset images are to be cut from.
            target_images_dir (Union[Path, str): images directory of target dataset
            target_labels_dir (Union[Path, str): labels directory of target dataset
            new_img_size (Union[int, Tuple[int, int]]): size (side length of square or rows, cols)
            img_bands (Optional[List[int]], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
            label_bands (Optional[List[int]], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
        """

        super().__init__(
            source_assoc=source_assoc, 
            target_images_dir=target_images_dir, 
            target_labels_dir=target_labels_dir, 
            img_bands=img_bands, 
            label_bands=label_bands)                        
        
        # Check new_img_size arg type
        if not isinstance(new_img_size, int) or (isinstance(new_img_size, tuple) and len(new_img_size)==2 and all(isinstance(entry, int) for entry in new_img_size)): 
            raise TypeError("new_img_size needs to be an integer or a pair of integers!")

        if isinstance(new_img_size, tuple):
            self.new_img_size_rows = new_img_size[0]
            self.new_img_size_cols = new_img_size[1]
        else:
            self.new_img_size_rows = new_img_size
            self.new_img_size_cols = new_img_size

        if not self.new_img_size_rows > 0:
            logger.error("new_img_size needs to have positive side length(s)")
            raise ValueError("new_img_size needs to have positive side length(s)")
        if not self.new_img_size_cols > 0: 
            logger.error("new_img_size needs to have positive side length(s)")
            raise ValueError("new_img_size needs to have positive side length(s)")

    def _get_windows_transforms_img_names(self, 
            source_img_name : str,
            target_assoc : Optional[ImgPolygonAssociator] = None, 
            new_imgs_dict : Optional[dict] = None,
            **kwargs : Any
            ) -> List[Tuple[Window, Affine, str]]:

        source_img_path = self.source_assoc._images_dir / source_img_name

        with rio.open(source_img_path) as src:

            if not src.height % self.new_img_size_rows == 0:
                logger.warning("number of rows in source image not divisible by number of rows in new images")
            if not src.width % self.new_img_size_cols == 0:
                logger.warning("number of columns in source image not divisible by number of columns in new images")

        windows_transforms_img_names = []

        # Iterate through grid ...
        for i in range(src.width // self.new_img_size_cols):
            for j in range(src.height // self.new_img_size_rows):
                
                # ... remember windows, ...
                window = Window(
                            i * self.new_img_size_cols, 
                            j* self.new_img_size_rows, 
                            width=self.new_img_size_cols, 
                            height=self.new_img_size_rows)
                
                # ... transforms ...
                window_transform = src.window_transform(window)
                
                # ... and image names.
                new_img_name = f"{Path(source_img_name).stem}_{j}_{i}.tif"

                windows_transforms_img_names.append((window, window_transform, new_img_name))

        return windows_transforms_img_names
