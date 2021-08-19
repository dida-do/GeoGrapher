""" 
SingleImgCutter that cuts a small image (or several contiguous such images if the polygon does not fit into a single one) around each polygon in the image accepted by the polygon filter predicate.
"""
from typing import Union, List, Optional, Tuple
from rs_tools.cut.type_aliases import ImgSize
import logging
from pathlib import Path
import math
import random
from shapely.geometry import Polygon, box
from geopandas import GeoDataFrame 
import rasterio as rio
from rasterio.io import DatasetReader
from rasterio.windows import Window
from affine import Affine

from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.cut.single_img_cutter_base import SingleImgCutter
from rs_tools.cut.polygon_filter_predicates import PolygonFilterPredicate
from rs_tools.graph import BipartiteGraph
from rs_tools.utils.utils import transform_shapely_geometry

logger = logging.getLogger(__name__)

class ToImgGridCutter(SingleImgCutter):
    """
    SingleImgCutter that cuts an image into a grid of images.
    """

    def __init__(self, 
                source_assoc: ImgPolygonAssociator, 
                target_data_dir : Union[Path, str], 
                polygon_filter_predicate: PolygonFilterPredicate, 
                new_img_size: ImgSize, 
                img_bands: Optional[List[int]], 
                label_bands: Optional[List[int]]) -> None:
        """
        Args:
            source_assoc (ImgPolygonAssociator): associator of dataset images are to be cut from.
            target_data_dir (Union[Path, str]): data directory of dataset where new images/labels will be created.
            polygon_filter_predicate (PolygonFilterPredicate): predicate to filter polygons. Ignored. 
            new_img_size (Union[int, Tuple[int, int]]): size (side length of square or rows, cols)
            img_bands (Optional[List[int]], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
            label_bands (Optional[List[int]], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
        """

        super().__init__(source_assoc=source_assoc, 
                        target_data_dir=target_data_dir, 
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
                                            source_img_name: str, 
                                            new_polygons_df: GeoDataFrame, 
                                            new_graph: BipartiteGraph):

        source_img_path = self.source_assoc.data_dir / "images" / source_img_name

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
