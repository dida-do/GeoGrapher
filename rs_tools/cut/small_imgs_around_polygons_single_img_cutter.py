""" 
SingleImgCutter that cuts a small image (or several contiguous such images if the polygon does not fit into a single one) around each polygon in the image accepted by the polygon filter predicate.
"""
from typing import Union, List, Optional, Tuple
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

class SmallImgsAroundPolygonsCutter(SingleImgCutter):
    """
    SingleImgCutter that cuts a small image (or several contiguous such images if the polygon does not fit into a single one) around each polygon in the image accepted by the polygon filter predicate.
    """

    def __init__(self, 
                source_assoc: ImgPolygonAssociator, 
                target_data_dir : Union[Path, str], 
                polygon_filter_predicate: PolygonFilterPredicate, 
                new_img_size: Union[int, Tuple[int, int]], 
                img_bands: Optional[List[int]], 
                labels_bands: Optional[List[int]], 
                mode: str, 
                random_seed: int = 42) -> None:
        """
        Args:
            source_assoc (ImgPolygonAssociator): associator of dataset images are to be cut from.
            target_data_dir (Union[Path, str]): data directory of dataset where new images/labels will be created.
            polygon_filter_predicate (PolygonFilterPredicate): predicate to filter polygons.
            new_img_size (Union[int, Tuple[int, int]]): size (side length of square or rows, cols)
            img_bands (Optional[List[int]], optional): list of bands to extract from source images. Defaults to None (i.e. all bands).
            label_bands (Optional[List[int]], optional):  list of bands to extract from source labels. Defaults to None (i.e. all bands).
            mode (str, optional): One of 'random' or 'centered'. If 'random' images (or minimal image grids) will be randomly chosen subject to constraint that they fully contain the polygons, if 'centered' will be centered on the polygons. Defaults to 'random'.
            random_seed (int, optional). random seed. Defaults to 42.

        Raises:
            ValueError: If the mode is unknown.
        """

        super().__init__(source_assoc=source_assoc, 
                        target_data_dir=target_data_dir, 
                        img_bands=img_bands, 
                        labels_bands=labels_bands, 
                        mode=mode)                        
        
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
            raise ValueError("new_img_size needs to have positive side length(s)")
        if not self.new_img_size_cols > 0: 
            raise ValueError("new_img_size needs to have positive side length(s)")

        self.polygon_filter_predicate = polygon_filter_predicate
        
        if mode not in {'random', 'centered'}:
            raise ValueError(f"mode is {mode}, needs to be one of 'random' or 'centered'!")
        self.mode = mode 

        random.seed(random_seed)


    def _get_windows_transforms_img_names(self, 
                                            source_img_name: str, 
                                            new_polygons_df: GeoDataFrame, 
                                            new_graph: BipartiteGraph):

        polygons_contained_in_img = self.source_assoc.polygons_contained_in_img(source_img_name)

        windows_transforms_img_names = []

        # for all polygons in the source_assoc contained in the img
        for polygon_name, polygon_geometry in (self.source_assoc.polygons_df.loc[polygons_contained_in_img, ['geometry']]).itertuples():

            if self.polygon_filter_predicate(polygon_name, new_polygons_df, self.source_assoc) == False:

                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # for cut_imgs_iter_over_imgs should drop polygon form new_polygons_df?
                pass

            else:

                windows_transforms_img_names_single_polygon = self._get_windows_transforms_img_names_single_polygon(polygon_name, polygon_geometry, source_img_name, (self.new_imgs_size_rows, self.new_imgs_size_cols))

                windows_transforms_img_names += windows_transforms_img_names_single_polygon

        return windows_transforms_img_names


    def _get_windows_transforms_img_names_single_polygon(self, 
                                        polygon_name: str, 
                                        polygon_geometry: Polygon, 
                                        source_img_name: str, 
                                        window_size: Tuple[int, int]) -> List[Tuple[Window, Affine, str]]:
        """
        Given a polygon and a GeoTiff image fully containing it return a list of windows, window transforms, and new img_names defining a minimal rectangular grid in the image covering the polygon.

        Args:
            polygon_name (str): polygon identifier
            polygon_geometry (Polygon): polygon fully contained in the GeoTiff
            polygon_crs_epsg_code (int): EPSG code of the polygon crs
            source_img_name (str): name of source image
            window_size (Tuple[int, int]): window size (row and col lengths in pixels)

        Returns:
            List[Tuple[Window, Affine, str]]: list of windows, window_transformations, and new image names
        """

        source_img_path = self.source_assoc.data_dir / "images" / source_img_name

        with rio.open(source_img_path) as src:

            row_off, col_off, num_small_imgs_in_row_direction, num_small_imgs_in_col_direction = \
                self._get_grid_row_col_offsets_num_windows_row_col_direction(img = src,                                
                                                                            polygon_name=polygon_name, 
                                                                            polygon_geometry=polygon_geometry)
            
            # The row and col offs and number of images in row and col direction define a grid. Iterate through the grid and accumulate windows, transforms, and img_names in a list:

            windows_transforms_img_names_single_polygon = []

            for img_row in range(num_small_imgs_in_row_direction):
                for img_col in range(num_small_imgs_in_row_direction):
            
                    # Define the square window with the calculated offsets.
                    window = rio.windows.Window(col_off=col_off + self.new_img_size_cols * img_col, 
                                                row_off=row_off + self.new_img_size_rows * img_row, 
                                                width=self.new_img_size_cols, 
                                                height=self.new_img_size_rows)

                    # Remember the transform for the new geotiff.
                    window_transform = src.window_transform(window)

                    # Generate new img name.
                    img_name_no_extension = Path(source_img_name).stem
                    
                    # (if there is only one window in the grid)
                    if num_small_imgs_in_row_direction==1 and num_small_imgs_in_col_direction==1:
                        new_img_name = f"img_name_no_extension_{polygon_name}.tif"
                    else:
                        new_img_name = f"img_name_no_extension_{polygon_name}_{img_row}_{img_col}.tif"

                    window_bounding_rectangle = box(*rio.windows.bounds(window, window_transform))

                    # append window if it intersects the polygon
                    if window_bounding_rectangle.intersects(polygon_geometry):

                        windows_transforms_img_names_single_polygon.append((window, window_transform, new_img_name))

            return windows_transforms_img_names_single_polygon


    def _get_grid_row_col_offsets_num_windows_row_col_direction(self, 
                                    img: DatasetReader,
                                    polygon_geometry: Polygon) -> Tuple[int, int, int, int]:
        """Return row and col offsets and number of windows in row and in col direction such that the resulting grid is minimal grid fully covering the polygon."""

        # transform polygons from assoc's crs to image source crs
        transformed_polygon_geometry = transform_shapely_geometry(polygon_geometry, 
                                                            from_epsg=self.polygon_crs_epsg_code, 
                                                            to_epsg=img.crs.to_epsg())

        # Find min and max row of rectangular envelope of polygon
        list_of_rectangle_corner_coords = list(transformed_polygon_geometry.envelope.exterior.coords)[:5]
        list_of_enveloping_rectangle_row_col_pairs = list(map(lambda pair: img.index(*pair), list_of_rectangle_corner_coords))

        tuple_of_rows_of_rectangle_corners = tuple(zip(*list_of_enveloping_rectangle_row_col_pairs))[0]
        tuple_of_cols_of_rectangle_corners = tuple(zip(*list_of_enveloping_rectangle_row_col_pairs))[1]

        min_row = min(*tuple_of_rows_of_rectangle_corners)
        max_row = max(*tuple_of_rows_of_rectangle_corners)
        min_col = min(*tuple_of_cols_of_rectangle_corners)
        max_col = max(*tuple_of_cols_of_rectangle_corners)

        num_small_imgs_in_row_direction = math.ceil((max_row - min_row) // self.new_img_size_rows)

        num_small_imgs_in_col_direction = math.ceil((max_col - min_col) // self.new_img_size_cols)

        # Choose row and col offset
        if self.mode == 'random':
            
            # ... choose row and col offsets randomly subject to constraint that the grid of image windows contains rectangular envelope of polygon. 
            row_off = random.randint(max(0, max_row - self.new_img_size_rows * num_small_imgs_in_row_direction), 
                                        min(min_row, img.height - self.new_img_size_rows * num_small_imgs_in_row_direction)) 

            col_off = random.randint(max(0, max_col - self.new_img_size_cols * num_small_imgs_in_col_direction), 
                                        min(min_col, img.width - self.new_img_size_cols * num_small_imgs_in_col_direction))

        elif self.mode == 'centered': 

            # ... to find the row, col offsets to center the polygon ...

            # ...we first find the centroid of the polygon in the img crs ...
            polygon_centroid_coords = transformed_polygon_geometry.envelope.centroid.coords[0]

            # ... extract the row, col of the centroid ...
            centroid_row, centroid_col = img.index(*polygon_centroid_coords)

            # and then choose offsets to center the polygon.
            row_off = centroid_row - (self.new_img_size_rows * num_small_imgs_in_row_direction // 2)
            col_off = centroid_col - (self.new_img_size_cols * num_small_imgs_in_col_direction // 2)
        
        else:
            
            raise ValueError(f"Unknown mode: {self.mode}")

        return row_off, col_off, num_small_imgs_in_row_direction, num_small_imgs_in_col_direction
