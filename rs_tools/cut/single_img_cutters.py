"""
TODO: remove source_assoc, target_data_dir args from SingleImgCutter?????

TODO: EXPLAIN 'CONTRACT' THAT SingleImgCutter should satisfy. (format of dice to be returned ...)
"""

from typing import Union, List, Tuple, Optional, Any
from pathlib import Path
import os
from abc import ABC, abstractmethod

import math
import random
from shapely.geometry.polygon import Polygon
from shapely.geometry import box
import rasterio as rio 
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from rasterio.io import DatasetReader
from rasterio.crs import CRS
from affine import Affine

from geopandas.geodataframe import GeoDataFrame
from rs_tools.graph import BipartiteGraph
from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.cut.polygon_filter_predicates import PolygonFilterPredicate
from rs_tools.cut.single_img_cutter_utils import write_window_to_geotif
from rs_tools.utils.utils import transform_shapely_geometry


class SingleImgCutter(ABC):
    def __init__(self, 
                source_assoc: ImgPolygonAssociator, 
                target_data_dir: Union[Path, str], 
                img_bands: Optional[List[int]], 
                label_bands: Optional[List[int]], 
                **kwargs: Any) -> dict:
        """
        Abstract base class for single image cutters. 

        To define an image cutter, override _get_windows_transforms_img_names method. 

        Args:
            source_assoc (ImgPolygonAssociator): source associator.
            target_data_dir (Union[Path, str]): data directory containing images and labels subdirectories in which the new images and labels will be created.
            new_img_size (Union[int, Tuple[int, int]]): size (rows, cols) of new images to be created in pixels. If only one integer given the image will be square. 
            img_bands (Optional[List[int]]): list of bands to extract from the image (note GeoTiff bands start at 1).
            labels_bands (Optional[List[int]]): list of bands to extract from the label (note GeoTiff bands start at 1).
    
        Raises:
            ValueError: if the mode is unknown.
            TypeError: if the new image size is not an integer or a pair of integers.
            ValueError: if the integer(s) defining the size of the new images are not positive.
        """

        self.source_assoc = source_assoc
        self.target_data_dir = target_data_dir
        
        if img_bands is None:
            self.img_bands = self._get_all_band_indices('images')
        else:
            self.img_bands = img_bands

        if self.label_bands is None:
            self.label_bands = self._get_all_band_indices('labels')
        else:
            self.label_bands = label_bands

        self.polygons_df_crs_epsg = self.source_assoc.polygons_df.crs.to_epsg()


    @abstractmethod
    def _get_windows_transforms_img_names(self, 
                source_img_name: str, 
                new_polygons_df: GeoDataFrame, 
                new_graph: BipartiteGraph) -> List[Tuple[Window, Affine, str]]:
        """
        Override to define subclass. 

        Args:
            source_img_name (str): name of img in self.source_img_path to be cut.
            new_polygons_df (GeoDataFrame): GeoDataFrame that will be the polygons_df of the associator of the new dataset of cut images that is being created by the calling dataset cutter. 
            new_graph (BipartiteGraph): the bipartite graph that is being built up for the target associator.

        Returns:
            List[Tuple[Window, Affine, str]]: list of rasterio windows, window transform, and new image names. 
        """
        pass


    def __call__(self, 
                source_img_name: str, 
                new_polygons_df: GeoDataFrame, 
                new_graph: BipartiteGraph) -> None:
        """
        Args:
            source_img_name (str): name of img in self.source_img_path to be cut.
            new_polygons_df (GeoDataFrame): GeoDataFrame that will be the polygons_df of the associator of the new dataset of cut images that is being created by the calling dataset cutter. 
            new_graph (BipartiteGraph): the bipartite graph that is being built up for the target associator.

        """
        # img and labels paths
        source_img_path = Path(self.source_assoc.data_dir / f"images/{source_img_name}")
        source_label_path = Path(self.source_assoc.data_dir / f"labels/{source_img_name}")        

        # dict to accumulate information about the newly created images
        imgs_from_cut_dict = {index_or_col_name: [] for index_or_col_name in [self.source_assoc.imgs_df.index.name] + list(self.source_assoc.imgs_df.columns)}

        windows_transforms_img_names = self._get_windows_transforms_img_name(source_img_name, 
                                                                            new_polygons_df, 
                                                                            new_graph)

        for window, window_transform, new_img_name in windows_transforms_img_names:

            # Make new image and label in target_data_dir ...
            img_bounds_in_img_crs, img_crs = self._make_new_img_and_label(window, window_transform, new_img_name)

            # ... gather all the information about the image in a dict ...
            single_new_img_info_dict = self._make_img_info_dict(new_img_name, img_bounds_in_img_crs, img_crs)

            # ... and accumulate that information. 
            for key in imgs_from_cut_dict.keys():
                imgs_from_cut_dict[key].append(single_new_img_info_dict[key])

            # Add connections to new_graph for the new image and modify new_polygons_df.
            self.source_assoc._add_img_to_graph_modify_polygons_df(new_img_name, 
                                                                img_bounding_rectangle=single_new_img_info_dict['geometry'], 
                                                                polygons_df=new_polygons_df, 
                                                                graph=new_graph)

        return imgs_from_cut_dict


    def _make_img_info_dict(self, 
                            imgs_from_cut_dict: dict, 
                            new_img_name: str, 
                            img_bounds_in_img_crs: Tuple[float, float, float, float], 
                            img_crs: CRS) -> None:
            
        img_bounding_rectangle_in_imgs_df_crs = box(*transform_bounds(img_crs, 
                                                                    self.source_assoc.imgs_df.crs,
                                                                    *img_bounds_in_img_crs))

        single_new_img_info_dict = {'img_name': new_img_name, 
                                    'geometry': img_bounding_rectangle_in_imgs_df_crs, 
                                    'orig_crs_epsg_code': img_crs.to_epsg(), 
                                    'img_processed?': True}
        
        # Copy over any remaining information about the img from self.source_assoc.imgs_df.
        for col in set(self.source_assoc.imgs_df.columns) - {'img_name', 'geometry', 'orig_crs_epsg_code', 'img_processed?'}:
            single_new_img_info_dict[col] = self.source_assoc.imgs_df.loc[new_img_name, col]

        return single_new_img_info_dict

            
    def _make_new_img_and_label(self, 
                                window: Window, 
                                window_transform: Affine, 
                                new_img_name: str) -> Tuple[Tuple[float, float, float, float], CRS]:

        dst_img_path = self.target_data_dir / f"images/{new_img_name}"
        dst_label_path = self.target_data_dir / f"labels/{new_img_name}"

        # write img window to destination img geotif
        img_bounds_in_img_crs, img_crs = write_window_to_geotif(self.source_img_path, 
                                                                dst_img_path, 
                                                                self.img_bands, 
                                                                window, 
                                                                window_transform)

        # write label window to destination label geotif
        label_bounds_in_img_crs, label_crs = write_window_to_geotif(self.source_label_path, 
                                dst_label_path, 
                                self.label_bands, 
                                window, 
                                window_transform)    

        assert img_crs == label_crs, "source image and label crs disagree!"
        assert label_bounds_in_img_crs == img_bounds_in_img_crs, "source image and label bounds disagree"

        return img_bounds_in_img_crs, img_crs


    def _write_window_to_geotif(self, 
                            src_img_path: Union[Path, str], 
                            dst_img_path: Union[Path, str], 
                            bands: List[int],
                            window: Window, 
                            window_transform: Affine) -> Tuple[Tuple[float, float, float, float], CRS]:
        """
        Write window from source GeoTiff to new GeoTiff.

        Args:
            src_img_path (Union[Path, str]): path of source GeoTiff
            dst_img_path (Union[Path, str]): path to GeoTiff to be created
            bands (List[int]): bands to extract from source GeoTiff
            window (Window): window to cut out from source GeoTiff
            window_transform (Affine): window transform of window

        Returns:
            Tuple[Tuple[float, float, float, float], CRS]: bounds and crs of new image
        """

        # Open source ...
        with rio.open(src_img_path) as src:

            # and destination ...
            with rio.open(Path(dst_img_path),
                            'w',
                            driver='GTiff',
                            height=window.height,
                            width=window.width,
                            count=len(bands),
                            dtype=src.profile["dtype"],
                            crs=src.crs,
                            transform=window_transform) as dst:

                # ... and go through the bands.
                for band in bands:
                    
                    # Read window for that band from source ...
                    new_img_band_raster = src.read(band, window=window)

                    # ... write to destination geotiff.
                    dst.write(new_img_band_raster, band)

        return dst.bounds, dst.crs

    def _get_all_band_indices(self, mode: str) -> List[int]:
        """
        Return list of all band indices of GeoTiffs. 

        It is assumed all images (or labels) in the data diretory have the same number of bands.

        Args:
            mode (str): 'images' or 'labels'

        Returns:
            List[int]: list of indices of all bands in GeoTiff 
        """

        img_or_label_dir = self.source_assoc.data_dir / mode
        img_or_label_name = [filename for filename in os.listdir(img_or_label_dir) if Path(filename).suffix == 'tif'][0]
        img_or_label_path = img_or_label_dir / img_or_label_name

        with rio.open(img_or_label_path) as src:
            bands = list(range(1, src.count + 1))

        return bands


class SmallImgsAroundPolygonsCutter(SingleImgCutter):
    """
    SingleImageCutter that cuts new small images around the polygons fully contained in it. 

    Go through all polygons in the source dataset/associator fully contained in the image. For each polygon if it is not filtered out by the polygon filter predicate create a single image fully containing the polygon or a grid of images if the polygon is too large to fit into a single new image. 
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
    

