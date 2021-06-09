"""
Abstract base class for single image cutters. 
"""
import logging
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
from rs_tools.utils.utils import transform_shapely_geometry

logger = logging.getLogger(__name__)

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

        A SingleImgCutter is a Callable that on a given call creates new images cut from a single source image, modifies the new_polygons_df and graph arguments to be used in the calling dataset cutting function to create the new associator organizing the newly created images, and returns a dict containing information about the created images to be used the calling function to create a imgs_df for the new associator (see __call__ docstring for details). 

        Args:
            source_assoc (ImgPolygonAssociator): source associator containing images/labels to be cut from.
            target_data_dir (Union[Path, str]): data directory containing images and labels subdirectories in which the new images and labels will be created.
            img_bands (Optional[List[int]]): list of bands to extract from the image (note GeoTiff bands start at 1).
            labels_bands (Optional[List[int]]): list of bands to extract from the label (note GeoTiff bands start at 1).
    
        Raises:
            ValueError: if the mode is unknown.
            TypeError: if the new image size is not an integer or a pair of integers.
            ValueError: if the integer(s) defining the size of the new images are not positive.
        """

        self.source_assoc = source_assoc
        self.target_data_dir = Path(target_data_dir)
        
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
        Return a list of rasterio windows, window transformations, and new image names. The returned list will be used to create the new images and labels. Override to subclass. 

        The new_polygons_df and new_graph arguments contain all the information available to decide which windows to select. They should not be modified by this method. 

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
                new_graph: BipartiteGraph) -> dict:
        """
        Cut new images from source image, update new_polygons_df and new_graph to account for the new images, and return a dict with keys the index and column names of the imgs_df to be created by the calling dataset cutter and values lists containing the new image names and corresponding entries for the new images. See small_imgs_around_polygons_cutter for an example. 

        Args:
            source_img_name (str): name of img in self.source_img_path to be cut.
            new_polygons_df (GeoDataFrame): GeoDataFrame that will be the polygons_df of the associator of the new dataset of cut images that is being created by the calling dataset cutter. 
            new_graph (BipartiteGraph): the bipartite graph that is being built up for the target associator.

        Returns:
            dict of lists that containing the data to be put in the imgs_df of the associator to be constructed for the created images. 
        """
        # img and labels paths
        source_img_path = Path(self.source_assoc.data_dir / f"images/{source_img_name}")
        source_label_path = Path(self.source_assoc.data_dir / f"labels/{source_img_name}")        

        # dict to accumulate information about the newly created images
        imgs_from_cut_dict = {index_or_col_name: [] for index_or_col_name in [self.source_assoc.imgs_df.index.name] + list(self.source_assoc.imgs_df.columns)}

        windows_transforms_img_names = self._get_windows_transforms_img_names(source_img_name, 
                                                                            new_polygons_df, 
                                                                            new_graph, 
                                                                            **self.kwargs)

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
                            new_img_name: str, 
                            img_bounds_in_img_crs: Tuple[float, float, float, float], 
                            img_crs: CRS) -> dict:
        """
        Return an img info dict for a single new image. 

        An img info dict contains the following key/value pairs:
            - key: the index name of the imgs_df to be created by calling dataset cutter,
                value: the image name of the new image.
            - keys: the columns names of the imgs_df to be created by calling dataset cutter,
                value: the entries to be written in those columns for the new image.
        
        Args:
            new_img_name (str): name of new image
            img_bounds_in_img_crs (Tuple[float, float, float, float]): image bounds
            img_crs (CRS): CRS of img

        Returns:
            dict: img info dict (see above)
        """
            
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
        """Make a new image and label with given image name from the given window and transform"""

        dst_img_path = self.target_data_dir / f"images/{new_img_name}"
        dst_label_path = self.target_data_dir / f"labels/{new_img_name}"

        # write img window to destination img geotif
        img_bounds_in_img_crs, img_crs = self._write_window_to_geotif(self.source_img_path, 
                                                                dst_img_path, 
                                                                self.img_bands, 
                                                                window, 
                                                                window_transform)

        # write label window to destination label geotif
        if self.source_label_path.is_file():
            label_bounds_in_img_crs, label_crs = self._write_window_to_geotif(
                                                    self.source_label_path, 
                                                    dst_label_path, 
                                                    self.label_bands, 
                                                    window, 
                                                    window_transform)    
        else:
            logger.info('No label cut for img {self.source_img_path.name} since it has no label.')

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
                for target_band, source_band in enumerate(bands, start=1):
                            
                    # Read window for that band from source ...
                    new_img_band_raster = src.read(source_band, window=window)

                    # ... write to new geotiff.
                    dst.write(new_img_band_raster, target_band)

        return dst.bounds, dst.crs

    def _get_all_band_indices(self, mode: str) -> List[int]:
        """
        Return list of all band indices of GeoTiffs. 

        It is assumed all images (or labels) in the data directory have the same number of bands.

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
