"""
Abstract base class for single image cutters. 
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Union, List, Tuple, Optional, Any
from pathlib import Path
import os
from abc import ABC, abstractmethod

from tqdm import tqdm
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
if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator
from rs_tools.cut.polygon_filter_predicates import PolygonFilterPredicate
from rs_tools.utils.utils import transform_shapely_geometry

logger = logging.getLogger(__name__)

class SingleImgCutter(ABC):
    def __init__(self, 
            source_assoc : ImgPolygonAssociator, 
            target_images_dir : Union[Path, str], 
            target_labels_dir : Union[Path, str], 
            img_bands : Optional[List[int]], 
            label_bands : Optional[List[int]], 
            **kwargs : Any):
        """
        Abstract base class for single image cutters. 

        To define an image cutter, override _get_windows_transforms_img_names method. 

        A SingleImgCutter is a Callable that on a given call creates new images cut from a single source image and returns a dict containing information about the created images to be used by the calling function to add to the imgs_df of the target associator (see __call__ docstring for details). 

        Args:
            source_assoc (ImgPolygonAssociator): source associator containing images/labels to be cut from.
            target_images_dir (Union[Path, str): images directory of target dataset
            target_labels_dir (Union[Path, str): labels directory of target dataset
            img_bands (Optional[List[int]]): list of bands to extract from the image (note GeoTiff bands start at 1).
            labels_bands (Optional[List[int]]): list of bands to extract from the label (note GeoTiff bands start at 1).
    
        Raises:
            ValueError: if the mode is unknown.
            TypeError: if the new image size is not an integer or a pair of integers.
            ValueError: if the integer(s) defining the size of the new images are not positive.
        """

        self.source_assoc = source_assoc
        self.target_images_dir = Path(target_images_dir)
        self.target_labels_dir = Path(target_labels_dir)
        
        if img_bands is None:
            self.img_bands = self._get_all_band_indices('images')
        else:
            self.img_bands = img_bands

        if label_bands is None:
            self.label_bands = self._get_all_band_indices('labels')
        else:
            self.label_bands = label_bands

        self.polygons_df_crs_epsg = self.source_assoc.polygons_df.crs.to_epsg()

        self.kwargs = kwargs 

    @abstractmethod
    def _get_windows_transforms_img_names(self, 
            source_img_name : str, 
            target_assoc : Optional[ImgPolygonAssociator] = None, 
            new_imgs_dict : Optional[dict] = None,
            **kwargs : Any
            ) -> List[Tuple[Window, Affine, str]]:
        """
        Return a list of rasterio windows, window transformations, and new image names. The returned list will be used to create the new images and labels. Override to subclass. 

        Args:
            source_img_name (str): name of img in source dataset to be cut.
            target_assoc (ImgPolygonAssociator): associator of target dataset
            new_imgs_dict (dict): dict with keys index or column names of target_assoc.imgs_df and values lists of entries correspondong to images containing information about cut images not yet appended to target_assoc.imgs_df
            kwargs (Any): keyword arguments to be used in subclass implementations.

        Returns:
            List[Tuple[Window, Affine, str]]: list of rasterio windows, window transform, and new image names. 
        """

        pass

    def __call__(self, 
            img_name : str, 
            target_assoc : Optional[ImgPolygonAssociator] = None, 
            new_imgs_dict : Optional[dict] = None,
            **kwargs : Any
            ) -> dict:
        """
        Cut new images from source image and return a dict with keys the index and column names of the imgs_df to be created by the calling dataset cutter and values lists containing the new image names and corresponding entries for the new images. See small_imgs_around_polygons_cutter for an example. 

        Args:
            img_name (str): name of img in source dataset to be cut.
            target_assoc (ImgPolygonAssociator): associator of target dataset
            new_imgs_dict (dict): dict with keys index or column names of target_assoc.imgs_df and values lists of entries correspondong to images containing information about cut images not yet appended to target_assoc.imgs_df
            kwargs (Any): optional keyword arguments for _get_windows_transforms_img_names

        Returns:
            dict of lists that containing the data to be put in the imgs_df of the associator to be constructed for the created images. 

        Note:
            The __call__ function should be able to access the information contained in the target (and source) associator but should *not* modify its arguments! Since create_or_update_dataset_from_iter_over_polygons and create_or_update_dataset_from_iter_over_imgs do not concatenate the information about the new images that have been cut to the target_assoc.imgs_df until after all polygons or images have been iterated over and we want to be able to use ImgSelectors _during_ such an iteration, we allow the call function to also depend on a new_imgs_dict argument which contains the information about the new images that have been cut. Unlike the target_assoc.imgs_df, the target_assoc.polygons_df and graph are updated during the iteration. One should thus think of the target_assoc and new_imgs_dict arguments together as the actual the target associator argument. 
        """
        
        # dict to accumulate information about the newly created images
        imgs_from_cut_dict = {index_or_col_name: [] for index_or_col_name in [self.source_assoc.imgs_df.index.name] + list(self.source_assoc.imgs_df.columns)}

        windows_transforms_img_names = self._get_windows_transforms_img_names(
            source_img_name=img_name, 
            target_assoc=target_assoc, 
            new_imgs_dict=new_imgs_dict,
            **kwargs)

        for window, window_transform, new_img_name in windows_transforms_img_names:

            # Make new image and label in target dataset ...
            img_bounds_in_img_crs, img_crs = self._make_new_img_and_label(
                                                new_img_name=new_img_name, 
                                                source_img_name=img_name, 
                                                window=window, 
                                                window_transform=window_transform)

            # ... gather all the information about the image in a dict ...
            single_new_img_info_dict = self._make_img_info_dict(
                                            new_img_name=new_img_name, 
                                            source_img_name=img_name, 
                                            img_bounds_in_img_crs=img_bounds_in_img_crs, 
                                            img_crs=img_crs)

            # ... and accumulate that information. 
            for key in imgs_from_cut_dict.keys():
                imgs_from_cut_dict[key].append(single_new_img_info_dict[key])

        return imgs_from_cut_dict

    def _make_img_info_dict(self, 
            new_img_name : str, 
            source_img_name : str, 
            img_bounds_in_img_crs : Tuple[float, float, float, float], 
            img_crs : CRS
            ) -> dict:
        """
        Return an img info dict for a single new image. 

        An img info dict contains the following key/value pairs:
            - key: the index name of the imgs_df to be created by calling dataset cutter,
                value: the image name of the new image.
            - keys: the columns names of the imgs_df to be created by calling dataset cutter,
                value: the entries to be written in those columns for the new image.
        
        Args:
            new_img_name (str): name of new image
            source_img_name (str): name of source image
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
            single_new_img_info_dict[col] = self.source_assoc.imgs_df.loc[source_img_name, col]

        return single_new_img_info_dict
            
    def _make_new_img_and_label(self, 
            new_img_name : str, 
            source_img_name : str, 
            window : Window, 
            window_transform : Affine, 
            ) -> Tuple[Tuple[float, float, float, float], CRS]:
        """
        Make a new image and label with given image name from the given window and transform

        Args:
            new_img_name (str): name of new image
            source_img_name (str): name of source image
            window (Window): 
            window_transform (Affine): 

        Returns:
            Tuple[Tuple[float, float, float, float], CRS]: tuple of bounds (in image CRS) and CRS of new image
        """
        
        source_img_path = self.source_assoc._images_dir / source_img_name
        source_label_path = self.source_assoc._labels_dir / source_img_name

        dst_img_path = self.target_images_dir / new_img_name
        dst_label_path = self.target_labels_dir / new_img_name

        # write img window to destination img geotif
        img_bounds_in_img_crs, img_crs = self._write_window_to_geotif(
                                            source_img_path, 
                                            dst_img_path, 
                                            self.img_bands, 
                                            window, 
                                            window_transform)

        # write label window to destination label geotif
        if source_label_path.is_file():
            label_bounds_in_img_crs, label_crs = self._write_window_to_geotif(
                                                    source_label_path, 
                                                    dst_label_path, 
                                                    self.label_bands, 
                                                    window, 
                                                    window_transform)    
            assert img_crs == label_crs, "source image and label crs disagree!"
            assert label_bounds_in_img_crs == img_bounds_in_img_crs, "source image and label bounds disagree"
        else:
            logger.info('No label cut for img {source_img_path.name} since it has no label.')

        return img_bounds_in_img_crs, img_crs

    def _write_window_to_geotif(self, 
            src_img_path : Union[Path, str], 
            dst_img_path : Union[Path, str], 
            bands : List[int],
            window : Window, 
            window_transform : Affine
            ) -> Tuple[Tuple[float, float, float, float], CRS]:
        """
        Write window from source GeoTiff to new GeoTiff.

        Args:
            src_img_path (Union[Path, str]): path of source GeoTiff
            dst_img_path (Union[Path, str]): path to GeoTiff to be created
            bands (List[int]): bands to extract from source GeoTiff
            window (Window): window to cut out from source GeoTiff
            window_transform (Affine): window transform of window

        Returns:
            Tuple[Tuple[float, float, float, float], CRS]: bounds (in image CRS) and CRS of new image
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

        assert mode in {'images', 'labels'}

        img_or_label_dir = self.source_assoc._images_dir if mode =='images' else self.source_assoc._labels_dir
        img_or_label_path = [filepath for filepath in img_or_label_dir.iterdir() if filepath.suffix == '.tif'][0]

        with rio.open(img_or_label_path) as src:
            bands = list(range(1, src.count + 1))

        return bands
