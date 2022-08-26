"""Abstract base class for single image cutters."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import rasterio as rio
from affine import Affine
from pydantic import BaseModel
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import box

from geographer.connector import Connector
from geographer.global_constants import RASTER_IMGS_INDEX_NAME
from geographer.img_bands_getter_mixin import ImgBandsGetterMixIn

logger = logging.getLogger(__name__)


class SingleImgCutter(ABC, BaseModel, ImgBandsGetterMixIn):
    """Base class for SingleImgCUtter."""

    @abstractmethod
    def _get_windows_transforms_img_names(
            self,
            source_img_name: str,
            source_connector: Connector,
            target_connector: Optional[Connector] = None,
            new_imgs_dict: Optional[dict] = None,
            **kwargs: Any) -> List[Tuple[Window, Affine, str]]:
        """Return a list of rasterio windows, window transformations, and new
        image names. The returned list will be used to create the new images
        and labels. Override to subclass.

        Args:
            source_img_name (str): name of img in source dataset to be cut.
            target_connector (Connector): connector of target dataset
            new_imgs_dict (dict): dict with keys index or column names of
                target_connector.raster_imgs and values lists of entries corresponding
                to images containing information about
            cut images not yet appended to target_connector.raster_imgs
            kwargs (Any): keyword arguments to be used in subclass implementations.

        Returns:
            List[Tuple[Window, Affine, str]]: list of rasterio windows, window
            transform, and new image names.
        """

    def __call__(self,
                 img_name: str,
                 source_connector: Connector,
                 target_connector: Optional[Connector] = None,
                 new_imgs_dict: Optional[dict] = None,
                 bands: Optional[Dict[str, Optional[List[int]]]] = None,
                 **kwargs: Any) -> dict:
        """Cut new images from source image and return a dict with keys the
        index and column names of the raster_imgs to be created by the calling
        dataset cutter and values lists containing the new image names and
        corresponding entries for the new images.

        Args:
            img_name: name of img in source dataset to be cut.
            target_connector: connector of target dataset
            new_imgs_dict: dict with keys index or column names of
                target_connector.raster_imgs and values lists of entries correspondong
                to images containing information
            about cut images not yet appended to target_connector.raster_imgs
            kwargs: optional keyword arguments for _get_windows_transforms_img_names

        Returns:
            dict of lists that containing the data to be put in the raster_imgs of the
            connector to be constructed for the created images.

        Note:
            The __call__ function should be able to access the information
            contained in the target (and source) connector but should *not*
            modify its arguments! Since create_or_update does not concatenate
            the information about the new images that have been cut to the
            target_connector.raster_imgs until after all features or images
            have been iterated over and we want to be able to use ImgSelectors
            _during_ such an iteration, we allow the call function to also depend
            on a new_imgs_dict argument which contains the information about the new
            images that have been cut. Unlike the target_connector.raster_imgs, the
            target_connector.vector_features and graph are updated during the
            iteration. One should thus think of the target_connector and new_imgs_dict
            arguments together as the actual the target connector argument.
        """

        # dict to accumulate information about the newly created images
        imgs_from_cut_dict = {
            index_or_col_name: []
            for index_or_col_name in [RASTER_IMGS_INDEX_NAME] +
            list(source_connector.raster_imgs.columns)
        }

        windows_transforms_img_names = self._get_windows_transforms_img_names(
            source_img_name=img_name,
            source_connector=source_connector,
            target_connector=target_connector,
            new_imgs_dict=new_imgs_dict,
            **kwargs)

        for window, window_transform, new_img_name in windows_transforms_img_names:

            # Make new image and label in target dataset ...
            img_bounds_in_img_crs, img_crs = self._make_new_img_and_label(
                new_img_name=new_img_name,
                source_img_name=img_name,
                source_connector=source_connector,
                target_connector=target_connector,
                window=window,
                window_transform=window_transform,
                bands=bands,
            )

            # ... gather all the information about the image in a dict ...
            single_new_img_info_dict = self._make_img_info_dict(
                new_img_name=new_img_name,
                source_img_name=img_name,
                source_connector=source_connector,
                img_bounds_in_img_crs=img_bounds_in_img_crs,
                img_crs=img_crs,
            )

            # ... and accumulate that information.
            for key in imgs_from_cut_dict.keys():
                imgs_from_cut_dict[key].append(single_new_img_info_dict[key])

        return imgs_from_cut_dict

    def _make_img_info_dict(
        self,
        new_img_name: str,
        source_img_name: str,
        source_connector: Connector,
        img_bounds_in_img_crs: Tuple[float, float, float, float],
        img_crs: CRS,
    ) -> dict:
        """Return an img info dict for a single new image.

        An img info dict contains the following key/value pairs:
            - keys: the columns names of the raster_img to be created by calling
                dataset cutter,
            - values: the entries to be written in those columns for the new image.

        Args:
            new_img_name: name of new image
            source_img_name: name of source image
            img_bounds_in_img_crs: image bounds
            img_crs: CRS of img

        Returns:
            dict: img info dict (see above)
        """

        img_bounding_rectangle_in_raster_imgs_crs = box(*transform_bounds(
            img_crs, source_connector.raster_imgs.crs, *img_bounds_in_img_crs))

        single_new_img_info_dict = {
            RASTER_IMGS_INDEX_NAME: new_img_name,
            'geometry': img_bounding_rectangle_in_raster_imgs_crs,
            'orig_crs_epsg_code': img_crs.to_epsg(),
            'img_processed?': True
        }

        # Copy over any remaining information about the img from
        # source_connector.raster_imgs.
        for col in set(source_connector.raster_imgs.columns) - {
                RASTER_IMGS_INDEX_NAME, 'geometry', 'orig_crs_epsg_code',
                'img_processed?'
        }:
            single_new_img_info_dict[col] = source_connector.raster_imgs.loc[
                source_img_name, col]

        return single_new_img_info_dict

    def _make_new_img_and_label(
        self,
        new_img_name: str,
        source_img_name: str,
        source_connector: Connector,
        target_connector: Connector,
        window: Window,
        window_transform: Affine,
        bands: Optional[Dict[str, Optional[List[int]]]],
    ) -> Tuple[Tuple[float, float, float, float], CRS]:
        """Make a new image and label with given image name from the given
        window and transform.

        Args:
            new_img_name: name of new image
            source_img_name: name of source image
            window: window
            window_transform: window transform

        Returns:
            tuple of bounds (in image CRS) and CRS of new image
        """

        for count, (source_images_dir, target_images_dir) in enumerate(
                zip(source_connector.image_data_dirs,
                    target_connector.image_data_dirs)):

            source_img_path = source_images_dir / source_img_name
            dst_img_path = target_images_dir / new_img_name

            if not source_img_path.is_file(
            ) and count > 0:  # count == 0 corresponds to images_dir
                continue
            else:
                img_bands = self._get_bands_for_img(bands, source_img_path)

                # write img window to destination img geotif
                bounds_in_img_crs, crs = self._write_window_to_geotif(
                    source_img_path, dst_img_path, img_bands, window,
                    window_transform)

            # make sure all images/labels/masks have same bounds and crs
            if count == 0:
                img_bounds_in_img_crs, img_crs = bounds_in_img_crs, crs
            else:
                assert crs == img_crs, f"new image and {target_images_dir.name} crs disagree!"
                assert bounds_in_img_crs == img_bounds_in_img_crs, f"new image and {target_images_dir.name} bounds disagree"

        return img_bounds_in_img_crs, img_crs

    def _write_window_to_geotif(
        self,
        src_img_path: Union[Path, str],
        dst_img_path: Union[Path, str],
        img_bands: List[int],
        window: Window,
        window_transform: Affine,
    ) -> Tuple[Tuple[float, float, float, float], CRS]:
        """Write window from source GeoTiff to new GeoTiff.

        Args:
            src_img_path: path of source GeoTiff
            dst_img_path: path to GeoTiff to be created
            img_bands: bands to extract from source GeoTiff
            window: window to cut out from source GeoTiff
            window_transform: window transform of window

        Returns:
            bounds (in image CRS) and CRS of new image
        """

        # Open source ...
        with rio.open(src_img_path) as src:

            # and destination ...
            Path(dst_img_path).parent.mkdir(exist_ok=True, parents=True)
            with rio.open(Path(dst_img_path),
                          'w',
                          driver='GTiff',
                          height=window.height,
                          width=window.width,
                          count=len(img_bands),
                          dtype=src.profile["dtype"],
                          crs=src.crs,
                          transform=window_transform) as dst:

                # ... and go through the bands.
                for target_band, source_band in enumerate(img_bands, start=1):

                    # Read window for that band from source ...
                    new_img_band_raster = src.read(source_band, window=window)

                    # ... write to new geotiff.
                    dst.write(new_img_band_raster, target_band)

        return dst.bounds, dst.crs
