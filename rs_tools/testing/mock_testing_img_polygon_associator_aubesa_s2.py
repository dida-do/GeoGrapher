"""Mock AuBeSa S2 associator for testing the download function.

Virtually 'downloads' (no files operations are actually done) from a
dataset of images in a source directory.
"""

import random
from pathlib import Path
from typing import List, Optional, Set, Union

import shapely
from geopandas.geodataframe import GeoDataFrame

from assoc import ImgPolygonAssociator
from assoc.errors import (ImgAlreadyExistsError, ImgDownloadError,
                          NoImgsForPolygonFoundError)
from assoc.tests.img_polygon_associator_artificial_data_test import \
    SEGMENTATION_CLASSES
from aubesa.aubassoc import raster_imgs_from_s2_tif_dir as raster_imgs_from_dir

# Parameters for download method of mock associator
PROBABILITY_OF_DOWNLOAD_ERROR = 0.1
PROBABILITY_IMG_ALREADY_DOWNLOADED = 0.1
"""
# Parameters to construct the associator with:
# 
TAILING_TYPES = ['ct', 'ht' ,'pt', 'it', 'rt', 'dt', 'ts', 'wr', 'h'] 
LABEL_TYPE = 'soft-categorical'

# polygons_df column and index names
IMGS_DF_INDEX_NAME_AND_TYPE = {'img_name': str}
IMGS_DF_INDEX_NAME = list(IMGS_DF_INDEX_NAME_AND_TYPE.keys())[0]
IMGS_DF_COLS_AND_TYPES = {'geometry': shapely.geometry, 
                                    'orig_crs_epsg_code': int, 
                                    'img_processed?': bool, 
                                    'timestamp': str} # YYYY-MM-DD HH:MM:SS
IMGS_DF_COLS_AND_INDEX_TYPES = {**IMGS_DF_INDEX_NAME_AND_TYPE, **IMGS_DF_COLS_AND_TYPES}

# raster_imgs column and index names 
POLYGONS_DF_INDEX_NAME_AND_TYPE = {'polygon_name': Union[str, int]}
POLYGONS_DF_INDEX_NAME = list(POLYGONS_DF_INDEX_NAME_AND_TYPE)[0]
POLYGONS_DF_COLS_AND_TYPES = {'geometry': shapely.geometry, 
                                        'have_img?': bool, 
                                        'have_img_downloaded?': bool,
                                        'download_exception': str,
                                        **{f"prob_seg_class_{seg_class}": float for seg_class in TAILING_TYPES},
                                        'visibility': int,
                                        'date': str}
POLYGONS_DF_COLS_AND_INDEX_TYPES = {**POLYGONS_DF_INDEX_NAME_AND_TYPE, **POLYGONS_DF_COLS_AND_TYPES}
"""

from aubesa.aubassoc.img_polygon_associator_aubesa_s2 import LABEL_TYPE


class TestImgPolygonAssociator(ImgPolygonAssociator):
    """ImgPolygonAssociator with mock download function that "downloads" images
    from an existing source dataset.

    Useful for debugging the (base) ImgPolygonAssociator's
    download_missing_imgs_for_polygons_df method.
    """

    def __init__(self,
                 data_dir: Union[Path, str],
                 source_data_dir: Union[Path, str],
                 raster_imgs: Optional[GeoDataFrame] = None,
                 polygons_df: Optional[GeoDataFrame] = None,
                 label_type: str = LABEL_TYPE,
                 segmentation_classes: List[str] = SEGMENTATION_CLASSES,
                 **kwargs):
        """
        Args:
            data_dir (Union[Path, str]): data directory
            source_data_dir (Union[Path, str]): data directory of source dataset
            raster_imgs (Optional[GeoDataFrame], optional): raster_imgs. Defaults to None.
            polygons_df (Optional[GeoDataFrame], optional): polygons_df. Defaults to None.
            label_type (str, optional): label type. Defaults to LABEL_TYPE.
            segmentation_classes (List[str], optional): segmentation classes. Defaults to SEGMENTATION_CLASSES.
        """

        self.source_data_dir = source_data_dir

        # Create source associator from imgs in source data dir and given polygons in polygons_df.
        source_raster_imgs = raster_imgs_from_dir(source_data_dir)
        self.source_assoc = ImgPolygonAssociator(
            Path("/home/rustam/whatever/"
                 ),  # no files will be created, so can use fake directory
            raster_imgs=source_raster_imgs,
            polygons_df=polygons_df,
            segmentation_classes=segmentation_classes,
            label_type=label_type,
            **kwargs)

        super().__init__(data_dir,
                         raster_imgs=raster_imgs,
                         polygons_df=polygons_df,
                         segmentation_classes=segmentation_classes,
                         label_type=label_type,
                         **kwargs)

    def integrate_new_polygons_df(self,
                                  new_polygons_df: GeoDataFrame,
                                  force_overwrite: bool = False) -> None:
        """Integrate the new polygons into source associator before integrating
        them into self."""

        # Integrate polygons_df into source associator, so the download function can find which images contain the new polygons.
        self.source_assoc.integrate_new_polygons_df(
            new_polygons_df=new_polygons_df, force_overwrite=force_overwrite)

        # Then just integrate the new polygons_df.
        return super().integrate_new_polygons_df(
            new_polygons_df, force_overwrite=force_overwrite)

    def download_missing_imgs_for_polygons_df(self,
                                              num_target_imgs_per_polygon: int,
                                              polygons_df: GeoDataFrame,
                                              add_labels: bool = False,
                                              **kwargs) -> None:
        """
        Call the parent class download function, but with add_labels = False (since there are no real images being 'downloaded').
        """

        return super().download_missing_imgs_for_polygons_df(
            num_target_imgs_per_polygon=num_target_imgs_per_polygon,
            polygons_df=polygons_df,
            add_labels=False,
            **kwargs)

    def _download_imgs_for_polygon(self, polygon_name: Union[str, int],
                                   polygon_geometry: shapely.geometry.Polygon,
                                   download_dir: Union[Path, str],
                                   previously_downloaded_imgs_set: Set[str],
                                   **kwargs) -> None:
        """'Download' an image fully containing a vector polygon or several
        images jointly containing it from the source_data_dir and return a dict
        with information to be updated in the associator, see below for
        details.

        Args:
            polygon_name: the name of the vector polygon.
            polygon_geometry: shapely geometry of polygon.
            download_dir: directory that the image file should be downloaded to.
            previously_downloaded_imgs_set: Set of previously downloaded img_names. In some use cases when it can't be guaranteed that an image can be downloaded that fully contains the polygon it can happen that attempts will be made to download an image that is already in the associator. Passing this argument allows the download function to make sure it doesn't try downloading an image that is already in the dataset.
            **kwargs: optional keyword arguments depending on the application.
        Returns:
             A dict with keys and values:
                'list_img_info_dicts': a list of dicts containing the information to be included in each row in the raster_imgs of the calling associator, one for each newly downloaded image. The keys should be the index and column names of the raster_imgs and the values the indices or entries of those columns in row that will correspond to the new image.
        """

        # Make sure the polygon is in self.source_assoc. This should be true by construction.
        if not polygon_name in self.source_assoc.polygons_df.index:
            raise Exception(
                f"Polygon {polygon_name} not in source associator. This shouldn't have happened, since the source associator should contain all polygons of the mock test associator."
            )

        # Find the images in self.source_assoc containing the polygon
        imgs_containing_polygon = list(
            self.source_assoc.imgs_containing_polygon(polygon_name))

        # If there isn't such an image ...
        if imgs_containing_polygon == []:

            # ... inform the calling download_missing_imgs_for_polygons_df by raising an error.
            raise NoImgsForPolygonFoundError(
                f"No images containing polygon {polygon_name} found in source dataset in {self.source_data_dir}!"
            )

        # Else, there is an image in the source dataset containing the polygon.
        else:

            # With some probability the API answers our query with an image that has already been downloaded...
            if imgs_containing_polygon and random.random(
            ) < PROBABILITY_IMG_ALREADY_DOWNLOADED:

                # ... in which case we raise an error.
                raise ImgAlreadyExistsError(
                    f"random.random() was less than PROBABILITY_IMG_ALREADY_DOWNLOADED= {PROBABILITY_IMG_ALREADY_DOWNLOADED}."
                )

            # Else, from not previously downloaded images ...
            remaining_imgs = [
                img for img in imgs_containing_polygon
                if img not in previously_downloaded_imgs_set
            ]

            if remaining_imgs:

                # ... choose one to 'download'.
                img_name = random.choice(remaining_imgs)

                # With some probabibility  ...
                if random.random() < PROBABILITY_OF_DOWNLOAD_ERROR:

                    # ... an error occurs when downloading, so we raise an ImgDownloadError.
                    raise ImgDownloadError(
                        f"random.random() was less than PROBABILITY_OF_DOWNLOAD_ERROR={PROBABILITY_OF_DOWNLOAD_ERROR}."
                    )

                # ... 'download' it, i.e. return the corresponding return dict.
                img_info_dict = {
                    'img_name':
                    img_name,
                    'img_processed?':
                    False,
                    'timestamp':
                    self.source_assoc.raster_imgs.loc[img_name, 'timestamp']
                }

                return {
                    'list_img_info_dicts': [img_info_dict],
                }

            else:

                raise NoImgsForPolygonFoundError(
                    f"No new images containing polygon {polygon_name} found in source dataset in {self.source_data_dir}!"
                )

    def _process_downloaded_img_file(self, img_name: str, in_dir: Union[Path,
                                                                        str],
                                     out_dir: Union[Path, str],
                                     convert_to_crs_epsg: int, **kwargs):
        """'Process' an image file downloaded by _download_imgs_for_polygon
        (i.e. do nothing) and return a dict with information to be updated in
        the associator, see below for details.

        Args:
            img_name (str): the image name (index identifiying the corresponding row in raster_imgs)
            in_dir (Union[Path, str]): the directory the image file was downloaded to
            out_dir (Union[Path, str]): the directory the processed image file should be in (usually data_dir/images)
            convert_to_crs_epsg (int): EPSG code of the crs the image bounding rectangle should be converted to
            **kwargs: optional keyword arguments depending on the application
        Returns:
            img_info_dict: a dict containing the information to be updated in the raster_imgs of the calling associator. The keys should be the index and column names of the raster_imgs and the values lists of indices or entries of those columns.
        """

        return {
            'img_name':
            img_name,
            'geometry':
            self.source_assoc.raster_imgs.loc[img_name, 'geometry'],
            'orig_crs_epsg_code':
            self.source_assoc.raster_imgs.loc[img_name, 'orig_crs_epsg_code'],
            'img_processed?':
            True
        }
