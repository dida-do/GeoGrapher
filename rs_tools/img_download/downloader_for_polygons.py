"""
Class for downloading images for polygons targeting a given number of images per polygon.
"""

import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Set, Union

from geopandas import GeoDataFrame
from pydantic import BaseModel, Field
from shapely.ops import unary_union
from tqdm.auto import tqdm
from rs_tools import ImgPolygonAssociator
from rs_tools.base_model_dict_conversion.save_base_model import \
    SaveAndLoadBaseModelMixIn
from rs_tools.errors import (
    ImgAlreadyExistsError,
    ImgDownloadError,
    NoImgsForPolygonFoundError,
)
from rs_tools.img_download.base_download_processor import ImgDownloadProcessor
from rs_tools.img_download.base_downloader_for_single_polygon import \
    ImgDownloaderForSinglePolygon
from rs_tools.utils.utils import concat_gdfs

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


class ImgDownloaderForPolygons(BaseModel, SaveAndLoadBaseModelMixIn):
    """
    Download images for polygons targeting a given number of images per polygon.
    """

    downloader_for_single_polygon: ImgDownloaderForSinglePolygon
    download_processor: ImgDownloadProcessor
    kwarg_defaults: dict = Field(default_factory=dict)

    def download(self,
                 assoc: Union[Path, str, ImgPolygonAssociator],
                 polygon_names: Optional[Union[str, int, List[int],
                                               List[str]]] = None,
                 target_img_count: int = 1,
                 add_labels: bool = True,
                 filter_out_polygons_contained_in_union_of_intersecting_imgs:
                 bool = False,
                 shuffle_polygons: bool = True,
                 **kwargs):
        """Download images for polygons so as to target a number of images per polygon.

        Sequentially considers the polygons for which the image count (number of images fully
        containing a given polygon) is less than num_target_imgs_per_polygon images in the associator's
        internal polygons_df or the optional polygons_df argument (if given), for each such polygon
        attempts to download num_target_imgs_per_polygon - image_count images fully containing the polygon
        (or several images jointly containing the polygon), creates the associated label(s) for the image(s)
        (assuming the default value True of add_labels is not changed), and integrates the new image(s)
        into the dataset/associator. Integrates images downloaded for a polygon into the dataset/associator
        immediately after downloading them and before downloading images for the next polygon. In particular,
        the image count is updated immediately after each download.

        Warning:
            The targeted number of downloads is determined by target_img_count
            and a polygons img_count. Since the img_count is the number of
            images in the dataset fully containing a polygon for "large"
            polygons the img_count will always remain zero and every call of the
            download_imgs method that includes this polygon will download
            target_img_count images (or image series). To avoid this, you can use
            the filter_out_polygons_contained_in_union_of_intersecting_imgs argument.

        Args:
            polygon_names (List[str], optional): Optional polygon_name or list of polygon_names to download images for. Defaults to None, i.e. consider all polygons in assoc.polygons_df.
            downloader (str): One of 'sentinel2' or 'jaxa'. Defaults, if possible, to previously used downloader.
            target_img_count (int): target for number of images per polygon in the dataset after downloading. The actual number of images for each polygon P that fully contain it could be lower if there are not enough images available or higher if after downloading num_target_imgs_per_polygon images for P P is also contained in images downloaded for other polygons.
            polygons_df (GeoDataFrame, optional): (Probably just best ignore this) GeoDataFrame of polygons conforming to the associator's format for polygon_df, defaults to the associator's internal polygons_df (i.e. assoc.polygons_df). If provided and not equal to assoc.polygons_df will download images for only those polygons and integrate the polygons in polygons_df into the associator after the images have been downloaded.
            add_labels (bool, optional): bool. Whether to add labels for the downloaded images. Defaults to True.
            filter_out_polygons_contained_in_union_of_intersecting_imgs (bool): Useful when dealing with 'large' polygons. Defaults to False.
            shuffle_polygons (bool): Whether to shuffle order of polygons for which images will be downloaded. Might in practice prevent an uneven distribution of the image count for repeated downloads. Defaults to True.
            kwargs (dict, optional): additional keyword arguments passed to downloader_for_single_polygon and download_processor. Defaults to self.kwarg_defaults.

        Note:
            Any kwargs given will be saved to self.default_kwargs and become default values.

        # TODO: move description?
        Kwargs (downloader='jaxa'):
            data_version (str): One of '1804', '1903', '2003', or '2012'.
                Defaults if possible to whichever choice you made last time,
                else to '1804'.
            download_mode (str): One of 'bboxvertices' (download images for
                vertices of the bbox of the polygon, preferred for
                small polygons, but will miss inbetween if the polygon spans
                more than two images in each axis), 'bboxgrid' (download images
                for each point on a grid defined by the bbox. Overshoots for
                small polygons, but works for large polygons). Defaults if possible
                to whichever choice you made last time, else to 'bboxvertices'.

        Returns:
            None

        Warning:
            It's easy to come up with examples where the image count distribution (i.e. distribution of images per polygon) becomes unbalanced particularly if num_target_imgs_per_polygon is large. These scenarios are not necessarily very likely, but possible. As an example, if one wants to download say 5 images images for a polygon that is not fully contained in any image in the dataset and if there does not exist an image we can download that fully contains it but there are 20 disjoint sets of images we can download that jointly cover the polygon then these 20 disjoint sets will all be downloaded.
        """

        self.kwarg_defaults.update(kwargs)

        if not isinstance(assoc, ImgPolygonAssociator):
            assoc = ImgPolygonAssociator.from_data_dir(assoc)
        assoc.images_dir.mkdir(parents=True, exist_ok=True)
        assoc.download_dir.mkdir(
            parents=True,
            exist_ok=True)  #TODO: assoc.data_dir / 'downloads' or whatever

        polygons_to_download = self._get_polygons_to_download(
            polygon_names=polygon_names,
            target_img_count=target_img_count,
            assoc=assoc,
            filter_out_polygons_contained_in_union_of_intersecting_imgs=
            filter_out_polygons_contained_in_union_of_intersecting_imgs,
        )

        if shuffle_polygons:
            random.shuffle(polygons_to_download)

        previously_downloaded_imgs_set = set(assoc.imgs_df.index)
        # (Will be used to make sure no attempt is made to download an image more than once.)

        # Dict to keep track of imgs we've downloaded. We'll append this to assoc.imgs_df as a (geo)dataframe later
        new_imgs_dict = defaultdict(list)

        pbar = tqdm(
            enumerate(assoc.polygons_df[[
                "geometry"
            ]].loc[polygons_to_download].itertuples(),
                      start=1))
        for count, (polygon_name, polygon_geometry) in pbar:

            # polygon_geometry = assoc.polygons_df.loc[polygon_name, 'geometry']

            pbar.set_description(
                f"Polygon {count}/{len(polygons_to_download)}", )
            log.debug(
                "download_missing_imgs_for_polygons_df: considering polygon %s.",
                polygon_name)

            # Since we process and connect each image after downloading it, we might not need to download
            # an image for a polygon that earlier was lacking an image if it is now contained in one of the already downloaded images,
            # so need to check again that there are not enough images for the polygon (since the iterator above is set when it is called
            # and won't know if the "img_count" column value has been changed in the meanwhile).
            num_img_series_to_download = target_img_count - assoc.polygons_df.loc[
                polygon_name, "img_count"]
            if num_img_series_to_download <= 0:
                log.debug(
                    "Skipping polygon %s since there now enough images fully containing it.",
                    polygon_name)
                continue

            while num_img_series_to_download > 0:

                # Try downloading an image series and save returned dict (of dicts)
                # containing information for polygons_df, assoc.imgs_df...
                try:

                    # DEBUG INFO
                    log.debug("attempting to download image for polygon. %s",
                              polygon_name)

                    return_dict = self.downloader_for_single_polygon.download(
                        polygon_name,
                        polygon_geometry,
                        assoc.download_dir,
                        previously_downloaded_imgs_set,  # _download_imgs_for_polygon should use this to make sure no attempt at downloading an already downloaded image is made.
                        **self.kwarg_defaults,
                    )

                # WHY DOES THIS NOT WORK?
                # except TypeError as exc:
                #     log.exception("Probably missing kwargs for downloader_for_single_polygon: {exc}")
                #     raise

                # ... unless either no images could be found ...
                except NoImgsForPolygonFoundError as exc:

                    # ... in which case we save it in assoc.polygons_df, ...
                    assoc.polygons_df.loc[polygon_name,
                                          'download_exception'] = repr(exc)

                    # ... log a warning, ...
                    log.warning(exc, exc_info=True)

                    # ... and break the while loop, ...
                    break

                # ... or a download error occured, ...
                except ImgDownloadError as exc:

                    assoc.polygons_df.loc[polygon_name,
                                          'download_exception'] = repr(exc)
                    log.warning(exc, exc_info=True)

                # ... or _download_imgs_for_polygon tried downloading a previously downloaded image.
                except ImgAlreadyExistsError as exc:

                    log.exception(
                        "_download_imgs_for_polygon tried downloading a previously downloaded image!"
                    )

                # If the download_method call was successful ...
                else:

                    # ... we first extract the information to be appended to assoc.imgs_df.
                    list_img_info_dicts = return_dict['list_img_info_dicts']
                    # (each img_info_dict contains the information for a new row of assoc.imgs_df)

                    # DEBUG INFO
                    log.debug(
                        "\nimg_polygon_associator: list_img_info_dicts is %s \n\n",
                        list_img_info_dicts)

                    # If no images were downloaded, ...
                    if list_img_info_dicts == []:
                        # ... we break the loop, since no further imgs can be found
                        break
                    # ... else ...
                    else:

                        self._run_safety_checks_on_downloaded_imgs(
                            previously_downloaded_imgs_set, polygon_name,
                            list_img_info_dicts)

                        # For each download ...
                        for img_info_dict in list_img_info_dicts:

                            # ... process it to an image ...
                            img_name = img_info_dict["img_name"]
                            single_img_processed_return_dict = self.download_processor.process(
                                img_name,
                                assoc.download_dir,
                                assoc.images_dir,
                                assoc.crs_epsg_code,
                                **self.kwarg_defaults,
                            )

                            # ... and update the img_info_dict with the returned information from processing. (This modifies list_img_info_dicts, too).
                            img_info_dict.update(
                                single_img_processed_return_dict)

                            # Connect the image: Add an image vertex to the graph, connect to all polygon vertices for which the intersection is non-empty and modify assoc.polygons_df where necessary ...
                            assoc._add_img_to_graph_modify_polygons_df(
                                img_name=img_name,
                                img_bounding_rectangle=img_info_dict[
                                    'geometry'])

                            # Finally, remember we downloaded the image.
                            previously_downloaded_imgs_set.add(img_name)
                        """
                        # Check the polygon is fully contained in the union of the downloaded images
                        # THIS MADE SENSE WHEN I WAS JUST DOWNLOADING ONE IMAGE PER POLYGON, BUT DOESN'T MAKE SENSE ANYMORE SINCE WE'RE SKIPPING IMAGES THAT WE'D LIKE TO USE FOR A POLYGON THAT ALREADY HAVE BEEN DOWNLOADED, SO WILL GET UNNECESSARY WARNINGS FOR THOSE POLYGONS. BUT COULD MODIFY DOWNLOAD FUNCTION TO RETURN A SET OF THOSE IMAGES SO WE CAN CHECK THIS IF WE WANT...
                        list_downloaded_img_bounding_rectangles = [img_info_dict['geometry'] for img_info_dict in list_img_info_dicts]
                        union_area_of_downloaded_images = unary_union(list_downloaded_img_bounding_rectangles)
                        if not polygon_geometry.within(union_area_of_downloaded_images):

                            downloaded_img_names = [img_info_dict['geometry'] for img_info_dict in list_img_info_dicts]

                            log.warning("Polygon %s not fully contained in the union of the images that were downloaded for it!", polygon_name)

                            assoc.polygons_df.loc[polygon_name, "download_exception"] += " Polygon %s not fully contained in images downloaded for it: %s", polygon_name, downloaded_img_names
                        """

                        # update new_imgs_dict
                        for img_info_dict in list_img_info_dicts:
                            for key in img_info_dict:
                                new_imgs_dict[key].append(img_info_dict[key])

                        num_img_series_to_download -= 1

        if len(new_imgs_dict) > 0:
            new_imgs_df = self._get_new_imgs_df(new_imgs_dict,
                                                assoc.crs_epsg_code)
            assoc.imgs_df = concat_gdfs([assoc.imgs_df, new_imgs_df])
            assoc.save()

            # Create labels, if necessary.
            if add_labels:
                assoc.make_labels()

    @staticmethod
    def _run_safety_checks_on_downloaded_imgs(
            previously_downloaded_imgs_set: Set[Union[str, int]],
            polygon_name: Union[str, int], list_img_info_dicts: List[dict]):
        """Check no images have been downloaded more than once"""

        # Extract the new image names ...
        new_img_names_list = [
            img_info_dict["img_name"] for img_info_dict in list_img_info_dicts
        ]

        # ... and make sure we have not downloaded an image twice for the same polygon.
        if len(new_img_names_list) != len(set(new_img_names_list)):
            duplicate_imgs_dict = {
                img_name: img_count
                for img_name, img_count in Counter(new_img_names_list).items()
                if img_count > 1
            }

            log.error(
                "Something is wrong with _download_imgs_for_polygon: it attempted to download the following images multiple times for polygon %s: %s",
                polygon_name, duplicate_imgs_dict)

            raise Exception(
                f"Something is wrong with _download_imgs_for_polygon: it attempted to download the following images multiple times for polygon {polygon_name}: {duplicate_imgs_dict}"
            )

            # Make sure we haven't downloaded an image that's already in the dataset.
            # (the _download_imgs_for_polygon method should have thrown an ImgAlreadyExistsError exception in this case, but we're checking again ourselves that this hasn't happened. )
        if set(new_img_names_list) & previously_downloaded_imgs_set:
            log.error(
                "Something is wrong with _download_imgs_for_polygon: it downloaded image(s) that have already been downloaded: %s",
                set(new_img_names_list) & previously_downloaded_imgs_set)

            raise Exception(
                f"Something is wrong with _download_imgs_for_polygon: it downloaded image(s) that have already been downloaded: {set(new_img_names_list) & previously_downloaded_imgs_set}"
            )

    def _get_polygons_to_download(
        self,
        polygon_names: Union[str, int, List[int], List[str]],
        target_img_count: int,
        assoc: ImgPolygonAssociator,
        filter_out_polygons_contained_in_union_of_intersecting_imgs: bool,
    ) -> List[Union[int, str]]:

        if polygon_names is None:
            polygons_to_download = list(assoc.polygons_df.loc[
                assoc.polygons_df['img_count'] < target_img_count].index)
        elif isinstance(polygon_names, (str, int)):
            polygons_to_download = [polygon_names]
        elif isinstance(polygon_names, list) and all(
                isinstance(element, (str, int)) for element in polygon_names):
            polygons_to_download = polygon_names
        else:
            raise TypeError(
                "The polygon_names argument should be a list of polygon names")

        if not set(polygons_to_download) <= set(assoc.polygons_df.index):
            raise ValueError(
                f"Polygons {set(polygons_to_download) - set(assoc.polygons_df.index)} missing from assoc.polygons_df"
            )

        polygons_to_download = self._filter_out_polygons_with_null_geometry(
            polygons_to_download, assoc)

        if filter_out_polygons_contained_in_union_of_intersecting_imgs:
            polygons_to_download = self._filter_out_polygons_contained_in_union_of_intersecting_imgs(
                polygons_to_download, assoc)

        return polygons_to_download

    def _filter_out_polygons_with_null_geometry(
        self,
        polygon_names: Union[str, int, List[int], List[str]],
        assoc: ImgPolygonAssociator,
    ) -> None:
        polygons_w_null_geometry_mask = assoc.polygons_df.geometry.values == None
        polygons_w_null_geometry = assoc.polygons_df[
            polygons_w_null_geometry_mask].index.tolist()
        if polygons_w_null_geometry != []:
            log.info(
                "download_imgs: skipping polygons with null geometry: %s.",
                polygons_w_null_geometry)
            return [
                polygon_name for polygon_name in polygon_names
                if polygon_name not in polygons_w_null_geometry
            ]
        else:
            return polygon_names

    def _filter_out_polygons_contained_in_union_of_intersecting_imgs(
        self,
        polygon_names: Union[str, int, List[int], List[str]],
        assoc: ImgPolygonAssociator,
    ) -> None:
        polygon_names = [
            polygon_name for polygon_name in polygon_names
            if not unary_union(assoc.imgs_df.loc[
                assoc.imgs_intersecting_polygon(polygon_name)].geometry.tolist(
                )).contains(assoc.polygons_df.loc[polygon_name].geometry)
        ]
        return polygon_names

    def _get_new_imgs_df(
        self,
        new_imgs_dict: dict,
        imgs_df_crs_epsg_code: int,
    ) -> GeoDataFrame:
        """Build and return imgs_df of new images from new_imgs_dict"""
        new_imgs_df = GeoDataFrame(new_imgs_dict)
        new_imgs_df.set_crs(epsg=imgs_df_crs_epsg_code, inplace=True)
        new_imgs_df.set_index("img_name", inplace=True)
        new_imgs_df = new_imgs_df.convert_dtypes(infer_objects=True,
                                                 convert_string=True,
                                                 convert_integer=True,
                                                 convert_boolean=True,
                                                 convert_floating=False)
        return new_imgs_df
