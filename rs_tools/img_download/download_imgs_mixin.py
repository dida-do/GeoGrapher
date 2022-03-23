import copy
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm.auto import tqdm

from rs_tools.errors import (ImgAlreadyExistsError, ImgDownloadError,
                             NoImgsForPolygonFoundError)
from rs_tools.global_constants import IMGS_DF_INDEX_NAME
from rs_tools.utils.utils import concat_gdfs

# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
log.setLevel(logging.DEBUG)


class DownloadImgsBaseMixIn(object):
    """
    Mix-in that implements a generic download method.
    """

    def download_imgs(
            self,
            polygon_names: Optional[Union[str, int, List[int],
                                          List[str]]] = None,
            downloader: Optional[str] = None,
            target_img_count: int = 1,
            add_labels: bool = True,
            filter_out_polygons_contained_in_union_of_intersecting_imgs:
        bool = False,
            shuffle_polygons: bool = True,
            **kwargs):
        """Download images for polygons.

        Sequentially considers the polygons for which the image count (number of images fully containing a given polygon) is less than num_target_imgs_per_polygon images in the associator's internal polygons_df or the optional polygons_df argument (if given), for each such polygon attempts to download num_target_imgs_per_polygon - image_count images fully containing the polygon (or several images jointly containing the polygon), creates the associated label(s) for the image(s) (assuming the default value True of add_labels is not changed), and integrates the new image(s) into the dataset/associator. Integrates images downloaded for a polygon into the dataset/associator immediately after downloading them and before downloading images for the next polygon. In particular, the image count is updated immediately after each download.

        Warning:
            The targeted number of downloads is determined by target_img_count
            and a polygons img_count. Since the img_count is the number of
            images in the dataset fully containing a polygon for "large"
            polygons the img_count will always remain zero and every call of the
            download_imgs method that includes this polygon will download
            target_img_count images (or image series). To avoid this, you can use the filter_out_polygons_contained_in_union_of_intersecting_imgs argument.

        Args:
            polygon_names (List[str], optional): Optional polygon_name or list of polygon_names to download images for. Defaults to None, i.e. consider all polygons in self.polygons_df.
            downloader (str): One of 'sentinel2' or 'jaxa'. Defaults, if possible, to previously used downloader.
            target_img_count (int): target for number of images per polygon in the dataset after downloading. The actual number of images for each polygon P that fully contain it could be lower if there are not enough images available or higher if after downloading num_target_imgs_per_polygon images for P P is also contained in images downloaded for other polygons.
            polygons_df (GeoDataFrame, optional): (Probably just best ignore this) GeoDataFrame of polygons conforming to the associator's format for polygon_df, defaults to the associator's internal polygons_df (i.e. self.polygons_df). If provided and not equal to self.polygons_df will download images for only those polygons and integrate the polygons in polygons_df into the associator after the images have been downloaded.
            add_labels (bool, optional): bool. Whether to add labels for the downloaded images. Defaults to True.
            filter_out_polygons_contained_in_union_of_intersecting_imgs (bool): Useful when dealing with 'large' polygons. Defaults to False.
            shuffle_polygons (bool): Whether to shuffle order of polygons for which images will be downloaded. Might in practice prevent an uneven distribution of the image count for repeated downloads. Defaults to True.

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

        Kwargs (downloader='sentinel2'):
            producttype (str): One of 'L1C'/'S2MSI1C' or 'L2A'/'S2MSI2A'. Defaults if possible to whichever choice you made last time.
            resolution (int): One of 10, 20, or 60. Defaults if possible to whichever choice you made last time.
            max_percent_cloud_coverage (int): Integer between 0 and 100. Defaults if possible to whichever choice you made last time.
            date (Any):  E.g. (“NOW-1DAY”, “NOW”) to avoid the long term archive.
                See https://sentinelsat.readthedocs.io/en/latest/api_reference.html
                Defaults if possible to whichever choice you made last time.
            area_relation : See https://sentinelsat.readthedocs.io/en/latest/api_reference.html Defaults if possible to whichever choice you made last time.

        Returns:
            None

        Warning:
            It's easy to come up with examples where the image count distribution (i.e. distribution of images per polygon) becomes unbalanced particularly if num_target_imgs_per_polygon is large. These scenarios are not necessarily very likely, but possible. As an example, if one wants to download say 5 images images for a polygon that is not fully contained in any image in the dataset and if there does not exist an image we can download that fully contains it but there are 20 disjoint sets of images we can download that jointly cover the polygon then these 20 disjoint sets will all be downloaded.
        """

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        downloader = self._get_and_remember_downloader(downloader)
        polygons_to_download = self._get_polygons_to_download(
            polygon_names, target_img_count)

        if filter_out_polygons_contained_in_union_of_intersecting_imgs:
            polygons_to_download = [
                polygon_name for polygon_name in polygons_to_download
                if not unary_union(self.imgs_df.loc[
                    self.imgs_intersecting_polygon(
                        polygon_name)].geometry.tolist()).contains(
                            self.polygons_df.loc[polygon_name].geometry)
            ]

        if shuffle_polygons == True:
            random.shuffle(polygons_to_download)

        # Set of previously downloaded images.
        previously_downloaded_imgs_set = set(self.imgs_df.index)
        # (Will be used to make sure no attempt is made to download an image more than once.)

        # Dict to keep track of imgs we've downloaded. We'll append this to self.imgs_df as a (geo)dataframe later
        new_imgs_dict = defaultdict(list)

        pbar = tqdm(
            enumerate(self.polygons_df[["geometry"
                                        ]].loc[polygons_to_download]))
        for count, polygon_name, polygon_geometry in pbar:

            # polygon_geometry = self.polygons_df.loc[polygon_name, 'geometry']

            pbar.set_description(
                f"Polygon {count}/{len(polygons_to_download)}")
            log.debug(
                f"download_missing_imgs_for_polygons_df: considering polygon {polygon_name}."
            )

            # Since we process and connect each image after downloading it, we might not need to download
            # an image for a polygon that earlier was lacking an image if it is now contained in one of the already downloaded images,
            # so need to check again that there are not enough images for the polygon (since the iterator above is set when it is called
            # and won't know if the "img_count" column value has been changed in the meanwhile).
            num_img_series_to_download = target_img_count - self.polygons_df.loc[
                polygon_name, "img_count"]
            if num_img_series_to_download <= 0:
                log.debug(
                    f"Skipping polygon {polygon_name} since there now enough images fully containing it."
                )
                continue

            while num_img_series_to_download > 0:

                # Try downloading an image series and save returned dict (of dicts)
                # containing information for polygons_df, self.imgs_df...
                try:

                    # DEBUG INFO
                    log.debug(
                        f"attempting to download image for polygon {polygon_name}."
                    )

                    download_method = getattr(
                        self, f"_download_imgs_for_polygon_{downloader}")
                    temporary_params_dict = copy.deepcopy(self._params_dict)
                    temporary_params_dict.update(kwargs)
                    return_dict = download_method(
                        polygon_name,
                        polygon_geometry,
                        self.download_dir,
                        previously_downloaded_imgs_set,  # _download_imgs_for_polygon should use this to make sure no attempt at downloading an already downloaded image is made.
                        **temporary_params_dict,
                    )

                # ... unless either no images could be found ...
                except NoImgsForPolygonFoundError as e:

                    # ... in which case we save it in self.polygons_df, ...
                    self.polygons_df.loc[polygon_name,
                                         'download_exception'] = repr(e)

                    # ... log a warning, ...
                    log.warning(e, exc_info=True)

                    # ... and break the while loop, ...
                    break

                # ... or a download error occured, ...
                except ImgDownloadError as e:

                    self.polygons_df.loc[polygon_name,
                                         'download_exception'] = repr(e)
                    log.warning(e, exc_info=True)

                # ... or _download_imgs_for_polygon tried downloading a previously downloaded image.
                except ImgAlreadyExistsError as e:

                    log.exception(
                        f"_download_imgs_for_polygon tried downloading a previously downloaded image!"
                    )

                # If the download_method call was successful ...
                else:

                    # ... we first extract the information to be appended to self.imgs_df.
                    list_img_info_dicts = return_dict['list_img_info_dicts']
                    # (each img_info_dict contains the information for a new row of self.imgs_df)

                    # DEBUG INFO
                    log.debug(
                        f"\nimg_polygon_associator: list_img_info_dicts is {list_img_info_dicts}\n\n"
                    )

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
                        for count, img_info_dict in enumerate(
                                list_img_info_dicts):

                            # ... process it to an image ...
                            img_name = img_info_dict["img_name"]
                            single_download_processor = getattr(
                                self,
                                f"_process_downloaded_img_file_{downloader}")
                            single_img_processed_return_dict = single_download_processor(
                                img_name, self.download_dir, self.images_dir,
                                self._params_dict['crs_epsg_code'],
                                **temporary_params_dict)

                            # ... and update the img_info_dict with the returned information from processing. (This modifies list_img_info_dicts, too).
                            img_info_dict.update(
                                single_img_processed_return_dict)

                            # Connect the image: Add an image vertex to the graph, connect to all polygon vertices for which the intersection is non-empty and modify self.polygons_df where necessary ...
                            self._add_img_to_graph_modify_polygons_df(
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

                            log.warning(f"Polygon {polygon_name} not fully contained in the union of the images that were downloaded for it!")

                            self.polygons_df.loc[polygon_name, "download_exception"] += f" Polygon {polygon_name} not fully contained in images downloaded for it: {downloaded_img_names}"
                        """

                        # update new_imgs_dict
                        for img_info_dict in list_img_info_dicts:
                            for key in img_info_dict:
                                new_imgs_dict[key].append(img_info_dict[key])

                        num_img_series_to_download -= 1

        if len(new_imgs_dict) > 0:
            # Extract accumulated information about the imgs we've downloaded from new_imgs into a dataframe ...
            new_imgs_df = GeoDataFrame(new_imgs_dict)
            new_imgs_df.set_crs(epsg=self._params_dict['crs_epsg_code'],
                                inplace=True)  # standard crs
            new_imgs_df.set_index("img_name", inplace=True)
            new_imgs_df = new_imgs_df.convert_dtypes(infer_objects=True,
                                                     convert_string=True,
                                                     convert_integer=True,
                                                     convert_boolean=True,
                                                     convert_floating=False)

            # ... and append it to self.imgs_df:
            self.imgs_df = concat_gdfs([self.imgs_df, new_imgs_df])

            self.save()

            # Create the label, if necessary.
            if add_labels:
                self.make_labels()

    def _run_safety_checks_on_downloaded_imgs(
            self, previously_downloaded_imgs_set: Set[Union[str, int]],
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
                f"Something is wrong with _download_imgs_for_polygon: it attempted to download the following images multiple times for polygon {polygon_name}: {duplicate_imgs_dict}"
            )

            raise Exception(
                f"Something is wrong with _download_imgs_for_polygon: it attempted to download the following images multiple times for polygon {polygon_name}: {duplicate_imgs_dict}"
            )

            # Make sure we haven't downloaded an image that's already in the dataset.
            # (the _download_imgs_for_polygon method should have thrown an ImgAlreadyExistsError exception in this case, but we're checking again ourselves that this hasn't happened. )
        if set(new_img_names_list) & previously_downloaded_imgs_set:
            log.error(
                f"Something is wrong with _download_imgs_for_polygon: it downloaded image(s) that have already been downloaded: {set(new_img_names_list) & previously_downloaded_imgs_set}"
            )

            raise Exception(
                f"Something is wrong with _download_imgs_for_polygon: it downloaded image(s) that have already been downloaded: {set(new_img_names_list) & previously_downloaded_imgs_set}"
            )

    def _get_polygons_to_download(
            self, polygon_names: Union[str, int, List[int], List[str]],
            target_img_count: int) -> List[Union[int, str]]:

        if polygon_names is None:
            polygons_to_download = list(self.polygons_df.loc[
                self.polygons_df['img_count'] < target_img_count].index)
        elif isinstance(polygon_names, (str, int)):
            polygons_to_download = [polygon_names]
        elif isinstance(polygon_names, list) and all(
                isinstance(element, (str, int)) for element in polygon_names):
            polygons_to_download = polygon_names
        else:
            raise TypeError(
                f"The polygon_names argument should be a list of polygon names"
            )

        if not set(polygons_to_download) <= set(self.polygons_df.index):
            raise ValueError(
                f"Polygons {set(polygons_to_download) - set(self.polygons_df.index)} missing from self.polygons_df"
            )

        # Remove polygons with null geometry
        polygons_w_null_geometry_mask = self.polygons_df.geometry.values == None
        polygons_w_null_geometry = self.polygons_df[
            polygons_w_null_geometry_mask].index.tolist()
        if polygons_w_null_geometry != []:
            polygons_to_download = [
                polygon_name for polygon_name in polygons_to_download
                if polygon_name not in polygons_w_null_geometry
            ]
            log.info(
                f"download_imgs: skipping polygons with null geometry: {polygons_w_null_geometry}."
            )

        return polygons_to_download

    def _get_and_remember_downloader(self, downloader):
        if downloader is None:
            try:
                # Use saved value
                downloader = getattr(self, 'downloader')
            except AttributeError:
                raise ValueError(
                    "Need to set downloader keyword argument (TODO!!!).")
        else:
            # Remember value
            self._params_dict['downloader'] = downloader
        return downloader
