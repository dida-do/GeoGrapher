"""
Class for downloading images for vector features targeting
a given number of images per vector feature.
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
from rs_tools.base_model_dict_conversion.save_load_base_model_mixin import \
    SaveAndLoadBaseModelMixIn
from rs_tools.errors import (
    ImgAlreadyExistsError,
    ImgDownloadError,
    NoImgsForPolygonFoundError,
)
from rs_tools.downloaders.base_download_processor import ImgDownloadProcessor
from rs_tools.downloaders.base_downloader_for_single_feature import \
    ImgDownloaderForSingleVectorFeature
from rs_tools.utils.utils import concat_gdfs

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


class ImgDownloaderForVectorFeatures(BaseModel, SaveAndLoadBaseModelMixIn):
    """
    Download images for vector features targeting a given number of images per feature.
    """

    download_dir: Path
    downloader_for_single_feature: ImgDownloaderForSingleVectorFeature
    download_processor: ImgDownloadProcessor
    kwarg_defaults: dict = Field(default_factory=dict)

    def download(self,
                 assoc: Union[Path, str, ImgPolygonAssociator],
                 feature_names: Optional[Union[str, int, List[int],
                                               List[str]]] = None,
                 target_img_count: int = 1,
                 filter_out_features_contained_in_union_of_intersecting_imgs:
                 bool = False,
                 shuffle: bool = True,
                 **kwargs):
        """Download images for vector features so as to target a number of images per vector feature.

        Sequentially considers the vector features for which the image count (number of images fully
        containing a given vector feature) is less than num_target_imgs_per_feature images in the associator's
        internal vector_features or the optional vector_features argument (if given), for each such vector feature
        attempts to download num_target_imgs_per_feature - image_count images fully containing the vector feature
        (or several images jointly containing the vector feature), and integrates the new image(s)
        into the dataset/associator. Integrates images downloaded for a vector feature into the dataset/associator
        immediately after downloading them and before downloading images for the next vector feature. In particular,
        the image count is updated immediately after each download.

        Warning:
            The targeted number of downloads is determined by target_img_count
            and a vector features img_count. Since the img_count is the number of
            images in the dataset fully containing a vector feature for "large"
            vector features (polygons) the img_count will always remain zero and every call of the
            download_imgs method that includes this vector feature will download
            target_img_count images (or image series). To avoid this, you can use
            the filter_out_features_contained_in_union_of_intersecting_imgs argument.

        Args:
            feature_names (List[str], optional): Optional feature_name or list of feature_names to download images for. Defaults to None, i.e. consider all vector features in assoc.vector_features.
            downloader (str): One of 'sentinel2' or 'jaxa'. Defaults, if possible, to previously used downloader.
            target_img_count (int): target for number of images per vector feature in the dataset after downloading. The actual number of images for each vector feature P that fully contain it could be lower if there are not enough images available or higher if after downloading num_target_imgs_per_feature images for P P is also contained in images downloaded for other vector features.
            vector_features (GeoDataFrame, optional): (Probably just best ignore this) GeoDataFrame of vector features conforming to the associator's format for vector_features, defaults to the associator's internal vector_features (i.e. assoc.vector_features). If provided and not equal to assoc.vector_features will download images for only those vector features and integrate the vector features in vector_features into the associator after the images have been downloaded.
            filter_out_vector features_contained_in_union_of_intersecting_imgs (bool): Useful when dealing with 'large' vector features. Defaults to False.
            shuffle (bool): Whether to shuffle order of vector features for which images will be downloaded. Might in practice prevent an uneven distribution of the image count for repeated downloads. Defaults to True.
            kwargs (dict, optional): additional keyword arguments passed to downloader_for_single_feature and download_processor. Defaults to self.kwarg_defaults.

        Note:
            Any kwargs given will be saved to self.default_kwargs and become default values.

        Returns:
            None

        Warning:
            In the case that the vector vector features are polygons it's easy to come up with examples where the
            image count distribution (i.e. distribution of images per polygon) becomes unbalanced particularly
            if num_target_imgs_per_feature is large. These scenarios are not necessarily very likely, but possible.
            As an example, if one wants to download say 5 images images for a polygon that is not fully contained
            in any image in the dataset and if there does not exist an image we can download that fully contains
            it but there are 20 disjoint sets of images we can download that jointly cover the polygon then
            these 20 disjoint sets will all be downloaded.
        """

        self.kwarg_defaults.update(kwargs)

        if not isinstance(assoc, ImgPolygonAssociator):
            assoc = ImgPolygonAssociator.from_data_dir(assoc)
        assoc.images_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(
            parents=True,
            exist_ok=True)  #TODO: assoc.data_dir / 'downloads' or whatever

        features_for_which_to_download = self._get_features_for_which_to_download(
            feature_names=feature_names,
            target_img_count=target_img_count,
            assoc=assoc,
            filter_out_features_contained_in_union_of_intersecting_imgs=
            filter_out_features_contained_in_union_of_intersecting_imgs,
        )

        if shuffle:
            random.shuffle(features_for_which_to_download)

        previously_downloaded_imgs_set = set(assoc.raster_imgs.index)
        # (Will be used to make sure no attempt is made to download an image more than once.)

        # Dict to keep track of imgs we've downloaded. We'll append this to assoc.raster_imgs as a (geo)dataframe later
        new_imgs_dict = defaultdict(list)

        pbar = tqdm(
            enumerate(assoc.vector_features[[
                "geometry"
            ]].loc[features_for_which_to_download].itertuples(),
                      start=1))
        for count, (feature_name, feature_geom) in pbar:

            # feature_geom = assoc.vector_features.loc[feature_name, 'geometry']

            pbar.set_description(
                f"Polygon {count}/{len(features_for_which_to_download)}", )
            log.debug(
                "download_missing_imgs_for_vector_features: considering vector feature %s.",
                feature_name)

            # Since we process and connect each image after downloading it, we might not need to download
            # an image for a vector feature that earlier was lacking an image if it is now contained in one of the already downloaded images,
            # so need to check again that there are not enough images for the vector feature (since the iterator above is set when it is called
            # and won't know if the "img_count" column value has been changed in the meanwhile).
            num_img_series_to_download = target_img_count - assoc.vector_features.loc[
                feature_name, "img_count"]
            if num_img_series_to_download <= 0:
                log.debug(
                    "Skipping vector feature %s since there now enough images fully containing it.",
                    feature_name)
                continue

            while num_img_series_to_download > 0:

                # Try downloading an image series and save returned dict (of dicts)
                # containing information for vector_features, assoc.raster_imgs...
                try:

                    # DEBUG INFO
                    log.debug(
                        "attempting to download image for vector feature. %s",
                        feature_name)

                    return_dict = self.downloader_for_single_feature.download(
                        feature_name,
                        feature_geom,
                        self.download_dir,
                        previously_downloaded_imgs_set,  # downloader_for_single_feature should use this to make sure no attempt at downloading an already downloaded image is made.
                        **self.kwarg_defaults,
                    )

                # WHY DOES THIS NOT WORK?
                # except TypeError as exc:
                #     log.exception("Probably missing kwargs for downloader_for_single_feature: {exc}")
                #     raise

                # ... unless either no images could be found ...
                except NoImgsForPolygonFoundError as exc:

                    # ... in which case we save it in assoc.vector_features, ...
                    assoc.vector_features.loc[feature_name,
                                              'download_exception'] = repr(exc)

                    # ... log a warning, ...
                    log.warning(exc, exc_info=True)

                    # ... and break the while loop, ...
                    break

                # ... or a download error occured, ...
                except ImgDownloadError as exc:

                    assoc.vector_features.loc[feature_name,
                                              'download_exception'] = repr(exc)
                    log.warning(exc, exc_info=True)

                # ... or downloader_for_single_feature tried downloading a previously downloaded image.
                except ImgAlreadyExistsError as exc:

                    log.exception(
                        "downloader_for_single_feature tried downloading a previously downloaded image!"
                    )

                # If the download_method call was successful ...
                else:

                    # ... we first extract the information to be appended to assoc.raster_imgs.
                    list_img_info_dicts = return_dict['list_img_info_dicts']
                    # (each img_info_dict contains the information for a new row of assoc.raster_imgs)

                    # DEBUG INFO
                    log.debug("list_img_info_dicts is %s \n\n",
                              list_img_info_dicts)

                    # If no images were downloaded, ...
                    if list_img_info_dicts == []:
                        # ... we break the loop, since no further imgs can be found
                        break
                    # ... else ...
                    else:

                        self._run_safety_checks_on_downloaded_imgs(
                            previously_downloaded_imgs_set, feature_name,
                            list_img_info_dicts)

                        # For each download ...
                        for img_info_dict in list_img_info_dicts:

                            # ... process it to an image ...
                            img_name = img_info_dict["img_name"]
                            single_img_processed_return_dict = self.download_processor.process(
                                img_name,
                                self.download_dir,
                                assoc.images_dir,
                                assoc.crs_epsg_code,
                                **self.kwarg_defaults,
                            )

                            # ... and update the img_info_dict with the returned information from processing. (This modifies list_img_info_dicts, too).
                            img_info_dict.update(
                                single_img_processed_return_dict)

                            # Connect the image: Add an image vertex to the graph, connect to all vector_features vertices for which the intersection is non-empty and modify assoc.vector_features where necessary ...
                            assoc._add_img_to_graph_modify_vector_features(
                                img_name=img_name,
                                img_bounding_rectangle=img_info_dict[
                                    'geometry'])

                            # Finally, remember we downloaded the image.
                            previously_downloaded_imgs_set.add(img_name)
                        """
                        # Check the vector feature is fully contained in the union of the downloaded images
                        # THIS MADE SENSE WHEN I WAS JUST DOWNLOADING ONE IMAGE PER POLYGON, BUT DOESN'T MAKE SENSE ANYMORE SINCE WE'RE SKIPPING IMAGES THAT WE'D LIKE TO USE FOR A POLYGON THAT ALREADY HAVE BEEN DOWNLOADED, SO WILL GET UNNECESSARY WARNINGS FOR THOSE POLYGONS. BUT COULD MODIFY DOWNLOAD FUNCTION TO RETURN A SET OF THOSE IMAGES SO WE CAN CHECK THIS IF WE WANT...
                        list_downloaded_img_bounding_rectangles = [img_info_dict['geometry'] for img_info_dict in list_img_info_dicts]
                        union_area_of_downloaded_images = unary_union(list_downloaded_img_bounding_rectangles)
                        if not feature_geom.within(union_area_of_downloaded_images):

                            downloaded_img_names = [img_info_dict['geometry'] for img_info_dict in list_img_info_dicts]

                            log.warning("Polygon %s not fully contained in the union of the images that were downloaded for it!", feature_name)

                            assoc.vector_features.loc[feature_name, "download_exception"] += " Polygon %s not fully contained in images downloaded for it: %s", feature_name, downloaded_img_names
                        """

                        # update new_imgs_dict
                        for img_info_dict in list_img_info_dicts:
                            for key in img_info_dict:
                                new_imgs_dict[key].append(img_info_dict[key])

                        num_img_series_to_download -= 1

        if len(new_imgs_dict) > 0:
            new_raster_imgs = self._get_new_raster_imgs(
                new_imgs_dict, assoc.crs_epsg_code)
            assoc.raster_imgs = concat_gdfs(
                [assoc.raster_imgs, new_raster_imgs])
            assoc.save()

    def save(self, file_path: Union[Path, str]):
        """
        Save downloader. By convention, the downloader should be saved to the
        associator subdirectory of the data directory it is supposed to operate on.
        """
        self._save(file_path)

    @staticmethod
    def _run_safety_checks_on_downloaded_imgs(
            previously_downloaded_imgs_set: Set[Union[str, int]],
            feature_name: Union[str, int], list_img_info_dicts: List[dict]):
        """Check no images have been downloaded more than once"""

        # Extract the new image names ...
        new_img_names_list = [
            img_info_dict["img_name"] for img_info_dict in list_img_info_dicts
        ]

        # ... and make sure we have not downloaded an image twice for the same vector feature.
        if len(new_img_names_list) != len(set(new_img_names_list)):
            duplicate_imgs_dict = {
                img_name: img_count
                for img_name, img_count in Counter(new_img_names_list).items()
                if img_count > 1
            }

            log.error(
                "Something is wrong with downloader_for_single_feature: it attempted to download the following images multiple times for vector feature %s: %s",
                feature_name, duplicate_imgs_dict)

            raise Exception(
                f"Something is wrong with downloader_for_single_feature: it attempted to download the following images multiple times for vector feature {feature_name}: {duplicate_imgs_dict}"
            )

            # Make sure we haven't downloaded an image that's already in the dataset.
            # (the downloader_for_single_feature method should have thrown an ImgAlreadyExistsError exception in this case, but we're checking again ourselves that this hasn't happened. )
        if set(new_img_names_list) & previously_downloaded_imgs_set:
            log.error(
                "Something is wrong with downloader_for_single_feature: it downloaded image(s) that have already been downloaded: %s",
                set(new_img_names_list) & previously_downloaded_imgs_set)

            raise Exception(
                f"Something is wrong with downloader_for_single_feature: it downloaded image(s) that have already been downloaded: {set(new_img_names_list) & previously_downloaded_imgs_set}"
            )

    def _get_features_for_which_to_download(
        self,
        feature_names: Union[str, int, List[int], List[str]],
        target_img_count: int,
        assoc: ImgPolygonAssociator,
        filter_out_features_contained_in_union_of_intersecting_imgs: bool,
    ) -> List[Union[int, str]]:

        if feature_names is None:
            features_for_which_to_download = list(assoc.vector_features.loc[
                assoc.vector_features['img_count'] < target_img_count].index)
        elif isinstance(feature_names, (str, int)):
            features_for_which_to_download = [feature_names]
        elif isinstance(feature_names, list) and all(
                isinstance(element, (str, int)) for element in feature_names):
            features_for_which_to_download = feature_names
        else:
            raise TypeError(
                "The feature_names argument should be a list of vector feature names"
            )

        if not set(features_for_which_to_download) <= set(
                assoc.vector_features.index):
            raise ValueError(
                f"Polygons {set(features_for_which_to_download) - set(assoc.vector_features.index)} missing from assoc.vector_features"
            )

        features_for_which_to_download = self._filter_out_features_with_null_geometry(
            features_for_which_to_download, assoc)

        if filter_out_features_contained_in_union_of_intersecting_imgs:
            features_for_which_to_download = self._filter_out_features_contained_in_union_of_intersecting_imgs(
                features_for_which_to_download, assoc)

        return features_for_which_to_download

    def _filter_out_features_with_null_geometry(
        self,
        feature_names: Union[str, int, List[int], List[str]],
        assoc: ImgPolygonAssociator,
    ) -> None:
        features_w_null_geometry_mask = assoc.vector_features.geometry.values == None
        features_w_null_geometry = assoc.vector_features[
            features_w_null_geometry_mask].index.tolist()
        if features_w_null_geometry != []:
            log.info(
                "download_imgs: skipping vector features with null geometry: %s.",
                features_w_null_geometry)
            return [
                feature_name for feature_name in feature_names
                if feature_name not in features_w_null_geometry
            ]
        else:
            return feature_names

    def _filter_out_features_contained_in_union_of_intersecting_imgs(
        self,
        feature_names: Union[str, int, List[int], List[str]],
        assoc: ImgPolygonAssociator,
    ) -> None:
        feature_names = [
            feature_name for feature_name in feature_names
            if not unary_union(assoc.raster_imgs.loc[
                assoc.imgs_intersecting_feature(feature_name)].geometry.tolist(
                )).contains(assoc.vector_features.loc[feature_name].geometry)
        ]
        return feature_names

    def _get_new_raster_imgs(
        self,
        new_imgs_dict: dict,
        raster_imgs_crs_epsg_code: int,
    ) -> GeoDataFrame:
        """Build and return raster_imgs of new images from new_imgs_dict"""
        new_raster_imgs = GeoDataFrame(new_imgs_dict)
        new_raster_imgs.set_crs(epsg=raster_imgs_crs_epsg_code, inplace=True)
        new_raster_imgs.set_index("img_name", inplace=True)
        new_raster_imgs = new_raster_imgs.convert_dtypes(
            infer_objects=True,
            convert_string=True,
            convert_integer=True,
            convert_boolean=True,
            convert_floating=False)
        return new_raster_imgs