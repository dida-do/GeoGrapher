from typing import Optional, Sequence, Union, List, Tuple
from pathlib import Path
from collections import Counter
import logging
import copy
import random
from tqdm import tqdm
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from rs_tools.errors import ImgAlreadyExistsError, NoImgsForPolygonFoundError, ImgDownloadError


# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)


class DownloadImgsMixIn(object):
    """
    Mix-in that implements a generic download method. 
    
    The download_imgs method depends on the _download_imgs_for_polygon and _process_downloaded_img_file methods that need to be implemented in a subclass for each data source (e.g. Sentinel-2). 
    """

    def download_imgs(self, 
            polygon_names : Optional[List[str]]=None,
            target_img_count : int=1,
            add_labels : bool=True,
            shuffle_polygons : bool=True,
            **kwargs):
        """ 
        Download images for polygons.

        Sequentially considers the polygons for which the image count (number of images fully containing a given polygon) is less than num_target_imgs_per_polygon images in the associator's internal polygons_df or the optional polygons_df argument (if given), for each such polygon attempts to download num_target_imgs_per_polygon - image_count images fully containing the polygon (or several images jointly containing the polygon), creates the associated label(s) for the image(s) (assuming the default value True of add_labels is not changed), and integrates the new image(s) into the dataset/associator. Integrates images downloaded for a polygon into the dataset/associator immediately after downloading them and before downloading images for the next polygon. In particular, the image count is updated immediately after each download. 

        Args:
            polygon_names (List[str], optional): Optional list of polygons to download images for. Defaults to None, i.e. consider all polygons in self.polygons_df.
            target_img_count (int): target for number of images per polygon in the dataset after downloading. The actual number of images for each polygon P that fully contain it could be lower if there are not enough images available or higher if after downloading num_target_imgs_per_polygon images for P P is also contained in images downloaded for other polygons. 
            polygons_df (GeoDataFrame, optional): (Probably just best ignore this) GeoDataFrame of polygons conforming to the associator's format for polygon_df, defaults to the associator's internal polygons_df (i.e. self.polygons_df). If provided and not equal to self.polygons_df will download images for only those polygons and integrate the polygons in polygons_df into the associator after the images have been downloaded. 
            add_labels (bool, optional): bool. Whether to add labels for the downloaded images. Defaults to True.
            shuffle_polygons (bool): Whether to shuffle order of polygons for which images will be downloaded. Might in practice prevent an uneven distribution of the image count for repeated downloads. Defaults to True.
        
        Returns:
            None

        Warning:
            It's easy to come up with examples where the image count distribution (i.e. distribution of images per polygon) becomes unbalanced particularly if num_target_imgs_per_polygon is large. These scenarios are not necessarily very likely, but possible. As an example, if one wants to download say 5 images images for a polygon that is not fully contained in any image in the dataset and if there does not exist an image we can download that fully contains it but there are 20 disjoint sets of images we can download that jointly cover the polygon then these 20 disjoint sets will all be downloaded. 
        """

        # Make sure images_dir exists
        self.images_dir.mkdir(parents=True, exist_ok=True)

        if polygon_names is None:

            polygons_to_download = list(self.polygons_df.loc[self.polygons_df['img_count'] < target_img_count].index)
            target_img_counts = [target_img_count] * len(polygons_to_download)

        # list of polygon names
        elif isinstance(polygon_names, list) and all(isinstance(element, str) for element in polygon_names):

            polygons_to_download = polygon_names
            target_img_counts = [target_img_count] * len(polygons_to_download)

            if not set(polygon_names) <= set(self.polygons_df.index):
                raise ValueError(f"Polygons {set(polygon_names) - set(self.polygons_df.index)} missing from self.polygons_df")

        else:
            raise TypeError(f"The polygon_names argument should be a list of polygon names (i.e. strings).")

        polygon_names_and_target_img_counts = list(
                                                zip(
                                                    polygons_to_download, 
                                                    target_img_counts
                                                ))
                                                    
        if shuffle_polygons == True:
            random.shuffle(polygon_names_and_target_img_counts)

        # Set of previously downloaded images.
        previously_downloaded_imgs_set = set(self.imgs_df.index) 
        # (Will be used to make sure no attempt is made to download an image more than once.)

        # Dict to keep track of imgs we've downloaded. We'll append this to self.imgs_df as a (geo)dataframe later
        new_imgs_dict = {index_or_col_name: [] for index_or_col_name in [self.imgs_df.index.name] + list(self.imgs_df.columns)}

        # Go through polygons for which not enough images have been downloaded yet.
        for count, (polygon_name, target_img_count) in tqdm(enumerate(polygon_names_and_target_img_counts)):

            polygon_geometry = self.polygons_df.loc[polygon_name, 'geometry'] 

            log.debug(f"download_missing_imgs_for_polygons_df: considering polygon {polygon_name}.")
            log.info(f"Polygon {count}/{len(polygon_names)}")

            # Since we process and connect each image after downloading it, we might not need to download 
            # an image for a polygon that earlier was lacking an image if it is now contained in one of the already downloaded images, so need to check again that there are not enough images for the polygon (since the iterator above is set when it is called and won't know if the "img_count" column value has been changed in the meanwhile).
            num_img_series_to_download = target_img_count - self.polygons_df.loc[polygon_name, "img_count"]
            if num_img_series_to_download <= 0:
                log.debug(f"download_missing_imgs_for_polygons_df: skipping polygon {polygon_name} since there enough images fully containing it.")

                pass
            
            else:

                # Dict of possible keyword args for download function.
                # We use deepcopy here so that a call to download_missing_imgs_for_polygons_df 
                # can not modify self._params_dict.
                temporary_params_dict = copy.deepcopy(self._params_dict) 
                temporary_params_dict.update(kwargs)
                
                while num_img_series_to_download > 0:

                    # Try downloading an image and save returned dict (of dicts) containing information for polygons_df, self.imgs_df...  
                    try:      

                        # DEBUG INFO
                        log.debug(f"attempting to download image for polygon {polygon_name}.")

                        return_dict = self._download_imgs_for_polygon(
                                        polygon_name,
                                        polygon_geometry,  
                                        self.download_dir,
                                        previously_downloaded_imgs_set, # _download_imgs_for_polygon should use this to make sure no attempt at downloading an already downloaded image is made.
                                        **temporary_params_dict)
                    
                    # ... unless either no images could be found or a download error occured, ...
                    except NoImgsForPolygonFoundError as e:

                        # ... in which case we save it in self.polygons_df, ...
                        self.polygons_df.loc[polygon_name, 'download_exception'] = repr(e)

                        # ... log a warning, ...
                        log.warning(e, exc_info=True)

                        # ... and break the while loop.
                        break

                    except ImgDownloadError as e:

                        self.polygons_df.loc[polygon_name, 'download_exception'] = repr(e)
                        log.warning(e, exc_info=True)
                    
                    # ... or _download_imgs_for_polygon tried downloading a previously downloaded image ...
                    except ImgAlreadyExistsError as e:

                        # ... in which case we log the exception. 
                        log.exception(f"_download_imgs_for_polygon tried downloading a previously downloaded image!")

                    # If nothing went wrong ...
                    else:

                        # ... we first extract the information to be appended to self.imgs_df.
                        list_img_info_dicts = return_dict['list_img_info_dicts']
                        # (each img_info_dict contains the information for a new row of self.imgs_df)

                        # DEBUG INFO
                        log.debug(f"\nimg_polygon_associator: list_img_info_dicts is {list_img_info_dicts}\n\n")
                        
                        # If at least one image was downloaded, ...
                        if list_img_info_dicts != [{}]:
                        
                            # ... extract the new image names ...
                            new_img_names_list = [img_info_dict[self.imgs_df.index.name] for img_info_dict in list_img_info_dicts]

                            # ... and make sure we have not downloaded an image twice for the same polygon.
                            if len(new_img_names_list) != len(set(new_img_names_list)):
                                
                                duplicate_imgs_dict = {img_name: img_count for img_name, img_count in Counter(new_img_names_list).items() if img_count>1}

                                log.error(f"Something is wrong with _download_imgs_for_polygon: it attempted to download the following images multiple times for polygon {polygon_name}: {duplicate_imgs_dict}")

                                raise Exception(f"Something is wrong with _download_imgs_for_polygon: it attempted to download the following images multiple times for polygon {polygon_name}: {duplicate_imgs_dict}")
                            
                            # Make sure we haven't downloaded an image that's already in the dataset.
                            # (the _download_imgs_for_polygon method should have thrown an ImgAlreadyExistsError exception in this case, but we're checking again ourselves that this hasn't happened. )
                            if set(new_img_names_list) & previously_downloaded_imgs_set:

                                log.error(f"Something is wrong with _download_imgs_for_polygon: it downloaded image(s) that have already been downloaded: {set(new_img_names_list) & previously_downloaded_imgs_set}")

                                raise Exception(f"Something is wrong with _download_imgs_for_polygon: it downloaded image(s) that have already been downloaded: {set(new_img_names_list) & previously_downloaded_imgs_set}")

                            # For each download ...
                            for count, img_info_dict in enumerate(list_img_info_dicts):
                                
                                # ... process it to an image ...
                                img_name = img_info_dict[self.imgs_df.index.name]
                                single_img_processed_return_dict = self._process_downloaded_img_file(
                                                                        img_name, 
                                                                        self.download_dir, 
                                                                        self.images_dir,
                                                                        self._params_dict['crs_epsg_code'],
                                                                        **self._params_dict)

                                # ... and update the img_info_dict with the returned information from processing. (This modifies list_img_info_dicts, too).
                                img_info_dict.update(single_img_processed_return_dict)
                                                            
                                # Connect the image: Add an image vertex to the graph, connect to all polygon vertices for which the intersection is non-empty and modify self.polygons_df where necessary ...
                                self._add_img_to_graph_modify_polygons_df(
                                    img_name=img_name, 
                                    img_bounding_rectangle=img_info_dict['geometry'])

                                # ... and create the label, if necessary.
                                if add_labels==True:
                                    self._make_geotif_label(self, img_name, log) # the self arg is needed, see import

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

                            # Go through all images downloaded/processed for this polygon.
                            for img_info_dict in list_img_info_dicts: 

                                # After checking img_info_dict contains the columns/index we want 
                                # (so we don't for example fill in missing columns with nonsensical default values)...
                                if set(img_info_dict.keys()) != set(self.imgs_df.columns) | {self.imgs_df.index.name}:
                                
                                    keys_not_in_cols_or_index = {key for key in img_info_dict.keys() if key not in set(self.imgs_df.columns) | {self.imgs_df.index.name}}

                                    cols_or_index_not_in_keys = {x for x in set(self.imgs_df.columns) | {self.imgs_df.index.name} if x not in img_info_dict}

                                    raise Exception(f"img_info_dict keys not equal to imgs_df columns and index name. \n Keys not in cols or index name {keys_not_in_cols_or_index} \n Columns or index not in keys: {cols_or_index_not_in_keys}")
                                
                                # ... accumulate the information in new_imgs_dict, which we will convert to a dataframe and append to imgs_df after we've gone through all new polygons.
                                for key in new_imgs_dict:

                                    new_imgs_dict[key].append(img_info_dict[key])

                                num_img_series_to_download -= 1
                        
        # Extract accumulated information about the imgs we've downloaded from new_imgs into a dataframe ...
        new_imgs_df = GeoDataFrame(new_imgs_dict)
        new_imgs_df.set_crs(epsg=self._params_dict['crs_epsg_code'], inplace=True) # standard crs
        new_imgs_df.set_index("img_name", inplace=True)

        # ... and append it to self.imgs_df:
        data_frames_list = [self.imgs_df, new_imgs_df]  
        self.imgs_df = GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)


    def _download_imgs_for_polygon(self,
            polygon_name: str,
            polygon_geometry: Polygon,
            download_dir: Union[str, Path],
            previously_downloaded_imgs_set: Sequence[str],
            **kwargs):
        """
        Not implemented, overwrite/implement in a subclass. Should download an image fully containing a vector polygon or several images jointly containing it and return a dict with information to be updated in the associator, see below for details.

        Args:
            polygon_name (str): the name of the vector polygon.
            polygon_geometry (Polygon): shapely geometry of polygon.
            download_dir (Union[Path, str]): directory that the image file should be downloaded to.
            previously_downloaded_imgs_set (Set[str]): Set of previously downloaded img_names. In some use cases when it can't be guaranteed that an image can be downloaded that fully contains the polygon it can happen that attempts will be made to download an image that is already in the associator. Passing this argument allows the download function to make sure it doesn't try downloading an image that is already in the dataset.
            **kwargs (Any): optional keyword arguments depending on the application.
            
        Returns:
            A dict with a key 'list_img_info_dicts' and value a list of dicts containing the information to be included in each row in the imgs_df of the calling associator, one for each newly downloaded image. The keys should be the index and column names of the imgs_df and the values the indices or entries of those columns in row that will correspond to the new image. We return a dict instead of directly returning the list to be backwards compatible.
        """

        raise NotImplementedError
    

    def _process_downloaded_img_file(self,
            img_name: str,
            in_dir: Union[str, Path],
            out_dir: Union[str, Path],
            convert_to_crs_epsg: int,
            **kwargs):
        """
        Not implemented, overwrite/implement in a subclass. Processes an image file downloaded by _download_imgs_for_polygon. Needs to return a dict with information to be updated in the associator, see below for details.
        
            Args:
                -img_name: the image name (index identifiying the corresponding row in imgs_df) 
                -in_dir: the directory the image file was downloaded to
                -out_dir: the directory the processed image file should be in (i.e. self.images_dir)
                -convert_to_crs_epsg: EPSG code of the crs the image (if georeferenced, e.g. as a GeoTiff) 
                    should be converted to.
                -**kwargs: optional keyword arguments depending on the application
            Returns:
                -img_info_dict: a dict containing the information to be updated in the imgs_df of the calling associator. The keys should be the index and column names of the imgs_df and the values lists of indices or entries of those columns.
        """

        raise NotImplementedError