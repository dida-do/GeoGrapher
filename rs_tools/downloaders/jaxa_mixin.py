"""
Downloader mix in that obtains digital elevation model (DEM) data 
from jaxa.jp's ALOS data-source.

Background: https://www.eorc.jaxa.jp/ALOS/en/index.htm
ALOS product description (file-format, etc): 
https://www.eorc.jaxa.jp/ALOS/en/aw3d30/aw3d30v3.2_product_e_e1.0.pdf
The data is assumed to be stored on the FTP server:
ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_vXXXX/
(port: 46287)

different versions exist
––– 1804
––– 1903
––– 2003
––– 2012

"""

from __future__ import annotations
from typing import Optional, Set, Union
from shapely.geometry.polygon import Polygon
from shapely.geometry import box
from pathlib import Path
import rasterio as rio
import tarfile
import shutil
import urllib.request as request
from contextlib import closing
import os
from datetime import datetime
import math
import numpy as np
from assoc.utils.utils import transform_shapely_geometry

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

JAXA_DATA_VERSIONS = ['1804', '1903', '2003', '2012']     # (attn: only 1804 has been tested so far)


class JAXADownloaderMixIn(object):
    """
    Download JAXA DEM (digital elevation) data.
    """

    @property
    def jaxa_data_version(self):
        return self._params_dict['jaxa_data_version']

    @jaxa_data_version.setter
    def jaxa_data_version(self, new_jaxa_version : str):
        self._params_dict['jaxa_data_version'] = new_jaxa_version


    def _download_imgs_for_polygon_jaxa(
            self,
            polygon_name : str,
            polygon_geometry : Polygon,
            download_dir : Union[Path, str],
            previously_downloaded_imgs_set : Set[str],
            data_version : str = None,
            download_mode : str = None,
            **kwargs):
        """
        Downloads DEM data from jaxa.jp's ftp-server for a given polygon and
        returns dict-structure compatible with the image-polygon-associator.

        Note:
            - Only operates on a single polygon
            - Does not collate multiple images currently
            - Only returns a single exception / error-code
                (not one per file downloaded)

        Warning:
            The downloader has only been tested for the 1804 jaxa_data_version.

        Explanation:
            The 'bboxvertices' download_mode will download images for
            vertices of the bbox of the polygon. This is preferred for
            small polygons, but will miss regions inbetween if a polygon spans
            more than two images in each axis. The 'bboxgrid' mode will download
            images for each point on a grid defined by the bbox. This overshoots
            for small polygons, but works for large polygons.

        Args:
            polygon_name (str): the name of the polygon
            polygon_geometry (shapely polygon): 
            download_dir (Path or str): directory that the image file should be downloaded to
            jaxa_data_version (str): One of '1804', '1903', '2003', or '2012'.
                1804 is the only version that has been tested. 
                Defaults if possible to whichever choice you made last time.
            jaxa_download_mode (str): One of 'bboxvertices', 'bboxgrid'.
                Defaults if possible to whichever choice you made last time.
            **kwargs (Any): currently ignored
        
        Returns: 
            dict of dicts according to the associator convention
            (containing list_img_info_dict).

        Raises:
            log.warning: when a file cannot be found or opened on jaxa's-ftp
            (download_exception = 'file_not_available_on_JAXA_ftp')
        """

        for jaxa_specific_keyword_arg, value in {
                                                ('data_version', data_version),
                                                ('download_mode', download_mode),
                                            }:

            if value is None:
                try:
                    # Use saved value
                    value = getattr(self, f"jaxa_{jaxa_specific_keyword_arg}")
                except (AttributeError, KeyError):
                    raise ValueError(f"Need to set {jaxa_specific_keyword_arg} keyword argument for the jaxa downloader.")
            else:
                # Remember value
                setattr(self, f"jaxa_{jaxa_specific_keyword_arg}", value)

        if data_version not in JAXA_DATA_VERSIONS:
            raise ValueError(f"Unknown data_version {data_version}. Should be one of {', '.join(JAXA_DATA_VERSIONS)}")

        jaxa_file_and_folder_names = set()

        if download_mode == 'bboxvertices':

            # obtain all files that intersect with the vertices of the bounding box of the polygon
            # ATTENTION: if polygon extends beyond 2 files in width or height this might miss files in between
            # the ones containing the outer points

            for (x,y) in polygon_geometry.envelope.exterior.coords:

                jaxa_folder_name = '{}/'.format(self._obtain_jaxa_index(x // 5 * 5, y // 5 * 5))
                jaxa_file_name = '{}.tar.gz'.format(self._obtain_jaxa_index(x, y))

                jaxa_file_and_folder_names |= {(jaxa_file_name, jaxa_folder_name)}

        elif download_mode == 'bboxgrid':

            minx, miny, maxx, maxy = polygon_geometry.envelope.exterior.bounds

            deltax = math.ceil(maxx - minx)
            deltay = math.ceil(maxy - miny)

            for countx in range(deltax + 1):
                for county in range(deltay + 1):

                    x = minx + countx
                    y = miny + county

                    jaxa_file_name = f"{self._obtain_jaxa_index(x, y)}.tar.gz"
                    jaxa_folder_name = f"{self._obtain_jaxa_index(x // 5 * 5, y // 5 * 5)}/"

                    jaxa_file_and_folder_names |= {(jaxa_file_name, jaxa_folder_name)}

        else:
            raise ValueError(f"Unknown download_mode: {download_mode}")

        list_img_info_dicts = [] # to collect information per downloaded file for associator

        # download_exception = str(None)
        # have_img_downloaded = False

        for jaxa_file_name, jaxa_folder_name in jaxa_file_and_folder_names:

            if jaxa_file_name[:-7] + '_DSM.tif' in previously_downloaded_imgs_set:
                # in this case skip download, don't store in list_img_info_dicts
                log.info('Skipping download for image ' + jaxa_file_name)
                continue
            else:
                # downloading from jaxa's FTP
                log.info(   f'Downloading from ftp.eorc.jaxa.jp (v{data_version}) for polygon {polygon_name} ' \
                            f'to: {os.path.join(download_dir, jaxa_file_name)}')
                try:
                    with closing(request.urlopen('ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_v' + data_version + '/' \
                                                    + jaxa_folder_name + jaxa_file_name)) as remote_source:
                        with open(os.path.join(download_dir, jaxa_file_name), 'wb') as local_file:
                            shutil.copyfileobj(remote_source, local_file)
                except Exception as e:
                    log.exception(f'File could not be found on JAXA ftp or could not be opened: {e.args}')
                    # download_exception = 'file_not_available_on_JAXA_ftp'

                else:
                    # extracting .tar file and deleting it afterwards
                    tar = tarfile.open(os.path.join(download_dir, jaxa_file_name), "r:gz")
                    tar.extractall(path=download_dir, members=[tar.getmembers()[1]]) # extract only DSM.tif from archive
                    tar.close()
                    os.remove(os.path.join(download_dir, jaxa_file_name)) # delete .tar file
                    # move needed DEM-file outside of unzipped folder (existing versions are overwritten)
                    shutil.move( str(Path(download_dir) / jaxa_file_name[:-7] / (jaxa_file_name[:-7] + '_AVE_DSM.tif')),
                                str(Path(download_dir) / (jaxa_file_name[:-7] + '_DSM.tif')) )
                    os.rmdir(os.path.join(Path(download_dir), jaxa_file_name[:-7])) # remove folder, only works when it is empty

                    # -- skipped -- only relevant when not iterating over polygons -------
                    # check if polygon is within boundaries of DEM file
                    # if download_exception == str(None):     # only if DEM was obtained successfully
                    #     with rio.open(str(Path(download_dir) / (jaxa_file_name[:-7] + '_AVE_DSM.tif'))) as src:
                    #         img_bounding_rectangle = box(*src.bounds)
                    #         if img_bounding_rectangle.contains(polygon_geometry) == False:
                    #             log.warning('Polygon ' + polygon_name + ' not fully contained in downloaded DEM. Processing errors might occur.')
                    #             log.warning('Polygon ' + polygon_name + ' bounds: ' + str(polygon_geometry.envelope.exterior))
                    #             log.warning('DEM ' + jaxa_file_name + ' bounds: ' + str(img_bounding_rectangle))
                    #             download_exception = 'polygon_outside_file_bounds'

                    # have_img_downloaded = True
                    dateTimeObj = datetime.now()
                    list_img_info_dicts.append(
                        {
                            'img_name' : jaxa_file_name[:-7] + '_DSM.tif', # TODO: check relevance of _AVE_ file-name component, consider to rename/remove
                            'img_processed?': False,
                            'timestamp': dateTimeObj.strftime("%Y-%m-%d-%H:%M:%S")
                        }
                    )

        return {
            'list_img_info_dicts' : list_img_info_dicts
        }


    def _process_downloaded_img_file_jaxa(
            self,
            img_name : str,
            in_dir : Union[Path, str],
            out_dir : Union[Path, str],
            convert_to_crs_epsg : int,
            **kwargs):
        """
        provides required postprocessing for downloaded jaxa DEM.
        Currently only copying from in_dir to out_dir is performed.

        Note:
        - Future update could include interpolation of DEM data to increase resolution for
        better visualization

        Args:
        all arguments are provided in the contex of the higher-level function ImgPolygonAssociator
        which calls this function
        """

        # obtain target coordinate-ref-system code from source file:
        geotif_filename = Path(in_dir) / img_name
        with rio.open(geotif_filename) as src:
            orig_crs_epsg_code = src.crs.to_epsg()
            img_bounding_rectangle = box(*src.bounds)

        # move file
        shutil.move( Path(in_dir) / img_name, Path(out_dir) / img_name )

        # setting relevant assoc properties
        img_info_dict = {
            'orig_crs_epsg_code' : orig_crs_epsg_code, # alternatively directly use 4326 for WGS84 (used for jaxa)
            'img_name' : img_name,
            'img_processed?' : True,
            'geometry' : transform_shapely_geometry(img_bounding_rectangle, orig_crs_epsg_code, 4326)
        }

        log.info('Successfully downloaded & processed file ' + img_name)
        return img_info_dict

    def _obtain_jaxa_index(
            self,
            x : Optional[float] = None,
            y : Optional[float] = None,
            nx : int = 3,
            ny : int = 3):
        """
        Creates string for filename corresponding to jaxas naming-convention to download from ftp server

        :param x: float, longitude (W/E), can be 'None' (will be ignored then in string-creation)
        :param y: float, latitude (N/S), can be 'None' (will be ignored then in string-creation)
        :param nx, ny: int, number of digits used for naming (filled with leading 0's)
        :return: string, filename (but not filetype eg .tif) containing the coordinates x,y
        """
        if x is not None:
            xf = '{ew}{x:0{nx}d}'.format(ew='W' if x < 0 else 'E', x=int(abs(np.floor(x))), nx=nx)
        else:
            xf = ''
        if y is not None:
            yf = '{ns}{y:0{ny}d}'.format(ns='S' if y < 0 else 'N', y=int(abs(np.floor(y))), ny=ny)
        else:
            yf = ''
        out = yf + xf
        return out