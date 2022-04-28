"""
ImgDownloaderForSinglePolygon that downloads digital elevation model (DEM) data
from jaxa.jp's ALOS data-source.

See here https://www.eorc.jaxa.jp/ALOS/en/index.htm for an overview of the ALOS data.
A detailed product description for ALOS (file-format, etc) can be found in:
https://www.eorc.jaxa.jp/ALOS/en/aw3d30/aw3d30v3.2_product_e_e1.0.pdf
The data is assumed to be stored on the FTP server:
ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_vXXXX/
(port: 46287)

There are different versions of the ALOS data: 1804, 1903, 2003, 2012. Only the 1804 version has been tested.
"""

from typing import Literal, Dict, Union, Any
import logging
import math
import os
import shutil
import tarfile
import urllib.request as request
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Union

import numpy as np
from shapely.geometry.polygon.base import BaseGeometry
from rs_tools.downloaders.base_downloader_for_single_feature import ImgDownloaderForSingleVectorFeature
from rs_tools.connector import Connector

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

JAXA_DATA_VERSIONS = ['1804', '1903', '2003',
                      '2012']  # (attn: only 1804 has been tested so far)


class JAXADownloaderForSingleVectorFeature(ImgDownloaderForSingleVectorFeature
                                           ):
    """Download JAXA DEM (digital elevation) data."""

    def download(
        self,
        geom_name: Union[int, str],
        geom_geometry: BaseGeometry,
        connector: Connector,
        download_dir: Path,
        previously_downloaded_imgs_set: Set[Union[str, int]],
        data_version: str = None,
        download_mode: str = None,
        **kwargs,
    ) -> Dict[Union[Literal['img_name', 'img_processed?'], str], Any]:
        """Downloads DEM data from jaxa.jp's ftp-server for a given (vector) geometry and
        returns dict-structure compatible with the connector.

        Warning:
            The downloader has only been tested for the 1804 jaxa_data_version.

        Explanation:
            The 'bboxvertices' download_mode will download images for
            vertices of the bbox of the (vector) geometry. This is preferred for
            small (vector) geometries, but will miss regions inbetween if a (vector) geometry spans
            more than two images in each axis. The 'bboxgrid' mode will download
            images for each point on a grid defined by the bbox. This overshoots
            for small geometries, but works for large geometries.

        Args:
            geom_name (Union[str, int]): the name of the vector geometry
            geom_geometry (shapely geometry):
            download_dir (Path or str): directory that the image file should be downloaded to
            jaxa_data_version (str): One of '1804', '1903', '2003', or '2012'.
                1804 is the only version that has been tested.
                Defaults if possible to whichever choice you made last time.
            jaxa_download_mode (str): One of 'bboxvertices', 'bboxgrid'.
                Defaults if possible to whichever choice you made last time.
            **kwargs (Any): other kwargs, ignored.

        Returns:
            dict of dicts according to the connector convention
            (containing list_img_info_dict).

        Raises:
            log.warning: when a file cannot be found or opened on jaxa's-ftp
            (download_exception = 'file_not_available_on_JAXA_ftp')
        """

        if data_version not in JAXA_DATA_VERSIONS:
            raise ValueError(
                f"Unknown data_version {data_version}. Should be one of {', '.join(JAXA_DATA_VERSIONS)}"
            )

        jaxa_file_and_folder_names = set()
        if download_mode == 'bboxvertices':

            for (x, y) in geom_geometry.envelope.exterior.coords:

                jaxa_folder_name = '{}/'.format(
                    self._obtain_jaxa_index(x // 5 * 5, y // 5 * 5))
                jaxa_file_name = '{}.tar.gz'.format(
                    self._obtain_jaxa_index(x, y))

                jaxa_file_and_folder_names |= {(jaxa_file_name,
                                                jaxa_folder_name)}

        elif download_mode == 'bboxgrid':

            minx, miny, maxx, maxy = geom_geometry.envelope.exterior.bounds

            deltax = math.ceil(maxx - minx)
            deltay = math.ceil(maxy - miny)

            for countx in range(deltax + 1):
                for county in range(deltay + 1):

                    x = minx + countx
                    y = miny + county

                    jaxa_file_name = f"{self._obtain_jaxa_index(x, y)}.tar.gz"
                    jaxa_folder_name = f"{self._obtain_jaxa_index(x // 5 * 5, y // 5 * 5)}/"

                    jaxa_file_and_folder_names |= {(jaxa_file_name,
                                                    jaxa_folder_name)}

        else:
            raise ValueError(f"Unknown download_mode: {download_mode}")

        list_img_info_dicts = [
        ]  # to collect information per downloaded file for connector

        for jaxa_file_name, jaxa_folder_name in jaxa_file_and_folder_names:

            # Skip download if file has already been downloaded ...
            if jaxa_file_name[:
                              -7] + '_DSM.tif' in previously_downloaded_imgs_set:
                # in this case skip download, don't store in list_img_info_dicts
                log.info('Skipping download for image %s', jaxa_file_name)
                continue
            # ... else, download.
            else:
                log.info(
                    'Downloading from ftp.eorc.jaxa.jp (v%s) for geometry %s',
                    data_version, geom_name)
                log.info('Downloading to: %s',
                         os.path.join(download_dir, jaxa_file_name))
                try:
                    with closing(request.urlopen('ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_v' + data_version + '/' \
                                                    + jaxa_folder_name + jaxa_file_name)) as remote_source:
                        with open(os.path.join(download_dir, jaxa_file_name),
                                  'wb') as local_file:
                            shutil.copyfileobj(remote_source, local_file)
                except Exception as exc:
                    log.warning(
                        'File %s in folder %s could not be found on JAXA ftp or could not be opened: %s',
                        jaxa_file_name, jaxa_folder_name, exc.args)
                    # continue

                else:
                    # Extract downloaded .tar file ...
                    tar = tarfile.open(
                        os.path.join(download_dir, jaxa_file_name), "r:gz")
                    tar.extractall(path=download_dir,
                                   members=[
                                       tar.getmembers()[1]
                                   ])  # extract only DSM.tif from archive
                    tar.close()
                    # ... and after extracting delete it.
                    os.remove(os.path.join(download_dir, jaxa_file_name))

                    shutil.move(
                        str(download_dir / jaxa_file_name[:-7] /
                            (jaxa_file_name[:-7] + '_AVE_DSM.tif')),
                        str(download_dir / (jaxa_file_name[:-7] + '_DSM.tif')))
                    shutil.rmtree(download_dir / jaxa_file_name[:-7],
                                  ignore_errors=True)

                    date_time_now = datetime.now().strftime(
                        "%Y-%m-%d-%H:%M:%S")
                    img_info_dict = {
                        'img_name': jaxa_file_name[:-7] + '_DSM.tif',
                        'img_processed?': False,
                        'timestamp': date_time_now,
                    }
                    list_img_info_dicts.append(img_info_dict)

        return {'list_img_info_dicts': list_img_info_dicts}

    def _obtain_jaxa_index(self,
                           x: Optional[float] = None,
                           y: Optional[float] = None,
                           nx: int = 3,
                           ny: int = 3):
        """Creates string for filename corresponding to jaxas naming-convention
        to download from ftp server.

        :param x: float, longitude (W/E), can be 'None' (will be ignored then in string-creation)
        :param y: float, latitude (N/S), can be 'None' (will be ignored then in string-creation)
        :param nx, ny: int, number of digits used for naming (filled with leading 0's)
        :return: string, filename (but not filetype eg .tif) containing the coordinates x,y
        """
        if x is not None:
            xf = '{ew}{x:0{nx}d}'.format(ew='W' if x < 0 else 'E',
                                         x=int(abs(np.floor(x))),
                                         nx=nx)
        else:
            xf = ''
        if y is not None:
            yf = '{ns}{y:0{ny}d}'.format(ns='S' if y < 0 else 'N',
                                         y=int(abs(np.floor(y))),
                                         ny=ny)
        else:
            yf = ''
        out = yf + xf
        return out
