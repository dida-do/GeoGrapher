""" 
downloader to be used by image-polygon-associator library that
obtains digital elevation model (DEM) data from jaxa.jp's ALOS data-source

background: https://www.eorc.jaxa.jp/ALOS/en/index.htm
ALOS product description (file-format, etc): https://www.eorc.jaxa.jp/ALOS/en/aw3d30/aw3d30v3.2_product_e_e1.0.pdf
data is assumed to be stored on FTP server: ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_vXXXX/
port: 46287

different versions exist, to access replace XXXX with one of numbers the below:
––– 1804
––– 1903
––– 2003
––– 2012

"""

from shapely.geometry import box
from pathlib import Path
import rasterio as rio
import tarfile
import shutil
import urllib.request as request
from contextlib import closing
import os
from datetime import datetime
import numpy as np
from assoc.utils.utils import transform_shapely_geometry
from assoc import img_polygon_associator as ipa

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

JAXA_DATA_VERSION = ['1804', '1903', '2003', '2012'][0]     # select which data to use here (attn: only 1804 has been tested so far)
OBJECT_TYPES = ['ct', 'ht' ,'pt', 'it', 'rt', 'dt', 'ts', 'wr', 'h'] # TODO: refactor into single project-wide config file
STANDARD_CRS_EPSG_CODE = 4326 # WGS84 # TODO: refactor into single project-wide config file

def obtain_index(x=None, y=None, nx=3, ny=3):
    '''
    Creates string for filename corresponding to jaxas naming-convention to download from ftp server

    :param x: float, longitude (W/E), can be 'None' (will be ignored then in string-creation)
    :param y: float, latitude (N/S), can be 'None' (will be ignored then in string-creation)
    :param nx, ny: int, number of digits used for naming (filled with leading 0's)
    :return: string, filename (but not filetype eg .tif) containing the coordinates x,y
    '''
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

class AubJAXAImgPolygonAssociator(ipa.ImgPolygonAssociator):
    '''
    Extends Standard IPA class to handle jaxa-data
    '''
    def __init__(self, data_dir,
                        imgs_df=None, 
                        polygons_df=None, 
                        segmentation_classes=OBJECT_TYPES,
                        crs_epsg_code=STANDARD_CRS_EPSG_CODE,
                        label_type=None,
                ):
        super().__init__(data_dir=data_dir,
                        imgs_df=imgs_df, 
                        polygons_df=polygons_df, 
                        crs_epsg_code=crs_epsg_code, 
                        segmentation_classes=segmentation_classes,
                        label_type=label_type)
        

    def _download_imgs_for_polygon( self,
                                    polygon_name,
                                    polygon_geometry, 
                                    download_dir,  
                                    previously_downloaded_imgs_set,   
                                    **kwargs):
        '''
        Downloads DSM-data from jaxa.jp's ftp-server for a given polygon and returns dict-structure compatible 
        with image-polygon-associator.
        
        Note:
        – Only operates on a single polygon
        – Downloads up to 4 files per polygon (but not more if polygon is exceptionally large)
        - Does not collate multiple images currently
        – Only returns a single exception / error-code (not one per file downloaded)
        
        Args:
        :param polygon_name: str, the name of the polygon
        :param polygon_geometry: shapely object, geometry of a polygon 
        :param download_dir: Path variable or str, directory that the image file should be downloaded to 
        :param **kwargs: ignored currently
        :return: dict of dicts according to assoc-convention (list_img_info_dict and polygon_info_dict),

        :raises log.warning: when a file cannot be found or opened on jaxa's-ftp (download_exception = 'file_not_available_on_JAXA_ftp')

        '''

        # obtain all files that intersect with the vertices of the bounding box of the polygon
        # ATTENTION: if polygon extends beyond 2 files in width or height this might miss files in between 
        # the ones containing the outer points
        jaxa_folder_names = []
        jaxa_file_names = []
        for (x,y) in polygon_geometry.envelope.exterior.coords:
            jaxa_folder_names.append('{}/'.format(obtain_index(x // 5 * 5, y // 5 * 5)))
            jaxa_file_names.append('{}.tar.gz'.format(obtain_index(x, y)))
        
        # only keep unique entries, ie up to 4 different files
        jaxa_folder_names = list(set(jaxa_folder_names))
        jaxa_file_names = list(set(jaxa_file_names))
        
        list_img_info_dicts = [] # to collect information per downloaded file for associator

        download_exception = str(None)
        have_img_downloaded = False

        for jaxa_file_name, jaxa_folder_name in zip(jaxa_file_names, jaxa_folder_names):
            
            if jaxa_file_name[:-7] + '_DSM.tif' in previously_downloaded_imgs_set:
                # in this case skip download, don't store in list_img_info_dicts
                log.info('Skipping download for image ' + jaxa_file_name)
                continue
            else:
                # downloading from jaxa's FTP
                log.info(   f'Downloading from ftp.eorc.jaxa.jp (v{JAXA_DATA_VERSION}) for polygon {polygon_name} ' \
                            f'to: {os.path.join(download_dir, jaxa_file_name)}')
                try:
                    with closing(request.urlopen('ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_v' + JAXA_DATA_VERSION + '/' \
                                                    + jaxa_folder_name + jaxa_file_name)) as remote_source:
                        with open(os.path.join(download_dir, jaxa_file_name), 'wb') as local_file:
                            shutil.copyfileobj(remote_source, local_file)
                except Exception as e:
                    log.exception(f'File could not be found or opened: {e.args}')
                    download_exception = 'file_not_available_on_JAXA_ftp'
                    
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

                    have_img_downloaded = True
                    dateTimeObj = datetime.now()
                    list_img_info_dicts.append(
                        {
                            'img_name' : jaxa_file_name[:-7] + '_DSM.tif', # TODO: check relevance of _AVE_ file-name component, consider to rename/remove
                            'img_processed?': False,
                            'timestamp': dateTimeObj.strftime("%Y-%m-%d-%H:%M:%S")
                        }
                    )


        
        polygon_info_dict = {
            'have_img_downloaded?' : have_img_downloaded, # TODO: what happens when this is set to false?
            'download_exception' : download_exception     # TODO: where is this processed? what happens for the specified cases?
                                                          # TODO: possibility / usefulness of storing multiple exceptions (per image)
        }

        return {
            'list_img_info_dicts' : list_img_info_dicts,
            'polygon_info_dict' : polygon_info_dict
        }


    def _process_downloaded_img_file(self, img_name, in_dir, out_dir, convert_to_crs_epsg, **kwargs):
        '''
        provides required postprocessing for downloaded jaxa DEM.
        Currently only copying from in_dir to out_dir is performed.

        Note:
        - Future update could include interpolation of DEM data to increase resolution for
        better visualization
        
        Args:
        all arguments are provided in the contex of the higher-level function ImgPolygonAssociator
        which calls this function

        '''

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
