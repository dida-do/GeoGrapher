"""
ImgPolygonAssociator that can download Sentinel-2 images.
"""

import os
from pathlib import Path
import pathlib
import logging
from tqdm import tqdm
from shapely import wkt
import shapely
from zipfile import ZipFile
from sentinelsat import SentinelAPI
import random
from dotenv import load_dotenv


import rs_tools.img_polygon_associator as ipa 
from didatools.remote_sensing.data_preparation.sentinel_2_preprocess import safe_to_geotif_L2A
from rs_tools.utils.utils import transform_shapely_geometry
from rs_tools.errors import ImgAlreadyExistsError, NoImgsForPolygonFoundError, ImgDownloadError



MAX_PERCENT_CLOUD_COVERAGE=10
PRODUCTTYPE='L2A' # or 'L1C'
STANDARD_CRS_EPSG_CODE = 4326 # WGS84 
RESOLUTION = 10 # possible values for Sentinel-2 L2A: 10, 20, 60 (in meters). See here https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
DATA_DIR_SUBDIRS = [Path("images"), Path("labels"), Path("safe_files")]
LABEL_TYPE = 'categorical'



# logger
log = logging.getLogger(__name__)


class ImgPolygonAssociatorS2(ipa.ImgPolygonAssociator):
    """
    ImgPolygonAssociator for Sentinel-2 images.

    Requires environment variables sentinelAPIusername and sentinelAPIpassword to set up the sentinel API. Assumes imgs_df has columns 'geometry', 'timestamp', 'orig_crs_epsg_code', and 'img_processed?'. Subclass/modify if you need other columns. 
    """

    def __init__(self, data_dir,
                        imgs_df=None, 
                        polygons_df=None, 
                        segmentation_classes=None,
                        crs_epsg_code=STANDARD_CRS_EPSG_CODE,
                        producttype=PRODUCTTYPE, 
                        resolution=RESOLUTION,
                        max_percent_cloud_coverage=MAX_PERCENT_CLOUD_COVERAGE,
                        label_type=LABEL_TYPE
                        ):
        """
        - data_dir: The data directory of the associator. 

        - imgs_df: optional imgs_df to initialize associator with. If not given, the associator will assume it can load an imgs_df.geojson file from data_dir.

        - polygons_df: polygons_df to initialize associator with. If not given, the associator will assume it can load an imgs_df.geojson file from data_dir.

        - segmentation_classes: list of segmentation classes, i.e. tailing types for the AuBeSa project.

        - standard_crs_epsg_code: the EPSG code of the coordinate reference system (crs) used to store the geometries in the imgs_df and polygons_df GeoDataFrames.
            
        - producttype (str): Sentinel-2 product type, "L2A" or "L1C".
        
        - resolution (int): resolution of Sentinel-2 images, one of 10, 20, 60 (for L2A, in meters).

        - max_percent_cloud_coverage (int): maximum allowable cloud coverage percentage when querying for a Sentinel-2 image. Should be between 0 and 100.

        - label_type:
        """

        super().__init__(data_dir=data_dir,
                                                    imgs_df=imgs_df, 
                                                    polygons_df=polygons_df, 
                                                    crs_epsg_code=crs_epsg_code, 
                                                    segmentation_classes=segmentation_classes,
                                                    label_type=label_type, 
                                                    producttype=producttype, 
                                                    resolution=resolution, 
                                                    max_percent_cloud_coverage=max_percent_cloud_coverage)


    def _download_imgs_for_polygon(self, polygon_name,
                                polygon_geometry, 
                                download_dir,
                                previously_downloaded_imgs_set,
                                **kwargs):
        """
        Downloads a sentinel-2 image fully containing the polygon, returns a dict in the format needed by the associator.

        :raises NoImgsForPolygonFoundError: Raised if no downloadable images with cloud coverage less than or equal to max_percent_cloud_coverage could be found for the polygon.
        :raises ImgAlreadyExistsError: Raised if the image selected to download already exists in the associator. 
        """

        # To set up sentinel API, ...
        # ... load environment variables ...
        load_dotenv()

        # ... extract username and password ...
        username = os.environ.get("sentinelAPIusername")
        password = os.environ.get("sentinelAPIpassword")
        if username is None or password is None:
            log.error(f"sentinelAPI: missing username or password")

        # ... and instantiate the API.
        api = SentinelAPI(username, password)

        # Return dicts with values to be collected in calling associator.
        polygon_info_dict = {'download_exception': str(None)}
        img_info_dict = {}

        # Determine missing args for the sentinel query.
        rectangle_wkt = wkt.dumps(polygon_geometry.envelope)
        area_relation='Contains'
        date = ("NOW-364DAYS", "NOW") # anything older is in long-term archive
        # Allow for producttype shorthand.
        if kwargs['producttype'] in {'L2A', 'S2MSI2A'}:
            producttype = 'S2MSI2A'
        elif kwargs['producttype'] in {'L1C', 'S2MSI1C'}:
            producttype = 'S2MSI1C'
        else:
            raise Exception(f"ImgPolygonAssociator.__init__: unknown producttype: {kwargs['producttype']}")
    
        # (One less than) the allowable cloud coverage we'll start querying with. We'll increment this until we find products or reach max allowable.
        cloud_coverage_counter = -1 
        
        products={}

        # Increase cloud coverage until query to API returns an image:
        while cloud_coverage_counter < kwargs['max_percent_cloud_coverage'] and len(products)==0:

            cloud_coverage_counter+=1

            try:

                # Query, remember results
                products = api.query(area=rectangle_wkt, 
                                        date=date, 
                                        area_relation=area_relation,
                                        producttype=producttype, 
                                        cloudcoverpercentage=(0,cloud_coverage_counter))

            # The sentinelsat API can throw an exception if there are no results for a query instead of returning an empty dict ...
            except:

                # ... so in that case we set the result by hand:
                products = {}

        # If we couldn't find anything, remember that, so we can deal with it later.
        if len(products) == 0:

            raise NoImgsForPolygonFoundError(f"No images for polygon {polygon_name} found with cloud coverage less than or equal to {kwargs['max_percent_cloud_coverage']}!")

        # If the query was succesful ...
        else:

            # ... we select a random product, ...
            product_id = random.choice(list(products.keys()))                

            # ... and determine the file name.
            product_metadata = api.get_product_odata(product_id, full=True)
            # (this key might have to be 'filename' (minus the .SAFE at the end) for L1C products?)
            try:
                img_name =  product_metadata['title'] + ".tif"
            except:
                raise Exception(f"Couldn't get the filename. Are you trying to download L1C products? Try changing the key for the products dict in the line of code above this...")

            # If the file has been downloaded before (really, this should not happen, since EXPLAIN!), throw an error ...
            if img_name in previously_downloaded_imgs_set:

                raise ImgAlreadyExistsError(f"_download_imgs_for_polygon wanted to download image {img_name} for polygon {polygon_name}, but this image was already downloaded previously by the associator. Something is wrong!")

            # ... if not, ...
            else:

                # ...  attempt to download it.
                try:

                    api.download(product_id, directory_path=download_dir)

                # If the download failed, remember that:
                except:

                    raise ImgDownloadError(f"An error occured while downloading img {img_name} with product id {product_id}.")

                # If the download was succesful...
                else:

                    # ...remember we downloaded an image for the polygon.
                    polygon_info_dict['have_img_downloaded?'] = True

                    # And assemble the information to updated in the returned img_info_dict:

                    img_info_dict['img_name'] = img_name
                    img_info_dict['img_processed?'] = False
                    img_info_dict['timestamp'] = product_metadata['Date'].strftime("%Y-%m-%d-%H:%M:%S")

        return {'list_img_info_dicts': [img_info_dict], 'polygon_info_dict': polygon_info_dict}


    def _process_downloaded_img_file(self, img_name, in_dir, out_dir, convert_to_crs_epsg, **kwargs) -> dict:
        """        
        Extracts downloaded sentinel-2 zip file to a .SAFE directory, then processes/converts to a GeoTiff image, deletes the zip file, puts the GeoTiff image in the right directory, and returns information about the img in a dict.
        """

        # file names and paths
        filename_no_extension = Path(img_name).stem
        zip_filename = filename_no_extension + ".zip"
        safe_path = os.path.abspath(in_dir / Path("safe_files") / (filename_no_extension + ".SAFE"))
        zip_path = os.path.abspath(in_dir / zip_filename)

        # extract zip to .SAFE and delete zip             
        with ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(in_dir / Path("safe_files/"))
        os.remove(zip_path)

        # convert SAFE to GeoTiff
        conversion_dict = safe_to_geotif_L2A(safe_root=safe_path, 
                            resolution=kwargs['resolution'],
                            outdir=out_dir)

        orig_crs_epsg_code, img_bounding_rectangle_orig_crs = conversion_dict["crs_epsg_code"], conversion_dict["img_bounding_rectangle"]

        # convert img_bounding_rectangle to the standard crs
        img_bounding_rectangle = transform_shapely_geometry(img_bounding_rectangle_orig_crs, from_epsg=orig_crs_epsg_code, to_epsg=convert_to_crs_epsg)
        # try using rio.warp.transform_bounds?? maybe this is more accurate when comparing to APIfootprint??

        # return a dict
        return_dict = {'img_name': img_name, 
                        'geometry': img_bounding_rectangle, 
                        'orig_crs_epsg_code': orig_crs_epsg_code, 
                        'img_processed?': True}

        return return_dict

