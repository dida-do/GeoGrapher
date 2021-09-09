"""
ImgPolygonAssociator that can download Sentinel-2 images.
"""

from typing import Union, List, Optional, Dict, Type, Sequence, Any
import os
from pathlib import Path
import logging
from shapely import wkt
from zipfile import ZipFile
from sentinelsat import SentinelAPI
from dotenv import load_dotenv
from geopandas import GeoDataFrame, GeoSeries

from img_polygon_associator import ImgPolygonAssociator
from rs_tools.utils.utils import transform_shapely_geometry
from rs_tools.errors import ImgAlreadyExistsError, NoImgsForPolygonFoundError, ImgDownloadError
from didatools.remote_sensing.data_preparation.sentinel_2_preprocess import safe_to_geotif_L2A


MAX_PERCENT_CLOUD_COVERAGE=10
PRODUCTTYPE='L2A' # or 'L1C'
STANDARD_CRS_EPSG_CODE = 4326 # WGS84 
RESOLUTION = 10 # possible values for Sentinel-2 L2A: 10, 20, 60 (in meters). See here https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
DATA_DIR_SUBDIRS = [Path("images"), Path("labels"), Path("safe_files")]
LABEL_TYPE = 'categorical'



# logger
log = logging.getLogger(__name__)


class ImgPolygonAssociatorS2(ImgPolygonAssociator):
    """
    ImgPolygonAssociator for Sentinel-2 images.

    Requires environment variables sentinelAPIusername and sentinelAPIpassword to set up the sentinel API. Assumes imgs_df has columns 'geometry', 'timestamp', 'orig_crs_epsg_code', and 'img_processed?'. Subclass/modify if you need other columns. 
    """

    def __init__(self, 

        # args w/o default values
        segmentation_classes : Sequence[str], 
        label_type : str,
        
        # polygons_df args. Exactly one value needs to be set (i.e. not None).
        polygons_df : Optional[GeoDataFrame] = None,
        polygons_df_cols : Optional[Union[List[str], Dict[str, Type]]] = None,
        
        # imgs_df args. Exactly one value needs to be set (i.e. not None).
        imgs_df : Optional[GeoDataFrame] = None, 
        imgs_df_cols : Optional[Union[List[str], Dict[str, Type]]] = None,
        
        # remaining non-path args w/ default values
        add_background_band_in_labels : bool = False, 
        crs_epsg_code : int = STANDARD_CRS_EPSG_CODE, 
        producttype : str = PRODUCTTYPE,
        resolution : int = RESOLUTION,
        max_percent_cloud_coverage : int = MAX_PERCENT_CLOUD_COVERAGE,

        # path args
        data_dir : Optional[Union[Path, str]] = None, # either this arg or all the path args below must be set (i.e. not None)
        images_dir : Optional[Union[Path, str]] = None, 
        labels_dir : Optional[Union[Path, str]] = None, 
        assoc_dir : Optional[Union[Path, str]] = None, 
        download_dir : Optional[Union[Path, str]] = None, 
        
        # optional kwargs
        **kwargs : Any
        ):            
        """
        See the docstring for the parent class __init__ for description of the arguments not mentioned here. 

        Args:
            producttype (str) : Sentinel-2 product type, "L2A" or "L1C", defaults to PRODUCTTYPE

            resolution (int) :  resolution of Sentinel-2 images, one of 10, 20, 60 (for L2A, in meters). Defaults to RESOLUTION

            max_percent_cloud_coverage (int): maximum allowable cloud coverage percentage when querying for a Sentinel-2 image. Should be between 0 and 100. Defults to MAX_PERCENT_CLOUD_COVERAGE

            **kwargs (Any): Optional keyword arguments.
        """

        super().__init__(
            segmentation_classes=segmentation_classes, 
            label_type=label_type, 
            polygons_df=polygons_df, 
            polygons_df_cols=polygons_df_cols, 
            imgs_df=imgs_df, 
            imgs_df_cols=imgs_df_cols, 
            add_background_band_in_labels=add_background_band_in_labels, 
            crs_epsg_code=crs_epsg_code, 
            data_dir=data_dir, 
            images_dir=images_dir, 
            labels_dir=labels_dir, 
            assoc_dir=assoc_dir, 
            download_dir=download_dir,
            producttype=producttype,
            resolution=resolution,
            max_percent_cloud_coverage=max_percent_cloud_coverage, 
            **kwargs)


    def _download_imgs_for_polygon(self,
            polygon_name: str,
            polygon_geometry: GeoSeries,
            download_dir: Union[str, Path],
            previously_downloaded_imgs_set: List[str],
            **kwargs) -> dict:
        """
        Downloads a sentinel-2 image fully containing the polygon, returns a dict in the format needed by the associator.

        Args:
            polygon_name: The name of the polygon, only relevant for print statements and errors.  #TODO: I'd make that one optional
            polygon_geometry: The areas the images shall be downloaded for.
            download_dir: Directory to save the downloaded Sentinel-2 products.
            previously_downloaded_imgs_set: A list of already downloaded products, will be used prevent double downloads.

        Returns:
            info_dicts: A dictionary containing information about the images and polygons. ({'list_img_info_dicts': [img_info_dict], 'polygon_info_dict': polygon_info_dict})

        Raises:
            LookupError: Raised if an unkknown product type is given.
            NoImgsForPolygonFoundError: Raised if no downloadable images with cloud coverage less than or equal to max_percent_cloud_coverage could be found for the polygon.
            KeyError: Raised if the product name could not be extracted correctly.
            ImgAlreadyExistsError: Raised if the image selected to download already exists in the associator.
            ImgDownloadError: Raised if an error occurred while trying to download a product.

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
            raise LookupError(f"ImgPolygonAssociator.__init__: unknown producttype: {kwargs['producttype']}")
    
        try:

            # Query, remember results
            products = api.query(area=rectangle_wkt, 
                                    date=date, 
                                    area_relation=area_relation,
                                    producttype=producttype, 
                                    cloudcoverpercentage=(0,kwargs['max_percent_cloud_coverage']))

            products = {k: v for k, v in products.items() if api.is_online(k)}

        # The sentinelsat API can throw an exception if there are no results for a query instead of returning an empty dict ...
        except:

            # ... so in that case we set the result by hand:
            products = {}

        # If we couldn't find anything, remember that, so we can deal with it later.
        if len(products) == 0:

            raise NoImgsForPolygonFoundError(f"No images for polygon {polygon_name} found with cloud coverage less than or equal to {kwargs['max_percent_cloud_coverage']}!")

        # If the query was succesful ...
        products_list = list(products.keys())
        products_list = sorted(products_list, key=lambda x: products[x]["cloudcoverpercentage"])
        for product_id in products_list:
            # ... and determine the file name.
            product_metadata = api.get_product_odata(product_id, full=True)
            # (this key might have to be 'filename' (minus the .SAFE at the end) for L1C products?)
            try:
                img_name =  product_metadata['title'] + ".tif"
            except:
                raise Exception(f"Couldn't get the filename. Are you trying to download L1C products? Try changing the key for the products dict in the line of code above this...")

            # If the file has been downloaded before (really, this should not happen, since EXPLAIN!), throw an error ...
            if img_name not in previously_downloaded_imgs_set:
                try:
                    api.download(product_id, directory_path=download_dir)
                    zip_path = download_dir / (product_metadata['title'] + ".zip")
                    with ZipFile(zip_path) as zip_ref:
                        assert zip_ref.testzip() is None

                    # And assemble the information to be updated in the returned img_info_dict:
                    img_info_dict['img_name'] = img_name
                    img_info_dict['img_processed?'] = False
                    img_info_dict['timestamp'] = product_metadata['Date'].strftime("%Y-%m-%d-%H:%M:%S")

                    return {'list_img_info_dicts': [img_info_dict]}

                except:
                    log.warn(f"failed to download {product_metadata['title']}")

        raise NoImgsForPolygonFoundError(f"Either no images were found for {polygon_name} found or all images failed to download.")


    def _process_downloaded_img_file(self,
            img_name: str,
            in_dir: Union[str, Path],
            out_dir: Union[str, Path],
            convert_to_crs_epsg: int,
            **kwargs) -> dict:
        """
        Extracts downloaded sentinel-2 zip file to a .SAFE directory, then processes/converts to a GeoTiff image, deletes the zip file, puts the GeoTiff image in the right directory, and returns information about the img in a dict.

        Args:
            img_name: The name of the image.
            in_dir: The directory containing the zip file.
            out_dir: The directory to save the
            convert_to_crs_epsg: The EPSG code to use to create the image bounds property.  # TODO: this name might not be appropriate as it suggests that the image geometries will be converted into that crs.

        Returns:
            return_dict: Contains information about the downloaded product.

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


