"""
ImgPolygonAssociator that can download Sentinel-2 images.
"""

from typing import Union, List, Optional, Dict, Any, Tuple
import itertools
from collections import OrderedDict
import os
from pathlib import Path
import logging
import numpy as np
import geopandas as gpd
import rasterio as rio
from shapely.geometry import box
from shapely import wkt
from zipfile import ZipFile
from sentinelsat import SentinelAPI
from dotenv import load_dotenv
from geopandas import GeoSeries
from scipy.ndimage import zoom

from rs_tools.img_download.sentinel2_safe_unpacking import safe_to_geotif_L2A
from rs_tools.utils.utils import transform_shapely_geometry
from rs_tools.errors import NoImgsForPolygonFoundError


# MAX_PERCENT_CLOUD_COVERAGE=10
# PRODUCTTYPE='L2A' # or 'L1C'
# STANDARD_CRS_EPSG_CODE = 4326 # WGS84
# RESOLUTION = 10 # possible values for Sentinel-2 L2A: 10, 20, 60 (in meters). See here https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
# DATA_DIR_SUBDIRS = [Path("images"), Path("labels"), Path("safe_files")]
# LABEL_TYPE = 'categorical'
NO_DATA_VAL = 0 # No data value for sentinel 2



# logger
log = logging.getLogger(__name__)


class Sentinel2DownloaderMixIn(object):
    """
    Downloader for Sentinel-2 images.

    Requires environment variables sentinelAPIusername and sentinelAPIpassword to set up the sentinel API. Assumes imgs_df has columns 'geometry', 'timestamp', 'orig_crs_epsg_code', and 'img_processed?'. Subclass/modify if you need other columns.
    """

    @property
    def sentinel2_producttype(self) -> str:
        return self._params_dict['sentinel2_producttype']

    @sentinel2_producttype.setter
    def sentinel2_producttype(self, new_sentinel2_producttype : str):
        self._params_dict['sentinel2_producttype'] = new_sentinel2_producttype


    @property
    def sentinel2_resolution(self):
        return self._params_dict['sentinel2_resolution']

    @sentinel2_resolution.setter
    def sentinel2_resolution(self, new_sentinel2_resolution : str):
        self._params_dict['sentinel2_resolution'] = new_sentinel2_resolution


    @property
    def sentinel2_max_percent_cloud_coverage(self):
        return self._params_dict['sentinel2_max_percent_cloud_coverage']

    @sentinel2_max_percent_cloud_coverage.setter
    def sentinel2_max_percent_cloud_coverage(self, new_sentinel2_max_percent_cloud_coverage : str):
        self._params_dict['sentinel2_max_percent_cloud_coverage'] = new_sentinel2_max_percent_cloud_coverage


    def _download_imgs_for_polygon_sentinel2(self,
            polygon_name: str,
            polygon_geometry: GeoSeries,
            download_dir: Union[str, Path],
            previously_downloaded_imgs_set: List[str],
            producttype : Optional[str] = None,
            resolution : Optional[int] = None,
            max_percent_cloud_coverage : Optional[int] = None,
            date : Optional[Any] = None, # See here for type https://sentinelsat.readthedocs.io/en/latest/api_reference.html
            area_relation : Optional[str] = None,
            **kwargs) -> dict:
        """
        Downloads a sentinel-2 image fully containing the polygon, returns a dict in the format needed by the associator.

        Args:
            polygon_name: The name of the polygon, only relevant for print statements and errors.  #TODO: I'd make that one optional
            polygon_geometry: The areas the images shall be downloaded for.
            download_dir: Directory to save the downloaded Sentinel-2 products.
            previously_downloaded_imgs_set: A list of already downloaded products, will be used prevent double downloads.
            producttype (str): One of 'L1C'/'S2MSI1C' or 'L2A'/'S2MSI2A'
            resolution (int): One of 10, 20, or 60.
            max_percent_cloud_coverage (int): Integer between 0 and 100.
            date (Any):  See https://sentinelsat.readthedocs.io/en/latest/api_reference.html
            area_relation : See https://sentinelsat.readthedocs.io/en/latest/api_reference.html

        Returns:
            info_dicts: A dictionary containing information about the images and polygons. ({'list_img_info_dicts': [img_info_dict], 'polygon_info_dict': polygon_info_dict})

        Raises:
            LookupError: Raised if an unkknown product type is given.
            NoImgsForPolygonFoundError: Raised if no downloadable images with cloud coverage less than or equal to max_percent_cloud_coverage could be found for the polygon.
            KeyError: Raised if the product name could not be extracted correctly.
            ImgAlreadyExistsError: Raised if the image selected to download already exists in the associator.
            ImgDownloadError: Raised if an error occurred while trying to download a product.

        """

        for s2_specific_keyword_arg, value in {
                                                ('producttype', producttype),
                                                ('resolution', resolution),
                                                ('max_percent_cloud_coverage', max_percent_cloud_coverage),
                                                ('area_relation', area_relation),
                                                ('date', date)
                                            }:

            if value is None:
                try:
                    # Use saved value
                    value = getattr(self, f"sentinel2_{s2_specific_keyword_arg}")
                except (AttributeError, KeyError):
                    raise ValueError(f"Need to set {s2_specific_keyword_arg} keyword argument for the sentinel2 downloader.")
            else:
                # Remember value
                setattr(self, f"sentinel2_{s2_specific_keyword_arg}", value)

        # Run some safety checks on the arg values
        if resolution  not in {10, 20, 60}:
            raise ValueError(f"Unknown resolution: {resolution}")
        if max_percent_cloud_coverage < 0 or max_percent_cloud_coverage > 100:
            raise ValueError(f"Unknown max_percent_cloud_coverage: {max_percent_cloud_coverage}")
        if producttype  not in {'L1C', 'S2MSI1C', 'L2A', 'S2MSI2A'}:
            raise ValueError(f"Unknown producttype: {producttype}")

        # producttype abbreviations
        if producttype == 'L1C':
            producttype = 'S2MSI1C'
        if producttype == 'L2A':
            producttype = 'S2MSI2A'

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
        # area_relation='Contains'
        # date = ("NOW-364DAYS", "NOW") # anything older is in long-term archive
        # Allow for producttype shorthand.
        if producttype in {'L2A', 'S2MSI2A'}:
            producttype = 'S2MSI2A'
        elif producttype in {'L1C', 'S2MSI1C'}:
            producttype = 'S2MSI1C'
        else:
            raise LookupError(f"ImgPolygonAssociator.__init__: unknown producttype: {producttype}")

        try:

            # Query, remember results
            products = api.query(area=rectangle_wkt,
                                    date=date,
                                    area_relation=area_relation,
                                    producttype=producttype,
                                    cloudcoverpercentage=(0,max_percent_cloud_coverage))

            products = {k: v for k, v in products.items() if api.is_online(k)}

        # The sentinelsat API can throw an exception if there are no results for a query instead of returning an empty dict ...
        except:

            # ... so in that case we set the result by hand:
            products = {}

        # If we couldn't find anything, remember that, so we can deal with it later.
        if len(products) == 0:

            raise NoImgsForPolygonFoundError(f"No images for polygon {polygon_name} found with cloud coverage less than or equal to {max_percent_cloud_coverage}!")

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


    def _process_downloaded_img_file_sentinel2(self,
            img_name: str,
            in_dir: Union[str, Path],
            out_dir: Union[str, Path],
            convert_to_crs_epsg: int,
            resolution: int,
            **kwargs) -> dict:
        """
        Extracts downloaded sentinel-2 zip file to a .SAFE directory, then processes/converts to a GeoTiff image, deletes the zip file, puts the GeoTiff image in the right directory, and returns information about the img in a dict.

        Args:
            img_name: The name of the image.
            in_dir: The directory containing the zip file.
            out_dir: The directory to save the
            convert_to_crs_epsg: The EPSG code to use to create the image bounds property.  # TODO: this name might not be appropriate as it suggests that the image geometries will be converted into that crs.
            resolution: int

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
                            resolution=resolution,
                            outdir=out_dir)

        orig_crs_epsg_code, img_bounding_rectangle_orig_crs = conversion_dict["crs_epsg_code"], conversion_dict["img_bounding_rectangle"]

        # convert img_bounding_rectangle to the standard crs
        img_bounding_rectangle = transform_shapely_geometry(img_bounding_rectangle_orig_crs, from_epsg=orig_crs_epsg_code, to_epsg=convert_to_crs_epsg)
        # try using rio.warp.transform_bounds?? maybe this is more accurate when comparing to APIfootprint??

        # return a dict
        return_dict = {'img_name': img_name,
                        'geometry': img_bounding_rectangle,
                        'orig_crs_epsg_code': int(orig_crs_epsg_code),
                        'img_processed?': True}

        return return_dict
    
    # def _safe_to_geotif_L2A(
    #         self,
    #         safe_root : Union[Path, str],
    #         resolution : Union[int, str],
    #         outdir : Optional[Union[Path, str]] = None,
    #         TCI : bool = True,
    #         bands_order : List[str] = ["B01", "B02", "B03", "B04", "B05", "B06",
    #                     "B07","B8A","B08", "B09", "B11", "B12", "WVP", "AOT"],
    #         jp2_masks_order : List[str] = ["CLDPRB"],
    #         gml_mask_order : List[str] = [("CLOUDS", "B00")]
    #         ) -> dict:
    #     """
    #     Converts a .SAFE file with L2A sentinel-2 data to geotif and returns a dict 
    #     with the crs epsg code and a shapely polygon defined by the image bounds.

    #     band structure of final geotif:
    #         if TCI: 1-3 TCI RGB
    #         else bands_order (only the available ones will be stored though) , jp2_masks_order, gml_mask_order

    #     resolution: the desired resolution

    #     jp2_masks are only available up to a resolution of 20 m, so for 10m the 20m mask ist taken

    #     safe_root is the safe folder root

    #     TCI = true color image

    #     bands_order and jp2_masks_order are lists of strings
    #     gml_mask_order is a list of tuples -> [0] mask name as string, [1] band for which to get the mask 
    #     """

    #     #assert resolution is within available
    #     assert resolution in [10, 20, 60, "10", "20", "60"]

    #     #define output file
    #     if outdir and os.path.isdir(outdir):
    #         outfile = os.path.join(outdir, os.path.split(safe_root)[-1][:-4]+"tif")
    #     else:
    #         outfile = safe_root[:-4] + "tif"

    #     granule_dir = os.path.join(safe_root, "GRANULE")
    #     masks_dir = os.path.join(
    #         granule_dir, "{}/QI_DATA/".format(os.listdir(granule_dir)[0]))

    #     if resolution in [10, "10"]:
    #         jp2_masks_paths = [masks_dir+f for f in os.listdir(masks_dir) if f.split("_")[-2] in jp2_masks_order
    #                         and f.split("_")[-1][:2] == "20"]
    #     else:
    #         jp2_masks_paths = [masks_dir+f for f in os.listdir(masks_dir) if f.split("_")[-2] in jp2_masks_order
    #                         and f.split("_")[-1][:2] == str(resolution)]

    #     jp2_path = os.path.join(
    #         granule_dir, "{}/IMG_DATA/R{}m/".format(os.listdir(granule_dir)[0], resolution))
    #     img_data_bands = [
    #         jp2_path + f for f in os.listdir(jp2_path) if not f.split("_")[-2] in ["TCI", "SCL"]]

    #     tci_path = [
    #         jp2_path + f for f in os.listdir(jp2_path) if f.split("_")[-2] == "TCI"][0]

    #     #set up rasterio loaders
    #     bands = {}
    #     max_width = 0
    #     max_height = 0
    #     for file in img_data_bands+jp2_masks_paths:

    #         band = rio.open(file, driver="JP2OpenJPEG")
    #         max_width = max(max_width, band.width)
    #         max_height = max(max_height, band.height)
    #         bands[file.split("_")[-2]] = band

    #     gml_mask_paths = {(f.split("_")[-2], f.split(
    #         "_")[-1].split(".")[-2]): masks_dir+f for f in os.listdir(masks_dir) if (f.split("_")[-2], f.split(
    #             "_")[-1].split(".")[-2]) in gml_mask_order and f.split(".")[-1] == "gml"}

    #     #reader for tci
    #     tci_band = rio.open(tci_path, driver="JP2OpenJPEG")

    #     #number of bands in final geotif
    #     count = len([b for b in bands if b in bands_order or b in jp2_masks_order] +
    #                 [g for g in gml_mask_paths if g in gml_mask_order])+3*TCI

    #     #write geotif
    #     with rio.open(outfile,
    #                 "w",
    #                 driver="GTiff",
    #                 width=max_width,
    #                 height=max_width,
    #                 count=count,
    #                 crs=bands["B02"].crs,
    #                 transform=bands["B02"].transform,
    #                 dtype=bands["B02"].dtypes[0]) as dst:

    #         dst.nodata = NO_DATA_VAL

    #         #write gml masks
    #         for idx, (_, gml_path) in enumerate(gml_mask_paths.items()):
    #             try:
    #                 shapes = gpd.read_file(gml_path)["geometry"].values
    #                 mask, _, _ = rio.mask.raster_geometry_mask(
    #                     bands["B02"], shapes, crop=False, invert=True)
    #             except ValueError:
    #                 mask = np.full(shape=bands["B02"].read(1).shape,fill_value=0.0,dtype=np.uint16)

    #             dst.write(mask.astype(np.uint16), len(bands)+3*TCI+idx+1)

    #         #write jp2 bands
    #         for idx, dst_reader in enumerate([bands[b] for b in bands_order+jp2_masks_order if b in bands]):

    #             img = dst_reader.read(1)
    #             if not dst_reader.dtypes[0] == bands["B02"].dtypes[0]:
    #                 img = (img*(65535.0/255.0)).astype(np.uint16)
    #             dst.write(img, 3*TCI+idx+1)

    #             dst_reader.close()

    #         #write tci
    #         if TCI:
    #             for i in range(3):
    #                 img = (tci_band.read(i+1)*(65535.0/255.0)).astype(np.uint16)
    #                 dst.write(img, i+1)

    #         crs_epsg_code = dst.crs.to_epsg()
    #         img_bounding_rectangle = box(*dst.bounds)

    #         return {'crs_epsg_code': crs_epsg_code, 'img_bounding_rectangle': img_bounding_rectangle}
