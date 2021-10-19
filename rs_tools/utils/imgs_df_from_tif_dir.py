""" 
Create associator imgs_df from a directory containing GeoTiff images.
"""

from typing import Optional, Union, List
import pathlib
from pathlib import Path
from geopandas import GeoDataFrame
from shapely.geometry import box
import rasterio as rio 

from rs_tools.utils.utils import transform_shapely_geometry


def imgs_df_from_tif_dir(
        images_dir : Union[pathlib.Path, str], 
        image_names : Optional[List[str]]=None,
        imgs_df_crs_epsg_code : Optional[int] = None
        ) -> GeoDataFrame:
    """
    Build and return an associator imgs_df from a directory of GeoTiff images (or from a data directory). Only the index (imgs_df_index_name, defaults to img_name), geometry column (coordinates of the img_bounding_rectangle, and orig_crs_epsg_code (epsg code of crs the GeoTiff image is in) columns will be populated, custom columns will have to be populated by a custom written function.

    Args:
        images_dir (Union[pathlib.Path, str]): path to directory containing GeoTiff images
        image_names (List[str], optional): optional list of image names. Defaults to None, i.e. all images in images_dir.
        imgs_df_crs_epsg_code (int, optional): EPSG code of imgs_df crs to be returned. If None, will use standard crs. Defaults to None.

    Returns:
        GeoDataFrame: imgs_df conforming to the associator imgs_df format with index imgs_df_index_name and columns geometry and orig_crs_epsg_code .
    """

    # stupid hack to avoid (not really) circular importing python can't deal with.
    from rs_tools.global_constants import STANDARD_CRS_EPSG_CODE

    images_dir = Path(images_dir)

    if imgs_df_crs_epsg_code == None:
        imgs_df_crs_epsg_code = STANDARD_CRS_EPSG_CODE
        
    if image_names is None:
        image_paths = list(images_dir.iterdir())
    else:
        image_paths = [images_dir / img_name for img_name in image_names]

    # dict to keep track of information about the imgs that we will make the imgs_df from.
    new_imgs_dict = {index_or_col_name: [] for index_or_col_name in {'img_name', 'geometry', 'orig_crs_epsg_code'}}

    # for all images in dir ...
    for img_path in image_paths:

        # ... open them in rasterio ...
        with rio.open(img_path) as src:

            # ... extract information ...
            orig_crs_epsg_code = src.crs.to_epsg()
            img_bounding_rectangle_orig_crs = box(*src.bounds)
            img_bounding_rectangle_imgs_df_crs = transform_shapely_geometry(
                                                    img_bounding_rectangle_orig_crs, 
                                                    orig_crs_epsg_code, 
                                                    imgs_df_crs_epsg_code)

            # and put the information into a dict.
            img_info_dict = {
                                'img_name': img_path.name, 
                                'geometry': img_bounding_rectangle_imgs_df_crs, 
                                'orig_crs_epsg_code': orig_crs_epsg_code
                            }

        #  Add information about the image to new_imgs_dict ...
        for key in new_imgs_dict.keys(): 
            new_imgs_dict[key].append(img_info_dict[key])

    # ... and create a imgs_df GeoDatFrame from new_imgs_dict:
    new_imgs_df = GeoDataFrame(new_imgs_dict)
    new_imgs_df.set_crs(epsg=imgs_df_crs_epsg_code, inplace=True)
    new_imgs_df.set_index('img_name', inplace=True)

    return new_imgs_df
            













