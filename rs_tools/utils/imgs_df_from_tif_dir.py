import os
from pathlib import Path
from shapely.geometry import box
import geopandas as gpd
import rasterio as rio 

# rs_tools.img_polygon_associator 
from rs_tools.utils.utils import transform_shapely_geometry



def imgs_df_from_tif_dir(imgs_dir_path, imgs_df_crs_epsg_code=None, imgs_df_index_name=None):
    """
    Builds and returns an associator imgs_df from a directory of GeoTiff images (or from a data directory). Only the index (imgs_df_index_name, defaults to img_name), geometry column (coordinates of the img_bounding_rectangle, and orig_crs_epsg_code (epsg code of crs the GeoTiff image is in) columns will be populated, custom columns will have to be populated by a custom written function.

    Args:
        - imgs_path (pathlib.Path or str): path of the directory that the images are in (assumes the dir has no images subdir), or path to a data_dir with an images subdir.
        - imgs_df_crs_epsg_code (int): epsg code of imgs_df crs to be returned. 
        - imgs_df_index_name (str): index name of imgs_df GeoDataFrame.
    
    Returns:
        - imgs_df with index imgs_df_index_name and columns geometry and orig_crs_epsg_code 
    """

    # stupid hack to avoid (not really) circular importing python can't deal with.
    from rs_tools.img_polygon_associator import STANDARD_CRS_EPSG_CODE, IMGS_DF_INDEX_NAME

    if imgs_df_crs_epsg_code == None:
        imgs_df_crs_epsg_code = STANDARD_CRS_EPSG_CODE

    if imgs_df_index_name == None:
        imgs_df_index_name = IMGS_DF_INDEX_NAME

    # ensure path is pathlib.Path
    imgs_dir_path = Path(imgs_dir_path)

    # if path was data_dir, go to images subdir
    if (imgs_dir_path / Path("images")).is_dir():
        imgs_dir_path = imgs_dir_path / Path("images")

    # dict to keep track of information about the imgs that we will make the imgs_df from.
    new_imgs_dict = {index_or_col_name: [] for index_or_col_name in {imgs_df_index_name, 'geometry', 'orig_crs_epsg_code'}}

    # for all images in dir ...
    for img_filename in os.listdir(imgs_dir_path):

        # ... open them in rasterio ...
        with rio.open(imgs_dir_path / img_filename) as src:

            # ... extract information ...

            orig_crs_epsg_code = src.crs.to_epsg()

            img_bounding_rectangle_orig_crs = box(*src.bounds)

            img_bounding_rectangle_imgs_df_crs = transform_shapely_geometry(img_bounding_rectangle_orig_crs, 
                                                                        orig_crs_epsg_code, 
                                                                        imgs_df_crs_epsg_code)


            # and put the information into a dict.
            img_info_dict = {imgs_df_index_name: img_filename, 
                                'geometry': img_bounding_rectangle_imgs_df_crs, 
                                'orig_crs_epsg_code': orig_crs_epsg_code}

        
        #  Add information about the image to new_imgs_dict ...
        for key in new_imgs_dict.keys(): 
            new_imgs_dict[key].append(img_info_dict[key])

    # ... and create a imgs_df GeoDatFrame from new_imgs_dict:
    new_imgs_df = gpd.GeoDataFrame(new_imgs_dict)
    new_imgs_df.set_crs(epsg=imgs_df_crs_epsg_code, inplace=True)
    new_imgs_df.set_index(imgs_df_index_name, inplace=True)

    return new_imgs_df
            













