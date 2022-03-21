"""Create associator imgs_df from a directory containing GeoTiff images."""

import pathlib
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import rasterio as rio
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, box
from tqdm import tqdm

from rs_tools.utils.utils import transform_shapely_geometry


def default_read_in_img_for_img_df_function(
        img_path: Path) -> Tuple[int, Polygon]:
    """Read in the crs code and the bounding rectangle that defines a GeoTIFF
    image.

    ..note::

    Args:
        img_path (Path): location of the image

    Returns:
        int: crs code of the image
        Polygon: bounding rectangle of the image
    """

    if img_path.suffix in [".tif", ".tiff"]:

        # ... open them in rasterio ...
        with rio.open(img_path, "r") as src:

            # ... extract information ...

            orig_crs_epsg_code = src.crs.to_epsg()

            img_bounding_rectangle_orig_crs = box(*src.bounds)

    else:
        orig_crs_epsg_code, img_bounding_rectangle_orig_crs = None, None

    return orig_crs_epsg_code, img_bounding_rectangle_orig_crs


def imgs_df_from_imgs_dir(
    images_dir: Union[pathlib.Path, str],
    imgs_df_crs_epsg_code: Optional[int] = None,
    image_names: Optional[List[str]] = None,
    imgs_datatype: str = "tif",
    read_in_img_for_img_df_function: Callable[[Path], Tuple[
        int, Polygon]] = default_read_in_img_for_img_df_function
) -> GeoDataFrame:
    """Builds and returns an associator imgs_df from a directory of images (or
    from a data directory). Only the index (imgs_df_index_name, defaults to
    img_name), geometry column (coordinates of the img_bounding_rectangle, and
    orig_crs_epsg_code (epsg code of crs the image is in) columns will be
    populated, custom columns will have to be populated by a custom written
    function.

    Args:
        - images_dir (Union[pathlib.Path, str]): path of the directory that the images are in (assumes the dir has no images subdir), or path to a data_dir with an images subdir.
        - imgs_df_crs_epsg_code (int): epsg code of imgs_df crs to be returned.
        - image_names (List[str], optional): optional list of image names. Defaults to None, i.e. all images in images_dir.
        - imgs_datatype (str): datatype suffix of the images
        - read_in_img_for_img_df_function (Callable[[Path], Tuple[
        int, Polygon]]): function that reads in the crs code and the bounding rectangle for the images

    Returns:
        GeoDataFrame: imgs_df conforming to the associator imgs_df format with index imgs_df_index_name and columns geometry and orig_crs_epsg_code
    """

    # stupid hack to avoid (not really) circular importing python can't deal with.
    from rs_tools.global_constants import STANDARD_CRS_EPSG_CODE

    images_dir = Path(images_dir)

    if imgs_df_crs_epsg_code == None:
        imgs_df_crs_epsg_code = STANDARD_CRS_EPSG_CODE

    if image_names is None:
        image_paths = images_dir.glob(f"*.{imgs_datatype}")
    else:
        image_paths = [images_dir / img_name for img_name in image_names]

    # dict to keep track of information about the imgs that we will make the imgs_df from.
    new_imgs_dict = {
        index_or_col_name: []
        for index_or_col_name in
        {'img_name', 'geometry', 'orig_crs_epsg_code'}
    }

    # for all images in dir ...
    for img_path in tqdm(image_paths, desc='building imgs_df'):

        orig_crs_epsg_code, img_bounding_rectangle_orig_crs = read_in_img_for_img_df_function(
            img_path=img_path)

        if orig_crs_epsg_code is None or img_bounding_rectangle_orig_crs is None:
            continue

        img_bounding_rectangle_imgs_df_crs = transform_shapely_geometry(
            img_bounding_rectangle_orig_crs, orig_crs_epsg_code,
            imgs_df_crs_epsg_code)

        # and put the information into a dict.
        img_info_dict = {
            'img_name': img_path.name,
            'geometry': img_bounding_rectangle_imgs_df_crs,
            'orig_crs_epsg_code': int(orig_crs_epsg_code)
        }

        #  Add information about the image to new_imgs_dict ...
        for key in new_imgs_dict.keys():
            new_imgs_dict[key].append(img_info_dict[key])

    # ... and create a imgs_df GeoDatFrame from new_imgs_dict:
    new_imgs_df = GeoDataFrame(new_imgs_dict)
    new_imgs_df.set_crs(epsg=imgs_df_crs_epsg_code, inplace=True)
    new_imgs_df.set_index('img_name', inplace=True)

    return new_imgs_df
