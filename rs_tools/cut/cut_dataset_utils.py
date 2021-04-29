"""
Utils for/code shared by cut_imgs_iter_over_polygons and cut_imgs_iter_over_imgs.
"""
from typing import Callable
import logging
import random
import rasterio as rio
from pathlib import Path
from shapely.geometry import box
import geopandas as gpd

from rs_tools.utils.utils import transform_shapely_geometry
from rs_tools.graph import BipartiteGraph

logger = logging.getLogger(__name__)

# Type alias for the filter predicate functions
Filter = Callable[[str, gpd.GeoDataFrame, ipa.ImgPolygonAssociator], bool]

def have_no_img_for_polygon(polygon_name: str, new_polygons_df: gpd.GeoDataFrame, source_assoc: ipa.ImgPolygonAssociator) -> bool:
    """
    Polygon filter predicate that tests whether an image has already been created for the polygon. Returns True if the polygon's 'have_img? ' value is False, returns False otherwise. 

    Using this polygon filter predicate in new_assoc_from_iter_over_imgs together with e.g. small_imgs_centered_around_polygons_cutter assures that the new associator contains exactly one new small image per polygon (and not more, as might happen if some polygons in the the source associator/data set are contained in multiple images.)
    """

    return new_polygons_df.loc[polygon_name, 'have_img?'] == False

def small_imgs_centered_around_polygons_cutter(img_name:str, 
                                        source_assoc:ipa.ImgPolygonAssociator,
                                        target_data_dir:Union[str, Path],
                                        polygon_filter_predicate:Filter,
                                        new_polygons_df:gpd.GeoDataFrame, 
                                        new_graph:BipartiteGraph,  
                                        img_size: int = 256, 
                                        img_bands: list = None, 
                                        label_bands: list = None, 
                                        centered: bool = False,
                                        random_seed: int = 42) -> dict:
    """
    img_cutter that cuts an img (and its label) into small imgs surrounding (a subset of the) polygons fully contained in the img. 
    
    Args: 
        - img_name: name of the img to be cut
        - source_assoc: source associator
        - target_data_dir: target data directory
        - polygon_filter_predicate: predicate to filter polygons. 

            Args:
                - polygon_name
                - new_polygons_df
                - source_assoc
                
            Returns:
                - True or False, whether to filter polygon or not

        - new_polygons_df: will be polygons_df of new associator
        - new_graph: will be graph of new associator
        - img_size: the img size (just one number)
        - bands: list of bands to extract
        - centered: bool. If True, will choose the image bounds of the new small cut image so that the polygon under is centered in it. If False, will choose the img bounds randomly subject to the constraint that the polygon under consideration is fully contained in it.

    Returns:
        imgs_from_single_cut_dict: dict with keys the names of the index and columns of the new_imgs_df 
        for the new associator being created. The values are lists of entries corresponding to 
        the rows for the newly created imgs.

    Given an img in the source associator with name img_name, small_imgs_centered_around_polygons_cutter creates an image
    and corresponding label in the target_data_dir for each polygon satisfying the polygon_filter_predicate condition, updates the new_graph and new_polygons_df as necessary, and returns a dict containing information about the created imgs that can be put into an associator's imgs_df (see explanation of return value above, and docstring for new_assoc_from_iter_over_imgs).

    TODO: numpy vs GeoTiff? which bands?
    """

    random.seed(random_seed)

    imgs_from_cut_dict = {index_or_col_name: [] for index_or_col_name in [source_assoc.imgs_df.index.name] + list(source_assoc.imgs_df.columns)}

    polygons_contained_in_img = source_assoc.polygons_contained_in_img(img_name)

    # for all polygons in the source_assoc contained in the img
    for polygon_name, polygon_geometry in (source_assoc.polygons_df.loc[polygons_contained_in_img, ['geometry']]).itertuples():

        if not polygon_filter_predicate(polygon_name, new_polygons_df, source_assoc):

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # for cut_imgs_iter_over_imgs should drop polygon form new_polygons_df?
            pass

        else:

            # cut both image and raster label
            for subdir in ["images", "labels"]:

                # open the img (or label)
                with rio.open(source_assoc.data_dir / Path(subdir) / img_name) as src:

                    # transform polygons from assoc's crs to image source crs
                    transformed_polygon_geometry = transform_shapely_geometry(polygon_geometry, 
                                                                        from_epsg=source_assoc.polygons_df.crs.to_epsg(), 
                                                                        to_epsg=src.crs.to_epsg())

                    # find row, col offset of random window containing polygon:

                    list_of_rectangle_corner_coords = list(transformed_polygon_geometry.envelope.exterior.coords)[:5]
                    list_of_enveloping_rectangle_row_col_pairs = list(map(lambda pair: src.index(*pair), list_of_rectangle_corner_coords))

                    tuple_of_rows_of_rectangle_corners = tuple(zip(*list_of_enveloping_rectangle_row_col_pairs))[0]
                    tuple_of_cols_of_rectangle_corners = tuple(zip(*list_of_enveloping_rectangle_row_col_pairs))[1]

                    min_row = min(*tuple_of_rows_of_rectangle_corners)
                    max_row = max(*tuple_of_rows_of_rectangle_corners)
                    min_col = min(*tuple_of_cols_of_rectangle_corners)
                    max_col = max(*tuple_of_cols_of_rectangle_corners)

                    if not centered:

                        # If the height of the polygon is less than img_size ...
                        if max_row - min_row <= img_size:
                            
                            # ... choose row offset randomly subject to constraint that cut image contains polygon. 
                            row_off = random.randint(max(0, max_row - img_size), min(min_row, src.height - img_size)) 
                        
                        # Else, ...
                        else:

                            # ... choose row offset randomly so that at most 1/3 of the rows do not contain any points of the polygon...
                            row_off = random.randint(max(0, min_row - (img_size // 3)), min(src.height, max_row + (img_size // 3)))

                            # .... and log a warning. 
                            logger.warning(f"small_imgs_centered_around_polygons_cutter: {polygon_name} has width {max_col - min_col} which is greater than img_size {img_size}.")

                        # Do the same for the columns:
                        # If the width of the polygon is less than img_size ...
                        if max_col - min_col <= img_size:
                            
                            # ... choose col offset randomly subject to constraint that cut image contains polygon. 
                            col_off = random.randint(max(0, max_col - img_size), min(min_col, src.width - img_size))
                            
                        # Else, ...
                        else:

                            # ... choose col offset randomly so that at most 1/3 of the cols do not contain any points of the polygon...
                            col_off = random.randint(max(0, min_col - (img_size // 3)), min(src.height, max_col + (img_size // 3)))

                            # .... and log a warning. 
                            logger.warning(f"small_imgs_centered_around_polygons_cutter: {polygon_name} has height {max_row - min_row} which is greater than img_size {img_size}.")

                    # In the centered case ...
                    else: 

                        # ... to find the row, col offsets to center the polygon ...

                        # ...we first find the centroid of the polygon in the img crs ...
                        polygon_centroid_coords = transformed_polygon_geometry.envelope.centroid.coords[0]

                        # ... extract the row, col of the centroid ...
                        centroid_row, centroid_col = src.index(*polygon_centroid_coords)

                        # and then choose offsets to center the polygon.
                        row_off = centroid_row - (img_size // 2)
                        col_off = centroid_col - (img_size // 2)

                        # If the polygon is too high ...
                        if max_row - min_row > img_size:

                            # ... log a warning. 
                            logger.warning(f"small_imgs_centered_around_polygons_cutter: {polygon_name} has height {max_row - min_row} which is greater than img_size {img_size}.")
                            
                        # If the polygon is too wide ...
                        if max_col - min_col > img_size:

                            # ... log a warning.
                            logger.warning(f"small_imgs_centered_around_polygons_cutter: {polygon_name} has width {max_col - min_col} which is greater than img_size {img_size}.")
                                            
                    # Define the square window with the calculated offsets.
                    window = rio.windows.Window(col_off=col_off, 
                                        row_off=row_off, 
                                        width=img_size, 
                                        height=img_size)
                    
                    # Remember the transform for the new geotiff.
                    window_transform = src.window_transform(window)

                    # Generate new img (or label) name.
                    img_name_no_extension = Path(img_name).stem
                    new_img_name = img_name_no_extension + "_" + polygon_name + ".tif"

                    # Set the bands.
                    # If we are considering the images ...
                    if subdir == "images":
                        
                    # ... then if the img_bands arg was given ...
                    if img_bands is not None:

                        # ...set band variable to that. 
                        bands = img_bands 
                    
                    # If img_bands was not given (i.e. img_bands is None) ...
                    else:

                        # ... default to all img bands
                        bands = list(range(1, src.count + 1)) # or src.count???
                    
                    # Similarly, when considering the labels ...
                    else: 

                        # ... then if the label_bands arg was given ...
                        if label_bands is not None:
                            
                            # ...set band variable to that. 
                            bands = label_bands 
                            
                        # If label_bands was not given (i.e. label_bands is None) ...
                        else:
                            
                            # ... default to all label bands. 
                            bands = list(range(1, src.count + 1)) # or src.count???

                    # To create new img or label with filename new_img_name and transform window_transform, open the target file with rasterio ...
                    with rio.open(target_data_dir / subdir / Path(new_img_name),
                                        'w',
                                        driver='GTiff',
                                        height=img_size,
                                        width=img_size,
                                        count=len(bands),
                                        dtype=src.profile["dtype"],
                                        crs=src.crs,
                                        transform=window_transform) as dst:
                        
                        # DEBUG INFO:
                        # print(f"creating new img: {target_data_dir / subdir / Path(new_img_name)}")

                        # DEBUG INFO
                        # print(f"subdir: {subdir}, bands: {bands}")  

                        # ... and go through the bands.
                        for band in bands:
                            
                            # Read window for that band from source ...
                            new_img_band_raster = src.read(band, window=window)

                            # ... write to new geotiff.
                            dst.write(new_img_band_raster, band)

                        # DEBUG INFO
                        # new_img_path = target_data_dir / subdir / Path(new_img_name)
                        # print(f"Does new img exist?: {new_img_path.is_file()} \n")

                        # Put the information about the image ...
                        img_bounds_in_img_crs = dst.bounds
                        img_bounding_rectangle_in_std_crs = box(*rio.warp.transform_bounds(src.crs, 
                                                                                        source_assoc.imgs_df.crs,
                                                                                        *img_bounds_in_img_crs))

                        orig_crs_epsg_code = src.crs.to_epsg()

            # ... in a dict ...
            single_new_img_info_dict = {'img_name': new_img_name, 
                                        'geometry': img_bounding_rectangle_in_std_crs, 
                                        'orig_crs_epsg_code': orig_crs_epsg_code, 
                                        'img_processed?': True, 
                                        'timestamp': source_assoc.imgs_df.loc[img_name, 'timestamp']}

            # ... and accumulate into the dict containing the information for all new images.
            for key in imgs_from_cut_dict.keys():
                imgs_from_cut_dict[key].append(single_new_img_info_dict[key])

            # On  the first pass of for subdir in ["images", "labels"] (i.e. when subdir = "images") add connections to new_graph for the new image and modify new_polygons_df.
            source_assoc._add_img_to_graph_modify_polygons_df(new_img_name, 
                                                                img_bounding_rectangle=img_bounding_rectangle_in_std_crs, 
                                                                polygons_df=new_polygons_df, 
                                                                graph=new_graph)

    # Return that dict, so we can build a geodataframe from the information.
    return imgs_from_cut_dict


# TODO: should inherit from callable ABC
def alwaystrue(polygon_or_img_name: str, new_polygons_df: gpd.GeoDataFrame, source_assoc: ipa.ImgPolygonAssociator) -> bool:
    """polygon_filter_predicate or img_filter predicate that always returns True."""
    return True  

