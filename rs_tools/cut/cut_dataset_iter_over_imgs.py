"""
Customizable very general function to create a new remote sensing imagery dataset (images, labels, and associator) from an existing one by iterating over (a subset of) the images and cutting/splitting them (in a way possibly depending on the associator, the polygons contained in the image ...).

TODO: implement img_to_grid_imgs_splitter so we can test
TODO: Clean up: Go through each line, check logic, separate two modes of cutting/splitting img (iter over imgs and over polygons), add documentation, comments!
TODO: Code reuse in new_assoc_from_iter_over_imgs and new_dataset_from_iter_over_polygons. Could remove duplicate code but would it make things less clear?
TODO: Update split associator from updated source associator/data set.
"""

import copy
from pathlib import Path
import random
import rasterio as rio
from shapely.geometry import box
import geopandas as gpd

import assoc.img_polygon_associator as ipa
from assoc.cut.cut_dataset_utils import small_imgs_centered_around_polygons_splitter, alwaystrue


def new_dataset_from_iter_over_imgs(source_data_dir, target_data_dir, img_splitter, img_filter_predicate=alwaystrue, polygon_filter_predicate=alwaystrue, img_bands=None, label_bands=None, **kwargs):
    """
    Create a new data set (images, labels and associator) in target_data_dir from the data set in source_data_dir by iterating over the images and labels in the source_data_dir and splitting them up according to the img_splitter, which can split a single image in a very general way depending on e.g. the polygons contained in the image and more generally the information in source associator. We can restrict to a subset of the polygons or images and labels in the source data_dir by filtering using an img_filter_predicate and/or a polygon_filter_predicate, which can also depend on the information in the source associator.

    Args:
        - source_data_dir: a data directory, i.e. a directory conforming to the conventions of an ImgPolygonAssociator. 
        - target_data_dir: directory where the new dataset and associator will be created. Is created if t does not exist yet. 
        - img_splitter: Function that splits an image
        
            Args: 
                - img_name: name of the img to be split
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
                - img_bands: list of bands to extract from the images, the img_splitter should default this to all possible bands
                - label_bands: list of bands to extract from the labels, the img_splitter should default this to all possible bands                - **kwargs: additional keyword arguments that might be needed by an img_splitter.

            Returns:
                imgs_from_single_split_dict: dict with keys the names of the index and columns of the new_imgs_df 
                for the new associator being created. The values are lists of entries corresponding to 
                the rows for the newly created imgs.

            Given an img in the associator with name img_name, the img_splitter splits the img 
            and the corresponding label in the images and labels subdirs of source_data_dir
            into subimages in some meaningful way, given the information in the components of the current associator 
            old_polygons_df and old_graph, saves them to the images and labels subdirs of the target_data_dir, 
            and modifies new_polygons_df and new_graph appropriately so that they can form the 
            components of a new associator in target_data_dir together with the new_imgs_df accumulating 
            the information in the imgs_from_single_split_dicts from the various calls
            on all the old imgs to be split.

        - polygon_filter_predicate: predicate to filter polygons. See above in args for img_splitter.
        - img_filter_predicate: predicate to filter images, similar to polygon_filter_predicate, except that instead of polygon_name, it accepts an img_name as argument
            but takes as argument a named tuple with names the columns and index of an imgs_df of an associator
        - kwargs: additional keyword arguments that might be needed by an img_splitter.

    Returns:
        - None

    """
    # load source associator
    source_assoc = ipa.ImgPolygonAssociator(data_dir = source_data_dir)

    # the dataframes and graph of the new associator:
    new_polygons_df = gpd.GeoDataFrame(columns=source_assoc.polygons_df.columns, 
                                        index=source_assoc.polygons_df.index, 
                                        data=copy.deepcopy(source_assoc.polygons_df.values), 
                                        crs=source_assoc.polygons_df.crs) # deepcopy, just to be safe
    new_polygons_df['have_img?'] = False
    new_polygons_df['have_img_downloaded?'] = False

    new_graph = ipa.empty_graph()
    for polygon_name in new_polygons_df.index:
        new_graph.add_vertex(polygon_name, 'polygons')

    # dict to keep track of information which will be in the new imgs_df (geo)dataframe we'll create after splitting
    new_imgs_dict = {index_or_col_name: [] for index_or_col_name in [source_assoc.imgs_df.index.name] + list(source_assoc.imgs_df.columns)}

    # make sure target_data_dir exists and has right subdirectories, create if necessary
    for subdir in ipa.DATA_DIR_SUBDIRS:
        Path(target_data_dir / subdir).mkdir(parents=True, exist_ok=True)

    # for each img in source associator
    for img_name in source_assoc.imgs_df.index:
        
        # split the img if the filter condition is met:
        if img_filter_predicate(img_name, new_polygons_df, source_assoc) == True:

            # split img and remember information to be put in rows of new imgs_df in return dict
            imgs_from_single_split_dict = img_splitter(img_name=img_name, 
                                                        source_assoc=source_assoc, 
                                                        polygon_filter_predicate=polygon_filter_predicate, 
                                                        new_polygons_df=new_polygons_df, 
                                                        new_graph=new_graph, 
                                                        target_data_dir=target_data_dir, 
                                                        img_bands=None, 
                                                        label_bands=None,
                                                        **kwargs)
                                                        
            # make sure img_splitter returned dict with same keys as needed by new_imgs_dict
            assert set(imgs_from_single_split_dict.keys()) == set(source_assoc.imgs_df.columns) | {source_assoc.imgs_df.index.name}

            # accumulate information for the new imgs in new_imgs_dict
            for key in new_imgs_dict.keys(): 
                        new_imgs_dict[key] += (imgs_from_single_split_dict[key])
        
    # extract accumulated information about the new imgs into a new_imgs_df dataframe...
    new_imgs_df = gpd.GeoDataFrame(new_imgs_dict)
    new_imgs_df.set_crs(epsg=source_assoc.__params_dict__['standard_crs_epsg_code']) # standard crs
    new_imgs_df.set_index("img_name", inplace=True)

    # save all the components of the new associator
    new_imgs_df.to_file(target_data_dir / "imgs_df.geojson", driver="GeoJSON")
    new_polygons_df.to_file(target_data_dir / "polygons_df.geojson", driver="GeoJSON")
    new_graph.save_to_file(target_data_dir / "graph.json")


def update_split_assoc(source_data_dir, target_data_dir, img_splitter, img_filter_predicate=alwaystrue, polygon_filter_predicate=alwaystrue, img_bands=None, label_bands=None):
    """

    TODO: Adapt code from cut_dataset_iter_over_polygons.
    """

    raise NotImplementedError


def new_dataset_cut_imgs_into_grids(source_data_dir, target_data_dir, size=256):
    """
    Create a new dataset (images, labels, and associator) in target_data_dir from dataset in source_data_dir by cutting each image (and label) in the source dataset into a grid of small images.

    Args:
        - source_data_dir: a data directory, i.e. a directory conforming to the conventions of an ImgPolygonAssociator. 
        - target_data_dir: directory where the new dataset will be created. Will be created if it doesn't exist already.
        - size: size in pixels of the new square images to be created (length in pixels)
    Returns:
        - None
    """

    new_dataset_from_iter_over_imgs(source_data_dir, target_data_dir, img_splitter=img_to_grid_imgs_splitter)


def img_to_grid_imgs_splitter(img_name, 
                                        source_assoc,
                                        target_data_dir,
                                        polygon_filter_predicate,
                                        new_polygons_df, 
                                        new_graph,  
                                        img_size=256, 
                                        img_bands=None, 
                                        label_bands=None):
    """
    img_splitter that splits an img into a grid of small imgs.
    
    Args: 
        - img_name: name of the img to be split
        - source_assoc: source associator
        - target_data_dir: target data directory
        - polygon_filter_predicate: predicate to filter polygons. Irrelevant for this splitter.
        - new_polygons_df: will be polygons_df of new associator
        - new_graph: will be graph of new associator
        - img_size: the img size (just one number, number of pixels in side length of square)
        - img_bands: list of bands to extract from the images, the img_splitter should default this to all possible bands
        - label_bands: list of bands to extract from the labels, the img_splitter should default this to all possible bands
    Returns:
        imgs_from_single_split_dict: dict with keys the names of the index and columns of the new_imgs_df 
        for the new associator being created. The values are lists of entries corresponding to 
        the rows for the newly created imgs.

    Given an img in the source associator with name img_name small_imgs_centered_around_polygons_splitter creates an image (and corresponding label) in the target_data_dir by splitting the image (or label) into a grid of small sqaure images and returns a dict containing information about the created imgs that can be put into an associator's imgs_df (see explanation of return value above, and docstring for new_assoc_from_iter_over_imgs).

    TODO: numpy vs GeoTiff? which bands?
    """

    raise NotImplementedError







