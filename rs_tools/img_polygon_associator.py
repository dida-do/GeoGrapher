"""
dunder to  
label_makers
TODO: Do we still need polygon_info_dict?

The ImgPolygonAssociator class organizes and handles remote sensing datasets.
"""


from rs_tools.label_makers import _make_geotif_label_categorical, _make_geotif_label_onehot, _make_geotif_label_soft_categorical
from json.decoder import JSONDecodeError
import os
import copy
import pathlib
import logging
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import transform, unary_union
import rasterio as rio
from collections import Counter

from rs_tools.img_polygon_associator_class import ImgPolygonAssociatorClass
from rs_tools.graph import BipartiteGraph, empty_bipartite_graph
from rs_tools.utils.utils import transform_shapely_geometry
from rs_tools.errors import ImgAlreadyExistsError, NoImgsForPolygonFoundError, ImgDownloadError




STANDARD_CRS_EPSG_CODE = 4326 # WGS84 
DATA_DIR_SUBDIRS = [Path("images"), Path("labels")] # for sentinel-2, also Path("safe_files")
IMGS_DF_INDEX_NAME="img_name"
POLYGONS_DF_INDEX_NAME="polygon_name"


# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)


class ImgPolygonAssociator(ImgPolygonAssociatorClass):
    """
    Organize and handle remote sensing datasets consisting of shapely polygons and images/labels.

    The ImgPolygonAssociator class can build up, handle, and organize datasets consisting of shapely vector polygon labels (as well as tabular information about them in the form of a GeoDataFrame) and remote sensing raster images and potentially (semantic) segmentation pixel labels (e.g. GeoTiffs or .npy files) (as well as tabular information about the images and pixel labels in the form of a GeoDataFrame) by providing a two-way linkage between the polygons and the images/pixel labels automatically keeping track of which polygons are contained in which images/pixel labels.
    
    Attributes:

    - polygons_df: A GeoDataFrame containing the vector polygon labels. Should have index name 'polygon_name' giving the unique identifier for the vector polygon (which usually should be a string or an int) and columns
        - 'geometry': shapely.geometry.Polygon. The vector polygon label (in a standard crs)
        - 'have_img?': bool. True if there is a processed image containing the polygon in the dataset.
        - 'have_img_downloaded?': bool. True if an image containing the polygon has been downloaded, but not necessarily processed.
        - If the label_type argument is 'categorical' or 'onehot' there should be a 'type' column whose entries are the segmenation class (one of the entries in the segmentation_classes list) the polygon belongs to. If the label_type argument is "soft-categorical" there should be columns prob_seg_class_[class] for each segmentation class.
        - other columns as needed for one's application.

    - imgs_df: A GeoDatFrame containing tabular information about the images. Should have index name 'img_name' and indices of type str giving the unique identifier of an image and columns
        - 'geometry': shapely.geometry.Polygon. Polygon defining the image bounds (in the associator's standardized crs)
        - 'orig_crs_epsg_code': int. The EPSG code of the crs the georeferenced image is in. 
        - other columns as needed for one's application.
            
    - data_dir: A data directory.

    - crs_epsg_code: EPSG code of the coordinate reference system (crs) the associator (i.e. the associator's imgs_df and polygons_df) is in. Defaults to 4326 (WGS84). Setting this attribute will automatically set the associator's imgs_df and polygons_df crs's. 
    """

    def __init__(self,
                    data_dir, 
                    imgs_df=None, # should be either given or will be loaded from file
                    polygons_df=None, # should be either given or will be loaded from file
                    segmentation_classes=None,# should be either given or will be inferred for a saved associator from param_dict.json file
                    label_type=None, # should be either given or will be inferred for a saved associator from param_dict.json file
                    add_background_band_in_labels=None, # should be either given or will be inferred for a saved associator from param_dict.json file
                    crs_epsg_code=STANDARD_CRS_EPSG_CODE, 
                    polygons_df_index_name=POLYGONS_DF_INDEX_NAME, 
                    imgs_df_index_name=IMGS_DF_INDEX_NAME,
                    **kwargs):
        """
        :param data_dir: The data directory of the associator. This is the only non-optional argument. 
        :type data_dir: str or pathlib.Path

        :param imgs_df: Imgs_df to initialize associator with. If not given, the associator will assume it can load an imgs_df.geojson file from data_dir. The associator needs either both the imgs_df and polygons_df arguments, or there needs to be an existing associator in the data_dir it can load. 
        :type imgs_df: geopandas.GeoDataFrame, optional 

        :param polygons_df: Polygons_df to initialize associator with. If not given, the associator will assume it can load an imgs_df.geojson file from data_dir. The associator needs either both the imgs_df and polygons_df arguments, or needs there to be an existing associator in the data_dir it can load. 
        :type polygons_df: geopandas.GeoDataFrame, optional

        :param segmentation_classes: List of segmentation classes. If not given, will attempt to load from file (param_dict.json in data_dir).
        :type segmentation_classes: list of str, optional

        :param label_type: The type of label to be created, one of 'categorical', 'onehot', or 'soft-categorical'. 
        :type label_type: str

        :param add_background_band_in_labels: Only relevant if the label_type is 'one-hot' or 'soft-categorical'. If True, will add a background segmentation class band when creating one-hot or soft-categorical labels. If False, will not. 
        :type add_background_band_in_labels: bool

        :param crs_epsg_code: The EPSG code of the coordinate reference system (crs) the associator is in, by which we mean the crs used to store the geometries in the imgs_df and polygons_df GeoDataFrames. Defaults to 4326 (i.e. WGS84). If not given, will attempt to load from file (param_dict.json in data_dir).
        :type crs_epsg_code: int, optional

        :param imgs_df_index_name: Name of index of imgs_df. If not given, will attempt to infer or use default. 
        :type imgs_df_index_name: str, optional
    
        :param polygons_df_index_name: Name of index of polygons_df. If not given, will attempt to infer or use default.
        :type polygons_df_index_name: str, optional

        :param \**kwargs: optional keyword arguments depending on the application, to be passed to e.g. _download_imgs_for_polygon and _process_downloaded_img_file. 
        """

        # Note to reviewer: Sorry, this is a bit messy!

        super().__init__()

        self._data_dir = data_dir
        
        # file paths
        self._imgs_df_path = data_dir / Path("imgs_df.geojson")
        self._polygons_df_path = data_dir / Path("polygons_df.geojson")
        self._graph_path = data_dir / Path("graph.json")
        self._params_dict_path = data_dir / Path("params_dict.json")

        # Try loading params dict from disk ...
        try:
            with open(self._params_dict_path, "r") as read_file:
                self._params_dict = json.load(read_file)
        except FileNotFoundError as e:
            log.info(f"__init__: No params_dict.json found.")
            self._params_dict = {}
        except JSONDecodeError as e:
            log.exception(f"{data_dir}/params_dict.json corrupted.")

        # ... then update the dict with the args/params from initialization, 
        # making sure to replace paths by strings, so we can save the dict as a json.
        if_path_to_str = lambda val: str(val) if isinstance(val, pathlib.PurePath) else val
        kwargs = {key: if_path_to_str(val) for key, val in kwargs.items()}
        self._params_dict.update(kwargs)

        # For the required parameter args whose default value is not None ... 
        for param_name, param_val in zip(
                                        ['polygons_df_index_name', 
                                        'imgs_df_index_name',
                                        'segmentation_classes',
                                        'label_type',
                                        'add_background_band_in_labels',
                                        'crs_epsg_code'],
                                        [polygons_df_index_name, 
                                        imgs_df_index_name,
                                        segmentation_classes,
                                        label_type,
                                        add_background_band_in_labels,
                                        crs_epsg_code]):
            # ... add them to param dict if they don't yet exist in the dict ...
            if param_name not in self._params_dict:

                if param_val is not None:

                    self._params_dict[param_name] = param_val 
            
                else:

                    raise Exception(f"Need value not equal to None for {param_name} argument.")

            # ... else, ...
            else:
                # ... if they conflict with the values in the dict ...
                if param_val is not None and self._params_dict[param_name] != param_val:
                    
                    # ... log a warning ...
                    log.warning(f"param {param_name} value {param_val} differs from value {self._params_dict[param_name]}in associator's params_dict file.")
                    
                    # ... and then update the dict.
                    self._params_dict[param_name] = param_val

        """
        # If segmentation classes arg not given, ... 
        if segmentation_classes is None:
            # ... check it exists in the _params_dict ...
            if 'segmentation_classes' not in self._params_dict:
                raise Exception(f"Necessary segmentation class param/arg neither on file (probably there is no params_dict file) nor given as argument!")
        # ... else if it exists in the _params_dict but the values don't match. ...
        elif 'segmentation_classes' in self._params_dict and self._params_dict['segmentation_classes'] != segmentation_classes:

                # ... raise an exception.
                raise Exception(f"param segmentation_classes value {segmentation_classes} differs from value {self._params_dict['segmentation_classes']}in associator's params_dict file.")
        # Set the segmentation_classes value in the _params_dict.
        else:
            self._params_dict['segmentation_classes'] = segmentation_classes
        """

        # Choose appropriate label maker according to label type.
        # (categorical case)
        if self._params_dict['label_type'] == 'categorical':

            self._make_geotif_label = _make_geotif_label_categorical

        # (onehot case)
        elif self._params_dict['label_type'] == 'onehot':

            self._make_geotif_label = _make_geotif_label_onehot

        # (soft-categorical case)
        elif self._params_dict['label_type'] == 'soft-categorical':
        
            self._make_geotif_label = _make_geotif_label_soft_categorical

        else:

            log.error(f"Unknown label_type: {self._params_dict['label_type']}")
            raise Exception(f"Unknown label_type: {self._params_dict['label_type']}")

        # Check that either ...
        if not ( \
                # ... the associator files (imgs_df, polygons_df, graph) already exist ...
                (self._imgs_df_path.is_file() and self._polygons_df_path.is_file() and self._graph_path.is_file()) \
                ^ # ... or that ...
                # ... the imgs_df and polygons_df arguments were given. 
                ((imgs_df is not None) and (polygons_df is not None))):
            
            # If not, then alert the user.
            raise Exception(f"ImgPolygonAssociator: __init__: Need either existing associator files in data_dir (imgs_df, polygons_df etc.), or both imgs_df and polygons_df arguments and a data_dir without associator files.")

        # If the associator files exist, load them from file:
        if self._imgs_df_path.is_file() and self._polygons_df_path.is_file() and self._graph_path.is_file():

            # load imgs_df ...
            self.imgs_df = gpd.read_file(self._imgs_df_path)
            self.imgs_df.set_index(self._params_dict['imgs_df_index_name'], inplace=True)

            # ... polygons_df ...
            self.polygons_df = gpd.read_file(self._polygons_df_path)
            self.polygons_df.set_index(self._params_dict['polygons_df_index_name'], inplace=True)

            # ..., and the graph.
            self._graph = BipartiteGraph(file_path=self._graph_path)

        # Else, ...
        else:

            # ... start with an empty associator, i.e. empty imgs_df and polygons_df dataframes ...
            self.imgs_df = empty_imgs_df_same_format_as(imgs_df)
            self.polygons_df = empty_polygons_df_same_format_as(polygons_df)
            # ... and an empty graph.
            self._graph = empty_graph()
            # We will integrate imgs_df and polygons_df into the associator later after some safety checks. 

        # Check crs.
        if self.imgs_df.crs.to_epsg() != self._params_dict['crs_epsg_code']: # standard crs
            self.imgs_df = self.imgs_df.to_crs(epsg=self._params_dict['crs_epsg_code'])
        if self.polygons_df.crs.to_epsg() != self._params_dict['crs_epsg_code']:
            self.polygons_df = self.polygons_df.to_crs(epsg=self._params_dict['crs_epsg_code'])

        # Check index names.
        assert self.polygons_df.index.name == self._params_dict['polygons_df_index_name']
        assert self.imgs_df.index.name == self._params_dict['imgs_df_index_name']
        
        # Make sure polygons_df has columns necessary for associator to work.
        assert set({'have_img?', 'have_img_downloaded?'}) <= set(self.polygons_df.columns) 

        # If given as arguments, integrate polygons_df and imgs_df into the associator: 
        if imgs_df is not None:
            self.integrate_new_imgs_df(imgs_df)
        if polygons_df is not None:
            self.integrate_new_polygons_df(polygons_df)


    @property
    def crs_epsg_code(self):
        """EPSG code of associator's crs. Setting will set associator's imgs_df and polygons_df crs automatically. """
        return self._params_dict['crs_epsg_code']


    @crs_epsg_code.setter
    def crs_epsg_code(self, epsg_code):
        
        # set value in params dict
        self._params_dict['crs_epsg_code'] = epsg_code
        
        # reproject imgs_df and polygons_df GeoDataFrames
        self.polygons_df.to_crs(epsg=epsg_code)
        self.imgs_df.to_crs(epsg=epsg_code)


    @property
    def data_dir(self):
        """data directory"""
        return self._data_dir

    
    @data_dir.setter
    def data_dir(self, data_dir):

        self._data_dir = data_dir
        
        # file paths
        self._imgs_df_path = data_dir / Path("imgs_df.geojson")
        self._polygons_df_path = data_dir / Path("polygons_df.geojson")
        self._graph_path = data_dir / Path("graph.json")
        self._params_dict_path = data_dir / Path("params_dict.json")


    def save(self):
        """
        Save associator to disk.

        Saves associator to disk in the data_dir: imgs_df to imgs_df.geojson, polygons_df to polygons_df.geojson, the internal graph to graph.json, and the params_dict to params.json). 
            - Args: None
            - Returns: None
        """

        log.info(f"Saving associator to disk...")
        
        # Make sure data_dir exists.
        Path(self._data_dir).mkdir(parents=True, exist_ok=True)

        # Save all the components of the associator.
        self.imgs_df.to_file(Path(self._imgs_df_path), driver="GeoJSON")
        self.polygons_df.to_file(Path(self._polygons_df_path), driver="GeoJSON")
        self._graph.save_to_file(Path(self._graph_path))
        with open(self._params_dict_path, "w") as write_file:
                json.dump(self._params_dict, write_file)


    def have_img_for_polygon(self, polygon_name):
        """
        Return whether there is an image in the dataset fully containing the polygon. 

        :param polygon_name: 
        :returns: `True` if there is an image in the dataset fully containing the polygon, False otherwise.
        :rtype: bool
        """

        return self.polygons_df.loc[polygon_name, 'have_img?']


    def rectangle_bounding_img(self, img_name):
        """
        Return the shapely polygon of the rectangle bounding the image in coordinates in the associator's (standard) crs.

        :param img_name: the img_name/identifier of the image
        :returns: shapely polygon giving the bounds of the image in the standard crs of the associator
        """
        
        return self.imgs_df.loc[img_name, 'geometry']


    def polygons_intersecting_img(self, img_name):
        """
        Given an image, return an iterator of the names of all polygons 
        which have non-empty intersection with it.

        Args:
            - img_name, the img_name/identifier of the image
        Returns:
            - list of the polygon_names/identifiers of all polygons in associator with non-empty intersection with the image
        """
        
        return self._graph.vertices_opposite(vertex=img_name, vertex_color='imgs')


    def imgs_intersecting_polygon(self, polygon_name):
        """
        Given a polygon, return an iterator of the names of all images 
        which have non-empty intersection with it.

        Args:
            - polygon_name, the img_name/identifier of the polygon
        Returns:
            - list of the polygon_names/identifiers of all polygons in associator with non-empty intersection with the image
        """
        
        return self._graph.vertices_opposite(vertex=polygon_name, vertex_color='polygons')        


    def polygons_contained_in_img(self, img_name):
        """
        Given an image, return an iterator of the names of all polygons 
        which it fully contains.

        Args:
            - img_name, the img_name/identifier of the image
        Returns:
            - list of the polygon_names/identifiers of all polygons in associator contained in the image
        """
        
        return self._graph.vertices_opposite(vertex=img_name, vertex_color='imgs', edge_data='contains')


    def imgs_containing_polygon(self, polygon_name):
        """
        Given a ploygon, return an iterator of the names of all images 
        in which it us fully contained.

        Args:
            - polygon_name, the img_name/identifier of the polygon
        Returns:
            - list of the img_names/identifiers of all images in associator containing the polygon
        """
        
        return self._graph.vertices_opposite(vertex=polygon_name, vertex_color='polygons', edge_data='contains')


    def does_img_contain_polygon(self, img_name, polygon_name):
        """
        Args:
            - img_name/identifier, polygon_name/identifier 
        Returns:
            - True or False depending on whether the image contains the polygon or not
        """
        
        return polygon_name in self.polygons_contained_in_img(img_name)

    
    def is_polygon_contained_in_img(self, polygon_name, img_name):
        """
        Args:
            - polygon_name/identifier, img_name/identifier
        Returns:
            - True or False depending on whether the image contains the polygon or not
        """
        
        return self.does_img_contain_polygon(img_name, polygon_name)


    def does_img_intersect_polygon(self, img_name, polygon_name):
        """
        Args:
            - img_name/identifier, polygon_name/identifier 
        Returns:
            - True or False depending on whether the image and the polygon intersect or not
        """

        return (polygon_name in self.polygons_intersecting_img(img_name))


    def does_polygon_intersect_img(self, polygon_name, img_name):
        """
        Args:
            - polygon_name/identifier, img_name/identifier
        Returns:
            - True or False depending on whether the image and the polygon intersect or not
        """

        return self.does_img_intersect_polygon(img_name, polygon_name)


    def integrate_new_polygons_df(self, new_polygons_df, force_overwrite=False):
        """
        Add (or overwrite) polygons in new_polygons_df to the associator (i.e. append to the associator's polygons_df) keeping track of which polygons are contained in which images. 

        Args:
            - new_polygons_df: GeoDataFrame of polygons conforming to the associator's polygons_df format
            - force_overwrite (default value: False): whether to overwrite existing rows for polygons 
                in new_polygons_df whose polygon_name/identifier already is exists in the associator's polygons_df. If False will drop rows for which the polygon_name/identifier is already in the associator's polygons_df
        Returns:
            - None
        """        

        # First, make sure that the coordinate reference systems agree, ...
        if new_polygons_df.crs != self.polygons_df.crs:
            
            log.error(f"integrate_new_polygons_df: crs of new_polygons_df arg doesn't agree with crs of self.polygons_df.")
            
            raise Exception(f"integrate_new_polygons_df: crs of new_polygons_df arg doesn't agree with crs of self.polygons_df.")

        # ...that the columns agree, ...
        if set(new_polygons_df.columns) != set(self.polygons_df.columns):
            
            new_polygons_df_cols_not_in_self = set(new_polygons_df.columns) - set(self.polygons_df.columns)

            self_cols_not_in_new_polygons_df = set(self.polygons_df.columns) - set(new_polygons_df.columns)

            log.error(f"integrate_new_polygons_df: columns of new_polygons_df arg and self.polygons_df don't agree.")
            
            if new_polygons_df_cols_not_in_self != {}:
                log.error(f"columns that are in new_polygons_df but not in self.polygons_df: {new_polygons_df_cols_not_in_self}")
            
            if self_cols_not_in_new_polygons_df != {}:
                log.error(f"columns that are in self.polygons_df but not in new_polygons_df: {self_cols_not_in_new_polygons_df}")

            raise Exception(f"integrate_new_polygons_df: columns of new_polygons_df arg and self.polygons_df don't agree.")

        # ...and that the index names agree. 
        if new_polygons_df.index.name != self.polygons_df.index.name:
            
            log.error(f"integrate_new_polygons_df: index name {new_polygons_df.index.name} of new_polygons_df=new_polygons_df does not agree with index name {new_polygons_df.index.name} of self.polygons_df.index.name.")

            raise Exception(f"integrate_new_polygons_df: index name {new_polygons_df.index.name} of new_polygons_df=new_polygons_df does not agree with index name {new_polygons_df.index.name} of self.polygons_df.index.name.")

        # For each new polygon...
        for polygon_name in new_polygons_df.index:
            
            # ... if it already is in the associator ...
            if self._graph.exists_vertex(polygon_name, 'polygons'): # or: polygon_name in self.polygons_df.index
                
                # ... if necessary. ...
                if force_overwrite == True:
                    
                    # ... we overwrite the row in the associator's polygons_df ...
                    self.polygons_df.loc[polygon_name] = new_polygons_df.loc[polygon_name].copy()
                    
                    # ... and drop the row from new_polygons_df, so it won't be in self.polygons_df twice after we concatenate polygons_df to self.polygons_df. ...
                    new_polygons_df.drop(polygon_name, inplace=True) 
                    
                    # Then, we recalculate the connections. ...
                    self._remove_polygon_from_graph_modify_polygons_df(polygon_name)
                    self._add_polygon_to_graph(polygon_name)
                
                # Else ...
                else:
                    
                    # ... we drop the row from new_polygons_df...
                    new_polygons_df.drop(polygon_name, inplace=True)

                    log.info(f"integrate_new_polygons_df: dropping row for {polygon_name} from input polygons_df since is already in the associator! (force_overwrite arg is set to{force_overwrite})")

            # If it is not in the associator ...
            else:
                
                # ... add a vertex for the new polygon to the graph and add all connections to existing images. ...
                self._add_polygon_to_graph(polygon_name, polygons_df=new_polygons_df)

        # Finally, append new_polygons_df to the associator's (self.)polygons_df.
        data_frames_list = [self.polygons_df, new_polygons_df]
        self.polygons_df = gpd.GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)


    def integrate_new_imgs_df(self, new_imgs_df):
        """
        Add image data in new_imgs_df to the associator keeping track of which polygons are contained in which images.

        Args:
            - new_imgs_df: GeoDataFrame of image information conforming to the associator's imgs_df format
        Returns:
            - None
        """        

        # First, make sure that the coordinate reference systems agree, ...
        if new_imgs_df.crs != self.imgs_df.crs:
            
            log.error(f"integrate_new_imgs_df: crs of new_imgs_df arg doesn't agree with crs of self.imgs_df.")
            
            raise Exception(f"integrate_new_imgs_df: crs of new_imgs_df arg doesn't agree with crs of self.imgs_df.")

        # ...that the columns agree, ...
        if set(new_imgs_df.columns) != set(self.imgs_df.columns):

            new_imgs_df_cols_not_in_self = set(new_imgs_df.columns) - set(self.imgs_df.columns)

            self_cols_not_in_new_imgs_df = set(self.imgs_df.columns) - set(new_imgs_df.columns)

            log.error(f"integrate_new_imgs_df: columns of new_imgs_df arg and self.imgs_df don't agree.")
            
            if new_imgs_df_cols_not_in_self != {}:
                log.error(f"columns that are in new_imgs_df but not in self.imgs_df: {new_imgs_df_cols_not_in_self}")
            
            if self_cols_not_in_new_imgs_df != {}:
                log.error(f"columns that are in self.imgs_df but not in new_imgs_df: {self_cols_not_in_new_imgs_df}")
            
            raise Exception(f"integrate_new_imgs_df: columns of new_imgs_df arg self.imgs_df don't agree.")

        # ...and that the index names agree. 
        if new_imgs_df.index.name != self.imgs_df.index.name:
            
            log.error(f"integrate_new_imgs_df: index name {new_imgs_df.index.name} of new_imgs_df={new_imgs_df} does not agree with index name {new_imgs_df.index.name} of self.imgs_df.index.name.")
            
            raise Exception(f"integrate_new_imgs_df: index name {new_imgs_df.index.name} of new_imgs_df={new_imgs_df} does not agree with index name {new_imgs_df.index.name} of self.imgs_df.index.name.")

        # go through all new imgs...
        for img_name in new_imgs_df.index:
            
            # ... check if it is already in associator.
            if self._graph.exists_vertex(img_name, 'imgs'): # or: img_name in self.imgs_df.index
                
                # drop row from new_imgs_df, so it won't be in self.imgs_df twice after we concat new_imgs_df to self.imgs_df
                new_imgs_df.drop(img_name, inplace=True) 
                log.info(f"integrate_new_imgs_df: dropping row for {img_name} from input imgs_df since an image with that name is already in the associator!")
                 
            else:
                
                # add new img vertex to the graph, add all connections to existing images, 
                # and modify self.polygons_df 'have_img?', 'have_img_downloaded?' values where needed
                img_bounding_rectangle=new_imgs_df.loc[img_name, 'geometry']
                self._add_img_to_graph_modify_polygons_df(img_name, img_bounding_rectangle=img_bounding_rectangle)

        # append new_imgs_df
        data_frames_list = [self.imgs_df, new_imgs_df]
        self.imgs_df = gpd.GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)


    def drop_polygons(self, polygon_names):
        """
        Drop polygons from associator (i.e. remove rows from the associator's polygons_df)
        
        Args:
            - polygon_names: list or list-like. polygon_names/identifiers of polygons to be dropped.
        Returns:
            - None
        """

        # make sure we don't interpret a string as a list of characters in the iteration below:
        if type(polygon_names) == str:
            polygon_names = [polygon_names]
        assert pd.api.types.is_list_like(polygon_names)

        # remove the polygon vertices (along with their edges) 
        for polygon_name in polygon_names:
            self._graph.delete_vertex(polygon_name, 'polygons', force_delete_with_edges=True)
            
        # drop row from self.polygons_df
        self.polygons_df.drop(polygon_names, inplace=True)


    def drop_imgs(self, img_names, remove_imgs_from_disk=True):
        """
        Drop images from associator and dataset, i.e. remove rows from the associator's imgs_df, delete the corresponding vertices in the graph, and delete the image from disk (unless remove_imgs_from_disk is set to False).
        
        Args:
            - img_names: list or list-like. img_names/identifiers of images to be dropped.
            - remove_imgs_from_disk: bool. If true, delete images and labels from disk (if they exist).
        Returns:
            - None
        """

        # make sure we don't interpret a string as a list of characters in the iteration below:
        if type(img_names) == str:
            img_names = [img_names]
        assert pd.api.types.is_list_like(img_names)

        # drop row from self.imgs_df
        self.imgs_df.drop(img_names, inplace=True)

        # remove all vertices from graph and modify polygons_df if necessary
        for img_name in img_names:
            self._remove_img_from_graph_modify_polygons_df(img_name)

        # remove imgs and labels from disk
        if remove_imgs_from_disk==True:
            for subdir in DATA_DIR_SUBDIRS:
                for img_name in img_names:
                    Path(self._data_dir / Path(subdir) / Path(img_name)).unlink(missing_ok=True)


    def download_missing_imgs_for_polygons_df(self, 
                            polygons_df=None,
                            add_labels=True,
                            **kwargs):
        """ 
        Sequentially considers the polygons not contained in any image in the 
        associator's internal polygons_df or the optional polygons_df argument (if given), for each such polygon downloads either one image fully containing the polygon or several images jointly containing the polygon, creates the associated label(s) for the image(s) (assuming the default value True of add_labels is not changed), and integrates the new image(s) into the dataset/associator. If the optional polygons_df argument is provided will append polygons_df to the associator's internal polygons_df. Integrates images downloaded for a polygon into the dataset/associator immediately after downloading them and before downloading images for the next polygon. In particular, connects the newly downloaded image(s) to all polygons in the associator (and in polygons_df, if given), so that if a newly downloaded image contains a polygon (distinct from the one it was downloaded for) that has yet to be considered no attempt will be made to download images for that polygon. 

        Args:
            - polygons_df (optional, probably just best ignore this): GeoDataFrame of polygons conforming to the associator's format for polygon_df, defaults to the associator's internal polygons_df (i.e. self.polygons_df). If provided and not equal to self.polygons_df will download images for only those polygons and integrate the polygons in polygons_df into the associator after the images have been downloaded. 
            - add_labels (optional, default: True): bool. Whether to add labels for the downloaded images. 
        Returns:
            - None
        """

        # Make sure images subdir exists in data_dir
        Path(Path(self._data_dir) / "images").mkdir(parents=True, exist_ok=True)

        # Check if any polygons in polygons_df are already in the associator.
        if polygons_df is not None and polygons_df is not self.polygons_df:
            if (polygons_df.index.isin(self.polygons_df.index)).any() == True:
                log.error(f"download_missing_imgs_for_polygons_df: polygons_df contains polygons already in associator!")
                raise Exception(f"polygons_df contains polygons already in associator!")
        
        # Default polygons_df to self.polygons_df.
        if polygons_df is None:
            polygons_df = self.polygons_df 
        
        # Check crs.
        assert polygons_df.crs.to_epsg() == self._params_dict['crs_epsg_code'] 
    
        # Dict to keep track of imgs we've downloaded. We'll append this to self.imgs_df as a (geo)dataframe later
        new_imgs_dict = {index_or_col_name: [] for index_or_col_name in [self.imgs_df.index.name] + list(self.imgs_df.columns)}

        # Go through polygons for which no image has been downloaded yet.
        for polygon_name, polygon_geometry in tqdm(polygons_df.loc[(polygons_df['have_img_downloaded?'] == False)].loc[:,['geometry']].itertuples()): 

            # DEBUG INFO
            log.debug(f"download_missing_imgs_for_polygons_df: considering polygon {polygon_name}.")

            # Since we process and connect each image after downloading it, we might not need to download 
            # an image for a polygon that earlier was lacking an image if it is now contained in one of the already downloaded images, so need to check again that there is no image for the polygon (since the iterator above is set when it is called and won't know if the "have_img_downloaded?" column calue has been changed in the meanwhile).
            if polygons_df.loc[polygon_name, "have_img_downloaded?"] == True:
            
                # DEBUG INFO
                log.debug(f"download_missing_imgs_for_polygons_df: skipping polygon {polygon_name} since its have_img_downloaded? entry is True.")

                pass
            
            else:

                # Dict of possible keyword args for download function.
                # We use deepcopy here so that a call to download_missing_imgs_for_polygons_df 
                # can not modify self._params_dict.
                temporary_params_dict = copy.deepcopy(self._params_dict) 
                temporary_params_dict.update(kwargs)
                
                # Set of previously downloaded images.
                previously_downloaded_imgs_set = set(self.imgs_df.index) | set(new_imgs_dict[self.imgs_df.index.name])
                # (Will be used to make sure no attempt is made to download an image more than once.)

                # Try downloading an image and save returned dict (of dicts) containing information for polygons_df, self.imgs_df...  
                try:      

                    # DEBUG INFO
                    log.debug(f"attempting to download image for polygon {polygon_name}.")

                    return_dict = self._download_imgs_for_polygon(polygon_name,
                                                polygon_geometry,  
                                                Path(self._data_dir),
                                                previously_downloaded_imgs_set, # _download_imgs_for_polygon should use this to make sure no attempt at downloading an already downloaded image is made.
                                                **temporary_params_dict)
                
                # ... unless either no images could be found or a download error occured, ...
                except (NoImgsForPolygonFoundError, ImgDownloadError) as e:

                    # ... in which case we remember it in the polygons_df ...
                    self.polygons_df.loc[polygon_name, 'download_exception'] = repr(e)

                    # ... and log a warning, ...
                    log.warning(e, exc_info=True)
                
                # ... or _download_imgs_for_polygon tried downloading a previously downloaded image ...
                except ImgAlreadyExistsError as e:

                    # ... in which case we log the exception. 
                    log.exception(f"_download_imgs_for_polygon tried downloading a previously downloaded image!")

                # If nothing went wrong ...
                else:

                    # ... we first extract the information to be appended to self.imgs_df.
                    list_img_info_dicts = return_dict['list_img_info_dicts']
                    # (each img_info_dict contains the information for a new row of self.imgs_df)

                    # DEBUG INFO
                    log.debug(f"\nimg_polygon_associator: list_img_info_dicts is {list_img_info_dicts}\n\n")
                    
                    # We then update polygons_df.
                    polygon_info_dict = return_dict['polygon_info_dict']
                    for key in polygon_info_dict.keys():
                        polygons_df.loc[polygon_name, key] = polygon_info_dict[key]
                    
                    # If at least one image was downloaded, ...
                    if list_img_info_dicts != [{}]:
                    
                        # ... extract the new image names ...
                        new_img_names_list = [img_info_dict[self.imgs_df.index.name] for img_info_dict in list_img_info_dicts]

                        # ... and make sure we have not downloaded an image twice for the same polygon.
                        if len(new_img_names_list) != len(set(new_img_names_list)):
                            
                            duplicate_imgs_dict = {img_name: img_count for img_name, img_count in Counter(new_img_names_list).items() if img_count>1}

                            log.error(f"Something is wrong with _download_imgs_for_polygon: it attempted to download the following images multiple times for polygon {polygon_name}: {duplicate_imgs_dict}")

                            raise Exception(f"Something is wrong with _download_imgs_for_polygon: it attempted to download the following images multiple times for polygon {polygon_name}: {duplicate_imgs_dict}")
                        
                        # Make sure we haven't downloaded an image that's already in the dataset.
                        # (the _download_imgs_for_polygon method should have thrown an ImgAlreadyExistsError exception in this case, but we're checking again ourselves that this hasn't happened. )
                        if set(new_img_names_list) & previously_downloaded_imgs_set:

                            log.error(f"Something is wrong with _download_imgs_for_polygon: it downloaded image(s) that have already been downloaded: {set(new_img_names_list) & previously_downloaded_imgs_set}")

                            raise Exception(f"Something is wrong with _download_imgs_for_polygon: it downloaded image(s) that have already been downloaded: {set(new_img_names_list) & previously_downloaded_imgs_set}")

                        # For each download ...
                        for count, img_info_dict in enumerate(list_img_info_dicts):
                            
                            # ... process it to an image ...
                            img_name = img_info_dict[self.imgs_df.index.name]
                            single_img_processed_return_dict = self._process_downloaded_img_file(img_name, 
                                                                                                    Path(self._data_dir), 
                                                                                                    Path(self._data_dir) / Path("images"),
                                                                                                    self._params_dict['crs_epsg_code'],
                                                                                                    **self._params_dict)

                            # ... and update the img_info_dict with the returned information from processing. (This modifies list_img_info_dicts, too).
                            img_info_dict.update(single_img_processed_return_dict)
                                                        
                            # Connect the image: Add an image vertex to the graph, connect to all polygon vertices for which the intersection is non-empty and modify polygons_df and self.polygons_df where necessary ...
                            p_frames = [polygons_df] if polygons_df is self.polygons_df else [polygons_df, self.polygons_df]
                            for p_frame in p_frames:
                                self._add_img_to_graph_modify_polygons_df(img_name=img_name, 
                                                    img_bounding_rectangle=img_info_dict['geometry'], 
                                                    polygons_df=p_frame)

                            # ... and create the label, if necessary.
                            if add_labels==True:
                                self._make_geotif_label(self, img_name, log) # the self arg is needed, see import

                        """
                        # Check the polygon is fully contained in the union of the downloaded images
                        # THIS MADE SENSE WHEN I WAS JUST DOWNLOADING ONE IMAGE PER POLYGON, BUT DOESN'T MAKE SENSE ANYMORE SINCE WE'RE SKIPPING IMAGES THAT WE'D LIKE TO USE FOR A POLYGON THAT ALREADY HAVE BEEN DOWNLOADED, SO WILL GET UNNECESSARY WARNINGS FOR THOSE POLYGONS. BUT COULD MODIFY DOWNLOAD FUNCTION TO RETURN A SET OF THOSE IMAGES SO WE CAN CHECK THIS IF WE WANT...
                        list_downloaded_img_bounding_rectangles = [img_info_dict['geometry'] for img_info_dict in list_img_info_dicts]
                        union_area_of_downloaded_images = unary_union(list_downloaded_img_bounding_rectangles)
                        if not polygon_geometry.within(union_area_of_downloaded_images):
                            
                            downloaded_img_names = [img_info_dict['geometry'] for img_info_dict in list_img_info_dicts]
                            
                            log.warning(f"Polygon {polygon_name} not fully contained in the union of the images that were downloaded for it!")
                            
                            self.polygons_df.loc[polygon_name, "download_exception"] += f" Polygon {polygon_name} not fully contained in images downloaded for it: {downloaded_img_names}"
                        """

                        # Go through all images downloaded/processed for this polygon.
                        for img_info_dict in list_img_info_dicts: 

                            # After checking img_info_dict contains the columns/index we want 
                            # (so we don't for example fill in missing columns with nonsensical default values)...
                            if set(img_info_dict.keys()) != set(self.imgs_df.columns) | {self.imgs_df.index.name}:
                            
                                keys_not_in_cols_or_index = {key for key in img_info_dict.keys() if key not in set(self.imgs_df.columns) | {self.imgs_df.index.name}}

                                cols_or_index_not_in_keys = {x for x in set(self.imgs_df.columns) | {self.imgs_df.index.name} if x not in img_info_dict}

                                raise Exception(f"img_info_dict keys not equal to imgs_df columns and index name. \n Keys not in cols or index name {keys_not_in_cols_or_index} \n Columns or index not in keys: {cols_or_index_not_in_keys}")
                            
                            # ... accumulate the information in new_imgs_dict, which we will convert to a dataframe and append to imgs_df after we've gone through all new polygons.
                            for key in new_imgs_dict:

                                new_imgs_dict[key].append(img_info_dict[key])
                        
        # Extract accumulated information about the imgs we've downloaded from new_imgs into a dataframe ...
        new_imgs_df = gpd.GeoDataFrame(new_imgs_dict)
        new_imgs_df.set_crs(epsg=self._params_dict['crs_epsg_code'], inplace=True) # standard crs
        new_imgs_df.set_index("img_name", inplace=True)

        # ... and append it to self.imgs_df:
        data_frames_list = [self.imgs_df, new_imgs_df]  
        self.imgs_df = gpd.GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)

        # If not already equal to it, append polygons_df to self.polygons_df.
        if polygons_df is not self.polygons_df: 
            data_frames_list = [self.polygons_df, polygons_df]  
            self.polygons_df = gpd.GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)


    def make_missing_geotif_labels(self):
        """
        Creates categorical GeoTiff pixel labels (i.e. one channel images where each pixel is an integer corresponding to either the background or a segmentation class, 0 indicating the background class, and k=1,2, ... indicating the k-th entry (starting from 1) of the segmentation_classes parameter of the associator) in the data directory's labels subdirectory for all GeoTiff images in the image subdirectory without a label. 
        
        Args:
            - None
        Returns: 
            - None:
        """
        log.info("\nCreating missing labels.\n")

        # First, make sure the labels subdir exists in data_dir.
        Path(Path(self._data_dir) / "labels").mkdir(parents=True, exist_ok=True)

        # Find the set of existing images in the dataset, ...
        existing_images = set(os.listdir(Path(self._data_dir) / Path("images")))

        # ... then if the set of images is a strict subset of the images in imgs_df ...
        if existing_images < set(self.imgs_df.index):
            
            # ... log a warning 
            log.warning(f"make_missing_geotif_labels: there more images in self.imgs_df that are not in the dataset's images subdirectory.")

        # ... and if it is not a subset, ...
        if not existing_images <= set(self.imgs_df.index):
            
            # ... log an error...
            log.error(f"make_missing_geotif_labels: there are images in the dataset's images subdirectory that are not in self.imgs_df.")
            
            # ...and raise an exception.
            raise Exception(f"make_missing_geotif_labels: there are images in the dataset's images subdirectory that are not in self.imgs_df.")

        # Find the set of existing labels. 
        existing_labels = set(os.listdir(Path(self._data_dir) / Path("labels")))

        # For each image without a label ...
        for img_name in tqdm(existing_images - existing_labels):
        
            # ... create a label.
            self._make_geotif_label(self, img_name, log)


    def print_graph(self):
        """
        Print the associator's internal graph.
        """
        print(self._graph)


    def _connect_img_to_polygon(self, img_name, polygon_name, contains_or_intersects=None, polygons_df=None, img_bounding_rectangle=None, graph=None):
        """
        Connect an image to a polygon, i.e. remember whether the image fully contains or just has
        non-empty intersection with the polygon, i.e. add an edge of the approriate type between the image and the polygon.
        """ 

        # default polygons_df
        if polygons_df is None:
            polygons_df=self.polygons_df

        # default graph
        if graph is None:
            graph = self._graph

        # default img_bounding_rectangle
        if img_bounding_rectangle is None:
            img_bounding_rectangle = self.imgs_df.loc[img_name, 'geometry']

        # first, check whether img and polygon have non-empty intersection
        polygon_geometry = polygons_df.loc[polygon_name, 'geometry']
        non_empty_intersection = polygon_geometry.intersects(img_bounding_rectangle)
        
        # if not, don't do anything
        if non_empty_intersection == False:
            log.info(f"_connect_img_to_polygon: not connecting, sinceimg  {img_name} and polygon {polygon_name} do not overlap.")
        
        # else, add an edge of the appropriate type
        else:
            if contains_or_intersects is None:
                contains_or_intersects = 'contains' if img_bounding_rectangle.contains(polygon_geometry) else 'intersects'
            else:
                assert contains_or_intersects == 'contains' if img_bounding_rectangle.contains(polygon_geometry) else 'intersects'
            
            graph.add_edge(img_name, 'imgs', polygon_name, contains_or_intersects)

            # if the image contains the polygon record it in self.polygons_df
            if contains_or_intersects == 'contains':
                polygons_df.loc[polygon_name, 'have_img?'] = True
                polygons_df.loc[polygon_name, 'have_img_downloaded?'] = True


    def _add_polygon_to_graph(self, polygon_name, polygons_df=None):
        """
        Connects a polygon to those images in self.imgs_df with which it has non-empty intersection.
        """

        # default polygons_df
        if polygons_df is None:
            polygons_df = self.polygons_df
        
        # add vertex if one does not yet exist
        if not self._graph.exists_vertex(polygon_name, 'polygons'):
            self._graph.add_vertex(polygon_name, 'polygons')
        
        # raise an exception if the polygon already has connections
        if list(self._graph.vertices_opposite(polygon_name, 'polygons')) != []:
            log.warning(f"_add_polygon_to_graph: !!!Warning (connect_polygon): polygon {polygon_name} already has connections! Probably _add_polygon_to_graph is being used wrongly. Check your code!")
        
        # go through all images and connect if intersection is non-empty
        for img_name, img_bounding_rectangle in self.imgs_df.loc[:, ['geometry']].itertuples():
            polygon_geometry = polygons_df.geometry.loc[polygon_name]
            if img_bounding_rectangle.intersects(polygon_geometry):
                contains_or_intersects = 'contains' if img_bounding_rectangle.contains(polygon_geometry) else 'intersects'
                self._connect_img_to_polygon(img_name, polygon_name, contains_or_intersects, polygons_df=polygons_df)


    def _add_img_to_graph_modify_polygons_df(self, img_name, img_bounding_rectangle=None, polygons_df=None, graph=None):
        """ 
        Connects an image to all polygons in polygons_df, creating the vertex if necessary. The default value for polygons_df is None, which we take to mean self.polygons_df. If img_bounding_rectangle is None, we assume we can get it from self. If the image already exists and already has connections a warning will be logged. imgs_df.
        """

        # default polygons_df
        if polygons_df is None:
            polygons_df = self.polygons_df

        # default graph:
        if graph is None:
            graph = self._graph

        # default img_bounding_rectangle
        if img_bounding_rectangle is None:
            img_bounding_rectangle = self.imgs_df.geometry.loc[img_name]

        # add vertex if it does not yet exist
        if not graph.exists_vertex(img_name, 'imgs'):
            graph.add_vertex(img_name, 'imgs')

        # check if img already has connections
        if list(graph.vertices_opposite(img_name, 'imgs')) != []:
            log.warning(f"!!!Warning (connect_img): image {img_name} already has connections!")

        # go through all polygons in polygons_df and connect by an edge if the polygon and img intersect
        for polygon_name, polygon_geometry in polygons_df.loc[:, ['geometry']].itertuples():
            if img_bounding_rectangle.intersects(polygon_geometry):
                contains_or_intersects = 'contains' if img_bounding_rectangle.contains(polygon_geometry) else 'intersects'
                self._connect_img_to_polygon(img_name, polygon_name, contains_or_intersects, polygons_df=polygons_df, img_bounding_rectangle=img_bounding_rectangle, graph=graph)


    def _remove_polygon_from_graph_modify_polygons_df(self, polygon_name, forget_have_img_downloaded=True):
        """
        Removes a polygon from the graph (i.e. removes the vertex and all incident edges) 
        and (if forget_have_img_downloaded == True) modifies the polygons_df fields 'have_img?' and 
        'have_img_downloaded?' for those polygons for which the image removed was the only image containing them. 
        """

        self._graph.delete_vertex(polygon_name, 'polygons', force_delete_with_edges=True)

        if forget_have_img_downloaded==True:
            self.polygons_df.loc[polygon_name, 'have_img?'] = False
            self.polygons_df.loc[polygon_name, 'have_img_downloaded?'] = False


    def _remove_img_from_graph_modify_polygons_df(self, img_name):
        """
        Removes an img from the graph (i.e. removes the vertex and all incident edges) 
        and modifies the polygons_df fields 'have_img?' and 'have_img_downloaded?' for those 
        polygons for which the image removed was the only image containing them. 
        """

        # set 'have_img?' and 'have_img_downloaded?' values in self.polygons_df that are affected by disconnecting img:
        # go through all polygons
        for polygon_name in self.polygons_contained_in_img(img_name):
            # check if the img is the only one containing the polygon
            if len(self.imgs_containing_polygon(polygon_name)) == 1:
                # set values
                self.polygons_df.loc[polygon_name, 'have_img?'] = False
                self.polygons_df.loc[polygon_name, 'have_img_downloaded?'] = False
                
        self._graph.delete_vertex(img_name, 'imgs', force_delete_with_edges=True)


    def _download_imgs_for_polygon(self, polygon_name, polygon_geometry, download_dir, previously_downloaded_imgs_set, **kwargs):
        """
        Not implemented, overwrite/implement in a subclass. Should download an image fully containing a vector polygon or several images jointly containing it and return a dict with information to be updated in the associator, see below for details.
            
            Args:
                -polygon_name: the name of the vector polygon. 
                -polygon_geometry: shapely geometry of polygon.
                -download_dir: directory that the image file should be downloaded to.
                -previously_downloaded_imgs_set: Set of previously downloaded img_names. In some use cases when it can't be guaranteed that an image can be downloaded that fully contains the polygon it can happen that attempts will be made to download an image that is already in the associator. Passing this argument allows the download function to make sure it doesn't try downloading an image that is already in the dataset.
                -**kwargs: optional keyword arguments depending on the application.
            Returns:
                - A dict with keys and values:
                    -'list_img_info_dicts': a list of dicts containing the information to be included in each row in the imgs_df of the calling associator, one for each newly downloaded image. The keys should be the index and column names of the imgs_df and the values the indices or entries of those columns in row that will correspond to the new image.
                    -'polygon_info_dict': a dict containing information (in particular values for the 'have_img_downloaded?' and 'download_exception' of the associators polygons_df) to be updated in polygons_df of the calling associator. The keys and values should be given by the column names and entries of polygons_df that should be updated for polygon polygon_name. 
        """

        raise NotImplementedError
    

    def _process_downloaded_img_file(self, img_name, in_dir, out_dir, convert_to_crs_epsg, **kwargs):
        """
        Not implemented, overwrite/implement in a subclass. Processes an image file downloaded by _download_imgs_for_polygon. Needs to return a dict with information to be updated in the associator, see below for details.
        
            Args:
                -img_name: the image name (index identifiying the corresponding row in imgs_df) 
                -in_dir: the directory the image file was downloaded to
                -out_dir: the directory the processed image file should be in (usually data_dir/images)
                -convert_to_crs_epsg: EPSG code of the crs the image (if georeferenced, e.g. as a GeoTiff) 
                    should be converted to.
                -**kwargs: optional keyword arguments depending on the application
            Returns:
                -img_info_dict: a dict containing the information to be updated in the imgs_df of the calling associator. The keys should be the index and column names of the imgs_df and the values lists of indices or entries of those columns.
        """

        raise NotImplementedError


###################### End of ImgPolygonAssociator class definition ######################################


def empty_gdf(df_index_name, df_cols_and_index_types, crs_epsg_code=STANDARD_CRS_EPSG_CODE):
    """Return a empty GeoDataFrame with specified index and column names and types and crs.

    Args:
        - df_index_name: name of the index of the new empty GeoDataFrame
        - df_cols_and_index_types: dict with keys the names of the index and columns of the GeoDataFrame and values the types of the indices/column entries.
        - crs_epsg_code: EPSG code of the crs the empty GeoDataFrame should have.
    Returns:
        - new_empty_df: the empty polygons_df GeoDataFrame.
    """

    new_empty_gdf_dict = {'geometry': gpd.GeoSeries([]),
                                **{index_or_col_name: pd.Series([], dtype=index_or_col_type) 
                                    for index_or_col_name, index_or_col_type in df_cols_and_index_types.items()
                                        if index_or_col_name != 'geometry'}}
    new_empty_gdf = gpd.GeoDataFrame(new_empty_gdf_dict, crs=f"EPSG:{crs_epsg_code}")
    new_empty_gdf.set_index(df_index_name, inplace=True)
    return new_empty_gdf


def empty_imgs_df(imgs_df_index_name, imgs_df_cols_and_index_types, crs_epsg_code=STANDARD_CRS_EPSG_CODE):
    """
    Return a generic empty imgs_df GeoDataFrame conforming to the ImgPolygonAssociator format.
    
    Args:
        - imgs_df_index_name: index name of the new empty imgs_df
        - imgs_df_cols_and_index_types: dict with keys the names of the index and columns of the new empty imgs_df and values the types of the index/column entries.
        - crs_epsg_code: EPSG code of the crs the empty imgs_df should have.
    Returns:
        - new_imgs_df: the empty imgs_df GeoDataFrame.
    """

    return empty_gdf(imgs_df_index_name, imgs_df_cols_and_index_types, crs_epsg_code=STANDARD_CRS_EPSG_CODE)


def empty_polygons_df(polygons_df_index_name, polygons_df_cols_and_index_types, crs_epsg_code=STANDARD_CRS_EPSG_CODE):
    """Return a generic empty polygons_df GeoDataFrame conforming to the ImgPolygonAssociator format.

    Return a generic empty polygons_df GeoDataFrame conforming to the ImgPolygonAssociator format.
    
    Args:
        - polygons_df_index_name_and_type: name of the index of the new empty polygons_df
        - polygons_df_cols_and_index_types: dict with keys the names of the index and columns of the new empty polygons_df and values the types of the indices/column entries.
        - crs_epsg_code: EPSG code of the crs the empty polygons_df should have.
    Returns:
        - new_polygons_df: the empty polygons_df GeoDataFrame.
    """

    return empty_gdf(polygons_df_index_name, polygons_df_cols_and_index_types, crs_epsg_code=STANDARD_CRS_EPSG_CODE)


def empty_polygons_df_same_format_as(polygons_df):
    """
    Creates an empty polygons_df of the same format (index name, columns, column types) as the polygons_df argument.
    """
    polygons_df_index_name = polygons_df.index.name

    polygons_df_cols_and_index_types = {polygons_df.index.name: polygons_df.index.dtype, 
                                        **polygons_df.dtypes.to_dict()}

    crs_epsg_code = polygons_df.crs.to_epsg()

    new_empty_polygons_df = empty_polygons_df(polygons_df_index_name, 
                                                polygons_df_cols_and_index_types, 
                                                crs_epsg_code=crs_epsg_code)

    return new_empty_polygons_df


def empty_imgs_df_same_format_as(imgs_df):
    """
    Creates an empty imgs_df of the same format (index name, columns, column types) as the imgs_df argument.
    """
    imgs_df_index_name = imgs_df.index.name

    imgs_df_cols_and_index_types = {imgs_df.index.name: imgs_df.index.dtype, 
                                        **imgs_df.dtypes.to_dict()}

    crs_epsg_code = imgs_df.crs.to_epsg()

    new_empty_imgs_df = empty_imgs_df(imgs_df_index_name, 
                                                imgs_df_cols_and_index_types, 
                                                crs_epsg_code=crs_epsg_code)

    return new_empty_imgs_df


def empty_assoc_same_format_as(target_data_dir, source_data_dir=None, source_assoc=None):
    """
    Creates an empty associator with data_dir target_data_dir of the same format as an existing one in source_data_dir or one given as source_assoc (same polygons_df and imgs_df columns and index names and paramaters).
    """

    # exactly one of source_data_dir or source_assoc should be given
    assert (source_data_dir != None) ^ (source_assoc != None)

    if source_assoc == None:
        source_assoc = ImgPolygonAssociator(source_data_dir)

    # new empty polygons_df
    new_empty_polygons_df = empty_polygons_df_same_format_as(source_assoc.polygons_df)                                        
    
    # new empty imgs_df
    new_empty_imgs_df = empty_imgs_df_same_format_as(source_assoc.imgs_df) 

    new_empty_assoc = ImgPolygonAssociator(data_dir=Path(target_data_dir), 
                        imgs_df=new_empty_imgs_df, 
                        polygons_df=new_empty_polygons_df, 
                        **source_assoc._params_dict)

    return new_empty_assoc
    

def empty_graph():
    """Return an empty bipartite graph to be used by ImgPolygonAssociator."""
    return empty_bipartite_graph(red='polygons', black='imgs')

    






                







