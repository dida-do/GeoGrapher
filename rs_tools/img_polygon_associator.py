"""
TODO: replace relative paths by attributes or queries of _params_dict e.g. self.data_dir / "images"  -> self._params_dict['images_dir']


The ImgPolygonAssociator class organizes and handles remote sensing datasets.
"""

from typing import Dict, Type, Any, List, TypeVar
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
from geopandas import GeoDataFrame
from shapely.ops import transform, unary_union
from shapely.geometry import Polygon
import rasterio as rio
from collections import Counter

from typing import Union, Optional, Sequence, List

from rs_tools.global_constants import *
from rs_tools.img_polygon_associator_class import ImgPolygonAssociatorClass
from rs_tools.graph import BipartiteGraph, empty_bipartite_graph
from rs_tools.utils.utils import transform_shapely_geometry, deepcopy_gdf
from rs_tools.utils.associator_utils import empty_gdf, empty_polygons_df_same_format_as, empty_imgs_df_same_format_as, empty_graph
from rs_tools.errors import ImgAlreadyExistsError, NoImgsForPolygonFoundError, ImgDownloadError

LABEL_TYPES = ['one-hot', 'categorical', 'soft-categorical']

IPAType = TypeVar('IPAType', bound='ImgPolygonAssociator')


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
        - 'img_count': int. Number of images in the dataset that fully contain the polygon.
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

        # args w/o default values
        data_dir : Union[Path, str], 
        segmentation_classes : Sequence[str], 
        label_type: str,

        # polygons_df args. Exactly one value needs to be set (i.e. not None).
        polygons_df : Optional[GeoDataFrame] = None,
        polygons_df_path : Optional[Union[Path, str]] = None, 
        polygons_df_cols : Optional[Union[List[str], Dict[str, Type]]] = None,
        
        # imgs_df args. Exactly one value needs to be set (i.e. not None).
        imgs_df : Optional[GeoDataFrame] = None, 
        imgs_df_path : Optional[Union[Path, str]] = None, 
        imgs_df_cols : Optional[Union[List[str], Dict[str, Type]]] = None,
        
        # remaining non-path args w/ default values
        add_background_band_in_labels : bool = False, 
        crs_epsg_code : int = STANDARD_CRS_EPSG_CODE, 

        # path args w/ default values
        images_dir : Optional[Union[Path, str]] = None, # will default to data_dir / "images"
        labels_dir : Optional[Union[Path, str]] = None, # will default to data_dir / "labels"

        # optional kwargs
        **kwargs : Any
        ):

        super().__init__()

        # build _params_dict from all args except for imgs_df, polygons_df and the corresponding columns
        self._params_dict = {}
        self._params_dict.update( 
            {
                "data_dir" : data_dir,
                "segmentation_classes" : segmentation_classes, 
                "label_type" : label_type, 
                "polygons_df_path" : polygons_df_path, 
                "imgs_df_path" : imgs_df_path, 
                "add_background_band_in_labels" : add_background_band_in_labels, 
                "crs_epsg_code" : crs_epsg_code, 
                "images_dir" : images_dir, 
                "labels_dir" : labels_dir,
                **kwargs
            })

        # get polygons_df and imgs_df
        polygons_df = self._init_get_df_from_args(
                        "polygons_df", 
                        polygons_df, 
                        polygons_df_path, 
                        polygons_df_cols, 
                        POLYGONS_DF_INDEX_NAME, 
                        crs_epsg_code)
        imgs_df = self._init_get_df_from_args(
                        "imgs_df",
                        imgs_df, 
                        imgs_df_path, 
                        imgs_df_cols, 
                        IMGS_DF_INDEX_NAME, 
                        crs_epsg_code)
        self._standardize_df_crss(
            polygons_df=polygons_df,
            imgs_df=imgs_df)

        # run safety checks on polygons_df, imgs_df and adjust format if necessary
        self._check_and_adjust_df(
            mode='polygons_df',
            df=polygons_df)
        self._check_and_adjust_df(
            mode='imgs_df',
            df=imgs_df)

        # set remaining associator components (polygons_df, imgs_df, _graph)
        self.imgs_df = empty_imgs_df_same_format_as(imgs_df)
        self.polygons_df = empty_polygons_df_same_format_as(polygons_df)
        self._graph = empty_graph()

        # set paths
        self._data_dir = Path(data_dir)
        self._init_set_paths( #  sets path attributes as well as path values in self._params_dict
            self._data_dir, 
            images_dir, 
            labels_dir)

        # label maker
        self._check_label_type(label_type)
        self._set_label_maker(label_type)

        # integrate dfs (also builds graph)
        self.integrate_new_polygons_df(polygons_df)
        self.integrate_new_imgs_df(imgs_df)


    @classmethod
    def from_json(
            cls : Type[IPAType], 
            json_path : Union[Path, str], 
            polygons_df : Optional[GeoDataFrame] = None,
            imgs_df : Optional[GeoDataFrame] = None
            ) -> IPAType:

        try:
            with open(json_path, "r") as read_file:
                kwargs = json.load(read_file)
        except FileNotFoundError:
            log.exception(f"No file under path {json_path} found!")
        except JSONDecodeError:
            log.exception(f"JSON file in {json_path=} corrupted!")

        new_assoc = cls(
            polygons_df=polygons_df, 
            imgs_df=imgs_df, 
            **kwargs)
        return new_assoc
        

    @classmethod
    def from_data_dir(
            cls : Type[IPAType], 
            data_dir : Union[Path, str], 
            polygons_df : Optional[GeoDataFrame] = None,
            imgs_df : Optional[GeoDataFrame] = None
            ) -> IPAType:
        """Initialize and return an associator from a data directory

        Args:
            data_dir (Union[Path, str]): data directory containing an associator's params_dict.json file
            polygons_df (Optional[GeoDataFrame], optional): polygons_df. Defaults to None.
            imgs_df (Optional[GeoDataFrame], optional): imgs_df. Defaults to None.

        Returns:
            IPAType: initialized associator
        """

        return cls.from_json(
                json_path=Path(data_dir) / "params_dict.json",
                polygons_df=polygons_df, 
                imgs_df=imgs_df)


    @classmethod
    def from_kwargs(cls, **kwargs : Any) -> 'ImgPolygonAssociator':
        """
        Initialize and return an associator from keyword arguments. 

        Ars:
            **kwargs (Any): keyword arguments

        Returns:
            initialized associator
        """
        return cls(**kwargs)


    def _check_and_adjust_df(self, 
            mode : str, 
            df : GeoDataFrame):
        """
        Check if a dataframe df (polygons_df or imgs_df) has the format required for an associator to work, if not either make adjustments if possible or raise a ValueError.

        Args:
            mode (str): One of "polygons_df" or "imgs_df"
            df (GeoDataFrame): polygons_df

        Raises:
            ValueError: If the df doesn't have the right index name
        """

        if mode == 'polygons_df':
            if 'img_count' not in set(df.columns):
                log.info("Adding 'img_count' column to polygons_df")
            df['img_count'] = 0
            
        if mode == 'polygons_df' and df.index.name != POLYGONS_DF_INDEX_NAME:
            raise ValueError(f"polygons_df.index.name is {df.index.name}, should be {POLYGONS_DF_INDEX_NAME}")
        if mode == 'imgs_df' and df.index.name != IMGS_DF_INDEX_NAME:
            raise ValueError(f"imgs_df.index.name is {df.index.name}, should be {IMGS_DF_INDEX_NAME}")


    def _standardize_df_crss(self, 
            polygons_df : GeoDataFrame, 
            imgs_df : GeoDataFrame):
        """
        Standardize polygons_df and imgs_df CRS (i.e. set their CRS to associator CRS).

        Args:
            polygons_df (GeoDataFrame): polygons_df
            imgs_df (GeoDataFrame): imgs_df
        """

        if imgs_df.crs.to_epsg() != self._params_dict['crs_epsg_code']: # standard crs
            imgs_df = imgs_df.to_crs(epsg=self._params_dict['crs_epsg_code'])
        if polygons_df.crs.to_epsg() != self._params_dict['crs_epsg_code']:
            polygons_df = polygons_df.to_crs(epsg=self._params_dict['crs_epsg_code'])


    @staticmethod
    def _init_get_df_from_args(
            mode : str, 
            df : Optional[GeoDataFrame], 
            geojson_path : Optional[Union[Path, str]], 
            df_cols : Optional[Union[List[str], Dict[str, Type]]], 
            df_index_name : str,
            crs_epsg_code : int,
            ) -> GeoDataFrame:
        """
        Extract dataframe from dataframe arguments. Used during initialization.

        Args:
            mode (str): One of "polygons_df" or "imgs_df"
            df (GeoDataFrame, optional): polygons_df or imgs_df
            geojson_path (Union[Path, str], optional): path to imgs_df or polygons_df geojson file
            df_cols (Union[List[str], Dict[str, Type]], optional): list of column names or dict of column names and types or empty dataframe (imgs_df or polygons_df)
            df_index_name (str, optional): index name of empty dataframe (imgs_df or polygons_df)
            crs_epsg_code (int): EPSG code of crs of df to be created, used if df_cols is not None.

        Returns:
            GeoDataFrame: polygons_df or imgs_df
        """

        if mode not in {"polygons_df", "imgs_df"}:
            raise ValueError(f"Unknown mode: {mode}")

        if not ((df is not None) ^ (geojson_path is not None) ^ (df_cols is not None)):
            raise ValueError(f"Exactly one of {mode}, {mode}_json_path, {mode}_cols should be set (i.e. not None).")

        if df is not None:
        
            return_df = df
        
        elif geojson_path is not None:
        
            return_df = gpd.read_file(geojson_path)
            return_df.set_index(df_index_name, inplace=True)
        
        elif df_cols is not None: 
            
            # build df_cols_and_index_types for empty_gdf
            df_cols_and_index_types = df_cols
            if isinstance(df_cols, list):
                df_cols_and_index_types = {col_name: None for col_name in df_cols_and_index_types}
            df_cols_and_index_types[df_index_name] = object

            return_df = empty_gdf(
                            df_index_name=df_index_name,
                            df_cols_and_index_types=df_cols_and_index_types, 
                            crs_epsg_code=crs_epsg_code)
                            
        return return_df
            

    def _init_set_paths(self, 
            data_dir: Path, 
            images_dir: Optional[Union[Path, str]], 
            labels_dir: Optional[Union[Path, str]]):
        """Set paths to image/label data and associator component files. Used during initialization."""
        
        # image/label directories
        self._images_dir = Path(images_dir) if images_dir is not None else data_dir / "images"
        self._labels_dir = Path(labels_dir) if labels_dir is not None else data_dir / "labels"

        self._params_dict['images_dir'] = self._images_dir
        self._params_dict['labels_dir'] = self._labels_dir
        
        # associator component files
        self._imgs_df_path = data_dir / "imgs_df.geojson"
        self._polygons_df_path = data_dir / "polygons_df.geojson"
        self._graph_path = data_dir / "graph.json"
        self._params_dict_path = data_dir / "params_dict.json"


    @staticmethod
    def _check_label_type(label_type : str):
        """Check label_type is allowed label type, raise ValueError if not"""
        if label_type not in LABEL_TYPES:
            raise ValueError(f"Unknown label_type: {label_type}")


    def _set_label_maker(self, 
            label_type : str):
        """
        Set associator label makers according to label_type.

        Args:
            label_type (str): label type
        """

        # (categorical case)
        if label_type == 'categorical':
            self._make_geotif_label = _make_geotif_label_categorical
        # (onehot case)
        elif label_type == 'onehot':
            self._make_geotif_label = _make_geotif_label_onehot
        # (soft-categorical case)
        elif label_type == 'soft-categorical':
            self._make_geotif_label = _make_geotif_label_soft_categorical
        else:
            log.error(f"Unknown label_type: {label_type}")
            raise ValueError(f"Unknown label_type: {label_type}")

    
    @staticmethod
    def _make_dict_json_serializable(input_dict: dict) -> dict:
        """
        Make dict serializable as JSON by replacing Path with strings

        Args:
            input_dict (dict): input dict with keys strings and values of arbitrary type

        Returns:
            dict:  dict with non-serializable values replaced by serializable ones (just Path -> str, for now)
        """

        def make_val_serializable(val): 
            return str(val) if isinstance(val, pathlib.PurePath) else val

        serializable_dict = {key : make_val_serializable(val) for key, val in input_dict.items()}
        
        return serializable_dict
        

    @property
    def crs_epsg_code(self) -> int:
        """
        int: EPSG code of associator's crs. 
        
        Setting will set associator's imgs_df and polygons_df crs automatically.
        """
        return self._params_dict['crs_epsg_code']


    @crs_epsg_code.setter
    def crs_epsg_code(self, epsg_code: int):
        # set value in params dict
        self._params_dict['crs_epsg_code'] = epsg_code
        
        # reproject imgs_df and polygons_df GeoDataFrames
        self.polygons_df.to_crs(epsg=epsg_code)
        self.imgs_df.to_crs(epsg=epsg_code)


    @property
    def data_dir(self) -> Path:
        """Path: data directory"""
        return self._data_dir

    
    @data_dir.setter
    def data_dir(self, data_dir: Union[str, Path]):
        self._data_dir = Path(data_dir)
        
        # file paths
        self._imgs_df_path = data_dir / Path("imgs_df.geojson")
        self._polygons_df_path = data_dir / Path("polygons_df.geojson")
        self._graph_path = data_dir / Path("graph.json")
        self._params_dict_path = data_dir / Path("params_dict.json")


    @property 
    def label_type(self) -> str:
        """Return label type"""
        return self._params_dict['label_type']

    @label_type.setter
    def label_type(self, new_label_type : str):
        """Set label type. WARNING! Does not remove old labels and recreate new ones."""

        self._check_label_type(new_label_type)
        old_label_type = self._params_dict['label_type'] 
        if new_label_type != old_label_type: 
            self._params_dict['label_type'] = new_label_type
            self._set_label_maker(new_label_type)

        log.warning(f"label_type has been changed to {label_type}. You might want to delete the old labels and recreate labels!")
            
    def save(self):
        """
        Save associator to disk.

        Saves associator to disk in the data_dir: imgs_df to imgs_df.geojson, polygons_df to polygons_df.geojson, the internal graph to graph.json, and the params_dict to params.json). 
        """

        log.info(f"Saving associator to disk...")
        
        # Make sure data_dir exists.
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Save all the components of the associator.
        self.imgs_df.to_file(Path(self._imgs_df_path), driver="GeoJSON")
        self.polygons_df.to_file(Path(self._polygons_df_path), driver="GeoJSON")
        self._graph.save_to_file(Path(self._graph_path))
        with open(self._params_dict_path, "w") as write_file:
            json.dump(
                self._make_dict_json_serializable(self._params_dict), 
                write_file)


    def have_img_for_polygon(self, polygon_name: str) -> bool:
        """
        Return whether there is an image in the dataset fully containing the polygon.

        Args:
            polygon_name (str): Name of polygon

        Returns:
            bool: `True` if there is an image in the dataset fully containing the polygon, False otherwise.
        """

        return self.polygons_df.loc[polygon_name, 'img_count'] > 0


    def rectangle_bounding_img(self, img_name: str) -> Polygon:
        """
        Return the shapely polygon of the rectangle bounding the image in coordinates in the associator's (standard) crs.

        Args:
            img_name (str): the img_name/identifier of the image

        Returns:
            Polygon: shapely polygon giving the bounds of the image in the standard crs of the associator
        """
        
        return self.imgs_df.loc[img_name, 'geometry']


    def polygons_intersecting_img(self, img_name : str) -> List[str]:
        """
        Given an image, return the list of (the names of) all polygons 
        which have non-empty intersection with it.

        Args:
            img_name (str): the img_name/identifier of the image

        Returns:
            list of strs of polygon_names/ids of all polygons in associator which have non-empty intersection with the image
        """
        
        return self._graph.vertices_opposite(vertex=img_name, vertex_color='imgs')


    def imgs_intersecting_polygon(self, polygon_name: str) -> List[str]:
        """
        Given a polygon, return an iterator of the names of all images 
        which have non-empty intersection with it.

        Args:
            polygon_name (str): the img_name/identifier of the polygon

        Returns:
            list of str: list of the polygon_names/identifiers of all polygons in associator with non-empty intersection with the image.
        """
        
        return self._graph.vertices_opposite(vertex=polygon_name, vertex_color='polygons')        


    def polygons_contained_in_img(self, img_name: str) -> List[str]:
        """
        Given an image, return an iterator of the names of all polygons 
        which it fully contains.

        Args:
            img_name (str): the img_name/identifier of the image

        Returns:
            list of str: list of the polygon_names/identifiers of all polygons in associator contained in the image.
        """
        
        return self._graph.vertices_opposite(vertex=img_name, vertex_color='imgs', edge_data='contains')


    def imgs_containing_polygon(self, polygon_name: str) -> List[str]:
        """
        Given a ploygon, return an iterator of the names of all images 
        in which it us fully contained.

        Args:
            polygon_name (str): the img_name/identifier of the polygon

        Returns:
            List[str]: list of the img_names/identifiers of all images in associator containing the polygon
        """
        
        return self._graph.vertices_opposite(vertex=polygon_name, vertex_color='polygons', edge_data='contains')


    def does_img_contain_polygon(self, img_name: str, polygon_name: str) -> bool:
        """
        Args:
            img_name (str): Name of image
            polygon_name (str): name of polygon
        
        Returns:
            bool: True or False depending on whether the image contains the polygon or not
        """
        
        return polygon_name in self.polygons_contained_in_img(img_name)

    
    def is_polygon_contained_in_img(self, polygon_name: str, img_name: str) -> bool:
        """
        Args:
            img_name (str): Name of image
            polygon_name (str): name of polygon

        Returns:
            bool: True or False depending on whether the polygon contains the image or not
        """
        
        return self.does_img_contain_polygon(img_name, polygon_name)


    def does_img_intersect_polygon(self, img_name: str, polygon_name: str) -> bool:
        """
        Args:
            img_name (str): Name of image
            polygon_name (str): name of polygon

        Returns:
            bool: True or False depending on whether the image intersects the polygon or not
        """

        return (polygon_name in self.polygons_intersecting_img(img_name))


    def does_polygon_intersect_img(self, polygon_name: str, img_name: str) -> bool:
        """
        Args:
            img_name (str): Name of image
            polygon_name (str): name of polygon

        Returns:
            bool: True or False depending on whether the polygon intersects the image or not
        """

        return self.does_img_intersect_polygon(img_name, polygon_name)


    def integrate_new_polygons_df(self, 
            new_polygons_df: GeoDataFrame, 
            force_overwrite: bool=False):
        """
        Add (or overwrite) polygons in new_polygons_df to the associator (i.e. append to the associator's polygons_df) keeping track of which polygons are contained in which images.

        Args:
            new_polygons_df (GeoDataFrame): GeoDataFrame of polygons conforming to the associator's polygons_df format
            force_overwrite (bool): whether to overwrite existing rows for polygons, default is False
        """        

        new_polygons_df = deepcopy_gdf(new_polygons_df) #  don't want to modify argument

        self._check_and_adjust_df(
            mode='polygons_df', 
            df=new_polygons_df)

        # Make sure that the coordinate reference systems agree, ...
        if new_polygons_df.crs != self.polygons_df.crs:
            
            log.error(f"integrate_new_polygons_df: crs of new_polygons_df arg doesn't agree with crs of self.polygons_df.")
            
            raise ValueError(f"integrate_new_polygons_df: crs of new_polygons_df arg doesn't agree with crs of self.polygons_df.")

        # ... and that the columns agree.
        if set(new_polygons_df.columns) != set(self.polygons_df.columns):
            
            new_polygons_df_cols_not_in_self = set(new_polygons_df.columns) - set(self.polygons_df.columns)

            self_cols_not_in_new_polygons_df = set(self.polygons_df.columns) - set(new_polygons_df.columns)

            log.error(f"integrate_new_polygons_df: columns of new_polygons_df arg and self.polygons_df don't agree.")
            
            if new_polygons_df_cols_not_in_self != {}:
                log.error(f"columns that are in new_polygons_df but not in self.polygons_df: {new_polygons_df_cols_not_in_self}")
            
            if self_cols_not_in_new_polygons_df != {}:
                log.error(f"columns that are in self.polygons_df but not in new_polygons_df: {self_cols_not_in_new_polygons_df}")

            raise ValueError(f"integrate_new_polygons_df: columns of new_polygons_df arg and self.polygons_df don't agree.")


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
        self.polygons_df = GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)


    def integrate_new_imgs_df(self, new_imgs_df: gpd.GeoDataFrame):
        """
        Add image data in new_imgs_df to the associator keeping track of which polygons are contained in which images.

        Args:
            new_imgs_df (gdf.GeoDataFrame): GeoDataFrame of image information conforming to the associator's imgs_df format
        """        

        new_imgs_df = deepcopy_gdf(new_imgs_df) #  don't want to modify argument

        self._check_and_adjust_df(
            mode='imgs_df', 
            df=new_imgs_df)

        # Make sure that the coordinate reference systems agree, ...
        if new_imgs_df.crs != self.imgs_df.crs:
            
            log.error(f"integrate_new_imgs_df: crs of new_imgs_df arg doesn't agree with crs of self.imgs_df.")
            
            raise ValueError(f"integrate_new_imgs_df: crs of new_imgs_df arg doesn't agree with crs of self.imgs_df.")

        # ... and that the columns agree.
        if set(new_imgs_df.columns) != set(self.imgs_df.columns):

            new_imgs_df_cols_not_in_self = set(new_imgs_df.columns) - set(self.imgs_df.columns)

            self_cols_not_in_new_imgs_df = set(self.imgs_df.columns) - set(new_imgs_df.columns)

            log.error(f"integrate_new_imgs_df: columns of new_imgs_df arg and self.imgs_df don't agree.")
            
            if new_imgs_df_cols_not_in_self != {}:
                log.error(f"columns that are in new_imgs_df but not in self.imgs_df: {new_imgs_df_cols_not_in_self}")
            
            if self_cols_not_in_new_imgs_df != {}:
                log.error(f"columns that are in self.imgs_df but not in new_imgs_df: {self_cols_not_in_new_imgs_df}")
            
            raise ValueError(f"integrate_new_imgs_df: columns of new_imgs_df arg self.imgs_df don't agree.")

        # go through all new imgs...
        for img_name in new_imgs_df.index:
            
            # ... check if it is already in associator.
            if self._graph.exists_vertex(img_name, 'imgs'): # or: img_name in self.imgs_df.index
                
                # drop row from new_imgs_df, so it won't be in self.imgs_df twice after we concat new_imgs_df to self.imgs_df
                new_imgs_df.drop(img_name, inplace=True) 
                log.info(f"integrate_new_imgs_df: dropping row for {img_name} from input imgs_df since an image with that name is already in the associator!")
                 
            else:
                
                # add new img vertex to the graph, add all connections to existing images, 
                # and modify self.polygons_df 'img_count' value
                img_bounding_rectangle=new_imgs_df.loc[img_name, 'geometry']
                self._add_img_to_graph_modify_polygons_df(
                    img_name, 
                    img_bounding_rectangle=img_bounding_rectangle)

        # append new_imgs_df
        data_frames_list = [self.imgs_df, new_imgs_df]
        self.imgs_df = gpd.GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)


    def drop_polygons(self, polygon_names: Sequence[str]):
        """
        Drop polygons from associator (i.e. remove rows from the associator's polygons_df)

        Args:
            polygon_names (Sequence[str]): polygon_names/identifiers of polygons to be dropped.
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


    def drop_imgs(self, 
        img_names: Sequence[str], 
        remove_imgs_from_disk: bool=True):
        """
        Drop images from associator and dataset, i.e. remove rows from the associator's imgs_df, delete the corresponding vertices in the graph, and delete the image from disk (unless remove_imgs_from_disk is set to False).

        Args:
            img_names (List[str]): img_names/ids of images to be dropped.
            remove_imgs_from_disk (bool): If true, delete images and labels from disk (if they exist).
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
                    (self.data_dir / subdir / img_name).unlink(missing_ok=True)


    def download_missing_imgs_for_polygons_df(self, 
            polygons_df: Optional[GeoDataFrame]=None,
            add_labels: bool=True,
            **kwargs):
        """ 
        Downloads images for polygons.

        Sequentially considers the polygons for which the image count (number of images fully containing a given polygon) is less than num_target_imgs_per_polygon images in the associator's internal polygons_df or the optional polygons_df argument (if given), for each such polygon attempts to download num_target_imgs_per_polygon - image_count images fully containing the polygon (or several images jointly containing the polygon), creates the associated label(s) for the image(s) (assuming the default value True of add_labels is not changed), and integrates the new image(s) into the dataset/associator. If the optional polygons_df argument is provided will append polygons_df to the associator's internal polygons_df. Integrates images downloaded for a polygon into the dataset/associator immediately after downloading them and before downloading images for the next polygon. In particular, the image count is updated immediately after each download. 

        Args:
            num_target_imgs_per_polygon (int): targetted number of images per polygon the download should achieve. The actual number of images for each polygon P that fully contain it could be lower if there are not enough images available or higher if after downloading num_target_imgs_per_polygon images for P P is also contained in images downloaded for other polygons. 
            polygons_df (GeoDataFrame, optional): (Probably just best ignore this) GeoDataFrame of polygons conforming to the associator's format for polygon_df, defaults to the associator's internal polygons_df (i.e. self.polygons_df). If provided and not equal to self.polygons_df will download images for only those polygons and integrate the polygons in polygons_df into the associator after the images have been downloaded. 
            add_labels (bool, optional): bool. Whether to add labels for the downloaded images. Defaults to True.
            shuffle_polygons (bool): Whether to shuffle order of polygons for which images will be downloaded. Might in practice prevent an uneven distribution of the image count for repeated downloads. Defaults to True.
        Returns:
            - None

        Warning:
            It's easy to come up with examples where funny things happen with image count distribution (i.e. distribution of images per polygon) particularly if num_target_imgs_per_polygon is large. These scenarios are not necessarily very likely, but possible. As an example, if one wants to download say 5 images images for a polygon that is not fully contained in any image in the dataset and if there does not exist an image we can download that fully contains it but there are 20 disjoint sets of images we can download that jointly cover the polygon then these 20 disjoint sets will all be downloaded. 
        """

        # Make sure images subdir exists in data_dir
        (self.data_dir / "images").mkdir(parents=True, exist_ok=True)

        # Check if any polygons in polygons_df are already in the associator.
        if polygons_df is not None and polygons_df is not self.polygons_df:
            if (polygons_df.index.isin(self.polygons_df.index)).any() == True:
                log.error(f"download_missing_imgs_for_polygons_df: polygons_df contains polygons already in associator!")
                raise ValueError(f"polygons_df contains polygons already in associator!")
        
        # Default polygons_df to self.polygons_df.
        if polygons_df is None:
            polygons_df = self.polygons_df 
        
        # Check crs.
        assert polygons_df.crs.to_epsg() == self._params_dict['crs_epsg_code'] 
    
        # Dict to keep track of imgs we've downloaded. We'll append this to self.imgs_df as a (geo)dataframe later
        new_imgs_dict = {index_or_col_name: [] for index_or_col_name in [self.imgs_df.index.name] + list(self.imgs_df.columns)}

        polygon_geoms_w_not_enough_imgs : list = list(polygons_df.loc[polygons_df['img_count'] < num_target_imgs_per_polygon].index)
        if shuffle_polygons == True:
            random.shuffle(polygon_geoms_w_not_enough_imgs)
        num_polygon_geoms_w_not_enough_imgs_df = len(polygon_geoms_w_not_enough_imgs)

        # Set of previously downloaded images.
        previously_downloaded_imgs_set = set(self.imgs_df.index) 
        # (Will be used to make sure no attempt is made to download an image more than once.)

        # Go through polygons for which not enough images have been downloaded yet.
        for count, polygon_name in tqdm(enumerate(polygon_geoms_w_not_enough_imgs)): 

            polygon_geometry = self.polygons_df.loc[polygon_name, 'geometry'] 

            log.debug(f"download_missing_imgs_for_polygons_df: considering polygon {polygon_name}.")
            log.info(f"Polygon {count}/{num_polygon_geoms_w_not_enough_imgs_df}")

            # Since we process and connect each image after downloading it, we might not need to download 
            # an image for a polygon that earlier was lacking an image if it is now contained in one of the already downloaded images, so need to check again that there are not enough images for the polygon (since the iterator above is set when it is called and won't know if the "img_count" column value has been changed in the meanwhile).
            num_img_series_to_download = num_target_imgs_per_polygon - polygons_df.loc[polygon_name, "img_count"]
            if num_img_series_to_download <= 0:
                log.debug(f"download_missing_imgs_for_polygons_df: skipping polygon {polygon_name} since there enough images fully containing it.")

                pass
            
            else:

                # Dict of possible keyword args for download function.
                # We use deepcopy here so that a call to download_missing_imgs_for_polygons_df 
                # can not modify self._params_dict.
                temporary_params_dict = copy.deepcopy(self._params_dict) 
                temporary_params_dict.update(kwargs)
                
                while num_img_series_to_download > 0:

                    # Try downloading an image and save returned dict (of dicts) containing information for polygons_df, self.imgs_df...  
                    try:      

                        # DEBUG INFO
                        log.debug(f"attempting to download image for polygon {polygon_name}.")

                        return_dict = self._download_imgs_for_polygon(polygon_name,
                                                    polygon_geometry,  
                                                    self.data_dir,
                                                    previously_downloaded_imgs_set, # _download_imgs_for_polygon should use this to make sure no attempt at downloading an already downloaded image is made.
                                                    **temporary_params_dict)
                    
                    # ... unless either no images could be found or a download error occured, ...
                    except NoImgsForPolygonFoundError as e:

                        # ... in which case we save it in the polygons_df, ...
                        self.polygons_df.loc[polygon_name, 'download_exception'] = repr(e)

                        # ... log a warning, ...
                        log.warning(e, exc_info=True)

                        # ... and break the while loop.
                        break

                    except ImgDownloadError as e:

                        self.polygons_df.loc[polygon_name, 'download_exception'] = repr(e)
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
                                                                                                        self.data_dir, 
                                                                                                        self.data_dir / "images",
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

                                # Finally, remember we downloaded the image. 
                                previously_downloaded_imgs_set.add(img_name)
                                    
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

                                num_img_series_to_download -= 1
                        
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
        Creates GeoTiff pixel labels in the data directory's labels subdirectory for all GeoTiff images in the image subdirectory without a label. 
        """
        self._compare_existing_imgs_to_imgs_df()

        log.info("\nCreating missing labels.\n")

        # Make sure the labels subdir exists in data_dir.
        self._params_dict['labels_dir'].mkdir(parents=True, exist_ok=True)
        
        # Find the set of existing images in the dataset, ...
        existing_images = {img_path.name for img_path in self._params_dict['images_dir'].iterdir()}
        existing_labels = {img_path.name for img_path in self._params_dict['labels_dir'].iterdir()}
        
        # For each image without a label ...
        for img_name in tqdm(existing_images - existing_labels):
        
            # ... create a label.
            self._make_geotif_label(self, img_name, log)


    def print_graph(self):
        """
        Print the associator's internal graph.
        """
        print(self._graph)


    def _connect_img_to_polygon(self,
            img_name : str,
            polygon_name : str,
            contains_or_intersects : Optional[str] = None,
            polygons_df : Optional[GeoDataFrame] = None,
            img_bounding_rectangle : Optional[Polygon] = None,
            graph : Optional[BipartiteGraph] = None):
        """
        Connect an image to a polygon in the graph.
        
        Remember (i.e. create a connection in the graph) whether the image fully contains or just has non-empty intersection with the polygon, i.e. add an edge of the approriate type between the image and the polygon.

        Args:
            img_name (str): Name of image to connect
            polygon_name (str): Name of polygon to connect
            contains_or_intersects (optional, str): Optional connection criteria
            polygons_df (optional, gdf.GeoDataFrame): Optional polygon dataframe
            img_bounding_rectangle (optional, Polygon): polygon decribing image footprint
            graph (optional, BipartiteGraph): optional bipartied graph
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

            # if the polygon is fully contained in the image increment the image counter in self.polygons_df
            if contains_or_intersects == 'contains':
                polygons_df.loc[polygon_name, 'img_count'] += 1


    def _add_polygon_to_graph(self,
            polygon_name : str,
            polygons_df : Optional[GeoDataFrame] = None):
        """
        Connects a polygon to those images in self.imgs_df with which it has non-empty intersection.

        Args:
            polygon_name (str): name/id of polygon to add
            polygons_df (GeoDataFrame, optional): Defaults to None (i.e. self.polygons_df).
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


    def _add_img_to_graph_modify_polygons_df(self,
            img_name: str,
            img_bounding_rectangle: Optional[Polygon]=None,
            polygons_df: Optional[gpd.GeoDataFrame]=None,
            graph: Optional[BipartiteGraph]=None):
        """ 
        Connects an image to all polygons in polygons_df, creating the vertex if necessary. The default value for polygons_df is None, which we take to mean self.polygons_df. If img_bounding_rectangle is None, we assume we can get it from self. If the image already exists and already has connections a warning will be logged. imgs_df.
        
        Args:
            img_name (str): Name of image to add
            img_bounding_rectangle (optional, Polygon): polygon decribing image footprint
            polygons_df (optional, gdf.GeoDataFrame): Optional polygons dataframe
            graph (optional, BipartiteGraph): optional bipartied graph
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


    def _remove_polygon_from_graph_modify_polygons_df(self,
            polygon_name : str,
            set_img_count_to_zero : bool = True):
        """
        Removes a polygon from the graph (i.e. removes the vertex and all incident edges) and (if set_img_count_to_zero == True) sets the polygons_df field 'img_count' to 0.

        Args:
            polygon_name (str): polygon name/id
            set_img_count_to_zero (bool): Whether to set img_count to 0.        
        """

        self._graph.delete_vertex(polygon_name, 'polygons', force_delete_with_edges=True)

        if set_img_count_to_zero==True:
            self.polygons_df.loc[polygon_name, 'img_count' ] = 0


    def _remove_img_from_graph_modify_polygons_df(self, img_name: str):
        """
        Removes an img from the graph (i.e. removes the vertex and all incident edges) and modifies the polygons_df fields 'img_count' for the polygons contained in the image.

        Args:
            img_name (str): name/id of image to remove
        """

        for polygon_name in self.polygons_contained_in_img(img_name):
            self.polygons_df.loc[polygon_name, 'img_count'] -= 1

        self._graph.delete_vertex(img_name, 'imgs', force_delete_with_edges=True)


    def _download_imgs_for_polygon(self,
            polygon_name: str,
            polygon_geometry: Polygon,
            download_dir: Union[str, Path],
            previously_downloaded_imgs_set: Sequence[str],
            **kwargs):
        """
        Not implemented, overwrite/implement in a subclass. Should download an image fully containing a vector polygon or several images jointly containing it and return a dict with information to be updated in the associator, see below for details.

        Args:
            polygon_name (str): the name of the vector polygon.
            polygon_geometry (Polygon): shapely geometry of polygon.
            download_dir (Union[Path, str]): directory that the image file should be downloaded to.
            previously_downloaded_imgs_set (Set[str]): Set of previously downloaded img_names. In some use cases when it can't be guaranteed that an image can be downloaded that fully contains the polygon it can happen that attempts will be made to download an image that is already in the associator. Passing this argument allows the download function to make sure it doesn't try downloading an image that is already in the dataset.
            **kwargs (Any): optional keyword arguments depending on the application.
            
        Returns:
            A dict with a key 'list_img_info_dicts' and value a list of dicts containing the information to be included in each row in the imgs_df of the calling associator, one for each newly downloaded image. The keys should be the index and column names of the imgs_df and the values the indices or entries of those columns in row that will correspond to the new image. We return a dict instead of directly returning the list to be backwards compatible.
        """

        raise NotImplementedError
    

    def _process_downloaded_img_file(self,
            img_name: str,
            in_dir: Union[str, Path],
            out_dir: Union[str, Path],
            convert_to_crs_epsg: int,
            **kwargs):
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


    def _compare_existing_imgs_to_imgs_df(self):
        """
        Safety check that compares the set of images in the data_dir's images subdirectory with the set of images in self.imgs_df

        :raises Exception: if there are images in the dataset's images subdirectory that are not in self.imgs_df.
        """

        # Find the set of existing images in the dataset, ...
        existing_images = {img_path.name for img_path in (self.data_dir / "images").iterdir()}

        # ... then if the set of images is a strict subset of the images in imgs_df ...
        if existing_images < set(self.imgs_df.index):
            
            # ... log a warning 
            log.warning(f"There more images in self.imgs_df that are not in the dataset's images subdirectory.")

        # ... and if it is not a subset, ...
        if not existing_images <= set(self.imgs_df.index):
            
            # ... log an error...
            message = f"There are images in the dataset's images subdirectory that are not in self.imgs_df."
            log.error(message)

            # ...and raise an exception.
            raise Exception(message)