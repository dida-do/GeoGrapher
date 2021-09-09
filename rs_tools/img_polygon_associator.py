"""
The ImgPolygonAssociator class organizes and handles remote sensing datasets.
"""

from __future__ import annotations
import rs_tools
from rs_tools.labels.label_type_conversion_utils import convert_polygons_df_soft_cat_to_cat
from typing import Dict, Tuple, Type, Any, List, TypeVar, Union, Optional, Sequence
from json.decoder import JSONDecodeError
import pathlib
import logging
import json
from pathlib import Path
import geopandas as gpd
from geopandas import GeoDataFrame

from rs_tools.add_drop_imgs_polygons_mixin import AddDropImgsPolygonsMixIn
from rs_tools.labels_mixin import LabelsMixIn
from rs_tools.download_imgs_mixin import DownloadImgsMixIn
from rs_tools.img_polygon_associator_base import ImgPolygonAssociatorBase
from rs_tools.global_constants import STANDARD_CRS_EPSG_CODE, DATA_DIR_SUBDIRS, IMGS_DF_INDEX_NAME, POLYGONS_DF_INDEX_NAME
from rs_tools.graph import BipartiteGraph
from rs_tools.utils.associator_utils import empty_gdf, empty_gdf_same_format_as, empty_graph

INFERRED_PATH_ATTR_FILENAMES = { # attribute self.key will be self._json_path / val
    '_polygons_df_path' : 'polygons_df.geojson',
    '_imgs_df_path' : 'imgs_df.geojson',
    '_params_dict_path' : 'params_dict.json',
    '_graph_path' : "graph.json", 
    '_cut_params_path' : 'cut_params.json', 
}
DEFAULT_ASSOC_DIR_NAME = 'associator' 
DEFAULT_IMAGES_DIR_NAME = 'images'
DEFAULT_LABELS_DIR_NAME = 'labels'
DEFAULT_DOWNLOAD_DIR_NAME = 'downloads'


IPAType = TypeVar('IPAType', bound='ImgPolygonAssociator')


# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)


class ImgPolygonAssociator(AddDropImgsPolygonsMixIn, LabelsMixIn, DownloadImgsMixIn, ImgPolygonAssociatorBase):
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

    - crs_epsg_code: EPSG code of the coordinate reference system (crs) the associator (i.e. the associator's imgs_df and polygons_df) is in. Defaults to 4326 (WGS84). Setting this attribute will automatically set the associator's imgs_df and polygons_df crs's. 
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

        Either all four of the images_dir, labels_dir, assoc_dir, and download_dir args or the data_dir arg should be given (i.e. not None). 

        Args:

            segmentation_classes (Sequence[str]): list of segmentation classes (excluding mask and background classes)
            label_type (str): Label type. One of 'categorical' (default), 'onehot', or 'soft-categorical'
            polygons_df (Optional[GeoDataFrame], optional): polygons_df. Defaults to None.
            polygons_df_cols (Optional[Union[List[str], Dict[str, Type]]], optional): column names (and optionally types) of empty polygons_df to start associator with. Defaults to None.
            imgs_df (Optional[GeoDataFrame], optional): imgs_df. Defaults to None.
            imgs_df_cols (Optional[Union[List[str], Dict[str, Type]]], optional): column names (and optionally types) of empty imgs_df to start associator with. Defaults to None.
            add_background_band_in_labels (bool, optional): Whether to add a background segmentation class band when creating labels. Only relevant for 'one-hot' or 'soft-categorical' labels. Defaults to False. 
            crs_epsg_code (int, optional): EPSG code associator works with. Defaults to STANDARD_CRS_EPSG_CODE
            data_dir (Optional[Union[Path, str]], optional): data directory containing images_dir, labels_dir, assoc_dir. 
            images_dir (Optional[Union[Path, str]], optional): path to directory containing images. 
            labels_dir (Optional[Union[Path, str]], optional): path to directory containing labels. 
            assoc_dir (Optional[Union[Path, str]], optional): path to directory containing (geo)json associator component files.
            download_dir (Optional[Union[Path, str]], optional): path to directory files are downloaded to. Can be used to store e.g. SAFE files for sentinel-2 data. 
            **kwargs (Any): optional keyword args for subclass implementations. 
        """        

        super().__init__()
        
        self._check_dir_args(
            data_dir=data_dir,
            images_dir=images_dir, 
            labels_dir=labels_dir, 
            assoc_dir=assoc_dir, 
            download_dir=download_dir
        )

        # set paths
        self._init_set_paths( 
            data_dir=data_dir, 
            images_dir=images_dir, 
            labels_dir=labels_dir, 
            assoc_dir=assoc_dir
        )

        # Check path args for consistency and see if we're loading the associator from disk or not
        load_from_disk = self._load_from_disk_or_not(
                            polygons_df=polygons_df, 
                            polygons_df_cols=polygons_df_cols, 
                            imgs_df=imgs_df, 
                            imgs_df_cols=imgs_df_cols)

        # build _params_dict from all args except for imgs_df, polygons_df and the corresponding column args
        self._params_dict = {}
        self._params_dict.update( 
            {
                "segmentation_classes" : segmentation_classes, 
                "label_type" : label_type, 
                "add_background_band_in_labels" : add_background_band_in_labels, 
                "crs_epsg_code" : crs_epsg_code, 
                **kwargs
            })

        # get polygons_df and imgs_df
        polygons_df = self._init_get_df(
                        "polygons_df", 
                        load_from_disk, 
                        polygons_df, 
                        polygons_df_cols, 
                        POLYGONS_DF_INDEX_NAME, 
                        crs_epsg_code)
        imgs_df = self._init_get_df(
                        "imgs_df",
                        load_from_disk, 
                        imgs_df, 
                        imgs_df_cols, 
                        IMGS_DF_INDEX_NAME, 
                        crs_epsg_code)
        self._standardize_df_crs(
            df=polygons_df, 
            df_name='polygons_df')
        self._standardize_df_crs(
            df=imgs_df, 
            df_name='imgs_df')

        # run safety checks on polygons_df, imgs_df and adjust format if necessary
        self._check_and_adjust_df_format(
            mode='polygons_df',
            df=polygons_df)
        self._check_and_adjust_df_format(
            mode='imgs_df',
            df=imgs_df)

        # set self.polygons_df, self.imgs_df, self._cut_params_dict
        self._set_remaining_assoc_components(
            load_from_disk=load_from_disk, 
            polygons_df=polygons_df, 
            imgs_df=imgs_df)

        # (attribute names of) all directories containing image data
        self._image_data_dirs = [self.images_dir, self.labels_dir] # in subclass implementation, can add e.g. mask_dir


    def __getattr__(self,  key):
        if key in self.__dict__._params_dict:
            return self.__dict__._params_dict[key]


    @classmethod
    def from_paths(
            cls : Type[IPAType], 
            assoc_dir : Union[Path, str], 
            images_dir : Union[Path, str], 
            labels_dir : Union[Path, str], 
            download_dir : Union[Path, str]
            ) -> IPAType:

        # read args from json
        try:
            params_dict_path = Path(assoc_dir) / INFERRED_PATH_ATTR_FILENAMES['_params_dict_path']
            with open(params_dict_path, "r") as read_file:
                kwargs = json.load(read_file)
        except FileNotFoundError as e:
            log.exception(f"Missing {INFERRED_PATH_ATTR_FILENAMES['_params_dict_path']} file found in {assoc_dir}!")
            raise e
        except JSONDecodeError:
            log.exception(f"The {INFERRED_PATH_ATTR_FILENAMES['_params_dict_path']} file in {assoc_dir} is corrupted!")

        new_assoc = cls(
            assoc_dir=assoc_dir, 
            images_dir=images_dir, 
            labels_dir=labels_dir,
            download_dir=download_dir,
            **kwargs)

        return new_assoc


    @classmethod
    def from_data_dir(
            cls : Type[IPAType], 
            data_dir : Union[Path, str], 
            ) -> IPAType:
        """Initialize and return an associator from a data directory

        Args:
            data_dir (Union[Path, str]): data directory containing 'associator_files', 'images', and 'labels' subdirectories

        Returns:
            IPAType: initialized associator
        """

        data_dir = Path(data_dir)

        images_dir, labels_dir, assoc_dir, download_dir = cls._get_default_dirs_from_data_dir(data_dir)

        assoc = cls.from_paths(
            images_dir=images_dir,
            labels_dir=labels_dir, 
            assoc_dir=assoc_dir, 
            download_dir=download_dir
        )

        return assoc


    @classmethod
    def from_kwargs(cls, **kwargs : Any) -> ImgPolygonAssociator:
        """
        Initialize and return an associator from keyword arguments. 

        Ars:
            **kwargs (Any): keyword arguments, see docstring for __init__ 

        Returns:
            initialized associator
        """
        return cls(**kwargs)


    @property
    def images_dir(self):
        return self._images_dir


    @property
    def labels_dir(self):
        return self._labels_dir


    @property
    def assoc_dir(self):
        return self._assoc_dir

    
    @property
    def download_dir(self):
        return self._download_dir


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
    def label_type(self) -> str:
        """Return label type"""
        return self._params_dict['label_type']

    @label_type.setter
    def label_type(self, new_label_type : str):
        """
        Set label type. 
        
        Warning:
            Does not delete old labels and create new ones.
        """

        self._check_label_type(new_label_type)

        old_label_type = self._params_dict['label_type'] 
        if old_label_type == new_label_type:
            pass
        else:
            if old_label_type == 'soft-categorical' and new_label_type == 'categorical': 
                self.polygons_df = convert_polygons_df_soft_cat_to_cat(self.polygons_df)
            elif old_label_type == 'categorical' and new_label_type == 'soft-categorical': 
                needed_cols = {f"prob_seg_class_{class_}" for class_ in self.all_classes}
                existing_cols = set(self.polygons_df.columns)
                if not needed_cols <= existing_cols:
                    message = f"Can't convert to soft-categorical label_type: Missing columns {existing_cols - needed_cols}"
                    log.error(message)
                    raise ValueError(message)
            elif {old_label_type, new_label_type} == {'categorical', 'onehot'}:
                pass
            else:
                raise ValueError(f"Conversion from label_type from {old_label_type} to {new_label_type} not implemented.")

            log.warning(f"label_type has been changed to {new_label_type}. You might want to delete the old labels and create new ones!")
            self._params_dict['label_type'] = new_label_type
    

    @property
    def all_classes(self):
        """Should include not just the segmentation classes but also e.g. mask or background classes."""
        return self.segmentation_classes + [self.background_class]


    @property
    def image_data_dirs(self) -> List[Path]:
        return self._image_data_dirs


    def save(self):
        """
        Save associator to disk.
        """

        log.info(f"Saving associator to disk...")
        
        # Make sure assoc_dir exists.
        self._assoc_dir.mkdir(parents=True, exist_ok=True)

        self.imgs_df.to_file(Path(self._imgs_df_path), driver="GeoJSON")
        self.polygons_df.to_file(Path(self._polygons_df_path), driver="GeoJSON")
        self._graph.save_to_file(Path(self._graph_path))
        # Save params dict
        with open(self._params_dict_path, "w") as write_file:
            save_params_dict = self._make_dict_json_serializable(self._params_dict) 
            json.dump(
                save_params_dict, 
                write_file)
        # Save cut params dict
        with open(self._cut_params_path, "w") as write_file:
            save_cut_params_dict = self._make_dict_json_serializable(self._cut_params_dict)
            json.dump(
                save_cut_params_dict,
                write_file)


    def empty_assoc_same_format_as(self, 
            data_dir : Optional[Union[Path, str]] = None, 
            assoc_dir : Optional[Union[Path, str]] = None, 
            images_dir : Optional[Union[Path, str]] = None, 
            labels_dir : Optional[Union[Path, str]] = None, # either all three above path args must be set or this one
            ) -> ImgPolygonAssociator:
        """
        Factory method that returns an empty associator of the same format (i.e. columns in polygons_df and imgs_df) as self with data_dir target_data_dir.

        Args:
            data_dir (Optional[Union[Path, str]], optional): data directory containing images_dir, labels_dir, assoc_dir. 
            images_dir (Optional[Union[Path, str]], optional): path to directory containing images. 
            labels_dir (Optional[Union[Path, str]], optional): path to directory containing labels. 
            assoc_dir (Optional[Union[Path, str]], optional): path to directory containing (geo)json associator component files.
            
        Returns: 
            new empty associator
        """

        self._check_dir_args(
            data_dir=data_dir, 
            assoc_dir=assoc_dir, 
            images_dir=images_dir, 
            labels_dir=labels_dir
        )

        if data_dir is not None:
            images_dir, labels_dir, assoc_dir, download_dir = self.__class__._get_default_dirs_from_data_dir(data_dir)

        new_empty_polygons_df = empty_gdf_same_format_as(self.polygons_df)                                        
        new_empty_imgs_df = empty_gdf_same_format_as(self.imgs_df) 

        new_empty_assoc = self.__class__.from_kwargs(
                                
                                # dir args
                                images_dir=images_dir, 
                                labels_dir=labels_dir,
                                assoc_dir=assoc_dir, 
                                download_dir=download_dir,

                                # empty dataframes
                                polygons_df=new_empty_polygons_df, 
                                imgs_df=new_empty_imgs_df, 

                                # remaining kwargs 
                                **self._params_dict)

        return new_empty_assoc
    

    def print_graph(self):
        """
        Print the associator's internal graph.
        """
        print(self._graph)


    def _load_from_disk_or_not(self, 
        polygons_df : Optional[GeoDataFrame], 
        polygons_df_cols : Optional[Union[List[str], Dict[str, Type]]], 
        imgs_df : Optional[GeoDataFrame], 
        imgs_df_cols : Optional[Union[List[str], Dict[str, Type]]],
        ) -> bool:
        """Check path args for consistency and return True if we are loading the associator components from disk."""
        
        polygons_df_from_disk : bool = polygons_df is None and polygons_df_cols is None
        imgs_df_from_disk : bool = imgs_df is None and imgs_df_cols is None

        both_from_disk = polygons_df_from_disk and imgs_df_from_disk
        neither_from_disk = not polygons_df_from_disk and not imgs_df_from_disk

        if (not both_from_disk) and (not neither_from_disk):
            raise ValueError(f"Either _both_ or none of polygons_df and imgs_df should be loaded from disk.")

        return both_from_disk


    def _set_remaining_assoc_components(self, 
            load_from_disk : bool, 
            polygons_df : GeoDataFrame, 
            imgs_df : GeoDataFrame):
        
        # First, set self.polygons_df, self.imgs_df, self._graph. If loading from disk ...
        if load_from_disk:

            # ... read graph from disk ...
            try:
                with open(self._graph_path, "r") as read_file:
                    graph = json.load(read_file)
            except JSONDecodeError:
                log.exception(f"graph.json file in {self._graph_path.parent} corrupted.")
            except FileNotFoundError:
                log.exception(f"Couldn't find graph.json in {self._graph_path.parent}.")
            
            # ... and set remaining components.
            self._graph = BipartiteGraph(file_path=self._graph_path)
            self.polygons_df = polygons_df
            self.imgs_df = imgs_df 

        else:

            # Else, start with an empty associator ...
            self._graph = empty_graph()
            self.polygons_df = empty_gdf_same_format_as(polygons_df)
            self.imgs_df = empty_gdf_same_format_as(imgs_df)
            
            # ... and build it up.
            self.add_to_polygons_df(polygons_df)
            self.add_to_imgs_df(imgs_df)

        # Set cut_params_dict 
        cut_params_path = self._cut_params_path
        if cut_params_path is None:
            self._cut_params_dict = {}
        else:
            # read cut_params_dict from disk            
            try:
                with open(cut_params_path, "r") as read_file:
                    self._cut_params_dict = json.load(read_file)
            except FileNotFoundError:
                log.info(f"No cut_params.json file in {cut_params_path.parent} found!")
                self._cut_params_dict = {}
            except JSONDecodeError:
                log.exception(f"JSON file in {cut_params_path} corrupted!")
                self._cut_params_dict = {}
            except:
                log.exception("An exception occured while reading the cut_params.json file in {cut_params_path.parent}.")
                self._cut_params_dict = {}
        

    def _check_and_adjust_df_format(self, 
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
            
        if mode == 'polygons_df' and df.index.name != POLYGONS_DF_INDEX_NAME:
            raise ValueError(f"polygons_df.index.name is {df.index.name}, should be {POLYGONS_DF_INDEX_NAME}")
        if mode == 'imgs_df' and df.index.name != IMGS_DF_INDEX_NAME:
            raise ValueError(f"imgs_df.index.name is {df.index.name}, should be {IMGS_DF_INDEX_NAME}")


    def _standardize_df_crs(self, 
            df : GeoDataFrame, 
            df_name : str):
        """
        Standardize CRS of dataframe (i.e. set to CRS of associator).

        Args:
            polygons_df (GeoDataFrame): polygons_df
            imgs_df (GeoDataFrame): imgs_df
        """

        if df.crs.to_epsg() != self.crs_epsg_code: # standard crs
            log.warning(f"Transforming {df_name} to crs: EPSG={self.crs_epsg_code}")
            df = df.to_crs(epsg=self.crs_epsg_code)
        

    def _init_get_df(self,
            mode : str, 
            load_from_disk : bool, 
            df : Optional[GeoDataFrame], 
            df_cols : Optional[Union[List[str], Dict[str, Type]]], 
            df_index_name : str,
            crs_epsg_code : int,
            ) -> GeoDataFrame:
        """
        Extract dataframe from arguments. Used during initialization.

        Args:
            mode (str): One of "polygons_df" or "imgs_df"
            load_from_disk (bool): Whether to load from disk
            df (GeoDataFrame, optional): polygons_df or imgs_df
            df_cols (Union[List[str], Dict[str, Type]], optional): list of column names or dict of column names and types or empty dataframe (imgs_df or polygons_df)
            df_index_name (str, optional): index name of empty dataframe (imgs_df or polygons_df)
            crs_epsg_code (int): EPSG code of crs of df to be created, used if df_cols is not None.

        Returns:
            GeoDataFrame: polygons_df or imgs_df
        """

        if mode not in {"polygons_df", "imgs_df"}:
            raise ValueError(f"Unknown mode: {mode}")

        if load_from_disk:
        
            df_json_path = getattr(self, f"_{mode}_path")
            return_df = gpd.read_file(df_json_path)
            return_df.set_index(df_index_name, inplace=True)

        elif df is not None:
        
            return_df = df
        
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
            

    def _check_dir_args(self, 
            data_dir : Union[Path, str], 
            images_dir : Union[Path, str], 
            labels_dir : Union[Path, str],
            assoc_dir : Union[Path, str],             
            download_dir : Union[Path, str] 
            ):

        component_dirs_all_not_None = images_dir is not None and labels_dir is not None and assoc_dir is not None and download_dir is not None

        if not (component_dirs_all_not_None ^ (data_dir is not None)):
            raise ValueError(f"Either the data_dir arg must be given (i.e. not None) or all of the images_dir, labels_dir, assoc_dir args, download_dir.")


    def _init_set_paths(self, 
            data_dir : Union[Path, str], 
            images_dir : Union[Path, str], 
            labels_dir : Union[Path, str],
            assoc_dir : Union[Path, str], 
            download_dir : Union[Path, str]
            ):
        """Set paths to image/label data and associator component files. Used during initialization."""
        
        if data_dir is not None:
            images_dir, labels_dir, assoc_dir, download_dir = self.__class__._get_default_dirs_from_data_dir(data_dir)

        self._images_dir = Path(images_dir)
        self._labels_dir = Path(labels_dir)
        self._assoc_dir = Path(assoc_dir)
        self._download_dir = Path(download_dir)

        # set inferred paths        
        for path_attr, filename in INFERRED_PATH_ATTR_FILENAMES.items():
            setattr(self, path_attr, self._assoc_dir / filename)


    @classmethod
    def _get_default_dirs_from_data_dir(cls, 
            data_dir : Union[Path, str]
            ) -> Tuple[Path, Path, Path]:
        
        data_dir = Path(data_dir)

        images_dir = data_dir / DEFAULT_IMAGES_DIR_NAME
        labels_dir = data_dir / DEFAULT_LABELS_DIR_NAME
        assoc_dir = data_dir / DEFAULT_ASSOC_DIR_NAME
        download_dir = data_dir / DEFAULT_DOWNLOAD_DIR_NAME

        return images_dir, labels_dir, assoc_dir, download_dir


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
        


    
    