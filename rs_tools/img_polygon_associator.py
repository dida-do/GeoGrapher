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

from rs_tools.global_constants import STANDARD_CRS_EPSG_CODE, IMGS_DF_INDEX_NAME, POLYGONS_DF_INDEX_NAME
from rs_tools.graph import BipartiteGraph
from rs_tools.utils.associator_utils import empty_gdf, empty_gdf_same_format_as, empty_graph
# Base class:
from rs_tools.img_polygon_associator_base import ImgPolygonAssociatorBase
# Mix-in classes:
from rs_tools.add_drop_imgs_polygons_mixin import AddDropImgsPolygonsMixIn
from rs_tools.labels_mixin import LabelsMixIn
from rs_tools.img_download.download_imgs_mixin import DownloadImgsBaseMixIn
from rs_tools.img_download import Sentinel2DownloaderMixIn, JAXADownloaderMixIn
from rs_tools.convert_dataset import CreateDSCombineRemoveSegClassesMixIn, CreateDSTiffToNpyMixIn, CreateDSCategoricalFromSoftCategoricalDatasetMixIn
from rs_tools.cut import CreateDSCutImgsAroundEveryPolygonMixIn, CreateDSCutEveryImgToGridMixIn, CreateDSCutIterOverImgsMixIn, CreateDSCutIterOverPolygonsMixIn
from rs_tools.update_from_source_dataset_mixin import UpdateFromSourceDSMixIn


INFERRED_PATH_ATTR_FILENAMES = { # attribute self.key will be self._json_path / val
    '_polygons_df_path' : 'polygons_df.geojson',
    '_imgs_df_path' : 'imgs_df.geojson',
    '_params_dict_path' : 'params_dict.json',
    '_graph_path' : "graph.json",
    '_update_from_source_dataset_dict_path' : 'update_from_source_dataset_dict.json',
}
DEFAULT_ASSOC_DIR_NAME = 'associator'
DEFAULT_IMAGES_DIR_NAME = 'images'
DEFAULT_LABELS_DIR_NAME = 'labels'
DEFAULT_DOWNLOAD_DIR_NAME = 'downloads'
NON_SEGMENTATION_POLYGON_CLASSES = ['background_class'] # polygon types that are not segmentation classes (e.g. polygons that define background regions or masks)


IPAType = TypeVar('IPAType', bound='ImgPolygonAssociator')


# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)


class ImgPolygonAssociator(
        AddDropImgsPolygonsMixIn,
        DownloadImgsBaseMixIn, # needs to before any Downloader mix ins
        Sentinel2DownloaderMixIn,
        JAXADownloaderMixIn,
        UpdateFromSourceDSMixIn, # Needs to be before any of the CreateDS mix ins
        CreateDSCombineRemoveSegClassesMixIn,
        CreateDSCategoricalFromSoftCategoricalDatasetMixIn,
        CreateDSTiffToNpyMixIn,
        CreateDSCutImgsAroundEveryPolygonMixIn,
        CreateDSCutEveryImgToGridMixIn,
        CreateDSCutIterOverImgsMixIn,
        CreateDSCutIterOverPolygonsMixIn,
        LabelsMixIn,
        ImgPolygonAssociatorBase): # needs to be last
    """
    Organize, build up and handle remote sensing datasets consisting of shapely polygons and images/labels.

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

        load_from_disk : bool,

        # args w/o default values
        segmentation_classes : Sequence[str],
        label_type : str,

        polygons_df : Optional[GeoDataFrame] = None,
        imgs_df : Optional[GeoDataFrame] = None,

        # remaining non-path args w/ default values
        background_class : Optional[str] = None,
        add_background_band_in_labels : bool = True,
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
        To initialize a new associator use either the from_scratch class method
        or the empty_assoc_same_format_as method. To initialize an existing
        associator use the from_data_dir or from_paths class methods.

        Warning:
            Note that many methods that create new dataset from existing ones
            won't work if you use a nonstandard directory format (i.e. set the
            images_dir, labels_dir, assoc_dir, download_dir from hand instead
            of setting the data_dir arg).

        Either all four of the images_dir, labels_dir, assoc_dir, and download_dir args or the data_dir arg should be given (i.e. not None).

        Args:

            load_from_disk (bool): whether to load an existing associator from disk or create a new one.
            segmentation_classes (Sequence[str]): list of segmentation classes (excluding mask and background classes)
            label_type (str): Label type. One of 'categorical' (default), 'onehot', or 'soft-categorical'
            polygons_df (Optional[GeoDataFrame], optional): polygons_df. Defaults to None, i.e. (if not loading from disk) an empty polygons_df.
            imgs_df (Optional[GeoDataFrame], optional): imgs_df. Defaults to None, i.e. (if not loading from disk) an empty imgs_df.
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

        self._check_no_non_segmentation_polygon_classes_are_segmentation_classes(
            segmentation_classes=segmentation_classes,
            background_class=background_class,
            **kwargs
        )

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
            assoc_dir=assoc_dir,
            download_dir=download_dir
        )

        # build _params_dict from all args except for imgs_df, polygons_df, the corresponding column args, and the path/dir args
        self._params_dict = {}
        self._params_dict.update(
            {
                "segmentation_classes" : segmentation_classes,
                "label_type" : label_type,
                "background_class" : background_class,
                "add_background_band_in_labels" : add_background_band_in_labels,
                "crs_epsg_code" : crs_epsg_code,
                **kwargs
            })

        # get polygons_df and imgs_df
        if load_from_disk:
            polygons_df = self._load_df_from_disk('polygons_df')
            imgs_df = self._load_df_from_disk('imgs_df')
        else:
            if polygons_df is None:
                polygons_df = self._get_empty_df('polygons_df')
            if imgs_df is None:
                imgs_df = self._get_empty_df('imgs_df')

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

        # set self.polygons_df, self.imgs_df, self._update_from_source_dataset_dict
        self._set_remaining_assoc_components(
            load_from_disk=load_from_disk,
            polygons_df=polygons_df,
            imgs_df=imgs_df)

        # safety check
        self._check_classes_in_polygons_df_contained_in_all_classes()

        # directories containing image data
        self._image_data_dirs = [self.images_dir, self.labels_dir] # in subclass implementation, can add e.g. mask_dir


    def __getattr__(self,  key):
        if key in self.__dict__['_params_dict']:
            return self.__dict__['_params_dict'][key]
        else:
            raise AttributeError(f"No such attribute: {key}")


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
            log.exception(f"Missing associator file {INFERRED_PATH_ATTR_FILENAMES['_params_dict_path']} found in {assoc_dir}!")
            raise e
        except JSONDecodeError:
            log.exception(f"The {INFERRED_PATH_ATTR_FILENAMES['_params_dict_path']} file in {assoc_dir} is corrupted!")

        new_assoc = cls(
            load_from_disk=True,
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
    def from_scratch(cls, **kwargs : Any) -> ImgPolygonAssociator:
        """
        Initialize and return a new associator from keyword arguments.

        Ars:
            **kwargs (Any): keyword arguments (except load_from_disk), see docstring for __init__

        Returns:
            initialized associator
        """
        kwargs.update({'load_from_disk': False})
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
                needed_cols = {f"prob_seg_class_{class_}" for class_ in self.all_polygon_classes}
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
    def all_polygon_classes(self):
        """Should include not just the segmentation classes but also e.g. mask or background classes."""

        answer = self.segmentation_classes.copy()
        for class_name in NON_SEGMENTATION_POLYGON_CLASSES:
            class_value = getattr(self, class_name)
            if class_value is not None:
                answer += [class_value]

        return answer


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

        self.imgs_df.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=False).to_file(Path(self._imgs_df_path), driver="GeoJSON")
        self.polygons_df.to_file(Path(self._polygons_df_path), driver="GeoJSON")
        self._graph.save_to_file(Path(self._graph_path))
        # Save params dict
        with open(self._params_dict_path, "w") as write_file:
            save_params_dict = self._make_dict_json_serializable(self._params_dict)
            json.dump(
                save_params_dict,
                write_file)
        # Save update_from_source_dict
        with open(self._update_from_source_dataset_dict_path, "w") as write_file:
            save_update_dict = self._make_dict_json_serializable(self._update_from_source_dataset_dict)
            json.dump(
                save_update_dict,
                write_file)


    def empty_assoc_same_format_as(self,
            data_dir : Optional[Union[Path, str]] = None, # either this arg or all four path args below must be set
            assoc_dir : Optional[Union[Path, str]] = None,
            images_dir : Optional[Union[Path, str]] = None,
            labels_dir : Optional[Union[Path, str]] = None,
            download_dir : Optional[Union[Path, str]] = None,
            copy_update_from_source_dataset_dict : bool = False
            ) -> ImgPolygonAssociator:
        """
        Factory method that returns an empty associator of the same format (i.e. columns in polygons_df and imgs_df) as self with data_dir target_data_dir.

        Args:
            data_dir (Optional[Union[Path, str]], optional): data directory containing images_dir, labels_dir, assoc_dir.
            images_dir (Optional[Union[Path, str]], optional): path to directory containing images.
            labels_dir (Optional[Union[Path, str]], optional): path to directory containing labels.
            assoc_dir (Optional[Union[Path, str]], optional): path to directory containing (geo)json associator component files.
            download_dir (Optional[Union[Path, str]], optional): path to download directory.
            copy_update_from_source_dataset_dict (bool): Whether to copy self._update_from_source_dataset_dict. Defaults to False

        Returns:
            new empty associator
        """

        self._check_dir_args(
            data_dir=data_dir,
            assoc_dir=assoc_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            download_dir=download_dir
        )

        if data_dir is not None:
            images_dir, labels_dir, assoc_dir, download_dir = self.__class__._get_default_dirs_from_data_dir(data_dir)

        new_empty_polygons_df = empty_gdf_same_format_as(self.polygons_df)
        new_empty_imgs_df = empty_gdf_same_format_as(self.imgs_df)

        new_empty_assoc = self.__class__.from_scratch(

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

        if copy_update_from_source_dataset_dict:
            new_empty_assoc._update_from_source_dataset_dict = self._update_from_source_dataset_dict

        return new_empty_assoc


    def print_graph(self):
        """
        Print the associator's internal graph.
        """
        print(self._graph)


    def _get_empty_df(self, df_name : str) -> GeoDataFrame:

        if df_name == 'polygons_df':
            index_name = POLYGONS_DF_INDEX_NAME
            cols_and_types = self._get_required_df_cols_and_types('polygons_df')
        elif df_name == 'imgs_df':
            index_name = IMGS_DF_INDEX_NAME
            cols_and_types = self._get_required_df_cols_and_types('imgs_df')

        df = empty_gdf(
                index_name=index_name,
                cols_and_types=cols_and_types,
                crs_epsg_code=self.crs_epsg_code)

        return df


    def _get_required_df_cols_and_types(self, df_name : str) -> dict:

        if df_name == 'polygons_df':

            if self.label_type in {'categorical', 'onehot'}:
                cols_and_types = {
                    'geometry' : None, # not used
                    'img_count' : int,
                    'type' : str
                }
            elif self.label_type == 'soft-categorical':
                cols_and_types = {
                    'geometry' : None, # type ignored by empty_gdf
                    'img_count' : int,
                    **{f"prob_seg_class_{class_}" : float for class_ in self.all_polygon_classes}
                }

        elif df_name == 'imgs_df':
                cols_and_types = {'geometry' : None} # type ignored by empty_gdf

        return cols_and_types


    def _get_required_df_cols(self, df_name : str) -> List[str]:
        return list(self._get_required_df_cols_and_types(df_name).keys())


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
        cut_params_path = self._update_from_source_dataset_dict_path
        if cut_params_path is None:
            self._update_from_source_dataset_dict = {}
        else:
            # read cut_params_dict from disk
            try:
                with open(cut_params_path, "r") as read_file:
                    self._update_from_source_dataset_dict = json.load(read_file)
            except FileNotFoundError:
                log.info(f"No cut_params.json file in {cut_params_path.parent} found!")
                self._update_from_source_dataset_dict = {}
            except JSONDecodeError:
                log.exception(f"JSON file in {cut_params_path} corrupted!")
                self._update_from_source_dataset_dict = {}
            except:
                log.exception("An exception occured while reading the cut_params.json file in {cut_params_path.parent}.")
                self._update_from_source_dataset_dict = {}


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


    def _load_df_from_disk(self,
            df_name : str
            ) -> GeoDataFrame:
        """Load polygons_df or imgs_df from disk"""

        if df_name == 'polygons_df':
            df_index_name = POLYGONS_DF_INDEX_NAME
        elif df_name == 'imgs_df':
            df_index_name = IMGS_DF_INDEX_NAME

        df_json_path = getattr(self, f"_{df_name}_path")
        return_df = gpd.read_file(df_json_path)
        return_df.set_index(df_index_name, inplace=True)

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

    @staticmethod
    def _check_no_non_segmentation_polygon_classes_are_segmentation_classes(
            segmentation_classes : List[str],
            background_class : str,
            **kwargs):
        """
        TODO

        Args:
            segmentation_classes (List[str]): [description]
            background_class (str): [description]
        """

        if not len(segmentation_classes) == len(set(segmentation_classes)):
            raise ValueError("segmentation_classes list contains duplicates.")

        non_segmentation_classes = {'background_class': background_class}
        for key, val in kwargs.items():
            if key in NON_SEGMENTATION_POLYGON_CLASSES and val is not None and val not in non_segmentation_classes:
                non_segmentation_classes[key] = val

        if not set(non_segmentation_classes.values()) & set(segmentation_classes) == set():
            bad_values = {class_name: value for class_name, value in non_segmentation_classes.items() if value in set(segmentation_classes)}
            raise ValueError(f"No non-segmentation polygon classes should be segmentation classes, but the following are: {bad_values}")


    def _check_classes_in_polygons_df_contained_in_all_classes(self,
            polygons_df : Optional[GeoDataFrame] = None,
            polygons_df_name : Optional[str] = None):
        """Check TODO"""

        if polygons_df is None:
            polygons_df = self.polygons_df
            polygons_df_name = 'self.polygons_df'
        elif polygons_df_name is None:
            ValueError(f"If the polygons_df argument is given, so should the polygons_df_name.")

        if self.label_type in {'categorical', 'onehot'}:
            polygon_classes_in_polygons_df = set(polygons_df['type'].unique())
        elif self.label_type == 'soft-categorical':
            polygon_classes_in_polygons_df = {col_name[15:] for col_name in polygons_df.columns if col_name.startswith('prob_seg_class_')}
        else:
            raise Exception(f"Unknown label_type: {self.label_type}")

        if not polygon_classes_in_polygons_df <= set(self.all_polygon_classes):
            raise ValueError(f"Unrecognized polygon classes in {polygons_df_name}: {polygon_classes_in_polygons_df - set(self.all_polygon_classes)}")
