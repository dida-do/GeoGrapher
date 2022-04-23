"""
TODO: docstring

The Connector class organizes and handles remote sensing
datasets.
"""

from __future__ import annotations

import json
import logging
import pathlib
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import (Any, List, Literal, Optional, Sequence, Tuple, Type,
                    TypeVar, Union)

import geopandas as gpd
from geopandas import GeoDataFrame

# Mix-in classes:
from rs_tools.add_drop_raster_imgs import AddDropRasterImgsMixIn
from rs_tools.add_drop_vector_features_mixin import AddDropVectorFeaturesMixIn
from rs_tools.graph.bipartite_graph_mixin import BipartiteGraphMixIn

from rs_tools.global_constants import (raster_imgs_INDEX_NAME,
                                       vector_features_INDEX_NAME,
                                       STANDARD_CRS_EPSG_CODE)
from rs_tools.graph import BipartiteGraph
from rs_tools.utils.connector_utils import (empty_gdf,
                                             empty_gdf_same_format_as,
                                             empty_graph)

INFERRED_PATH_ATTR_FILENAMES = {  # attribute self.key will be self.connector_dir / val
    "_vector_features_path": "vector_features.geojson",
    "_raster_imgs_path": "raster_imgs.geojson",
    "attrs_path": "attrs.json",
    "_graph_path": "graph.json",
}
DEFAULT_CONNECTOR_DIR_NAME = "connector"
DEFAULT_IMAGES_DIR_NAME = "images"
DEFAULT_LABELS_DIR_NAME = "labels"
NON_ML_TASK_FEATURE_CLASSES = [
    "background_class"
]  # vector feature classes not to be determined by a machine learning model (e.g. features that define background regions or masks)

ConnectorType = TypeVar("ConnectorType", bound="Connector")

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)


class Connector(
        AddDropVectorFeaturesMixIn,
        AddDropRasterImgsMixIn,
        BipartiteGraphMixIn,  # Needs to be last
):
    """
    The Connector connects vector features and raster data by building up
    a (bipartite) graph defined by the containment and intersection relations
    between the vector features and raster images.

    REWRITE!

    The Connector class can build up, handle, and organize datasets
    consisting of geometry labels (as well as tabular information
    about them in the form of a GeoDataFrame) and remote sensing raster images
    and potentially (semantic) segmentation pixel labels (e.g. GeoTiffs
    or .npy files) (as well as tabular information about the images and pixel labels
    in the form of a GeoDataFrame) by providing a two-way linkage between
    the geometries and the images/pixel labels automatically keeping track of which geometries
     are contained in which images/pixel labels.

    Attributes:

    - vector_features: GeoDataFrame of vector features. Should be indexed
        by unique identifiers (str or int) and contain the following columns:
        - 'geometry': shapely geometry of vector feature (in a standard crs)
        - 'img_count': int. Number of images in the dataset that fully contain the feature geometry.
        - other columns as needed for one's application.

    - raster_imgs: GeoDataFrame containing tabular information about the images. Should be indexed
        by the image names and contain the following columns:
        - 'geometry': shapely.geometry.Polygon. Polygon defining the image bounds (in the connector's standardized crs)
        - 'orig_crs_epsg_code': int. The EPSG code of the crs the georeferenced image is in.
        - other columns as needed for one's application.

    - crs_epsg_code: EPSG code of the coordinate reference system (crs) the connector
    (i.e. the connector's raster_imgs and vector_features) is in. Defaults to 4326 (WGS84). Setting
    this attribute will automatically set the connector's raster_imgs and vector_features crs's.
    """

    # yapf: disable
    def __init__(
        self,
        load_from_disk: bool,

        # args w/o default values
        vector_features: Optional[GeoDataFrame] = None,
        raster_imgs: Optional[GeoDataFrame] = None,

        # remaining non-path args w/ default values
        task_feature_classes: Optional[Sequence[str]] = None,
        background_class: Optional[str] = None,
        crs_epsg_code: int = STANDARD_CRS_EPSG_CODE,

        # path args
        data_dir: Optional[
            Union[Path, str]
        ] = None,  # either this arg or all the path args below must be set (i.e. not None)
        images_dir: Optional[Union[Path, str]] = None,
        labels_dir: Optional[Union[Path, str]] = None,
        connector_dir: Optional[Union[Path, str]] = None,

        # optional kwargs
        **kwargs: Any,

        # yapf: enable
    ):
        """To initialize a new connector use either the from_scratch class
        method or the empty_connector_same_format_as method. To initialize an
        existing connector use the from_data_dir or from_paths class methods.

        Warning:
            Note that many methods that create new dataset from existing ones
            won't work if you use a nonstandard directory format (i.e. set the
            images_dir, labels_dir, connector_dir from hand instead
            of setting the data_dir arg).

        Either all four of the images_dir, labels_dir, and connector_dir args
        or the data_dir arg should be given (i.e. not None).

        Args:

            load_from_disk (bool): whether to load an existing connector from disk or create a new one.
            task_feature_classes (Sequence[str]): list of feature classes for the machine learning task (excluding mask and background classes). Defaults to None, i.e. the single class "object"
            vector_features (Optional[GeoDataFrame], optional): vector_features. Defaults to None, i.e. (if not loading from disk) an empty vector_features.
            raster_imgs (Optional[GeoDataFrame], optional): raster_imgs. Defaults to None, i.e. (if not loading from disk) an empty raster_imgs.
            crs_epsg_code (int, optional): EPSG code connector works with. Defaults to STANDARD_CRS_EPSG_CODE
            data_dir (Optional[Union[Path, str]], optional): data directory containing images_dir, labels_dir, connector_dir.
            images_dir (Optional[Union[Path, str]], optional): path to directory containing images.
            labels_dir (Optional[Union[Path, str]], optional): path to directory containing labels.
            connector_dir (Optional[Union[Path, str]], optional): path to directory containing (geo)json connector component files.
            **kwargs (Any): optional keyword args for subclass implementations.
        """

        super().__init__()

        if task_feature_classes is None:
            task_feature_classes = ["object"]
        self._check_no_non_ml_task_geom_classes_are_task_feature_classes(
            task_feature_classes=task_feature_classes, background_class=background_class, **kwargs
        )

        self._check_dir_args(
            data_dir=data_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            connector_dir=connector_dir,
        )

        # set paths
        self._init_set_paths(
            data_dir=data_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            connector_dir=connector_dir,
        )

        # build attrs from all args except for raster_imgs, vector_features, the corresponding column args, and the path/dir args
        self.attrs = {}
        self.attrs.update(
            {
                "task_feature_classes": task_feature_classes,
                "background_class": background_class,
                "crs_epsg_code": crs_epsg_code,
                **kwargs,
            }
        )

        # get vector_features and raster_imgs
        if load_from_disk:
            vector_features = self._load_df_from_disk("vector_features")
            raster_imgs = self._load_df_from_disk("raster_imgs")
        else:
            if vector_features is None:
                vector_features = self._get_empty_df("vector_features")
            if raster_imgs is None:
                raster_imgs = self._get_empty_df("raster_imgs")

        vector_features = self._get_df_in_crs(df=vector_features, df_name="vector_features", crs_epsg_code=self.crs_epsg_code)
        raster_imgs = self._get_df_in_crs(df=raster_imgs, df_name="raster_imgs", crs_epsg_code=self.crs_epsg_code)

        # set self.vector_features, self.raster_imgs
        self._set_remaining_connector_components(
            load_from_disk=load_from_disk, vector_features=vector_features, raster_imgs=raster_imgs
        )

        # safety checks
        self._check_required_df_cols_exist(df=raster_imgs, df_name='self.raster_imgs', mode='raster_imgs')
        self._check_required_df_cols_exist(df=vector_features, df_name='self.vector_features', mode='vector_features')

        # directories containing image data
        self._image_data_dirs = [
            self.images_dir,
            self.labels_dir,
        ]  # in subclass implementation, can add e.g. mask_dir

    def __getattr__(self, key):
        if key in self.__dict__["attrs"]:
            return self.__dict__["attrs"][key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    @classmethod
    def from_paths(
        cls: Type[ConnectorType],
        connector_dir: Union[Path, str],
        images_dir: Union[Path, str],
        labels_dir: Union[Path, str],
    ) -> ConnectorType:
        """Initialize from paths"""

        # read args from json
        try:
            attrs_path = Path(connector_dir) / INFERRED_PATH_ATTR_FILENAMES["attrs_path"]
            with open(attrs_path, "r") as read_file:
                kwargs = json.load(read_file)
        except FileNotFoundError as exc:
            log.exception(
                "Missing connector file %s found in %s", INFERRED_PATH_ATTR_FILENAMES['attrs_path'], connector_dir)
            raise exc
        except JSONDecodeError:
            log.exception(
                "The %s file in %s is corrupted!", INFERRED_PATH_ATTR_FILENAMES['attrs_path'], connector_dir)

        new_connector = cls(
            load_from_disk=True,
            connector_dir=connector_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            **kwargs,
        )

        return new_connector

    @classmethod
    def from_data_dir(
        cls: Type[ConnectorType],
        data_dir: Union[Path, str],
    ) -> ConnectorType:
        """Initialize and return an connector from a data directory.

        Args:
            data_dir (Union[Path, str]): data directory containing 'connector_files', 'images', and 'labels' subdirectories

        Returns:
            IPAType: initialized connector
        """

        data_dir = Path(data_dir)

        images_dir, labels_dir, connector_dir = cls._get_default_dirs_from_data_dir(
            data_dir
        )

        connector = cls.from_paths(
            images_dir=images_dir,
            labels_dir=labels_dir,
            connector_dir=connector_dir,
        )

        return connector

    @classmethod
    def from_scratch(cls, **kwargs: Any) -> Connector:
        """Initialize and return a new connector from keyword arguments.

        Ars:
            **kwargs (Any): keyword arguments (except load_from_disk), see docstring for __init__

        Returns:
            initialized connector
        """
        kwargs.update({"load_from_disk": False})
        return cls(**kwargs)

    @property
    def images_dir(self):
        return self._images_dir

    @property
    def labels_dir(self):
        return self._labels_dir

    @property
    def connector_dir(self):
        return self._connector_dir

    @property
    def crs_epsg_code(self) -> int:
        """
        int: EPSG code of connector's crs.

        Setting will set connector's raster_imgs and vector_features crs automatically.
        """
        return self.attrs["crs_epsg_code"]

    @crs_epsg_code.setter
    def crs_epsg_code(self, epsg_code: int):
        # set value in params dict
        self.attrs["crs_epsg_code"] = epsg_code

        # reproject raster_imgs and vector_features GeoDataFrames
        self.vector_features = self.vector_features.to_crs(epsg=epsg_code)
        self.raster_imgs = self.raster_imgs.to_crs(epsg=epsg_code)

    @property
    def task_vector_feature_classes(self):
        return self.attrs["task_feature_classes"]

    @task_vector_feature_classes.setter
    def task_vector_feature_classes(self, new_task_feature_classes: List[str]):
        if not len(new_task_feature_classes) == len(set(new_task_feature_classes)):
            raise ValueError("no duplicates in list of task_feature_classes allowed")
        self.attrs["task_feature_classes"] = new_task_feature_classes

    @property
    def all_vector_feature_classes(self):
        """Return all allowed classes in vector_features, including those not related to the ML task (e.g. the background class)"""

        answer = self.task_vector_feature_classes.copy()
        for class_name in NON_ML_TASK_FEATURE_CLASSES:
            class_value = getattr(self, class_name)
            if class_value is not None:
                answer += [class_value]

        return answer

    @property
    def image_data_dirs(self) -> List[Path]:
        return self._image_data_dirs

    def save(self):
        """Save connector to disk."""

        log.info("Saving connector to disk...")

        # Make sure connector_dir exists.
        self._connector_dir.mkdir(parents=True, exist_ok=True)

        raster_imgs_non_geometry_columns = [col for col in self.raster_imgs.columns if col != "geometry"]
        self.raster_imgs[raster_imgs_non_geometry_columns] = self.raster_imgs[
            raster_imgs_non_geometry_columns
        ].convert_dtypes(
            infer_objects=True,
            convert_string=True,
            convert_integer=True,
            convert_boolean=True,
            convert_floating=False,
        )
        self.raster_imgs.index.name = raster_imgs_INDEX_NAME
        self.raster_imgs.to_file(Path(self._raster_imgs_path), driver="GeoJSON")
        self.vector_features.index.name = vector_features_INDEX_NAME
        self.vector_features.to_file(Path(self._vector_features_path), driver="GeoJSON")
        self._graph.save_to_file(Path(self._graph_path))
        # Save params dict
        with open(self.attrs_path, "w") as write_file:
            saveattrs = self._make_dict_json_serializable(self.attrs)
            json.dump(saveattrs, write_file)

    def empty_connector_same_format(
        self,
        data_dir: Optional[
            Union[Path, str]
        ] = None,  # either this arg or all four path args below must be set
        connector_dir: Optional[Union[Path, str]] = None,
        images_dir: Optional[Union[Path, str]] = None,
        labels_dir: Optional[Union[Path, str]] = None,
    ) -> Connector:
        """Factory method that returns an empty connector of the same format
        (i.e. columns in vector_features and raster_imgs) as self with data_dir
        target_data_dir.

        Args:
            data_dir (Optional[Union[Path, str]], optional): data directory containing images_dir, labels_dir, connector_dir.
            images_dir (Optional[Union[Path, str]], optional): path to directory containing images.
            labels_dir (Optional[Union[Path, str]], optional): path to directory containing labels.
            connector_dir (Optional[Union[Path, str]], optional): path to directory containing (geo)json connector component files.

        Returns:
            new empty connector
        """

        self._check_dir_args(
            data_dir=data_dir,
            connector_dir=connector_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
        )

        if data_dir is not None:
            (
                images_dir,
                labels_dir,
                connector_dir,
            ) = self.__class__._get_default_dirs_from_data_dir(data_dir)

        new_empty_vector_features = empty_gdf_same_format_as(self.vector_features)
        new_empty_raster_imgs = empty_gdf_same_format_as(self.raster_imgs)

        new_empty_connector = self.__class__.from_scratch(
            # dir args
            images_dir=images_dir,
            labels_dir=labels_dir,
            connector_dir=connector_dir,
            # empty dataframes
            vector_features=new_empty_vector_features,
            raster_imgs=new_empty_raster_imgs,
            # remaining kwargs
            **self.attrs,
        )

        return new_empty_connector

    def print_graph(self):
        """Print the connector's internal graph."""
        print(self._graph)

    def _get_empty_df(self, df_name: str) -> GeoDataFrame:

        if df_name == "vector_features":
            index_name = vector_features_INDEX_NAME
            cols_and_types = self._get_required_df_cols_and_types("vector_features")
        elif df_name == "raster_imgs":
            index_name = raster_imgs_INDEX_NAME
            cols_and_types = self._get_required_df_cols_and_types("raster_imgs")

        df = empty_gdf(
            index_name=index_name, cols_and_types=cols_and_types, crs_epsg_code=self.crs_epsg_code
        )

        return df

    def _get_required_df_cols_and_types(self, df_name: str) -> dict:

        # type of "geometry" column is ignored
        if df_name == "vector_features":
            cols_and_types = {"geometry": None, "img_count": int}
        elif df_name == "raster_imgs":
            cols_and_types = {"geometry": None}

        return cols_and_types

    def _set_remaining_connector_components(
        self, load_from_disk: bool, vector_features: GeoDataFrame, raster_imgs: GeoDataFrame
    ):

        if load_from_disk:

            self._graph = BipartiteGraph(file_path=self._graph_path)
            self.vector_features = vector_features
            self.raster_imgs = raster_imgs

        else:

            self._graph = empty_graph()
            self.vector_features = empty_gdf_same_format_as(vector_features)
            self.raster_imgs = empty_gdf_same_format_as(raster_imgs)

            self.add_to_vector_features(vector_features)
            self.add_to_raster_imgs(raster_imgs)


    def _check_required_df_cols_exist(
            self, df: GeoDataFrame, df_name: str,
            mode: Literal["vector_features", "raster_imgs"]) -> bool:
        """Check if required columns exist."""

        required_cols = list(self._get_required_df_cols_and_types(df_name).keys())

        if not set(required_cols) <= set(df.columns):

            missing_cols = set(required_cols) - set(df.columns)
            raise ValueError(
                f"{df_name} is missing required columns: {', '.join(missing_cols)}"
            )

    def _get_df_in_crs(self, df: GeoDataFrame, df_name: str, crs_epsg_code: int):
        """Standardize CRS of dataframe (i.e. set to CRS of connector).

        Args:
            vector_features (GeoDataFrame): vector_features
            raster_imgs (GeoDataFrame): raster_imgs
        """

        if df.crs.to_epsg() != crs_epsg_code:
            log.info("Transforming %s to crs: EPSG=%s", df_name, crs_epsg_code)
            return df.to_crs(epsg=crs_epsg_code)
        else:
            # faster than unnecessary df.to_crs(epsg=crs_epsg_code)
            return df

    def _load_df_from_disk(self, df_name: str) -> GeoDataFrame:
        """Load vector_features or raster_imgs from disk."""

        if df_name == "vector_features":
            df_index_name = vector_features_INDEX_NAME
        elif df_name == "raster_imgs":
            df_index_name = raster_imgs_INDEX_NAME

        df_json_path = getattr(self, f"_{df_name}_path")
        return_df = gpd.read_file(df_json_path)
        return_df.set_index(df_index_name, inplace=True)

        return return_df

    def _check_dir_args(
        self,
        data_dir: Union[Path, str],
        images_dir: Union[Path, str],
        labels_dir: Union[Path, str],
        connector_dir: Union[Path, str],
    ):

        component_dirs_all_not_None = (
            images_dir is not None
            and labels_dir is not None
            and connector_dir is not None
        )

        if not (component_dirs_all_not_None ^ (data_dir is not None)):
            raise ValueError(
                "Either the data_dir arg must be given (i.e. not None) or all of the images_dir, labels_dir, and connector_dir args."
            )

    def _init_set_paths(
        self,
        data_dir: Union[Path, str],
        images_dir: Union[Path, str],
        labels_dir: Union[Path, str],
        connector_dir: Union[Path, str],
    ):
        """Set paths to image/label data and connector component files.

        Used during initialization.
        """

        if data_dir is not None:
            (
                images_dir,
                labels_dir,
                connector_dir,
            ) = self.__class__._get_default_dirs_from_data_dir(data_dir)

        self._images_dir = Path(images_dir)
        self._labels_dir = Path(labels_dir)
        self._connector_dir = Path(connector_dir)

        # set inferred paths
        for path_attr, filename in INFERRED_PATH_ATTR_FILENAMES.items():
            setattr(self, path_attr, self._connector_dir / filename)

    @classmethod
    def _get_default_dirs_from_data_dir(cls, data_dir: Union[Path, str]) -> Tuple[Path, Path, Path]:

        data_dir = Path(data_dir)

        images_dir = data_dir / DEFAULT_IMAGES_DIR_NAME
        labels_dir = data_dir / DEFAULT_LABELS_DIR_NAME
        connector_dir = data_dir / DEFAULT_CONNECTOR_DIR_NAME

        return images_dir, labels_dir, connector_dir

    @staticmethod
    def _make_dict_json_serializable(input_dict: dict) -> dict:
        """Make dict serializable as JSON by replacing Path with strings.

        Args:
            input_dict (dict): input dict with keys strings and values of arbitrary type

        Returns:
            dict:  dict with non-serializable values replaced by serializable ones (just Path -> str, for now)
        """

        def make_val_serializable(val):
            return str(val) if isinstance(val, pathlib.PurePath) else val

        serializable_dict = {key: make_val_serializable(val) for key, val in input_dict.items()}

        return serializable_dict

    @staticmethod
    def _check_no_non_ml_task_geom_classes_are_task_feature_classes(
        task_feature_classes: List[str],
        background_class: str, **kwargs
    ):
        """TODO.

        Args:
            task_feature_classes (List[str]): [description]
            background_class (str): [description]
        """

        if not len(task_feature_classes) == len(set(task_feature_classes)):
            raise ValueError("task_feature_classes list contains duplicates.")

        non_task_feature_classes = {"background_class": background_class}
        for key, val in kwargs.items():
            if (
                key in NON_ML_TASK_FEATURE_CLASSES
                and val is not None
                and val not in non_task_feature_classes
            ):
                non_task_feature_classes[key] = val

        if not set(non_task_feature_classes.values()) & set(task_feature_classes) == set():
            bad_values = {
                class_name: value
                for class_name, value in non_task_feature_classes.items()
                if value in set(task_feature_classes)
            }
            raise ValueError(
                f"The following task_feature_classes are also classes unrelated to the machine learning task: {bad_values}"
            )
