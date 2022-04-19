"""
TODO: docstring: labels

The ImgPolygonAssociator class organizes and handles remote sensing
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
from rs_tools.add_drop_imgs_polygons_mixin import AddDropImgsPolygonsMixIn
from rs_tools.graph.bipartite_graph_mixin import BipartiteGraphMixIn

from rs_tools.global_constants import (IMGS_DF_INDEX_NAME,
                                       POLYGONS_DF_INDEX_NAME,
                                       STANDARD_CRS_EPSG_CODE)
from rs_tools.graph import BipartiteGraph
from rs_tools.utils.associator_utils import (empty_gdf,
                                             empty_gdf_same_format_as,
                                             empty_graph)

INFERRED_PATH_ATTR_FILENAMES = {  # attribute self.key will be self.assoc_dir / val
    "_polygons_df_path": "polygons_df.geojson",
    "_imgs_df_path": "imgs_df.geojson",
    "attrs_path": "attrs.json",
    "_graph_path": "graph.json",
}
DEFAULT_ASSOC_DIR_NAME = "associator"
DEFAULT_IMAGES_DIR_NAME = "images"
DEFAULT_LABELS_DIR_NAME = "labels"
NON_SEGMENTATION_POLYGON_CLASSES = [
    "background_class"
]  # polygon types that are not segmentation classes (e.g. polygons that define background regions or masks)

IPAType = TypeVar("IPAType", bound="ImgPolygonAssociator")

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)


class ImgPolygonAssociator(
        AddDropImgsPolygonsMixIn,
        BipartiteGraphMixIn,  # Needs to be last
):
    """Organize, build up and handle remote sensing datasets consisting of
    shapely polygons and images/labels.

    The ImgPolygonAssociator class can build up, handle, and organize datasets
    consisting of shapely vector polygon labels (as well as tabular information
    about them in the form of a GeoDataFrame) and remote sensing raster images
    and potentially (semantic) segmentation pixel labels (e.g. GeoTiffs
    or .npy files) (as well as tabular information about the images and pixel labels
    in the form of a GeoDataFrame) by providing a two-way linkage between
    the polygons and the images/pixel labels automatically keeping track of which polygons
     are contained in which images/pixel labels.

    Attributes:

    - polygons_df: GeoDataFrame containing the vector polygon labels. Should be indexed
        by unique identifiers (str or int) for the polygons and contain the following columns:
        - 'geometry': shapely.geometry.Polygon. The vector polygon label (in a standard crs)
        - 'img_count': int. Number of images in the dataset that fully contain the polygon.
        - other columns as needed for one's application.

    - imgs_df: GeoDatFrame containing tabular information about the images. Should be indexed
        by the image names and contain the following columns:
        - 'geometry': shapely.geometry.Polygon. Polygon defining the image bounds (in the associator's standardized crs)
        - 'orig_crs_epsg_code': int. The EPSG code of the crs the georeferenced image is in.
        - other columns as needed for one's application.

    - crs_epsg_code: EPSG code of the coordinate reference system (crs) the associator
    (i.e. the associator's imgs_df and polygons_df) is in. Defaults to 4326 (WGS84). Setting
    this attribute will automatically set the associator's imgs_df and polygons_df crs's.
    """

    # yapf: disable
    def __init__(
        self,
        load_from_disk: bool,

        # args w/o default values
        segmentation_classes: Sequence[str],
        polygons_df: Optional[GeoDataFrame] = None,
        imgs_df: Optional[GeoDataFrame] = None,

        # remaining non-path args w/ default values
        background_class: Optional[str] = None,
        crs_epsg_code: int = STANDARD_CRS_EPSG_CODE,

        # path args
        data_dir: Optional[
            Union[Path, str]
        ] = None,  # either this arg or all the path args below must be set (i.e. not None)
        images_dir: Optional[Union[Path, str]] = None,
        labels_dir: Optional[Union[Path, str]] = None,
        assoc_dir: Optional[Union[Path, str]] = None,

        # optional kwargs
        **kwargs: Any,

        # yapf: enable
    ):
        """To initialize a new associator use either the from_scratch class
        method or the empty_assoc_same_format_as method. To initialize an
        existing associator use the from_data_dir or from_paths class methods.

        Warning:
            Note that many methods that create new dataset from existing ones
            won't work if you use a nonstandard directory format (i.e. set the
            images_dir, labels_dir, assoc_dir from hand instead
            of setting the data_dir arg).

        Either all four of the images_dir, labels_dir, and assoc_dir args
        or the data_dir arg should be given (i.e. not None).

        Args:

            load_from_disk (bool): whether to load an existing associator from disk or create a new one.
            segmentation_classes (Sequence[str]): list of segmentation classes (excluding mask and background classes)
            polygons_df (Optional[GeoDataFrame], optional): polygons_df. Defaults to None, i.e. (if not loading from disk) an empty polygons_df.
            imgs_df (Optional[GeoDataFrame], optional): imgs_df. Defaults to None, i.e. (if not loading from disk) an empty imgs_df.
            crs_epsg_code (int, optional): EPSG code associator works with. Defaults to STANDARD_CRS_EPSG_CODE
            data_dir (Optional[Union[Path, str]], optional): data directory containing images_dir, labels_dir, assoc_dir.
            images_dir (Optional[Union[Path, str]], optional): path to directory containing images.
            labels_dir (Optional[Union[Path, str]], optional): path to directory containing labels.
            assoc_dir (Optional[Union[Path, str]], optional): path to directory containing (geo)json associator component files.
            **kwargs (Any): optional keyword args for subclass implementations.
        """

        super().__init__()

        self._check_no_non_segmentation_polygon_classes_are_segmentation_classes(
            segmentation_classes=segmentation_classes, background_class=background_class, **kwargs
        )

        self._check_dir_args(
            data_dir=data_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            assoc_dir=assoc_dir,
        )

        # set paths
        self._init_set_paths(
            data_dir=data_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            assoc_dir=assoc_dir,
        )

        # build attrs from all args except for imgs_df, polygons_df, the corresponding column args, and the path/dir args
        self.attrs = {}
        self.attrs.update(
            {
                "segmentation_classes": segmentation_classes,
                "background_class": background_class,
                "crs_epsg_code": crs_epsg_code,
                **kwargs,
            }
        )

        # get polygons_df and imgs_df
        if load_from_disk:
            polygons_df = self._load_df_from_disk("polygons_df")
            imgs_df = self._load_df_from_disk("imgs_df")
        else:
            if polygons_df is None:
                polygons_df = self._get_empty_df("polygons_df")
            if imgs_df is None:
                imgs_df = self._get_empty_df("imgs_df")

        self._standardize_df_crs(df=polygons_df, df_name="polygons_df")
        self._standardize_df_crs(df=imgs_df, df_name="imgs_df")

        # set self.polygons_df, self.imgs_df
        self._set_remaining_assoc_components(
            load_from_disk=load_from_disk, polygons_df=polygons_df, imgs_df=imgs_df
        )

        # safety checks
        self._check_required_df_cols_exist(df=imgs_df, df_name='self.imgs_df', mode='imgs_df')
        self._check_required_df_cols_exist(df=polygons_df, df_name='self.polygons_df', mode='polygons_df')

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
        cls: Type[IPAType],
        assoc_dir: Union[Path, str],
        images_dir: Union[Path, str],
        labels_dir: Union[Path, str],
    ) -> IPAType:
        """Initialize from paths"""

        # read args from json
        try:
            attrs_path = Path(assoc_dir) / INFERRED_PATH_ATTR_FILENAMES["attrs_path"]
            with open(attrs_path, "r") as read_file:
                kwargs = json.load(read_file)
        except FileNotFoundError as exc:
            log.exception(
                "Missing associator file %s found in %s", INFERRED_PATH_ATTR_FILENAMES['attrs_path'], assoc_dir)
            raise exc
        except JSONDecodeError:
            log.exception(
                "The %s file in %s is corrupted!", INFERRED_PATH_ATTR_FILENAMES['attrs_path'], assoc_dir)

        new_assoc = cls(
            load_from_disk=True,
            assoc_dir=assoc_dir,
            images_dir=images_dir,
            labels_dir=labels_dir,
            **kwargs,
        )

        return new_assoc

    @classmethod
    def from_data_dir(
        cls: Type[IPAType],
        data_dir: Union[Path, str],
    ) -> IPAType:
        """Initialize and return an associator from a data directory.

        Args:
            data_dir (Union[Path, str]): data directory containing 'associator_files', 'images', and 'labels' subdirectories

        Returns:
            IPAType: initialized associator
        """

        data_dir = Path(data_dir)

        images_dir, labels_dir, assoc_dir = cls._get_default_dirs_from_data_dir(
            data_dir
        )

        assoc = cls.from_paths(
            images_dir=images_dir,
            labels_dir=labels_dir,
            assoc_dir=assoc_dir,
        )

        return assoc

    @classmethod
    def from_scratch(cls, **kwargs: Any) -> ImgPolygonAssociator:
        """Initialize and return a new associator from keyword arguments.

        Ars:
            **kwargs (Any): keyword arguments (except load_from_disk), see docstring for __init__

        Returns:
            initialized associator
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
    def assoc_dir(self):
        return self._assoc_dir

    @property
    def crs_epsg_code(self) -> int:
        """
        int: EPSG code of associator's crs.

        Setting will set associator's imgs_df and polygons_df crs automatically.
        """
        return self.attrs["crs_epsg_code"]

    @crs_epsg_code.setter
    def crs_epsg_code(self, epsg_code: int):
        # set value in params dict
        self.attrs["crs_epsg_code"] = epsg_code

        # reproject imgs_df and polygons_df GeoDataFrames
        self.polygons_df = self.polygons_df.to_crs(epsg=epsg_code)
        self.imgs_df = self.imgs_df.to_crs(epsg=epsg_code)

    @property
    def segmentation_classes(self):
        return self.attrs["segmentation_classes"]

    @segmentation_classes.setter
    def segmentation_classes(self, new_segmentation_classes: List[str]):
        if not len(new_segmentation_classes) == len(set(new_segmentation_classes)):
            raise ValueError("no duplicates in list of segmentation_classes allowed")
        self.attrs["segmentation_classes"] = new_segmentation_classes

    @property
    def all_polygon_classes(self):
        """Should include not just the segmentation classes but also e.g. mask
        or background classes."""

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
        """Save associator to disk."""

        log.info("Saving associator to disk...")

        # Make sure assoc_dir exists.
        self._assoc_dir.mkdir(parents=True, exist_ok=True)

        imgs_df_non_geometry_columns = [col for col in self.imgs_df.columns if col != "geometry"]
        self.imgs_df[imgs_df_non_geometry_columns] = self.imgs_df[
            imgs_df_non_geometry_columns
        ].convert_dtypes(
            infer_objects=True,
            convert_string=True,
            convert_integer=True,
            convert_boolean=True,
            convert_floating=False,
        )
        self.imgs_df.index.name = IMGS_DF_INDEX_NAME
        self.imgs_df.to_file(Path(self._imgs_df_path), driver="GeoJSON")
        self.polygons_df.index.name = POLYGONS_DF_INDEX_NAME
        self.polygons_df.to_file(Path(self._polygons_df_path), driver="GeoJSON")
        self._graph.save_to_file(Path(self._graph_path))
        # Save params dict
        with open(self.attrs_path, "w") as write_file:
            saveattrs = self._make_dict_json_serializable(self.attrs)
            json.dump(saveattrs, write_file)

    def empty_assoc_same_format_as(
        self,
        data_dir: Optional[
            Union[Path, str]
        ] = None,  # either this arg or all four path args below must be set
        assoc_dir: Optional[Union[Path, str]] = None,
        images_dir: Optional[Union[Path, str]] = None,
        labels_dir: Optional[Union[Path, str]] = None,
    ) -> ImgPolygonAssociator:
        """Factory method that returns an empty associator of the same format
        (i.e. columns in polygons_df and imgs_df) as self with data_dir
        target_data_dir.

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
            labels_dir=labels_dir,
        )

        if data_dir is not None:
            (
                images_dir,
                labels_dir,
                assoc_dir,
            ) = self.__class__._get_default_dirs_from_data_dir(data_dir)

        new_empty_polygons_df = empty_gdf_same_format_as(self.polygons_df)
        new_empty_imgs_df = empty_gdf_same_format_as(self.imgs_df)

        new_empty_assoc = self.__class__.from_scratch(
            # dir args
            images_dir=images_dir,
            labels_dir=labels_dir,
            assoc_dir=assoc_dir,
            # empty dataframes
            polygons_df=new_empty_polygons_df,
            imgs_df=new_empty_imgs_df,
            # remaining kwargs
            **self.attrs,
        )

        return new_empty_assoc

    def print_graph(self):
        """Print the associator's internal graph."""
        print(self._graph)

    def _get_empty_df(self, df_name: str) -> GeoDataFrame:

        if df_name == "polygons_df":
            index_name = POLYGONS_DF_INDEX_NAME
            cols_and_types = self._get_required_df_cols_and_types("polygons_df")
        elif df_name == "imgs_df":
            index_name = IMGS_DF_INDEX_NAME
            cols_and_types = self._get_required_df_cols_and_types("imgs_df")

        df = empty_gdf(
            index_name=index_name, cols_and_types=cols_and_types, crs_epsg_code=self.crs_epsg_code
        )

        return df

    def _get_required_df_cols_and_types(self, df_name: str) -> dict:

        # type of "geometry" column is ignored
        if df_name == "polygons_df":
            cols_and_types = {"geometry": None, "img_count": int}
        elif df_name == "imgs_df":
            cols_and_types = {"geometry": None}

        return cols_and_types

    def _set_remaining_assoc_components(
        self, load_from_disk: bool, polygons_df: GeoDataFrame, imgs_df: GeoDataFrame
    ):

        if load_from_disk:

            self._graph = BipartiteGraph(file_path=self._graph_path)
            self.polygons_df = polygons_df
            self.imgs_df = imgs_df

        else:

            self._graph = empty_graph()
            self.polygons_df = empty_gdf_same_format_as(polygons_df)
            self.imgs_df = empty_gdf_same_format_as(imgs_df)

            self.add_to_polygons_df(polygons_df)
            self.add_to_imgs_df(imgs_df)


    def _check_required_df_cols_exist(
            self, df: GeoDataFrame, df_name: str,
            mode: Literal["polygons_df", "imgs_df"]) -> bool:
        """Check if required columns exist."""

        required_cols = list(self._get_required_df_cols_and_types(df_name).keys())

        if not set(required_cols) <= set(df.columns):

            missing_cols = set(required_cols) - set(df.columns)
            raise ValueError(
                f"{df_name} is missing required columns: {', '.join(missing_cols)}"
            )

    def _standardize_df_crs(self, df: GeoDataFrame, df_name: str):
        """Standardize CRS of dataframe (i.e. set to CRS of associator).

        Args:
            polygons_df (GeoDataFrame): polygons_df
            imgs_df (GeoDataFrame): imgs_df
        """

        if df.crs.to_epsg() != self.crs_epsg_code:  # standard crs
            log.warning("Transforming %s to crs: EPSG=%s", df_name, self.crs_epsg_code)
            df = df.to_crs(epsg=self.crs_epsg_code)

    def _load_df_from_disk(self, df_name: str) -> GeoDataFrame:
        """Load polygons_df or imgs_df from disk."""

        if df_name == "polygons_df":
            df_index_name = POLYGONS_DF_INDEX_NAME
        elif df_name == "imgs_df":
            df_index_name = IMGS_DF_INDEX_NAME

        df_json_path = getattr(self, f"_{df_name}_path")
        return_df = gpd.read_file(df_json_path)
        return_df.set_index(df_index_name, inplace=True)

        return return_df

    def _check_dir_args(
        self,
        data_dir: Union[Path, str],
        images_dir: Union[Path, str],
        labels_dir: Union[Path, str],
        assoc_dir: Union[Path, str],
    ):

        component_dirs_all_not_None = (
            images_dir is not None
            and labels_dir is not None
            and assoc_dir is not None
        )

        if not (component_dirs_all_not_None ^ (data_dir is not None)):
            raise ValueError(
                "Either the data_dir arg must be given (i.e. not None) or all of the images_dir, labels_dir, and assoc_dir args."
            )

    def _init_set_paths(
        self,
        data_dir: Union[Path, str],
        images_dir: Union[Path, str],
        labels_dir: Union[Path, str],
        assoc_dir: Union[Path, str],
    ):
        """Set paths to image/label data and associator component files.

        Used during initialization.
        """

        if data_dir is not None:
            (
                images_dir,
                labels_dir,
                assoc_dir,
            ) = self.__class__._get_default_dirs_from_data_dir(data_dir)

        self._images_dir = Path(images_dir)
        self._labels_dir = Path(labels_dir)
        self._assoc_dir = Path(assoc_dir)

        # set inferred paths
        for path_attr, filename in INFERRED_PATH_ATTR_FILENAMES.items():
            setattr(self, path_attr, self._assoc_dir / filename)

    @classmethod
    def _get_default_dirs_from_data_dir(cls, data_dir: Union[Path, str]) -> Tuple[Path, Path, Path]:

        data_dir = Path(data_dir)

        images_dir = data_dir / DEFAULT_IMAGES_DIR_NAME
        labels_dir = data_dir / DEFAULT_LABELS_DIR_NAME
        assoc_dir = data_dir / DEFAULT_ASSOC_DIR_NAME

        return images_dir, labels_dir, assoc_dir

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
    def _check_no_non_segmentation_polygon_classes_are_segmentation_classes(
        segmentation_classes: List[str],
        background_class: str, **kwargs
    ):
        """TODO.

        Args:
            segmentation_classes (List[str]): [description]
            background_class (str): [description]
        """

        if not len(segmentation_classes) == len(set(segmentation_classes)):
            raise ValueError("segmentation_classes list contains duplicates.")

        non_segmentation_classes = {"background_class": background_class}
        for key, val in kwargs.items():
            if (
                key in NON_SEGMENTATION_POLYGON_CLASSES
                and val is not None
                and val not in non_segmentation_classes
            ):
                non_segmentation_classes[key] = val

        if not set(non_segmentation_classes.values()) & set(segmentation_classes) == set():
            bad_values = {
                class_name: value
                for class_name, value in non_segmentation_classes.items()
                if value in set(segmentation_classes)
            }
            raise ValueError(
                f"No non-segmentation polygon classes should be segmentation classes, but the following are: {bad_values}"
            )
