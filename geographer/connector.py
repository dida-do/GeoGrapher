"""The Connector class organizes and handles remote sensing datasets."""

from __future__ import annotations

import json
import logging
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any, Literal, Sequence, Type, TypeVar

import geopandas as gpd
from geopandas import GeoDataFrame

# Mix-in classes:
from geographer.add_drop_rasters_mixin import AddDropRastersMixIn
from geographer.add_drop_vectors_mixin import AddDropVectorsMixIn
from geographer.global_constants import (
    RASTER_IMGS_INDEX_NAME,
    STANDARD_CRS_EPSG_CODE,
    VECTOR_FEATURES_INDEX_NAME,
)
from geographer.graph import BipartiteGraph
from geographer.graph.bipartite_graph_mixin import BipartiteGraphMixIn
from geographer.utils.connector_utils import (
    empty_gdf,
    empty_gdf_same_format_as,
    empty_graph,
)

DEFAULT_CONNECTOR_DIR_NAME = "connector"
DEFAULT_IMAGES_DIR_NAME = "rasters"
DEFAULT_LABELS_DIR_NAME = "labels"
INFERRED_PATH_ATTR_FILENAMES = {
    "_vectors_path": "vectors.geojson",
    "_rasters_path": "rasters.geojson",
    "attrs_path": "attrs.json",
    "_graph_path": "graph.json",
}
"""Attribute self.key will be self.connector_dir / val."""

ConnectorType = TypeVar("ConnectorType", bound="Connector")

log = logging.getLogger(__name__)


class Connector(
    AddDropVectorsMixIn,
    AddDropRastersMixIn,
    BipartiteGraphMixIn,  # Needs to be last
):
    """Dataset class that connects vector features and raster data.

    A ``Connector`` represents a remote sensing computer vision dataset
    composed of vector features and rasters. It connects the vector
    features and rasters by a bipartite graph encoding the containment
    or intersection relationships between them and is a container for
    tabular information about the vector features and rasters as well as
    for metadata about the dataset.
    """

    _non_task_vector_classes = [
        "background_class"
    ]  # vector feature classes not to be determined by a machine learning

    # model (e.g. vector features that define background regions or masks)

    # yapf: disable
    def __init__(
        self,
        load_from_disk: bool,

        # data dir
        data_dir: Path | str,

        # args w/o default values
        vectors: GeoDataFrame | None = None,
        rasters: GeoDataFrame | None = None,

        # remaining non-path args w/ default values
        task_vector_classes: Sequence[str] | None = None,
        background_class: str | None = None,
        crs_epsg_code: int = STANDARD_CRS_EPSG_CODE,
        raster_count_col_name: str = "raster_count",

        # optional kwargs
        **kwargs: Any,

        # yapf: enable
    ):
        """Initialize Connector.

        Note:
            We advise you to use the following more convenient constructor methods
            to initialize a ``Connector`` instead of using ``__init__`` directly.

            To initialize a new connector use
                - the :meth:`from_scratch` class method, or
                - the :meth:`empty_connector_same_format_as` method

            To initialize an existing connector use
                - the :meth:`from_data_dir` class method


        Args:
            load_from_disk: whether to load an existing connector from disk
                or create a new one.
            task_vector_classes: list of vector feature classes for the machine
                learning task (excluding mask and background classes). Defaults to
                None, i.e. the single class "object"
            vectors: vectors. Defaults to None, i.e. (if not loading
                from disk) an empty vectors.
            rasters: rasters. Defaults to None, i.e. (if not loading
                from disk) an empty rasters.
            crs_epsg_code: EPSG code connector works with.
                Defaults to STANDARD_CRS_EPSG_CODE
            data_dir: data directory containing rasters_dir, labels_dir, connector_dir.
            kwargs: optional keyword args for subclass implementations.
        """
        super().__init__()

        if task_vector_classes is None:
            task_vector_classes = ["object"]
        self._check_no_non_task_vector_classes_are_task_classes(
            task_vector_classes=task_vector_classes,
            background_class=background_class,
            **kwargs
        )

        # set paths
        self._init_set_paths(
            data_dir=data_dir,
        )

        # build attrs from all args except for rasters, vectors,
        # the corresponding column args, and the path/dir args
        self.attrs = {}
        self.attrs.update(
            {
                "task_vector_classes": task_vector_classes,
                "background_class": background_class,
                "crs_epsg_code": crs_epsg_code,
                "raster_count_col_name": raster_count_col_name,
                **kwargs,
            }
        )

        # get vectors and rasters
        if load_from_disk:
            vectors = self._load_df_from_disk("vectors")
            rasters = self._load_df_from_disk("rasters")
        else:
            if vectors is None:
                vectors = self._get_empty_df("vectors")
            if rasters is None:
                rasters = self._get_empty_df("rasters")

        vectors = self._get_df_in_crs(
            df=vectors,
            df_name="vectors",
            crs_epsg_code=self.crs_epsg_code)
        rasters = self._get_df_in_crs(
            df=rasters,
            df_name="rasters",
            crs_epsg_code=self.crs_epsg_code)

        # set self.vectors, self.rasters
        self._set_remaining_connector_components(
            load_from_disk=load_from_disk,
            vectors=vectors,
            rasters=rasters
        )

        # safety checks
        self._check_required_df_cols_exist(
            df=rasters,
            df_name='self.rasters',
            mode='rasters')
        self._check_required_df_cols_exist(
            df=vectors,
            df_name='self.vectors',
            mode='vectors')

        # directories containing raster data
        self._raster_data_dirs = [
            self.rasters_dir,
            self.labels_dir,
        ]  # in subclass implementation, can add e.g. mask_dir

    def __getattr__(self, key: str) -> Any:
        """Check for key in attrs dict."""
        if "attrs" in self.__dict__ and key in self.__dict__["attrs"]:
            return self.__dict__["attrs"][key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    @classmethod
    def from_data_dir(
        cls: Type[ConnectorType],
        data_dir: Path | str,
    ) -> ConnectorType:
        """Initialize a connector from a data directory.

        Args:
            data_dir: data directory containing 'connector_files', 'rasters', and
                'labels' subdirectories

        Returns:
            initialized connector
        """
        data_dir = Path(data_dir)

        rasters_dir, labels_dir, connector_dir = cls._get_default_dirs_from_data_dir(
            data_dir
        )

        # read args from json
        try:
            attrs_path = Path(connector_dir) / \
                INFERRED_PATH_ATTR_FILENAMES["attrs_path"]
            with open(attrs_path, "r") as file:
                kwargs = json.load(file)
        except FileNotFoundError:
            log.exception(
                "Missing connector file %s found in %s",
                INFERRED_PATH_ATTR_FILENAMES['attrs_path'],
                connector_dir)
            raise
        except JSONDecodeError:
            log.exception(
                "The %s file in %s is corrupted!",
                INFERRED_PATH_ATTR_FILENAMES['attrs_path'],
                connector_dir)
            raise

        new_connector = cls(
            load_from_disk=True,
            data_dir=data_dir,
            **kwargs,
        )

        return new_connector

    @classmethod
    def from_scratch(cls, **kwargs: Any) -> Connector:
        r"""Initialize a new connector.

        Ars:
            \**kwargs: same keyword arguments as in :meth:`__init__`
                except for load_from_disk

        Returns:
            initialized connector
        """
        kwargs.update({"load_from_disk": False})
        return cls(**kwargs)

    @property
    def vectors(self) -> GeoDataFrame:
        """Vector features geodataframe, see :ref:`vectors`."""
        return self._vectors

    @vectors.setter
    def vectors(self, new_vectors: GeoDataFrame) -> None:
        self._vectors = new_vectors

    @property
    def rasters(self) -> GeoDataFrame:
        """Raster rasters geodataframe, see :ref:`rasters`."""
        return self._rasters

    @rasters.setter
    def rasters(self, new_rasters: GeoDataFrame) -> None:
        self._rasters = new_rasters

    @property
    def data_dir(self) -> str:
        """Data directory."""  # noqa: D401
        return self._data_dir

    @property
    def rasters_dir(self) -> Path:
        """Directory containing the rasters."""
        return self._rasters_dir

    @property
    def labels_dir(self) -> Path:
        """Directory containing the segmentation labels."""
        return self._labels_dir

    @property
    def connector_dir(self) -> Path:
        """Directory in which the connector files are saved."""
        return self._connector_dir

    @property
    def crs_epsg_code(self) -> int:
        """EPSG code of connector's :term:`crs`.

        Setting ``crs_epsg_code`` will set automatically set the
        connector's ``rasters`` and ``vectors`` crs.
        """
        return self.attrs["crs_epsg_code"]

    @crs_epsg_code.setter
    def crs_epsg_code(self, epsg_code: int):
        # set value in params dict
        self.attrs["crs_epsg_code"] = epsg_code

        # reproject rasters and vectors GeoDataFrames
        self.vectors = self.vectors.to_crs(epsg=epsg_code)
        self.rasters = self.rasters.to_crs(epsg=epsg_code)

    @property
    def task_vector_classes(self):
        """All classes for the :term:`ML` task."""
        return self.attrs["task_vector_classes"]

    @task_vector_classes.setter
    def task_vector_classes(self, new_task_vector_classes: list[str]):
        if not len(new_task_vector_classes) == len(set(new_task_vector_classes)):
            raise ValueError("no duplicates in list of task_vector_classes allowed")
        self.attrs["task_vector_classes"] = new_task_vector_classes

    @property
    def all_vector_classes(self):
        """All allowed classes in vectors.

        Includes those not related to the :term:`ML` task (e.g. the
        background class)
        """
        answer = self.task_vector_classes.copy()
        for class_name in self._non_task_vector_classes:
            class_value = getattr(self, class_name)
            if class_value is not None:
                answer += [class_value]

        return answer

    @property
    def raster_data_dirs(self) -> list[Path]:
        """All directories containing raster data.

        Includes e.g. segmentation labels.
        """
        return self._raster_data_dirs

    @property
    def raster_count_col_name(self) -> str:
        """Name of column in vectors containing raster counts."""
        return self.attrs["raster_count_col_name"]

    @raster_count_col_name.setter
    def set_raster_count_col_name(self, new_col_name: str):
        """Set name of column in vectors containing raster counts."""
        self.attrs["raster_count_col_name"] = new_col_name

    @property
    def graph_str(self) -> str:
        """Return a string representation of the internal graph.

        Note that the representation might change if the internal
        representation changes.
        """
        return str(self._graph)

    def save(self):
        """Save connector to disk."""
        log.info("Saving connector to disk...")

        # Make sure connector_dir exists.
        self._connector_dir.mkdir(parents=True, exist_ok=True)

        rasters_non_geometry_columns = [
            col for col in self.rasters.columns
            if col != "geometry"
        ]
        self.rasters[rasters_non_geometry_columns] = self.rasters[
            rasters_non_geometry_columns
        ].convert_dtypes(
            infer_objects=True,
            convert_string=True,
            convert_integer=True,
            convert_boolean=True,
            convert_floating=False,
        )
        self.rasters.index.name = RASTER_IMGS_INDEX_NAME
        self.rasters.to_file(Path(self._rasters_path), driver="GeoJSON")
        self.vectors.index.name = VECTOR_FEATURES_INDEX_NAME
        self.vectors.to_file(Path(self._vectors_path), driver="GeoJSON")
        self._graph.save_to_file(Path(self._graph_path))
        # Save params dict
        with open(self.attrs_path, "w", encoding='utf-8') as write_file:
            saveattrs = self._replace_path_values(self.attrs)
            try:
                json.dump(saveattrs, write_file, ensure_ascii=False, indent=4)
            except TypeError as exc:
                raise TypeError(
                    "User defined attributes must be JSON-serializable."
                    ) from exc

    def empty_connector_same_format(
        self,
        data_dir: Path | str
    ) -> Connector:
        """Return an empty connector of the same format.

        Return an empty connector of the same format
        (i.e. same columns in vectors and rasters).

        Args:
            data_dir: data directory containing rasters_dir, labels_dir, connector_dir.
            rasters_dir: path to directory containing rasters.
            labels_dir: path to directory containing labels.
            connector_dir: path to directory containing (geo)json connector
                component files.

        Returns:
            new empty connector
        """
        if data_dir is not None:
            (
                rasters_dir,
                labels_dir,
                connector_dir,
            ) = self.__class__._get_default_dirs_from_data_dir(data_dir)

        new_empty_vectors = empty_gdf_same_format_as(self.vectors)
        new_empty_rasters = empty_gdf_same_format_as(self.rasters)

        new_empty_connector = self.__class__.from_scratch(
            data_dir=data_dir,
            # empty dataframes
            vectors=new_empty_vectors,
            rasters=new_empty_rasters,
            # remaining kwargs
            **self.attrs,
        )

        return new_empty_connector

    def _get_empty_df(self, df_name: str) -> GeoDataFrame:

        if df_name == "vectors":
            index_name = VECTOR_FEATURES_INDEX_NAME
            cols_and_types = self._get_required_df_cols_and_types("vectors")
        elif df_name == "rasters":
            index_name = RASTER_IMGS_INDEX_NAME
            cols_and_types = self._get_required_df_cols_and_types("rasters")

        df = empty_gdf(
            index_name=index_name,
            cols_and_types=cols_and_types,
            crs_epsg_code=self.crs_epsg_code
        )

        return df

    def _get_required_df_cols_and_types(self, df_name: str) -> dict:

        # type of "geometry" column is ignored
        if df_name.endswith("vectors"):
            cols_and_types = {"geometry": None, self.raster_count_col_name: int}
        elif df_name.endswith("rasters"):
            cols_and_types = {"geometry": None}

        return cols_and_types

    def _set_remaining_connector_components(
        self,
        load_from_disk: bool,
        vectors: GeoDataFrame,
        rasters: GeoDataFrame
    ):

        if load_from_disk:

            self._graph = BipartiteGraph(file_path=self._graph_path)
            self.vectors = vectors
            self._rasters = rasters

        else:

            self._graph = empty_graph()
            self._vectors = empty_gdf_same_format_as(vectors)
            self._rasters = empty_gdf_same_format_as(rasters)

            self.add_to_vectors(vectors)
            self.add_to_rasters(rasters)

    def _check_required_df_cols_exist(
            self, df: GeoDataFrame, df_name: str,
            mode: Literal["vectors", "rasters"]) -> bool:
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
            vectors: vectors
            rasters: rasters
        """
        if df.crs.to_epsg() != crs_epsg_code:
            log.info("Transforming %s to crs: EPSG=%s", df_name, crs_epsg_code)
            return df.to_crs(epsg=crs_epsg_code)
        else:
            # faster than unnecessary df.to_crs(epsg=crs_epsg_code)
            return df

    def _load_df_from_disk(self, df_name: str) -> GeoDataFrame:
        """Load vectors or rasters from disk."""
        if df_name == "vectors":
            df_index_name = VECTOR_FEATURES_INDEX_NAME
        elif df_name == "rasters":
            df_index_name = RASTER_IMGS_INDEX_NAME

        df_json_path = getattr(self, f"_{df_name}_path")
        return_df = gpd.read_file(df_json_path)
        return_df.set_index(df_index_name, inplace=True)

        return return_df

    def _init_set_paths(
        self,
        data_dir: Path | str,
    ):
        """Set paths to raster/label data and connector component files.

        Used during initialization.
        """
        (
            rasters_dir,
            labels_dir,
            connector_dir,
        ) = self.__class__._get_default_dirs_from_data_dir(data_dir)

        self._data_dir = Path(data_dir)
        self._rasters_dir = Path(rasters_dir)
        self._labels_dir = Path(labels_dir)
        self._connector_dir = Path(connector_dir)

        # set inferred paths
        for path_attr, filename in INFERRED_PATH_ATTR_FILENAMES.items():
            setattr(self, path_attr, self._connector_dir / filename)

    @classmethod
    def _get_default_dirs_from_data_dir(
            cls,
            data_dir: Path | str
            ) -> tuple[Path, Path, Path]:

        data_dir = Path(data_dir)

        rasters_dir = data_dir / DEFAULT_IMAGES_DIR_NAME
        labels_dir = data_dir / DEFAULT_LABELS_DIR_NAME
        connector_dir = data_dir / DEFAULT_CONNECTOR_DIR_NAME

        return rasters_dir, labels_dir, connector_dir

    @staticmethod
    def _replace_path_values(input_dict: dict) -> dict:
        """Replace data_dir, rasters_dir, etc Paths with strings.

        Args:
            input_dict: input dict with keys strings and values of arbitrary type

        Returns:
            dict
        """
        path_keys = {"data_dir", "rasters_dir", "labels_dir", "connector_dir"}

        output_dict = {
            key: str(val) if key in path_keys else val
            for key, val in input_dict.items()
        }

        return output_dict

    def _check_no_non_task_vector_classes_are_task_classes(
        self,
        task_vector_classes: list[str],
        background_class: str, **kwargs
    ):
        """Check non-task vector class and task classes are disjoint.

        Args:
            task_vector_classes: [description]
            background_class: [description]
        """
        if not len(task_vector_classes) == len(set(task_vector_classes)):
            raise ValueError("task_vector_classes list contains duplicates.")

        non_task_vector_classes = {"background_class": background_class}
        for key, val in kwargs.items():
            if (
                key in self._non_task_vector_classes
                and val is not None
                and val not in non_task_vector_classes
            ):
                non_task_vector_classes[key] = val

        if not set(non_task_vector_classes.values()).isdisjoint(
                set(task_vector_classes)):

            bad_values = {
                class_name: value
                for class_name, value in non_task_vector_classes.items()
                if value in set(task_vector_classes)
            }
            raise ValueError(
                "The following task_vector_classes are also classes unrelated "
                f"to the machine learning task: {bad_values}"
            )
