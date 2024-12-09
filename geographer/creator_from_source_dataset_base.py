"""ABC for creating or updating a dataset from an existing source dataset."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from geographer.base_model_dict_conversion.save_load_base_model_mixin import (
    SaveAndLoadBaseModelMixIn,
)
from geographer.connector import (
    DEFAULT_CONNECTOR_DIR_NAME,
    INFERRED_PATH_ATTR_FILENAMES,
    Connector,
)


class DSCreatorFromSource(ABC, SaveAndLoadBaseModelMixIn, BaseModel):
    """ABC for creating or updating a dataset from an existing one."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    source_data_dir: Path
    target_data_dir: Path
    name: str = Field(
        title="Name",
        description="Name of dataset creator. Used as part of filename when saving.",
    )
    _source_connector: Optional[Connector] = PrivateAttr(default=None)
    _target_connector: Optional[Connector] = PrivateAttr(default=None)

    @field_validator("source_data_dir", mode="before")
    def validate_source_data_dir(cls, value: Path) -> Path:
        """Ensure source_data_dir is a valid path."""
        if not value.is_dir():
            raise ValueError(f"Invalid source_data_dir: {value}")
        return value

    @model_validator(mode="after")
    def validate_connectors(self) -> "DSCreatorFromSource":
        """Initialize and validate connectors."""
        if self.source_data_dir:
            self._source_connector = Connector.from_data_dir(self.source_data_dir)

        if self.target_data_dir:
            connector_file_paths_exist = [
                (self.target_data_dir / DEFAULT_CONNECTOR_DIR_NAME / filename).is_file()
                for filename in INFERRED_PATH_ATTR_FILENAMES.values()
            ]

            if all(connector_file_paths_exist):
                self._target_connector = Connector.from_data_dir(self.target_data_dir)
            elif not any(connector_file_paths_exist):
                self._target_connector = (
                    self._source_connector.empty_connector_same_format(
                        self.target_data_dir
                    )
                )
            else:
                raise ValueError(
                    "Corrupted target dataset: only some of the connector files exist."
                )

        return self

    @abstractmethod
    def _create(self, *args, **kwargs) -> Connector:
        """Create a new dataset from source dataset."""

    @abstractmethod
    def _update(self, *args, **kwargs) -> Connector:
        """Update the target dataset from the source dataset."""

    def create(self, *args, **kwargs) -> Connector:
        """Create a new dataset by cutting the source dataset."""
        self._create(*args, **kwargs)
        self._after_creating_or_updating()
        self.target_connector.save()
        return self.target_connector

    def update(self, *args, **kwargs) -> Connector:
        """Update the target dataset from the source dataset."""
        self._update(*args, **kwargs)
        self._after_creating_or_updating()
        self.target_connector.save()
        return self.target_connector

    def save(self):
        """Save to update folder in source_data_dir."""
        json_file_path = self.target_connector.connector_dir / f"{self.name}.json"
        self._save(json_file_path)

    @property
    def source_connector(self):
        """Connector in source_data_dir."""
        return self._source_connector

    @property
    def target_connector(self):
        """Connector in target_data_dir."""
        return self._target_connector

    def _after_creating_or_updating(self):
        """Run hook after creating/updating.

        Can be used to e.g. save parameters to the target_connector.
        """

    def _add_missing_vectors_to_target(self):
        """Add missing vector features from source dataset to target dataset.

        Only checks vector feature names/indices, not whether entries
        differ.
        """
        source_vectors = self.source_connector.vectors
        target_vectors = self.target_connector.vectors
        vectors_to_add = source_vectors[
            ~source_vectors.index.isin(target_vectors.index)
        ]
        self.target_connector.add_to_vectors(vectors_to_add)

    def _create_target_dirs(self):
        """Create target_data_dir and subdirectories."""
        self.target_connector.connector_dir.mkdir(parents=True, exist_ok=True)
        for dir_ in self.target_connector.raster_data_dirs:
            dir_.mkdir(parents=True, exist_ok=True)


class DSCreatorFromSourceWithBands(DSCreatorFromSource, ABC):
    """ABC for creating/updating a dataset from an existing one.

    Includes a bands field.
    """

    bands: Optional[dict[str, Optional[list[int]]]] = Field(
        default=None,
        title="Dict of band indices",
        description="keys: raster directory names, values: list of band indices "
        "to keep, starting with 1",
    )
