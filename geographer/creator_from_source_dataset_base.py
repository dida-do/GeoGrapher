"""ABC for creating or updating a dataset from an existing source dataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Extra, Field

from geographer.base_model_dict_conversion.save_load_base_model_mixin import (
    SaveAndLoadBaseModelMixIn,
)
from geographer.connector import Connector


class DSCreatorFromSource(ABC, SaveAndLoadBaseModelMixIn, BaseModel):
    """ABC for creating or updating a dataset from an existing one."""

    source_data_dir: Path
    target_data_dir: Path
    name: str = Field(
        title="Name",
        description="Name of dataset creator. Used as part of filename when saving.",
    )
    _source_connector: Optional[Connector] = Field(
        default=None, exclude=True, description="Do not set by hand"
    )
    _target_connector: Optional[Connector] = Field(
        default=None, exclude=True, description="Do not set by hand"
    )

    class Config:
        """BaseModel Config."""

        arbitrary_types_allowed = True
        extra = Extra.allow
        underscore_attrs_are_private = True

    def __init__(self, **data):
        """Initialize from data.

        Args:
            data: data
        """
        super().__init__(**data)
        self._source_connector = None
        self._target_connector = None
        self._set_source_connector()
        self._set_target_connector()

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
        if (
            self._source_connector is None
            or self._source_connector.images_dir.parent != self.source_data_dir
        ):
            self._set_source_connector()
        return self._source_connector

    @property
    def target_connector(self):
        """Connector in target_data_dir."""
        if (
            self._target_connector is None
            or self._target_connector.images_dir.parent != self.target_data_dir
        ):
            self._set_target_connector()
        return self._target_connector

    def _after_creating_or_updating(self):
        """Run hook after creating/updating.

        Can be used to e.g. save parameters to the target_connector.
        """

    def _set_source_connector(self):
        """Set source connector."""
        self._source_connector = Connector.from_data_dir(self.source_data_dir)

    def _set_target_connector(self):
        """Set target connector."""
        try:
            target_connector = Connector.from_data_dir(self.target_data_dir)
        except FileNotFoundError:
            target_connector = self.source_connector.empty_connector_same_format(
                self.target_data_dir
            )
        finally:
            self._target_connector = target_connector

    def _add_missing_vector_features_to_target(self):
        """Add missing vector features from source dataset to target dataset.

        Only checks feature names/indices, not whether entries differ.
        """
        source_features = self.source_connector.vector_features
        target_features = self.target_connector.vector_features
        features_to_add = source_features[
            ~source_features.index.isin(target_features.index)
        ]
        self.target_connector.add_to_vector_features(features_to_add)

    def _create_target_dirs(self):
        """Create target_data_dir and subdirectories."""
        self.target_connector.connector_dir.mkdir(parents=True, exist_ok=True)
        for dir_ in self.target_connector.image_data_dirs:
            dir_.mkdir(parents=True, exist_ok=True)


class DSCreatorFromSourceWithBands(DSCreatorFromSource, ABC):
    """ABC for creating/updating a dataset from an existing one.

    Includes a bands field.
    """

    bands: Optional[Dict[str, Optional[List[int]]]] = Field(
        default=None,
        title="Dict of band indices",
        description="keys: image directory names, values: list of band indices"
        "starting at 1 to keep",
    )
