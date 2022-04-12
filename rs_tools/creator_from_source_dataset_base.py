"""
Base class for Creating or updating a dataset from an existing source dataset.
"""

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from rs_tools import ImgPolygonAssociator
from rs_tools.base_model_dict_conversion.save_load_base_model_mixin import SaveAndLoadBaseModelMixIn


class DSCreatorFromSource(ABC, SaveAndLoadBaseModelMixIn, BaseModel):
    """
    Base class for Creating or updating a dataset from an existing source dataset.
    """

    source_data_dir: Path
    target_data_dir: Path
    name: str = Field(
        title="Name",
        description=
        "Name of dataset creator. Used as part of filename when saving.")

    def __init__(self, **data):
        super().__init__(**data)
        self._source_assoc = None
        self._target_assoc = None
        self._set_source_assoc()
        self._set_target_assoc()

    @abstractmethod
    def _create(self, *args, **kwargs) -> ImgPolygonAssociator:
        """Create a new dataset from source dataset"""

    @abstractmethod
    def _update(self, *args, **kwargs) -> ImgPolygonAssociator:
        """Update the target dataset from the source dataset"""

    def create(self, *args, **kwargs) -> ImgPolygonAssociator:
        """Create a new dataset by cutting the source dataset"""
        self._create(*args, **kwargs)
        self._after_converting()

    def update(self, *args, **kwargs) -> ImgPolygonAssociator:
        """Update the target dataset from the source dataset"""
        self._update(*args, **kwargs)
        self._after_converting()

    def save(self):
        """Save converter to update folder in source_data_dir"""
        json_file_path = self.target_assoc.assoc_dir / self.name
        self._save(json_file_path)

    @property
    def source_assoc(self):
        """Associator in source_data_dir"""
        if self._source_assoc is None or self._source_assoc.images_dir.parent != self.source_data_dir:
            self._set_source_assoc()
        return self._source_assoc

    @property
    def target_assoc(self):
        """Associator in target_data_dir"""
        if self._target_assoc is None or self._target_assoc.images_dir.parent != self.target_data_dir:
            self._set_target_assoc()
        return self._target_assoc

    def _after_converting(self):
        """Can be used in a subclass to e.g. save parameters to the target_assoc"""

    def _set_source_assoc(self):
        """Set source associator"""
        self._source_assoc = ImgPolygonAssociator.from_data_dir(
            self.source_data_dir)

    def _set_target_assoc(self):
        """Set target associator"""
        try:
            target_assoc = ImgPolygonAssociator.from_data_dir(
                self.target_data_dir)
        except FileNotFoundError:
            target_assoc = self.empty_assoc_same_format_as(
                self.target_data_dir)
        finally:
            self._target_assoc = target_assoc

    def _create_target_dirs(self):
        """Create target_data_dir and subdirectories"""
        self.target_assoc.assoc_dir.mkdir(parents=True, exist_ok=True)
        for dir_ in self.target_assoc.image_data_dirs:
            dir_.mkdir(parents=True, exist_ok=True)

class DSCreatorFromSourceWithBands(DSCreatorFromSource, ABC):

    bands: Optional[Dict[str, Optional[List[int]]]] = Field(
        default=None,
        title="Dict of band indices",
        description=
        "keys: image directory names, values: list of band indices starting at 1 to keep"
    )
