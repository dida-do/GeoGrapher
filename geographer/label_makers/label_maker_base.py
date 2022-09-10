"""Base class for label makers.

Base class for label makers that generate labels (for any kind of
computer vision task) from a connector's vector_features.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from geographer.connector import Connector

from geographer.base_model_dict_conversion.save_load_base_model_mixin import (
    SaveAndLoadBaseModelMixIn,
)

# logger
log = logging.getLogger(__name__)


class LabelMaker(ABC, BaseModel, SaveAndLoadBaseModelMixIn):
    """Base class for label makers.

    Base class for label makers. that generate labels (for any kind of
    computer vision task) from a connector's vector_features.
    """

    @abstractmethod
    def make_labels(
        self,
        connector: Connector,
        img_names: Optional[list[str]] = None,
    ):
        """Create segmentation labels.

        Args:
            img_names: raster names to create labels for.
                Defaults to None (i.e. all raster without a label).
        """

    @abstractmethod
    def delete_labels(
        self,
        connector: Connector,
        img_names: Optional[list[str]] = None,
    ):
        """Delete (pixel) labels from the connector's labels_dir.

        Args:
            img_names: names of images for which to delete labels.
            Defaults to None, i.e. all labels.
        """

    def recompute_labels(
        self,
        connector: Connector,
        img_names: Optional[list[str]] = None,
    ):
        """Recompute labels.

        Equivalent to delete_labels followed by make_labels
        """
        self.delete_labels(connector, img_names)
        self.make_labels(connector, img_names)
