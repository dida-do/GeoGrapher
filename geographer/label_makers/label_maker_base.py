"""
Base class for label makers that generate labels (for
any kind of computer vision task) from a connector's vector_features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

from tqdm.auto import tqdm
if TYPE_CHECKING:
    from geographer.connector import Connector
from geographer.base_model_dict_conversion.save_load_base_model_mixin import SaveAndLoadBaseModelMixIn

# logger
log = logging.getLogger(__name__)


class LabelMaker(ABC, BaseModel, SaveAndLoadBaseModelMixIn):
    """
    Base class for label makers that generate labels (for
    any kind of computer vision task) from a connector's vector_features.
    """

    @abstractmethod
    def make_labels(
        self,
        connector: Connector,
        img_names: Optional[List[str]] = None,
    ):
        """
        Create segmentation labels.

        Args:
            img_names (List[str], optional): list of image names to create labels for.
                Defaults to None (i.e. all images without a label).
        """

    @abstractmethod
    def delete_labels(
        self,
        connector: Connector,
        img_names: Optional[List[str]] = None,
    ):
        """Delete (pixel) labels from the connector's labels_dir.

        Args:
            img_names (Optional[List[str]], optional): names of images for which to delete labels. Defaults to None, i.e. all labels.
        """

    def recompute_labels(
        self,
        connector: Connector,
        img_names: Optional[List[str]] = None,
    ):
        """Recompute labels. Equivalent to delete_labels followed by make_labels"""
        self.delete_labels(connector, img_names)
        self.make_labels(connector, img_names)
