"""Base class for label makers that generate segmentation labels from a
connector's vector_features."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from geographer.base_model_dict_conversion.save_load_base_model_mixin import \
    SaveAndLoadBaseModelMixIn
from geographer.connector import Connector
from geographer.label_makers.label_maker_base import LabelMaker

# logger
log = logging.getLogger(__name__)


class SegLabelMaker(LabelMaker, BaseModel, SaveAndLoadBaseModelMixIn):
    """Base class for label makers that generate segmentation labels from a
    connector's vector_features."""

    add_background_band: bool = Field(
        default=True, description="Whether to add implicit background band.")

    @abstractmethod
    def _make_label_for_img(self, connector: Connector, img_name: str):
        """Make label for single image."""
        pass

    @property
    @abstractmethod
    def label_type(self) -> str:
        """Return label type."""
        pass

    def _after_make_labels(self, connector: Connector):
        """Override this hook in subclass to apply custom logic after making
        labels."""
        pass

    def _run_safety_checks(self, connector: Connector):
        """Override to check e.g. if existing classes in vector_features
        contained in connector.all_vector_feature_classes."""
        pass

    def make_labels(
        self,
        connector: Connector,
        img_names: Optional[List[str]] = None,
    ):
        """Create segmentation labels.

        Args:
            img_names (List[str], optional): list of image names to create labels for.
                Defaults to None (i.e. all images without a label).
        """

        # safety checks
        self._run_safety_checks(connector)
        self._set_label_type_in_connector_attrs(connector)
        self._compare_existing_imgs_to_raster_imgs(connector)

        connector.labels_dir.mkdir(parents=True, exist_ok=True)

        existing_images = {
            img_path.name
            for img_path in connector.images_dir.iterdir() if
            img_path.is_file() and img_path.name in connector.raster_imgs.index
        }

        if img_names is None:
            # Find images without labels
            existing_labels = {
                img_path.name
                for img_path in connector.labels_dir.iterdir()
                if img_path.is_file()
                and img_path.name in connector.raster_imgs.index
            }
            img_names = existing_images - existing_labels
        elif not set(img_names) <= existing_images:
            raise FileNotFoundError(
                f"Can't make labels for missing images: {existing_images - img_names}"
            )

        for img_name in tqdm(img_names, desc='Making labels: '):
            self._make_label_for_img(connector=connector, img_name=img_name)

        connector.attrs['label_type'] = self.label_type
        self._after_make_labels(connector)
        connector.save()

    def delete_labels(
        self,
        connector: Connector,
        img_names: Optional[List[str]] = None,
    ):
        """Delete (pixel) labels from the connector's labels_dir.

        Args:
            img_names (Optional[List[str]], optional): names of images for which to delete labels. Defaults to None, i.e. all labels.
        """
        if img_names is None:
            img_names = connector.raster_imgs.index

        for img_name in tqdm(img_names, desc='Deleting labels: '):
            (connector.labels_dir / img_name).unlink(missing_ok=True)

    def _set_label_type_in_connector_attrs(self, connector: Connector):
        connector.attrs["label_type"] = self.label_type

    @staticmethod
    def _compare_existing_imgs_to_raster_imgs(connector: Connector):
        """Safety check: compare sets of images in images_dir and in
        connector.raster_imgs.

        Raises warnings if there is a discrepancy.
        """

        # Find the set of existing images in the dataset, ...
        existing_images = {
            img_path.name
            for img_path in connector.images_dir.iterdir()
            if img_path.is_file()
        }

        # ... then if the set of images is a strict subset of the images in raster_imgs ...
        if existing_images < set(connector.raster_imgs.index):

            # ... log a warning
            log.warning(
                "There images in connector.raster_imgs that are not in the images_dir %s.",
                connector.images_dir)

        # ... and if it is not a subset, ...
        if not existing_images <= set(connector.raster_imgs.index):

            # ... log an warning
            message = "Warning! There are images in the dataset's images subdirectory that are not in connector.raster_imgs."
            log.warning(message)
