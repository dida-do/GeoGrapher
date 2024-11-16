"""Base class for segmentation label makers."""

from __future__ import annotations

import logging
from abc import abstractmethod

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from geographer.base_model_dict_conversion.save_load_base_model_mixin import (
    SaveAndLoadBaseModelMixIn,
)
from geographer.connector import Connector
from geographer.label_makers.label_maker_base import LabelMaker

# logger
log = logging.getLogger(__name__)


class SegLabelMaker(LabelMaker, BaseModel, SaveAndLoadBaseModelMixIn):
    """Base class for segmentation label makers."""

    add_background_band: bool = Field(
        default=True, description="Whether to add implicit background band."
    )

    @abstractmethod
    def _make_label_for_raster(self, connector: Connector, raster_name: str):
        """Make label for single raster."""
        pass

    @property
    @abstractmethod
    def label_type(self) -> str:
        """Return label type."""
        pass

    def _after_make_labels(self, connector: Connector):
        """Run after making labels.

        Override this hook in subclass to apply custom logic after
        making labels.
        """
        pass

    def _run_safety_checks(self, connector: Connector):
        """Run safety checks. Hook.

        Override to check e.g. if existing classes in vectors contained
        in connector.all_vector_classes.
        """
        pass

    def make_labels(
        self,
        connector: Connector,
        raster_names: list[str] | None = None,
    ):
        """Create segmentation labels.

        Args:
            raster_names: list of raster names to create labels for.
                Defaults to None (i.e. all rasters without a label).
        """
        # safety checks
        self._run_safety_checks(connector)
        self._set_label_type_in_connector_attrs(connector)
        self._compare_existing_rasters_to_rasters(connector)

        connector.labels_dir.mkdir(parents=True, exist_ok=True)

        existing_rasters = {
            raster_path.name
            for raster_path in connector.rasters_dir.iterdir()
            if raster_path.is_file() and raster_path.name in connector.rasters.index
        }

        if raster_names is None:
            # Find rasters without labels
            existing_labels = {
                raster_path.name
                for raster_path in connector.labels_dir.iterdir()
                if raster_path.is_file() and raster_path.name in connector.rasters.index
            }
            raster_names = existing_rasters - existing_labels
        elif not set(raster_names) <= existing_rasters:
            raise FileNotFoundError(
                "Can't make labels for missing rasters: "
                f"{existing_rasters - raster_names}"
            )

        for raster_name in tqdm(raster_names, desc="Making labels: "):
            self._make_label_for_raster(connector=connector, raster_name=raster_name)

        connector.attrs["label_type"] = self.label_type
        self._after_make_labels(connector)
        connector.save()

    def delete_labels(
        self,
        connector: Connector,
        raster_names: list[str] | None = None,
    ):
        """Delete (pixel) labels from the connector's labels_dir.

        Args:
            raster_names: names of rasters for which to delete labels.
            Defaults to None, i.e. all labels.
        """
        if raster_names is None:
            raster_names = connector.rasters.index

        for raster_name in tqdm(raster_names, desc="Deleting labels: "):
            (connector.labels_dir / raster_name).unlink(missing_ok=True)

    def _set_label_type_in_connector_attrs(self, connector: Connector):
        connector.attrs["label_type"] = self.label_type

    @staticmethod
    def _compare_existing_rasters_to_rasters(connector: Connector):
        """Safety check.

        Compare sets of rasters in rasters_dir and in connector.rasters.

        Raises warnings if there is a discrepancy.
        """
        # Find the set of existing rasters in the dataset, ...
        existing_rasters = {
            raster_path.name
            for raster_path in connector.rasters_dir.iterdir()
            if raster_path.is_file()
        }

        # ... then if the set of rasters is a strict subset
        # of the rasters in rasters ...
        if existing_rasters < set(connector.rasters.index):
            # ... log a warning
            log.warning(
                "There are rasters in connector.rasters that "
                "are not in the rasters_dir %s.",
                connector.rasters_dir,
            )

        # ... and if it is not a subset, ...
        if not existing_rasters <= set(connector.rasters.index):
            # ... log an warning
            message = (
                "Warning! There are rasters in the dataset's rasters "
                "subdirectory that are not in connector.rasters."
            )
            log.warning(message)
