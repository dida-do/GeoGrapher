"""Base class for segmentation label makers"""

from abc import ABC, abstractmethod
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from tqdm.auto import tqdm
from rs_tools import ImgPolygonAssociator
from rs_tools.base_model_dict_conversion.save_load_base_model_mixin import SaveAndLoadBaseModelMixIn

# logger
log = logging.getLogger(__name__)


class SegLabelMaker(ABC, BaseModel, SaveAndLoadBaseModelMixIn):
    """Base class for segmentation label makers"""

    add_background_band: bool = Field(
        default=True, description="Whether to add implicit background band.")

    @abstractmethod
    def _make_label_for_img(self, assoc: ImgPolygonAssociator, img_name: str):
        """Make label for single image"""
        pass

    @property
    @abstractmethod
    def label_type(self):
        """Return label type"""
        pass

    def _after_make_labels(self, assoc: ImgPolygonAssociator):
        """Override this hook in subclass to apply custom logic after making labels"""
        pass

    def _run_safety_checks(self, assoc: ImgPolygonAssociator):
        """Override to check e.g. if existing classes in polygons_df contained in assoc.all_classes"""
        pass

    def make_labels(
        self,
        assoc: ImgPolygonAssociator,
        img_names: Optional[List[str]] = None,
    ):
        """
        Create segmentation labels.

        Args:
            img_names (List[str], optional): list of image names to create labels for.
                Defaults to None (i.e. all images without a label).
        """

        # safety checks
        self._run_safety_checks(assoc)
        self._compare_existing_imgs_to_imgs_df(assoc)

        assoc.labels_dir.mkdir(parents=True, exist_ok=True)

        existing_images = {
            img_path.name
            for img_path in assoc.images_dir.iterdir()
            if img_path.is_file() and img_path.name in self.imgs_df.index
        }

        if img_names is None:
            # Find images without labels
            existing_labels = {
                img_path.name
                for img_path in assoc.labels_dir.iterdir()
                if img_path.is_file() and img_path.name in self.imgs_df.index
            }
            img_names = existing_images - existing_labels
        elif not set(img_names) <= existing_images:
            raise FileNotFoundError(
                f"Can't make labels for missing images: {existing_images - img_names}"
            )

        for img_name in tqdm(img_names, desc='Making labels: '):
            self._make_label_for_img(assoc=self, img_name=img_name)

        assoc.attrs['label_type'] = self.label_type
        self._after_make_labels(assoc)
        assoc.save()

    def delete_labels(
        self,
        assoc: ImgPolygonAssociator,
        img_names: Optional[List[str]] = None,
    ):
        """Delete (pixel) labels from assoc's labels_dir.

        Args:
            img_names (Optional[List[str]], optional): names of images for which to delete labels. Defaults to None, i.e. all labels.
        """
        if img_names is None:
            img_names = assoc.imgs_df.index

        for img_name in tqdm(img_names, desc='Deleting labels: '):
            (assoc.labels_dir / img_name).unlink(missing_ok=True)

    @staticmethod
    def _compare_existing_imgs_to_imgs_df(assoc: ImgPolygonAssociator):
        """Safety check: compare sets of images in images_dir and in
        self.imgs_df.

        Raises warnings if there is a discrepancy.
        """

        # Find the set of existing images in the dataset, ...
        existing_images = {
            img_path.name
            for img_path in assoc.images_dir.iterdir() if img_path.is_file()
        }

        # ... then if the set of images is a strict subset of the images in imgs_df ...
        if existing_images < set(assoc.imgs_df.index):

            # ... log a warning
            log.warning(
                "There images in self.imgs_df that are not in the images_dir %s.",
                assoc.images_dir)

        # ... and if it is not a subset, ...
        if not existing_images <= set(assoc.imgs_df.index):

            # ... log an warning
            message = "Warning! There are images in the dataset's images subdirectory that are not in self.imgs_df."
            log.warning(message)
