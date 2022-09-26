"""Create a dataset with soft-categorical labels.

Create a dataset with soft-categorical labels from a dataset with
categorical labels.
"""

import logging
import shutil
from typing import Optional

from pydantic import Field
from tqdm.auto import tqdm

from geographer import Connector
from geographer.creator_from_source_dataset_base import DSCreatorFromSource
from geographer.img_bands_getter_mixin import ImgBandsGetterMixIn
from geographer.label_makers.label_maker_base import LabelMaker
from geographer.label_makers.label_type_conversion_utils import (
    convert_vector_features_soft_cat_to_cat,
)

log = logging.Logger(__name__)


class DSConverterSoftCatToCat(DSCreatorFromSource, ImgBandsGetterMixIn):
    """Create a dataset with soft-categorical labels.

    Assumes source dataset has categorical labels.
    """

    label_maker: Optional[LabelMaker] = Field(
        default=None,
        description="Optional LabelMaker. If given, will create labels"
        "in target dataset.",
    )

    def _create(self):
        self._create_or_update()

    def _update(self):
        self._create_or_update()

    def _create_or_update(self) -> Connector:

        if self.source_assoc.label_type != "soft-categorical":
            raise ValueError(
                "Only works with label_type soft-categorical\n"
                + f"Current label type: {self.source_assoc.label_type}"
            )

        # need this later
        geoms_that_will_be_added_to_target_dataset = set(
            self.source_assoc.geoms_df.index
        ) - set(self.target_assoc.geoms_df.index)

        # add geometriess to target dataset
        source_geoms_df_converted_to_soft_categorical_format = (
            convert_vector_features_soft_cat_to_cat(self.source_assoc.geoms_df)
        )
        self.target_assoc.add_to_geoms_df(
            source_geoms_df_converted_to_soft_categorical_format
        )

        # add images to target dataset
        self.target_assoc.add_to_raster_imgs(self.source_assoc.raster_imgs)

        # Determine which images to copy to target dataset
        imgs_that_already_existed_in_target_dataset = {
            img_name
            for img_name in self.target_assoc.raster_imgs.index
            if (self.target_assoc.images_dir / img_name).is_file()
        }
        imgs_in_source_images_dir = {
            img_name for img_name in self.source_assoc.raster_imgs.index
        }
        imgs_in_source_that_are_not_in_target = (
            imgs_in_source_images_dir - imgs_that_already_existed_in_target_dataset
        )

        # Copy those images
        for img_name in tqdm(
            imgs_in_source_that_are_not_in_target, desc="Copying images"
        ):
            source_img_path = self.source_assoc.images_dir / img_name
            target_img_path = self.target_assoc.images_dir / img_name
            shutil.copyfile(source_img_path, target_img_path)

        # For each image that already existed in the target dataset ...
        for img_name in imgs_that_already_existed_in_target_dataset:
            # ... if among the geometries intersecting it in the target dataset ...
            geoms_intersecting_img = set(
                self.target_assoc.geoms_intersecting_img(img_name)
            )
            # ... there is a *new* geometry ...
            if (
                geoms_intersecting_img & geoms_that_will_be_added_to_target_dataset
                != set()
            ):
                # ... then we need to update the label for it,
                # so we delete the current label.
                (self.target_assoc.labels_dir / img_name).unlink(missing_ok=True)

        self.target_assoc.save()

        # Finally, we make all missing categorical labels in target dataset.
        # make labels
        if self.label_maker is not None:
            self.label_maker.make_labels(connector=self.target_connector)

        return self.target_assoc
