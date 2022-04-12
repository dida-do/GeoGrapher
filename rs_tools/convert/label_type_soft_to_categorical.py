"""
Convert a dataset of GeoTiffs to NPYs.
"""

import logging
import shutil
from tqdm.auto import tqdm

from rs_tools.convert.convert_base import DSCreatorFromSource
from rs_tools.cut.img_bands_getter_mixin import ImgBandsGetterMixIn
from rs_tools.labels.label_type_conversion_utils import \
    convert_polygons_df_soft_cat_to_cat
from rs_tools import ImgPolygonAssociator

log = logging.Logger(__name__)


class DSConverterCatToSoftCat(DSCreatorFromSource, ImgBandsGetterMixIn):
    """Convert a dataset of GeoTiffs to NPYs."""

    def _create(self):
        self._create_or_update()

    def _update(self):
        self._create_or_update()

    def _create_or_update(self) -> ImgPolygonAssociator:

        if self.source_assoc.label_type != 'soft-categorical':
            raise ValueError(
                "Only works with label_type soft-categorical\n" +
                f"Current label type: {self.source_assoc.label_type}")

        # need this later
        polygons_that_will_be_added_to_target_dataset = set(
            self.source_assoc.polygons_df.index) - set(
                self.target_assoc.polygons_df.index)

        # add polygons to target dataset
        source_polygons_df_converted_to_soft_categorical_format = convert_polygons_df_soft_cat_to_cat(
            self.source_assoc.polygons_df)
        self.target_assoc.add_to_polygons_df(
            source_polygons_df_converted_to_soft_categorical_format)

        # add images to target dataset
        self.target_assoc.add_to_imgs_df(self.source_assoc.imgs_df)

        # Determine which images to copy to target dataset
        imgs_that_already_existed_in_target_dataset = {
            img_name
            for img_name in self.target_assoc.imgs_df.index
            if (self.target_assoc.images_dir / img_name).is_file()
        }
        imgs_in_source_images_dir = {
            img_name
            for img_name in self.source_assoc.imgs_df.index
        }
        imgs_in_source_that_are_not_in_target = imgs_in_source_images_dir - imgs_that_already_existed_in_target_dataset

        # Copy those images
        for img_name in tqdm(imgs_in_source_that_are_not_in_target,
                             desc='Copying images'):
            source_img_path = self.source_assoc.images_dir / img_name
            target_img_path = self.target_assoc.images_dir / img_name
            shutil.copyfile(source_img_path, target_img_path)

        # For each image that already existed in the target dataset ...
        for img_name in imgs_that_already_existed_in_target_dataset:
            # ... if among the polygons intersecting it in the target dataset ...
            polygons_intersecting_img = set(
                self.target_assoc.polygons_intersecting_img(img_name))
            # ... there is a *new* polygon ...
            if polygons_intersecting_img & polygons_that_will_be_added_to_target_dataset != set(
            ):
                # ... then we need to update the label for it, so we delete the current label.
                (self.target_assoc.labels_dir /
                 img_name).unlink(missing_ok=True)

        # Finally, we make all missing categorical labels in target dataset.
        self.target_assoc.make_labels()
        self.target_assoc.save()

        log.info(
            "_convert_or_update_dataset_soft_categorical_to_categorical: done!"
        )

        return self.target_assoc
