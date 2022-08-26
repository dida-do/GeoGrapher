"""Convert a dataset of GeoTiffs to NPYs."""

import logging
from typing import List, Literal

import numpy as np
import rasterio as rio
from pydantic import Field
from tqdm.auto import tqdm

from geographer.creator_from_source_dataset_base import \
    DSCreatorFromSourceWithBands
from geographer.img_bands_getter_mixin import ImgBandsGetterMixIn

log = logging.Logger(__name__)


class DSConverterGeoTiffToNpy(DSCreatorFromSourceWithBands,
                              ImgBandsGetterMixIn):
    """Convert a dataset of GeoTiffs to NPYs."""

    squeeze_label_channel_dim_if_single_channel: bool = Field(
        default=True,
        description="whether to squeeze the label channel dim/axis if possible"
    )
    channels_first_or_last_in_npy: Literal['last', 'first'] = Field(
        default='last',
        description=
        "Ignoring squeezing: 'last' -> (height, width, channels), 'first' -> (channels, height, width)."
    )

    def _create(self):
        self._create_or_update()

    def _update(self):
        self._create_or_update()

    def _create_or_update(self) -> None:

        # need this later
        geoms_that_will_be_added_to_target_dataset = set(
            self.source_assoc.geoms_df.index) - set(
                self.target_assoc.geoms_df.index)

        # build npy associator
        npy_raster_imgs = self._get_npy_raster_imgs()
        self.target_assoc.add_to_raster_imgs(npy_raster_imgs)
        self.target_assoc.add_to_geoms_df(self.source_assoc.geoms_df)

        # Determine which images to copy to target dataset
        imgs_that_already_existed_in_target_images_dir = {
            img_path.name
            for img_path in self.target_assoc.images_dir.iterdir()
        }

        # For each image that already existed in the target dataset ...
        for img_name in imgs_that_already_existed_in_target_images_dir:
            # ... if among the (vector) geometries intersecting it in the target dataset ...
            geoms_intersecting_img = set(
                self.target_assoc.geoms_intersecting_img(img_name))
            # ... there is a *new* (vector) geometry ...
            if geoms_intersecting_img & geoms_that_will_be_added_to_target_dataset != set(
            ):
                # ... then we need to update the label for it, so we delete the current label.
                (self.target_assoc.labels_dir /
                 img_name).unlink(missing_ok=True)

        # For the images_dir and labels_dir of the source tif and target npy dataset ...
        for tif_dir, npy_dir in zip(self.source_assoc.image_data_dirs,
                                    self.target_assoc.image_data_dirs):
            # ... go through all tif files. ...
            for tif_img_name in tqdm(self.source_assoc.raster_imgs.index,
                                     desc=f"Converting {tif_dir.name}"):
                # If the corresponding npy in the target image data dir does not exist ...
                if not (npy_dir /
                        self._npy_filename_from_tif(tif_img_name)).is_file():
                    # ... convert the tif: Open the tif file ...
                    with rio.open(tif_dir / tif_img_name) as src:

                        img_bands = self._get_bands_for_img(
                            self.bands,
                            tif_dir / tif_img_name,
                        )

                        # extract bands to list of arrays
                        seq_extracted_np_bands = [
                            src.read(band) for band in img_bands
                        ]

                        # new img path
                        new_npy_img_path = npy_dir / self._npy_filename_from_tif(
                            tif_img_name)

                        # axis along which to stack
                        if self.channels_first_or_last_in_npy == 'last':
                            axis = 2
                        else:  # 'first'
                            axis = 0

                        # stack band arrays into single tensor
                        np_img = np.stack(seq_extracted_np_bands, axis=axis)

                        # squeeze np_img if necessary
                        if str(
                                tif_dir.name
                        ) == "labels" and self.squeeze_label_channel_dim_if_single_channel:
                            if len(img_bands) == 1:
                                np_img = np.squeeze(np_img, axis=axis)

                        # save numpy array
                        np.save(new_npy_img_path, np_img)

        # ... and save the associator.
        self.target_assoc.save()
        self.save()

        return self.target_assoc

    def _get_npy_raster_imgs(self):
        npy_raster_imgs = self.source_assoc.raster_imgs
        # the line after the next destroys the index name of tif_assoc.raster_imgs,
        # so we remember it ...
        tif_assoc_raster_imgs_index_name = self.source_assoc.raster_imgs.index.name
        tif_img_name_list = npy_raster_imgs.index.tolist().copy(
        )  # (it's either the .tolist() or .copy() operation, don't understand why)
        npy_raster_imgs.index = list(
            map(self._npy_filename_from_tif, tif_img_name_list))
        # ... and then set it by hand
        npy_raster_imgs.index.name = tif_assoc_raster_imgs_index_name
        return npy_raster_imgs

    @staticmethod
    def _npy_filename_from_tif(tif_filename: str) -> str:
        """Return .npy filename from .tif filename."""
        return tif_filename[:-4] + ".npy"

    @staticmethod
    def _check_imgs_are_tifs(img_names: List[str]):
        """Make sure all images are GeoTiffs."""
        non_tif_imgs = list(
            filter(
                lambda s: not s.endswith('.tif'),
                img_names,
            ))
        if non_tif_imgs:
            raise ValueError("Only works with dataset of GeoTiff images!")
