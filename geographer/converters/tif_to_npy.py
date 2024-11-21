"""Convert a dataset of GeoTiffs to NPYs."""

import logging
from typing import Literal

import numpy as np
import rasterio as rio
from pydantic import Field
from tqdm.auto import tqdm

from geographer.creator_from_source_dataset_base import DSCreatorFromSourceWithBands
from geographer.raster_bands_getter_mixin import RasterBandsGetterMixIn

log = logging.Logger(__name__)


class DSConverterGeoTiffToNpy(DSCreatorFromSourceWithBands, RasterBandsGetterMixIn):
    """Convert a dataset of GeoTiffs to NPYs."""

    squeeze_label_channel_dim_if_single_channel: bool = Field(
        default=True,
        description="whether to squeeze the label channel dim/axis if possible",
    )
    channels_first_or_last_in_npy: Literal["last", "first"] = Field(
        default="last",
        description="Ignoring squeezing: 'last' -> (height, width, channels), "
        "'first' -> (channels, height, width).",
    )

    def _create(self):
        self._create_or_update()

    def _update(self):
        self._create_or_update()

    def _create_or_update(self) -> None:
        # need this later
        geoms_that_will_be_added_to_target_dataset = set(
            self.source_assoc.geoms_df.index
        ) - set(self.target_assoc.geoms_df.index)

        # build npy associator
        npy_rasters = self._get_npy_rasters()
        self.target_assoc.add_to_rasters(npy_rasters)
        self.target_assoc.add_to_geoms_df(self.source_assoc.geoms_df)

        # Determine which rasters to copy to target dataset
        rasters_that_already_existed_in_target_rasters_dir = {
            raster_path.name for raster_path in self.target_assoc.rasters_dir.iterdir()
        }

        # For each raster that already existed in the target dataset ...
        for raster_name in rasters_that_already_existed_in_target_rasters_dir:
            # ... if among the (vector) geometries intersecting
            # it in the target dataset ...
            geoms_intersecting_raster = set(
                self.target_assoc.geoms_intersecting_raster(raster_name)
            )
            # ... there is a *new* (vector) geometry ...
            if (
                geoms_intersecting_raster & geoms_that_will_be_added_to_target_dataset
                != set()
            ):
                # ... then we need to update the label for it,
                # so we delete the current label.
                (self.target_assoc.labels_dir / raster_name).unlink(missing_ok=True)

        # For the rasters_dir and labels_dir of the source tif
        # and target npy dataset ...
        for tif_dir, npy_dir in zip(
            self.source_assoc.raster_data_dirs, self.target_assoc.raster_data_dirs
        ):
            # ... go through all tif files. ...
            for tif_raster_name in tqdm(
                self.source_assoc.rasters.index, desc=f"Converting {tif_dir.name}"
            ):
                # If the corresponding npy in the target raster data dir
                # does not exist ...
                if not (
                    npy_dir / self._npy_filename_from_tif(tif_raster_name)
                ).is_file():
                    # ... convert the tif: Open the tif file ...
                    with rio.open(tif_dir / tif_raster_name) as src:
                        raster_bands = self._get_bands_for_raster(
                            self.bands,
                            tif_dir / tif_raster_name,
                        )

                        # extract bands to list of arrays
                        seq_extracted_np_bands = [
                            src.read(band) for band in raster_bands
                        ]

                        # new raster path
                        new_npy_raster_path = npy_dir / self._npy_filename_from_tif(
                            tif_raster_name
                        )

                        # axis along which to stack
                        if self.channels_first_or_last_in_npy == "last":
                            axis = 2
                        else:  # 'first'
                            axis = 0

                        # stack band arrays into single tensor
                        np_raster = np.stack(seq_extracted_np_bands, axis=axis)

                        # squeeze np_raster if necessary
                        if (
                            str(tif_dir.name) == "labels"
                            and self.squeeze_label_channel_dim_if_single_channel
                        ):
                            if len(raster_bands) == 1:
                                np_raster = np.squeeze(np_raster, axis=axis)

                        # save numpy array
                        np.save(new_npy_raster_path, np_raster)

        # ... and save the associator.
        self.target_assoc.save()
        self.save()

        return self.target_assoc

    def _get_npy_rasters(self):
        npy_rasters = self.source_assoc.rasters
        # the line after the next destroys the index name of tif_assoc.rasters,
        # so we remember it ...
        tif_assoc_rasters_index_name = self.source_assoc.rasters.index.name
        tif_raster_name_list = (
            npy_rasters.index.tolist().copy()
        )  # (it's either the .tolist() or .copy() operation, don't understand why)
        npy_rasters.index = list(map(self._npy_filename_from_tif, tif_raster_name_list))
        # ... and then set it by hand
        npy_rasters.index.name = tif_assoc_rasters_index_name
        return npy_rasters

    @staticmethod
    def _npy_filename_from_tif(tif_filename: str) -> str:
        """Return .npy filename from .tif filename."""
        return tif_filename[:-4] + ".npy"

    @staticmethod
    def _check_rasters_are_tifs(raster_names: list[str]):
        """Make sure all rasters are GeoTiffs."""
        non_tif_rasters = list(
            filter(
                lambda s: not s.endswith(".tif"),
                raster_names,
            )
        )
        if non_tif_rasters:
            raise ValueError("Only works with dataset of GeoTiff rasters!")
