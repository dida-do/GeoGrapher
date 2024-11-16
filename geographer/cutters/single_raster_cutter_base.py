"""Abstract base class for single raster cutters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

import rasterio as rio
from affine import Affine
from pydantic import BaseModel
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import box

from geographer.connector import Connector
from geographer.global_constants import RASTER_IMGS_INDEX_NAME
from geographer.raster_bands_getter_mixin import RasterBandsGetterMixIn

logger = logging.getLogger(__name__)


class SingleRasterCutter(ABC, BaseModel, RasterBandsGetterMixIn):
    """Base class for SingleRasterCUtter."""

    @abstractmethod
    def _get_windows_transforms_raster_names(
        self,
        source_raster_name: str,
        source_connector: Connector,
        target_connector: Connector | None = None,
        new_rasters_dict: dict | None = None,
        **kwargs: Any,
    ) -> list[Tuple[Window, Affine, str]]:
        """Return windows, window transforms, and new rasters.

        Return a list of rasterio windows, window transformations, and new
        raster names. The returned list will be used to create the new rasters
        and labels. Override to subclass.

        Args:
            source_raster_name: name of raster in source dataset to be cut.
            target_connector: connector of target dataset
            new_rasters_dict: dict with keys index or column names of
                target_connector.rasters and values lists of entries corresponding
                to rasters containing information about
            cut rasters not yet appended to target_connector.rasters
            kwargs: keyword arguments to be used in subclass implementations.

        Returns:
            list of rasterio windows, window transform, and new raster names.
        """

    def __call__(
        self,
        raster_name: str,
        source_connector: Connector,
        target_connector: Connector | None = None,
        new_rasters_dict: dict | None = None,
        bands: dict[str, list[int] | None] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Cut new rasters and return return_dict.

        Cut new rasters from source raster and return a dict with keys the
        index and column names of the rasters to be created by the calling
        dataset cutter and values lists containing the new raster names and
        corresponding entries for the new rasters.

        Args:
            raster_name: name of raster in source dataset to be cut.
            target_connector: connector of target dataset
            new_rasters_dict: dict with keys index or column names of
                target_connector.rasters and values lists of entries correspondong
                to rasters containing information
            about cut rasters not yet appended to target_connector.rasters
            kwargs: optional keyword arguments for _get_windows_transforms_raster_names

        Returns:
            dict of lists that containing the data to be put in the rasters of the
            connector to be constructed for the created rasters.

        Note:
            The __call__ function should be able to access the information
            contained in the target (and source) connector but should *not*
            modify its arguments! Since create_or_update does not concatenate
            the information about the new rasters that have been cut to the
            target_connector.rasters until after all vector features or rasters
            have been iterated over and we want to be able to use RasterSelectors
            _during_ such an iteration, we allow the call function to also depend
            on a new_rasters_dict argument which contains the information about the new
            rasters that have been cut. Unlike the target_connector.rasters, the
            target_connector.vectors and graph are updated during the
            iteration. One should thus think of the target_connector and
            new_rasters_dict arguments together as the actual the target connector
            argument.
        """
        # dict to accumulate information about the newly created rasters
        rasters_from_cut_dict = {
            index_or_col_name: []
            for index_or_col_name in [RASTER_IMGS_INDEX_NAME]
            + list(source_connector.rasters.columns)
        }

        windows_transforms_raster_names = self._get_windows_transforms_raster_names(
            source_raster_name=raster_name,
            source_connector=source_connector,
            target_connector=target_connector,
            new_rasters_dict=new_rasters_dict,
            **kwargs,
        )

        for (
            window,
            window_transform,
            new_raster_name,
        ) in windows_transforms_raster_names:
            # Make new raster and label in target dataset ...
            raster_bounds_in_raster_crs, raster_crs = self._make_new_raster_and_label(
                new_raster_name=new_raster_name,
                source_raster_name=raster_name,
                source_connector=source_connector,
                target_connector=target_connector,
                window=window,
                window_transform=window_transform,
                bands=bands,
            )

            # ... gather all the information about the raster in a dict ...
            single_new_raster_info_dict = self._make_raster_info_dict(
                new_raster_name=new_raster_name,
                source_raster_name=raster_name,
                source_connector=source_connector,
                raster_bounds_in_raster_crs=raster_bounds_in_raster_crs,
                raster_crs=raster_crs,
            )

            # ... and accumulate that information.
            for key in rasters_from_cut_dict.keys():
                rasters_from_cut_dict[key].append(single_new_raster_info_dict[key])

        return rasters_from_cut_dict

    def _make_raster_info_dict(
        self,
        new_raster_name: str,
        source_raster_name: str,
        source_connector: Connector,
        raster_bounds_in_raster_crs: Tuple[float, float, float, float],
        raster_crs: CRS,
    ) -> dict:
        """Return an raster info dict for a single new raster.

        An raster info dict contains the following key/value pairs:
            - keys: the columns names of the raster to be created by calling
                dataset cutter,
            - values: the entries to be written in those columns for the new raster.

        Args:
            new_raster_name: name of new raster
            source_raster_name: name of source raster
            raster_bounds_in_raster_crs: raster bounds
            raster_crs: CRS of raster

        Returns:
            dict: raster info dict (see above)
        """
        raster_bounding_rectangle_in_rasters_crs = box(
            *transform_bounds(
                raster_crs, source_connector.rasters.crs, *raster_bounds_in_raster_crs
            )
        )

        single_new_raster_info_dict = {
            RASTER_IMGS_INDEX_NAME: new_raster_name,
            "geometry": raster_bounding_rectangle_in_rasters_crs,
            "orig_crs_epsg_code": raster_crs.to_epsg(),
            "raster_processed?": True,
        }

        # Copy over any remaining information about the raster from
        # source_connector.rasters.
        for col in set(source_connector.rasters.columns) - {
            RASTER_IMGS_INDEX_NAME,
            "geometry",
            "orig_crs_epsg_code",
            "raster_processed?",
        }:
            single_new_raster_info_dict[col] = source_connector.rasters.loc[
                source_raster_name, col
            ]

        return single_new_raster_info_dict

    def _make_new_raster_and_label(
        self,
        new_raster_name: str,
        source_raster_name: str,
        source_connector: Connector,
        target_connector: Connector,
        window: Window,
        window_transform: Affine,
        bands: dict[str, list[int] | None] | None,
    ) -> Tuple[Tuple[float, float, float, float], CRS]:
        """Make a new raster and label.

        Make a new raster and label with given raster name from the given
        window and transform.

        Args:
            new_raster_name: name of new raster
            source_raster_name: name of source raster
            window: window
            window_transform: window transform

        Returns:
            tuple of bounds (in raster CRS) and CRS of new raster
        """
        for count, (source_rasters_dir, target_rasters_dir) in enumerate(
            zip(source_connector.raster_data_dirs, target_connector.raster_data_dirs)
        ):
            source_raster_path = source_rasters_dir / source_raster_name
            dst_raster_path = target_rasters_dir / new_raster_name

            if (
                not source_raster_path.is_file() and count > 0
            ):  # count == 0 corresponds to rasters_dir
                continue
            else:
                raster_bands = self._get_bands_for_raster(bands, source_raster_path)

                # write raster window to destination raster geotif
                bounds_in_raster_crs, crs = self._write_window_to_geotif(
                    source_raster_path,
                    dst_raster_path,
                    raster_bands,
                    window,
                    window_transform,
                )

            # make sure all rasters/labels/masks have same bounds and crs
            if count == 0:
                raster_bounds_in_raster_crs, raster_crs = bounds_in_raster_crs, crs
            else:
                assert (
                    crs == raster_crs
                ), f"new raster and {target_rasters_dir.name} crs disagree!"
                assert (
                    bounds_in_raster_crs == raster_bounds_in_raster_crs
                ), f"new raster and {target_rasters_dir.name} bounds disagree"

        return raster_bounds_in_raster_crs, raster_crs

    def _write_window_to_geotif(
        self,
        src_raster_path: Path | str,
        dst_raster_path: Path | str,
        raster_bands: list[int],
        window: Window,
        window_transform: Affine,
    ) -> Tuple[Tuple[float, float, float, float], CRS]:
        """Write window from source GeoTiff to new GeoTiff.

        Args:
            src_raster_path: path of source GeoTiff
            dst_raster_path: path to GeoTiff to be created
            raster_bands: bands to extract from source GeoTiff
            window: window to cut out from source GeoTiff
            window_transform: window transform of window

        Returns:
            bounds (in raster CRS) and CRS of new raster
        """
        # Open source ...
        with rio.open(src_raster_path) as src:
            # and destination ...
            Path(dst_raster_path).parent.mkdir(exist_ok=True, parents=True)
            with rio.open(
                Path(dst_raster_path),
                "w",
                driver="GTiff",
                height=window.height,
                width=window.width,
                count=len(raster_bands),
                dtype=src.profile["dtype"],
                crs=src.crs,
                transform=window_transform,
            ) as dst:
                # ... and go through the bands.
                for target_band, source_band in enumerate(raster_bands, start=1):
                    # Read window for that band from source ...
                    new_raster_band_raster = src.read(source_band, window=window)

                    # ... write to new geotiff.
                    dst.write(new_raster_band_raster, target_band)

        return dst.bounds, dst.crs
