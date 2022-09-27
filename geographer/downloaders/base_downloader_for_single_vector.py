"""Base class for downloaders for a single vector feature."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel
from shapely.geometry import Polygon


class RasterDownloaderForSingleVector(ABC, BaseModel):
    """Base class for downloaders for a single vector feature."""

    @abstractmethod
    def download(
        self,
        vector_name: Union[int, str],
        vector_geom: Polygon,
        download_dir: Path,
        previously_downloaded_rasters_set: set[Union[str, int]],
        **kwargs,
    ) -> dict[Union[Literal["raster_name", "raster_processed?"], str], Any]:
        """Download (a series of) raster(s) for a single vector feature.

        Args:
            vector_name: name of vector feature
            vector_geom: geometry of vector feature
            download_dir: directory to download to
            previously_downloaded_rasters_set: set of (names of)
            previously downloaded rasters
            kwargs: other keyword arguments

        Returns:
            Dict with a key 'list_raster_info_dicts': The corresponding value is a
            list of dicts containing (at least) the keys 'raster_name',
            'raster_processed?', each corresponding to the entries of rasters for the
            row defined by the raster.
        """
