"""Base class for downloaders for a single vector feature."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from shapely.geometry import Polygon

log = logging.getLogger(__name__)


class RasterDownloaderForSingleVector(ABC, BaseModel):
    """Base class for downloaders for a single vector feature."""

    @abstractmethod
    def download(
        self,
        vector_name: str | int,
        vector_geom: Polygon,
        download_dir: Path,
        previously_downloaded_rasters_set: set[str | int],
        **params: Any,
    ) -> dict[Literal["raster_name", "raster_processed?"] | str, Any]:
        """Download (a series of) raster(s) for a single vector feature.

        Args:
            vector_name:
                Name of vector feature
            vector_geom:
                Geometry of vector feature
            download_dir:
                Directory in which raw downloads are placed
            previously_downloaded_rasters_set:
                Set of (names of) previously downloaded rasters
            params:
                Additional keyword arguments. Corresponds to the downloader_params
                argument of the RasterDownloaderForVectors.download method.

        Returns:
            Dict with a key 'list_raster_info_dicts': The corresponding value is a
            list of dicts containing (at least) the keys 'raster_name',
            'raster_processed?', each corresponding to the entries of rasters for the
            row defined by the raster.
        """
