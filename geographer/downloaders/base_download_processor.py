"""Base class for processing a downloaded file."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

log = logging.getLogger(__name__)


class RasterDownloadProcessor(ABC, BaseModel):
    """Base class for download processors."""

    @abstractmethod
    def process(
        self,
        raster_name: str,
        download_dir: Path,
        rasters_dir: Path,
        return_bounds_in_crs_epsg_code: int,
        **params: Any,
    ) -> dict[
        Literal["raster_name", "geometry", "orig_crs_epsg_code"] | str, Any
    ]:
        """Process a single download.

        Args:
            raster_name:
                Name of raster
            download_dir:
                Directory containing download
            rasters_dir:
                Directory to place processed raster in
            crs_epsg_code:
                EPSG code of crs raster bounds should be returned in
            params:
                Additional keyword arguments. Corresponds to the processor_params
                argument of the RasterDownloaderForVectors.download method.

        Returns:
            return_dict:
                Contains information about the downloaded product.
                Keys should include: 'raster_name', 'geometry', 'orig_crs_epsg_code'.
        """
