"""Base class for processing a downloaded file."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel


class RasterDownloadProcessor(ABC, BaseModel):
    """Base class for download processors."""

    @abstractmethod
    def process(
        self,
        raster_name: str,
        download_dir: Path,
        rasters_dir: Path,
        return_bounds_in_crs_epsg_code: int,
        **kwargs: Any,
    ) -> dict[
        Union[Literal["raster_name", "geometry", "orig_crs_epsg_code"], str], Any
    ]:
        """Process a single download.

        Args:
            raster_name: name of raster
            download_dir: directory containing download
            rasters_dir: directory to place processed raster in
            crs_epsg_code: EPSG code of crs raster bounds should be returned in
            kwargs: other keyword arguments

        Returns:
            return_dict: Contains information about the downloaded product.
                Keys should include: 'raster_name', 'geometry', 'orig_crs_epsg_code'.
        """
