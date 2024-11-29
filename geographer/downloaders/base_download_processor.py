"""Base class for processing a downloaded file."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)


class RasterDownloadProcessor(ABC, BaseModel):
    """Base class for download processors."""

    default_process_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Default kwargs for the `process` method.",
    )

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

    @field_validator("default_process_kwargs")
    def validate_no_forbidden_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Validate default_process_kwargs contains no forbidden kwargs."""
        forbidden_keys = {
            "raster_name",
            "download_dir",
            "rasters_dir",
            "return_bounds_in_crs_epsg_code",
        }
        invalid_keys = forbidden_keys & set(value)

        if invalid_keys:
            msg = (
                "The following kwargs are set by RasterDownloaderForVectors "
                "and are not allowed: %s"
            )
            log.error(msg, {", ".join(invalid_keys)})
            raise ValueError(msg % {", ".join(invalid_keys)})

        return value
