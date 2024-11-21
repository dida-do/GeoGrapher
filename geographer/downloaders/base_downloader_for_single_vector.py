"""Base class for downloaders for a single vector feature."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator
from shapely.geometry import Polygon

log = logging.getLogger(__name__)


class RasterDownloaderForSingleVector(ABC, BaseModel):
    """Base class for downloaders for a single vector feature."""

    default_download_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Default kwargs for the `download` method.",
    )

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

    @field_validator("default_download_kwargs")
    def validate_no_forbidden_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Validate default_download_kwargs contains no forbidden kwargs."""
        forbidden_keys = {
            "vector_name",
            "vector_geom",
            "download_dir",
            "previously_downloaded_rasters_set",
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
