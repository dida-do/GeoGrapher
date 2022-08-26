"""Base class for processing a downloaded file."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Union, dict

from pydantic import BaseModel


class ImgDownloadProcessor(ABC, BaseModel):
    """Base class for download processors."""

    @abstractmethod
    def process(
        self,
        img_name: str,
        download_dir: Path,
        images_dir: Path,
        return_bounds_in_crs_epsg_code: int,
        **kwargs: Any,
    ) -> dict[Union[Literal["img_name", "geometry", "orig_crs_epsg_code"], str], Any]:
        """Process a single download.

        Args:
            img_name: name of image
            download_dir: directory containing download
            images_dir: directory to place processed image in
            crs_epsg_code: EPSG code of crs image bounds should be returned in
            kwargs: other keyword arguments

        Returns:
            return_dict: Contains information about the downloaded product.
                Keys should include: 'img_name', 'geometry', 'orig_crs_epsg_code'.
        """
