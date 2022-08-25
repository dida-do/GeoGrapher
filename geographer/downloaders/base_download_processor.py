"""Base class for processing a downloaded file."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Union

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
    ) -> Dict[Union[Literal['img_name', 'geometry', 'orig_crs_epsg_code'],
                    str], Any]:
        """Process a single download.

        Args:
            img_name (str): name of image
            download_dir (Path): directory containing download
            images_dir (Path): directory to place processed image in
            crs_epsg_code (int): EPSG code of crs image bounds should be returned in
            kwargs (Any): other keyword arguments

        Returns:
            return_dict: Contains information about the downloaded product.
            Keys should include: 'img_name', 'geometry', 'orig_crs_epsg_code'.
        """
