"""Base class for downloaders for a single vector feature"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Set, Union
from shapely.geometry import Polygon
from pydantic import BaseModel

from rs_tools.connector import Connector


class ImgDownloaderForSingleVectorFeature(ABC, BaseModel):
    """Base class for downloaders for a single vector feature"""

    @abstractmethod
    def download(
        self,
        feature_name: Union[int, str],
        feature_geom: Polygon,
        connector: Connector,
        download_dir: Path,
        previously_downloaded_imgs_set: Set[Union[str, int]],
        **kwargs,
    ) -> Dict[Union[Literal['img_name', 'img_processed?'], str], Any]:
        """Download an image or a series of images for a single vector feature.

        Args:
            feature_name (Union[int, str]): name of vector feature
            feature_geom (shapely.geometry.Polygon): geometry of vector feature
            connector (Connector): connector
            download_dir (Path): directory to download to
            previously_downloaded_imgs_set (Set[Union[str, int]]): set of (names of)
            previously downloaded images
            kwargs (Any): other keyword arguments

        Returns:
            Dict with a key 'list_img_info_dicts': The corresponding value is a list of dicts
            containing (at least) the keys 'img_name', 'img_processed?', each corresponding
            to the entries of raster_imgs for the row defined by the image.
        """
