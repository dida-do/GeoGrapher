"""Base class for downloaders for a single polygon"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Set, Union
from shapely.geometry import Polygon
from pydantic import BaseModel

from rs_tools.img_polygon_associator import ImgPolygonAssociator


class ImgDownloaderForSinglePolygon(ABC, BaseModel):
    """Base class for downloaders for a single polygon"""

    @abstractmethod
    def download(
        self,
        polygon_name: Union[int, str],
        polygon_geometry: Polygon,
        assoc: ImgPolygonAssociator,
        download_dir: Path,
        previously_downloaded_imgs_set: Set[Union[str, int]],
        **kwargs,
    ) -> Dict[Union[Literal['img_name', 'img_processed?'], str], Any]:
        """Download an image or a series of images for a single polygon.

        Args:
            polygon_name (Union[int, str]): name of polygon
            polygon_geometry (shapely.geometry.Polygon): polygon geometry
            assoc (ImgPolygonAssociator): associator
            download_dir (Path): directory to download to
            previously_downloaded_imgs_set (Set[Union[str, int]]): set of (names of)
            previously downloaded images
            kwargs (Any): other keyword arguments

        Returns:
            Dict with a key 'list_img_info_dicts': The corresponding value is a list of dicts
            containing (at least) the keys 'img_name', 'img_processed?', each corresponding
            to the entries of imgs_df for the row defined by the image.
        """
