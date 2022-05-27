"""Dataset cutter that cuts images in the source dataset to a grid of images"""

from pathlib import Path
from typing import List, Optional
import logging

from pydantic import Field
from geographer.cutters.cut_iter_over_imgs import DSCutterIterOverImgs
from geographer.cutters.type_aliases import ImgSize
from geographer.cutters.img_filter_predicates import ImgsNotPreviouslyCutOnly, ImgFilterPredicate
from geographer.cutters.single_img_cutter_grid import SingleImgCutterToGrid

logger = logging.getLogger(__name__)


class DSCutterEveryImgToGrid(DSCutterIterOverImgs):
    """Dataset cutter that cuts images in the source dataset to a grid of images"""

    new_img_size: ImgSize = Field(
        description=
        "Size of cutouts. Passed to ImgToGridCutter during __init__ only.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        source_data_dir: Path,
        target_data_dir: Path,
        name: str,
        new_img_size: ImgSize = 512,
        img_filter_predicate: ImgFilterPredicate = ImgsNotPreviouslyCutOnly(),
        cut_imgs: Optional[List[str]] = None,
    ) -> None:
        if cut_imgs is None:
            cut_imgs = []
        super().__init__(
            source_data_dir=source_data_dir,
            target_data_dir=target_data_dir,
            name=name,
            img_cutter=SingleImgCutterToGrid(new_img_size=new_img_size),
            img_filter_predicate=img_filter_predicate,
            cut_imgs=cut_imgs,
            new_img_size=new_img_size,
        )

    def _after_creating_or_updating(self):
        self.target_connector.attrs["img_size"] = self.new_img_size
        self.target_connector.save()
