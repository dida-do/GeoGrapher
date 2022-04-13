"""Dataset cutter that cuts images in the source dataset to a grid of images"""

from typing import List, Optional
import logging

from pydantic import Field
from rs_tools.cut.cut_iter_over_imgs import DSCutterIterOverImgs
from rs_tools.cut.type_aliases import ImgSize
from rs_tools.cut.img_filter_predicates import AlwaysTrue, ImgFilterPredicate
from rs_tools.cut.single_img_cutter_grid import SingleImgCutterToGrid

logger = logging.getLogger(__name__)


class DSCutterEveryImgToGrid(DSCutterIterOverImgs):
    """Dataset cutter that cuts images in the source dataset to a grid of images"""

    new_img_size: Field(
        ImgSize,
        description=
        "Size of cutouts. Passed to ImgToGridCutter during __init__ only.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        new_img_size: ImgSize = 512,
        img_filter_predicate: ImgFilterPredicate = AlwaysTrue(),
        cut_imgs: Optional[List[str]] = None,
    ) -> None:
        if cut_imgs is None:
            cut_imgs = []
        super().__init__(
            img_cutter=SingleImgCutterToGrid(new_img_size),
            img_filter_predicate=img_filter_predicate,
            cut_imgs=cut_imgs,
            new_img_size=new_img_size,
        )

    def _after_creating_or_updating(self):
        self.target_assoc._params_dict["img_size"] = self.new_img_size
        self.target_assoc.save()
