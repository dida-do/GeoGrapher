"""Mixin to cut datasets of GeoTiffs (or update previously cut datasets) by
cutting each image in the source dataset to a grid of images."""

import logging

from rs_tools.cut.cut_iter_over_imgs import DSCutterIterOverImgs
from rs_tools.cut.type_aliases import ImgSize
from rs_tools.cut.img_filter_predicates import AlwaysTrue, ImgFilterPredicate
from rs_tools.cut.single_img_cutter_grid import ImgToGridCutter

logger = logging.getLogger(__name__)


class CreateDSCutEveryImgToGrid(DSCutterIterOverImgs):
    """Dataset cutter that cuts images in the source dataset to a grid of images"""

    new_img_size: ImgSize

    def __init__(
        self,
        new_img_size: ImgSize = 512,
        img_filter_predicate: ImgFilterPredicate = AlwaysTrue(),
        cut_imgs: Optional[List[str]] = None,
    ) -> None:
        if cut_imgs is None:
            cut_imgs = []
        super().__init__(
            img_cutter=ImgToGridCutter(new_img_size),
            img_filter_predicate=img_filter_predicate,
            cut_imgs=cut_imgs,
            new_img_size=new_img_size,
        )

    def _after_cutting(self):
        self.target_assoc._params_dict["img_size"] = self.new_img_size
        self.target_assoc.save()
