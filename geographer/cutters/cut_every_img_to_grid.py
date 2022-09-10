"""Cut images in the source dataset to a grid of images.

Dataset cutter that cuts images in the source dataset to a grid of
images.
"""

import logging
from pathlib import Path
from typing import Optional

from geographer.cutters.cut_iter_over_imgs import DSCutterIterOverImgs
from geographer.cutters.img_filter_predicates import (
    ImgFilterPredicate,
    ImgsNotPreviouslyCutOnly,
)
from geographer.cutters.single_img_cutter_grid import SingleImgCutterToGrid
from geographer.cutters.type_aliases import ImgSize

logger = logging.getLogger(__name__)


def get_cutter_every_img_to_grid(
    source_data_dir: Path,
    target_data_dir: Path,
    name: str,
    new_img_size: ImgSize = 512,
    img_filter_predicate: Optional[ImgFilterPredicate] = None,
) -> DSCutterIterOverImgs:
    """Return dataset cutter that cuts every image to a grid.

    Return dataset cutter that cuts every image in the source dataset
    to a grid of images.

    Args:
        source_data_dir: source data dir
        target_data_dir: target data dir
        name: name of cutter, used when saving the cutter
        new_img_size: size of new images. Defaults to 512.
        img_filter_predicate: image filter predicate to select
            images. Defaults to None (i.e. cut all
            images that have not been previously cut).

    Returns:
        DSCutterIterOverImgs: dataset cutter
    """
    if img_filter_predicate is None:
        img_filter_predicate = ImgsNotPreviouslyCutOnly()
    img_cutter = SingleImgCutterToGrid(new_img_size=new_img_size)

    return DSCutterIterOverImgs(
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        name=name,
        img_cutter=img_cutter,
        img_filter_predicate=img_filter_predicate,
        cut_imgs=[],
    )
