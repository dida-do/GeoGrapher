"""Dataset cutter that cuts images in the source dataset to a grid of images"""

from pathlib import Path
from typing import List, Optional
import logging

from pydantic import Field
from geographer.creator_from_source_dataset_base import DSCreatorFromSourceWithBands
from geographer.cutters.cut_iter_over_imgs import DSCutterIterOverImgs
from geographer.cutters.type_aliases import ImgSize
from geographer.cutters.img_filter_predicates import ImgsNotPreviouslyCutOnly, ImgFilterPredicate
from geographer.cutters.single_img_cutter_grid import SingleImgCutterToGrid

logger = logging.getLogger(__name__)

def get_cutter_every_img_to_grid(
    source_data_dir: Path,
    target_data_dir: Path,
    name: str,
    new_img_size: ImgSize = 512,
    img_filter_predicate: Optional[ImgFilterPredicate] = None,
) -> DSCutterIterOverImgs:
    """
    Return dataset cutter that cuts every image in the source dataset to
    a grid of images.

    Args:
        source_data_dir (Path): source data dir
        target_data_dir (Path): target data dir
        name (str): name of cutter, used when saving the cutter
        new_img_size (ImgSize, optional): size of new images. Defaults to 512.
        img_filter_predicate (Optional[ImgFilterPredicate], optional): image
            filter predicate to select images. Defaults to None (i.e. cut all
            images that have not been previously cut).

    Returns:
        DSCutterIterOverImgs: dataset cutter
    """
    if img_filter_predicate is None:
        img_filter_predicate = ImgsNotPreviouslyCutOnly()
    img_cutter=SingleImgCutterToGrid(new_img_size=new_img_size)

    return DSCutterIterOverImgs(
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        name=name,
        img_cutter=img_cutter,
        img_filter_predicate=img_filter_predicate,
        cut_imgs=[],
    )
