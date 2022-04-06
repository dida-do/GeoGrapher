"""REWRITE DOCSTRINGS.

TODO: Do we need the random_seed field here? Should it be in the subclass.
TODO: bands!

Mixin that implements creating or updating a dataset of GeoTiffs by
cutting images around polygons from a source dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
from rs_tools.cut.cut_iter_over_polygons_mixin import DSCutterIterOverPolygons

from rs_tools.cut.type_aliases import ImgSize

if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator

from rs_tools.cut.img_selectors import ImgSelector, RandomImgSelector
from rs_tools.cut.polygon_filter_predicates import IsPolygonMissingImgs, PolygonFilterPredicate
from rs_tools.cut.single_img_cutter_around_polygon import \
    ImgsAroundPolygonCutter

logger = logging.getLogger(__name__)


class DSCutterImgsAroundEveryPolygon(DSCutterIterOverPolygons):

    new_img_size: Optional[ImgSize]
    min_new_img_size: Optional[ImgSize]
    scaling_factor: Union[None, float]
    target_img_count: int
    mode: str
    random_seed: int

    def __init__(self,
                 new_img_size: Optional[ImgSize] = 512,
                 min_new_img_size: Optional[ImgSize] = 64,
                 scaling_factor: Union[None, float] = 1.2,
                 target_img_count: int = 1,
                 mode: str = 'random',
                 random_seed: int = 10,
                 **data):

        is_polygon_missing_imgs: PolygonFilterPredicate = IsPolygonMissingImgs(
            target_img_count)
        random_img_selector: ImgSelector = RandomImgSelector(target_img_count)
        small_imgs_around_polygons_cutter = ImgsAroundPolygonCutter(
            mode=mode,
            new_img_size=new_img_size,
            min_new_img_size=min_new_img_size,
            scaling_factor=scaling_factor,
            random_seed=random_seed,
            img_bands=img_bands,
            label_bands=label_bands,
        )

        super().__init__(
            polygon_filter_predicate=is_polygon_missing_imgs,
            img_selector=random_img_selector,
            img_cutter=small_imgs_around_polygons_cutter,
            new_img_size=new_img_size,
            min_new_img_size=min_new_img_size,
            scaling_factor=scaling_factor,
            target_img_count=target_img_count,
            mode=mode,
            random_seed=random_seed,
            img_bands=img_bands,
            label_bands=label_bands,
            **data,
        )