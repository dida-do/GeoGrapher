"""
Dataset cutter that cuts out images around polygons.
"""

import logging
from typing import Optional, Union
from rs_tools.cut.type_aliases import ImgSize
from rs_tools.cut.cut_iter_over_polygons import DSCutterIterOverPolygons
from rs_tools.cut.img_selectors import ImgSelector, RandomImgSelector
from rs_tools.cut.polygon_filter_predicates import IsPolygonMissingImgs, PolygonFilterPredicate
from rs_tools.cut.single_img_cutter_around_polygon import \
    ImgsAroundPolygonCutter

logger = logging.getLogger(__name__)


class DSCutterImgsAroundEveryPolygon(DSCutterIterOverPolygons):
    """Dataset cutter that cuts out images around polygons."""

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
        """Dataset cutter that cuts out images around polygons.

        Args:
            new_img_size (Optional[ImgSize], optional): size of cutouts in 'centered' or 'random' mode. Defaults to 512.
            min_new_img_size (Optional[ImgSize], optional): Min size of cutouts for 'variable' mode. Defaults to 64.
            scaling_factor (Union[None, float], optional): Scaling factor for 'variable' mode. Defaults to 1.2.
            target_img_count (int, optional): Targetted number of images per polygon. Defaults to 1.
            mode (str, optional): One of 'random', 'centered', or 'variable'. Defaults to 'random'.
            random_seed (int, optional): Random seed. Defaults to 10.
        """

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