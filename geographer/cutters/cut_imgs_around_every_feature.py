"""
Dataset cutter that cuts out images around vector features.
"""

import logging
from typing import Optional, Union
from geographer.cutters.type_aliases import ImgSize
from geographer.cutters.cut_iter_over_features import DSCutterIterOverFeatures
from geographer.cutters.img_selectors import ImgSelector, RandomImgSelector
from geographer.cutters.feature_filter_predicates import IsFeatureMissingImgs, FeatureFilterPredicate
from geographer.cutters.single_img_cutter_around_feature import \
    SingleImgCutterAroundFeature

logger = logging.getLogger(__name__)


class DSCutterImgsAroundEveryFeature(DSCutterIterOverFeatures):
    """Dataset cutter that cuts out images around vector features."""

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
        """Dataset cutter that cuts out images around vector feature.

        Args:
            new_img_size (Optional[ImgSize], optional): size of cutouts in 'centered'
                or 'random' mode. Defaults to 512.
            min_new_img_size (Optional[ImgSize], optional): Min size of cutouts
                for 'variable' mode. Defaults to 64.
            scaling_factor (Union[None, float], optional): Scaling factor for 'variable' mode.
                Defaults to 1.2.
            target_img_count (int, optional): Targetted number of images per vector feature. Defaults to 1.
            mode (str, optional): One of 'random', 'centered', or 'variable'. Defaults to 'random'.
            random_seed (int, optional): Random seed. Defaults to 10.
        """

        is_feature_missing_imgs: FeatureFilterPredicate = IsFeatureMissingImgs(
            target_img_count)
        random_img_selector: ImgSelector = RandomImgSelector(target_img_count)
        small_imgs_around_features_cutter = SingleImgCutterAroundFeature(
            mode=mode,
            new_img_size=new_img_size,
            min_new_img_size=min_new_img_size,
            scaling_factor=scaling_factor,
            random_seed=random_seed,
        )

        super().__init__(
            feature_filter_predicate=is_feature_missing_imgs,
            img_selector=random_img_selector,
            img_cutter=small_imgs_around_features_cutter,
            new_img_size=new_img_size,
            min_new_img_size=min_new_img_size,
            scaling_factor=scaling_factor,
            target_img_count=target_img_count,
            mode=mode,
            random_seed=random_seed,
            **data,
        )

    def _after_creating_or_updating(self):
        if self.mode in {'random', 'centered'}:
            self.target_connector.attrs["img_size"] = self.new_img_size
            self.target_connector.save()
