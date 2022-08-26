"""Dataset cutter that cuts out images around vector features."""

import logging
from pathlib import Path
from typing import Literal, Optional, Union

from geographer.cutters.cut_iter_over_features import DSCutterIterOverFeatures
from geographer.cutters.feature_filter_predicates import (
    FeatureFilterPredicate,
    IsFeatureMissingImgs,
)
from geographer.cutters.img_selectors import ImgSelector, RandomImgSelector
from geographer.cutters.single_img_cutter_around_feature import (
    SingleImgCutterAroundFeature,
)
from geographer.cutters.type_aliases import ImgSize

logger = logging.getLogger(__name__)


def get_cutter_imgs_around_every_feature(
    source_data_dir: Union[Path, str],
    target_data_dir: Union[Path, str],
    name: str,
    mode: Literal["random", "centered", "variable"] = "random",
    new_img_size: Optional[ImgSize] = 512,
    min_new_img_size: Optional[ImgSize] = None,
    scaling_factor: Optional[float] = None,
    target_img_count: int = 1,
    bands: Optional[dict] = None,
    random_seed: int = 10,
) -> DSCutterIterOverFeatures:
    """Return dataset cutter that creates cutouts around vector features.

    In "random" or "centered" all cutouts will be of the same size.
    In "variable" mode the size of a cutout is the maximum of
    scaling_factor*size_of_bounding_box or min_new_img_size

    Args:
        source_data_dir: source data dir
        target_data_dir: target data dir
        name: name
        new_img_size: new image size for
            "random" and "centered" modes. Defaults to 512.
        min_new_img_size: lower bound on image
            size for "variable" mode. Defaults to None.
        scaling_factor: scaling factor for
            "variable" mode. Defaults to None.
        target_img_count: targeted number of images per vector
            feature to create. Defaults to 1.
        mode: On. Defaults to "random".
        bands: bands dict. Defaults to None.
        random_seed: random seed. Defaults to 10.

    Returns:
        DSCutterIterOverFeatures: dataset cutter
    """

    is_feature_missing_imgs: FeatureFilterPredicate = IsFeatureMissingImgs(
        target_img_count=target_img_count
    )
    random_img_selector: ImgSelector = RandomImgSelector(
        target_img_count=target_img_count
    )
    small_imgs_around_features_cutter = SingleImgCutterAroundFeature(
        mode=mode,
        new_img_size=new_img_size,
        min_new_img_size=min_new_img_size,
        scaling_factor=scaling_factor,
        random_seed=random_seed,
    )

    return DSCutterIterOverFeatures(
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        name=name,
        feature_filter_predicate=is_feature_missing_imgs,
        img_selector=random_img_selector,
        img_cutter=small_imgs_around_features_cutter,
        bands=bands,
    )
