# TODO: rename MOCK_DOWNLOAD_SOURCE_DATA_DIR?

from ftplib import all_errors
from functools import reduce
import os
import shutil
from typing import List
import pandas as pd
from geopandas import GeoDataFrame
from geographer import Connector
from geographer.converters.combine_remove_seg_classes import DSConverterCombineRemoveClasses
from geographer.label_makers import SegLabelMakerCategorical, SegLabelMakerSoftCategorical
from geographer.utils.utils import deepcopy_gdf
from tests.mock_download_test import MOCK_DOWNLOAD_SOURCE_DATA_DIR
from tests.utils import create_dummy_imgs, delete_dummy_images, get_test_dir

COMBINER_NAME = "combine_remove"

def class_probs_add_to_one(df: pd.DataFrame) -> bool:
    """Check whether class probabilities add up to one"""
    return all(df[[col for col in df.columns if col.startswith("prob_of_class_")]].sum(axis=1).round(2) == 1.0)

def test_combine_remove_seg_classes_categorical():

    ## create source dataset with categorical segmentation labels
    mock_download_source_data_dir = get_test_dir() / MOCK_DOWNLOAD_SOURCE_DATA_DIR
    mock_download_source_connector = Connector.from_data_dir(mock_download_source_data_dir)

    source_data_dir = get_test_dir() / "temp" / "combine_remove_source"
    shutil.rmtree(source_data_dir, ignore_errors=True)
    source_connector = mock_download_source_connector.empty_connector_same_format(data_dir=source_data_dir)

    # select images intersecting the most vector features
    num_imgs = 2
    img_names = [
        img_name for img_name in
        list(
            zip(
                *sorted(
                    [
                        (img, len(mock_download_source_connector.vector_features_intersecting_img(img)))
                        for img in mock_download_source_connector.raster_imgs.index
                    ],
                    key=lambda x: x[1],
                    reverse=True
                )
            )
        )[0]
        if img_name.endswith("_1.tif") # _1, _2 have identical bounds, assure distinct bounds
    ][:num_imgs]
    source_connector.add_to_raster_imgs(mock_download_source_connector.raster_imgs.loc[img_names])
    source_connector.add_to_vector_features(mock_download_source_connector.vector_features)

    # define task vector feature classes
    source_connector.vector_features["type"] = ""
    num_features = len(source_connector.vector_features)
    for n in range(0, 1 + num_features // 100):
        range_lower = n*100
        range_upper = min(num_features, (n+1)*100)
        source_connector.vector_features["type"].iloc[range_lower:range_upper] = str(n)
    source_connector.task_vector_feature_classes = source_connector.vector_features["type"].unique().tolist()

    source_connector.save()

    # create images and labels
    create_dummy_imgs(data_dir=source_data_dir, img_size=10980)
    categorical_seg_label_maker = SegLabelMakerCategorical()
    categorical_seg_label_maker.make_labels(
        source_connector
    )

    ## test combining/removing classes
    target_data_dir = get_test_dir() / "temp" / "combine_remove_target"
    shutil.rmtree(target_data_dir, ignore_errors=True)
    combiner = DSConverterCombineRemoveClasses(
        name=COMBINER_NAME,
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        classes=[["1", "2", "5"], "3"],
        label_maker=categorical_seg_label_maker)

    combiner.create()
    combiner.save()

    assert combiner.target_connector.vector_features["type"].value_counts().to_dict() == {
        "1+2+5": 234,
        "3": 100,
    }

    target_mask_125 = combiner.target_connector.vector_features["type"].isin(["1+2+5"])
    source_mask_1_2_5 = combiner.source_connector.vector_features["type"].isin(["1", "2", "5"])
    assert set(combiner.target_connector.vector_features[target_mask_125].index) == set(combiner.source_connector.vector_features[source_mask_1_2_5].index)

    target_mask_3 = combiner.target_connector.vector_features["type"].isin(["3"])
    source_mask_3 = combiner.source_connector.vector_features["type"].isin(["3"])
    assert set(combiner.target_connector.vector_features[target_mask_3].index) == set(combiner.source_connector.vector_features[source_mask_3].index)


def test_combine_remove_seg_classes_soft_categorical():
    ## create source dataset with soft-categorical segmentation labels
    source_data_dir = get_test_dir() / "temp" / "combine_remove_source"
    source_connector = Connector.from_data_dir(source_data_dir)
    # source dataset contains two images from test_combine_remove_seg_classes_categorical

    # select vector features to keep
    vector_features_contained_in_images = sorted(
        [
            feature for feature in source_connector.vector_features.index
            if source_connector.imgs_containing_vector_feature(feature)
        ]
    )
    vector_features_not_intersecting_images = sorted(
        [
            feature for feature in source_connector.vector_features.index
            if not source_connector.imgs_intersecting_vector_feature(feature)
        ]
    )
    vector_features_to_keep : List[str] = vector_features_contained_in_images[:3] + vector_features_not_intersecting_images[:1]
    vector_features_to_keep : GeoDataFrame = deepcopy_gdf(source_connector.vector_features.loc[vector_features_to_keep])

    source_connector.drop_vector_features(
        list(source_connector.vector_features.index)
    )
    categorical_seg_label_maker = SegLabelMakerCategorical()
    categorical_seg_label_maker.delete_labels(
        connector=source_connector
    )

    # convert vector_features_to_keep to soft-categorical format
    vector_features_to_keep.drop(columns="type", inplace=True)
    class_probabilities = {
        "prob_of_class_0": [0.2, 0.2, 0.5, 0.3],
        "prob_of_class_1": [0.2, 0.0, 0.0, 0.1],
        "prob_of_class_2": [0.1, 0.0, 0.0, 0.1],
        "prob_of_class_3": [0.4, 0.4, 0.0, 0.1],
        "prob_of_class_4": [0.0, 0.0, 0.5, 0.3],
        "prob_of_class_5": [0.1, 0.4, 0.0, 0.1],
    }
    assert class_probs_add_to_one(pd.DataFrame(data=class_probabilities))
    for col in class_probabilities:
        vector_features_to_keep[col] = class_probabilities[col]

    source_connector.add_to_vector_features(vector_features_to_keep)
    source_connector.attrs["label_type"] = "soft-categorical"
    source_connector.save()

    label_maker = SegLabelMakerSoftCategorical(add_background_band=True)

    target_data_dir = get_test_dir() / "temp" / "combine_remove_target"
    shutil.rmtree(target_data_dir, ignore_errors=True)
    combiner = DSConverterCombineRemoveClasses(
        name=COMBINER_NAME,
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        classes=[["1", "2", "5"], "3"],
        new_class_names=["new1", "new2"],
        label_maker=label_maker)
    combiner.create()
    combiner.save()

    assert {
        col for col in combiner.target_connector.vector_features.columns
        if col.startswith("prob_of_class_")
    } == {"prob_of_class_new1", "prob_of_class_new2"}
    assert class_probs_add_to_one(combiner.target_connector.vector_features)
    assert set(combiner.target_connector.vector_features.index) == {"10001", "10008", "10164"}
    assert combiner.target_connector.vector_features[
        ["prob_of_class_new1", "prob_of_class_new2"]].round(2).to_dict() == {
            'prob_of_class_new1': {'10001': 0.5, '10008': 0.5, '10164': 0.75},
            'prob_of_class_new2': {'10001': 0.5, '10008': 0.5, '10164': 0.25},
    }
    assert combiner.target_connector.background_class is None


def test_combine_remove_seg_classes_update():

    mock_download_source_data_dir = get_test_dir() / MOCK_DOWNLOAD_SOURCE_DATA_DIR
    mock_download_source_connector = Connector.from_data_dir(mock_download_source_data_dir)

    target_data_dir = get_test_dir() / "temp" / "combine_remove_target"
    combiner = DSConverterCombineRemoveClasses.from_json_file(target_data_dir / "connector" / f"{COMBINER_NAME}.json")

    # make sure we're working with the dataset created by test_combine_remove_seg_classes_soft_categorical as input
    assert set(combiner.source_connector.vector_features.index) == {'10001', '10008', '10053', '10164'}
    assert set(combiner.source_connector.raster_imgs.index) == {'188_18_1.tif', '189_18_1.tif'}
    assert set(os.listdir(combiner.source_connector.images_dir)) >= {'188_18_1.tif', '189_18_1.tif'}

    # make sure we're working with the target dataset created by test_combine_remove_seg_classes_soft_categorical
    assert set(combiner.target_connector.vector_features.index) == {"10001", "10008", "10164"}
    assert set(combiner.target_connector.raster_imgs.index) >= {'188_18_1.tif', '189_18_1.tif'}

    ## Add data to the source dataset

    # Let's add a new feature ('10073') from the mock download source dataset
    # to the source_dataset which will be contained in '188_18_1.tif'.
    new_vector_feature_name = sorted(
        list(
            set(mock_download_source_connector.vector_features_contained_in_img('188_18_1.tif'))\
            - set(combiner.source_connector.vector_features.index)\
            - set(mock_download_source_connector.vector_features_contained_in_img('189_18_1.tif'))
        )
    )[0]
    new_vector_feature_row = mock_download_source_connector.vector_features.loc[[new_vector_feature_name]]
    class_probabilities = {
        "prob_of_class_0": [0.5],
        "prob_of_class_1": [0.5],
        "prob_of_class_2": [0.0],
        "prob_of_class_3": [0.0],
        "prob_of_class_4": [0.0],
        "prob_of_class_5": [0.0],
    }
    assert class_probs_add_to_one(pd.DataFrame(data=class_probabilities))
    for col in class_probabilities:
        new_vector_feature_row[col] = class_probabilities[col]
    combiner.source_connector.add_to_vector_features(new_vector_feature_row)

    # Feature '10164' does not intersect any images in the source dataset.
    # Let's add the img '188_17_1.tif' containing it from the mock download source dataset
    # shutil.copy( # just create dummy image in source dataset!!
    #     src=mock_download_source_connector.images_dir / '188_17_1.tif',
    #     dst=combiner.source_connector.images_dir
    # )
    new_raster_img_row = mock_download_source_connector.raster_imgs.loc[['188_17_1.tif']]
    combiner.source_connector.add_to_raster_imgs(new_raster_img_row)
    combiner.source_connector.save()
    create_dummy_imgs(combiner.source_data_dir, img_size=10980, img_names=['188_17_1.tif'])

    # make sure label for '188_18_1.tif' (which now contains a new
    # vector feature after updating) was updated
    label_creation_time = os.path.getctime(combiner.target_connector.labels_dir / '188_18_1.tif')
    combiner.update()
    # since '188_18_1.tif' now contains a new vector feature the label should have been updated
    new_label_creation_time = os.path.getctime(combiner.target_connector.labels_dir / '188_18_1.tif')
    assert new_label_creation_time > label_creation_time, "label for '188_17_1.tif' which contains new feature wasn't updated"

    # make sure label for new image '188_17_1.tif' was created
    assert set(os.listdir(combiner.target_connector.labels_dir)) == {'188_18_1.tif', '189_18_1.tif', '188_17_1.tif'}


if __name__ == "__main__":
    # tests need to be run in order
    test_combine_remove_seg_classes_categorical()
    test_combine_remove_seg_classes_soft_categorical()
    test_combine_remove_seg_classes_update()




