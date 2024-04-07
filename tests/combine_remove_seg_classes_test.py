"""Test DSConverterCombineRemoveClasses.

TODO: rename MOCK_DOWNLOAD_SOURCE_DATA_DIR?
"""

import os
import shutil
from typing import List

import pandas as pd
from geopandas import GeoDataFrame
from mock_download_test import MOCK_DOWNLOAD_SOURCE_DATA_DIR
from utils import create_dummy_rasters, get_test_dir

from geographer import Connector
from geographer.converters import DSConverterCombineRemoveClasses
from geographer.label_makers import (
    SegLabelMakerCategorical,
    SegLabelMakerSoftCategorical,
)
from geographer.utils.utils import deepcopy_gdf

COMBINER_NAME = "combine_remove"


def class_probs_add_to_one(df: pd.DataFrame) -> bool:
    """Check whether class probabilities add up to one."""
    return all(
        df[[col for col in df.columns if col.startswith("prob_of_class_")]]
        .sum(axis=1)
        .round(2)
        == 1.0
    )


def test_combine_remove_seg_classes_categorical():
    """Test create method of DSConverterCombineRemoveClasses."""
    ## create source dataset with categorical segmentation labels
    mock_download_source_data_dir = get_test_dir() / MOCK_DOWNLOAD_SOURCE_DATA_DIR
    mock_download_source_connector = Connector.from_data_dir(
        mock_download_source_data_dir
    )

    source_data_dir = get_test_dir() / "temp" / "combine_remove_source"
    shutil.rmtree(source_data_dir, ignore_errors=True)
    source_connector = mock_download_source_connector.empty_connector_same_format(
        data_dir=source_data_dir
    )

    # select rasters intersecting the most vector features
    num_rasters = 2
    raster_names = [
        raster_name
        for raster_name in list(
            zip(
                *sorted(
                    [
                        (
                            raster,
                            len(
                                mock_download_source_connector.vectors_intersecting_raster(  # noqa: E501
                                    raster
                                )
                            ),
                        )
                        for raster in mock_download_source_connector.rasters.index
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        )[0]
        if raster_name.endswith(
            "_1.tif"
        )  # _1, _2 have identical bounds, assure distinct bounds
    ][:num_rasters]
    source_connector.add_to_rasters(
        mock_download_source_connector.rasters.loc[raster_names]
    )
    source_connector.add_to_vectors(mock_download_source_connector.vectors)

    # define task vector feature classes
    source_connector.vectors["type"] = ""
    num_vectors = len(source_connector.vectors)
    for n in range(0, 1 + num_vectors // 100):
        range_lower = n * 100
        range_upper = min(num_vectors, (n + 1) * 100)
        col_idx = source_connector.vectors.columns.get_loc("type")
        source_connector.vectors.iloc[range_lower:range_upper, col_idx] = str(n)
    source_connector.task_vector_classes = (
        source_connector.vectors["type"].unique().tolist()
    )

    source_connector.save()

    # create rasters and labels
    create_dummy_rasters(data_dir=source_data_dir, raster_size=10980)
    categorical_seg_label_maker = SegLabelMakerCategorical()
    categorical_seg_label_maker.make_labels(source_connector)

    ## test combining/removing classes
    target_data_dir = get_test_dir() / "temp" / "combine_remove_target"
    shutil.rmtree(target_data_dir, ignore_errors=True)
    combiner = DSConverterCombineRemoveClasses(
        name=COMBINER_NAME,
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        classes=[["1", "2", "5"], "3"],
        label_maker=categorical_seg_label_maker,
    )

    combiner.create()
    combiner.save()

    assert combiner.target_connector.vectors["type"].value_counts().to_dict() == {
        "1+2+5": 234,
        "3": 100,
    }

    target_mask_125 = combiner.target_connector.vectors["type"].isin(["1+2+5"])
    source_mask_1_2_5 = combiner.source_connector.vectors["type"].isin(["1", "2", "5"])
    assert set(  # noqa: BLK100
        combiner.target_connector.vectors[target_mask_125].index
    ) == set(combiner.source_connector.vectors[source_mask_1_2_5].index)

    target_mask_3 = combiner.target_connector.vectors["type"].isin(["3"])
    source_mask_3 = combiner.source_connector.vectors["type"].isin(["3"])
    assert set(combiner.target_connector.vectors[target_mask_3].index) == set(
        combiner.source_connector.vectors[source_mask_3].index
    )


def test_combine_remove_seg_classes_soft_categorical():
    """Test update method of DSConverterCombineRemoveClasses, soft-categorical case."""
    ## create source dataset with soft-categorical segmentation labels
    source_data_dir = get_test_dir() / "temp" / "combine_remove_source"
    source_connector = Connector.from_data_dir(source_data_dir)
    # source dataset contains two rasters
    # from test_combine_remove_seg_classes_categorical

    # select vector features to keep
    vectors_contained_in_rasters = sorted(
        [
            feature
            for feature in source_connector.vectors.index
            if source_connector.rasters_containing_vector(feature)
        ]
    )
    vectors_not_intersecting_rasters = sorted(
        [
            feature
            for feature in source_connector.vectors.index
            if not source_connector.rasters_intersecting_vector(feature)
        ]
    )
    vectors_to_keep: List[str] = (
        vectors_contained_in_rasters[:3] + vectors_not_intersecting_rasters[:1]
    )
    vectors_to_keep: GeoDataFrame = deepcopy_gdf(
        source_connector.vectors.loc[vectors_to_keep]
    )

    source_connector.drop_vectors(list(source_connector.vectors.index))
    categorical_seg_label_maker = SegLabelMakerCategorical()
    categorical_seg_label_maker.delete_labels(connector=source_connector)

    # convert vectors_to_keep to soft-categorical format
    vectors_to_keep.drop(columns="type", inplace=True)
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
        vectors_to_keep[col] = class_probabilities[col]

    source_connector.add_to_vectors(vectors_to_keep)
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
        label_maker=label_maker,
    )
    combiner.create()
    combiner.save()

    assert {
        col
        for col in combiner.target_connector.vectors.columns
        if col.startswith("prob_of_class_")
    } == {"prob_of_class_new1", "prob_of_class_new2"}
    assert class_probs_add_to_one(combiner.target_connector.vectors)
    assert set(combiner.target_connector.vectors.index) == {
        "10001",
        "10008",
        "10164",
    }
    assert combiner.target_connector.vectors[
        ["prob_of_class_new1", "prob_of_class_new2"]
    ].round(2).to_dict() == {
        "prob_of_class_new1": {"10001": 0.5, "10008": 0.5, "10164": 0.75},
        "prob_of_class_new2": {"10001": 0.5, "10008": 0.5, "10164": 0.25},
    }
    assert combiner.target_connector.background_class is None


def test_combine_remove_seg_classes_update():
    """Test update method of DSConverterCombineRemoveClasses."""
    mock_download_source_data_dir = get_test_dir() / MOCK_DOWNLOAD_SOURCE_DATA_DIR
    mock_download_source_connector = Connector.from_data_dir(
        mock_download_source_data_dir
    )

    target_data_dir = get_test_dir() / "temp" / "combine_remove_target"
    combiner = DSConverterCombineRemoveClasses.from_json_file(
        target_data_dir / "connector" / f"{COMBINER_NAME}.json"
    )

    # make sure we're working with the dataset created
    # by test_combine_remove_seg_classes_soft_categorical as input
    assert set(combiner.source_connector.vectors.index) == {
        "10001",
        "10008",
        "10053",
        "10164",
    }
    assert set(combiner.source_connector.rasters.index) == {
        "188_18_1.tif",
        "189_18_1.tif",
    }
    assert set(os.listdir(combiner.source_connector.rasters_dir)) >= {
        "188_18_1.tif",
        "189_18_1.tif",
    }

    # make sure we're working with the target dataset created
    # by test_combine_remove_seg_classes_soft_categorical
    assert set(combiner.target_connector.vectors.index) == {
        "10001",
        "10008",
        "10164",
    }
    assert set(combiner.target_connector.rasters.index) >= {
        "188_18_1.tif",
        "189_18_1.tif",
    }

    ## Add data to the source dataset

    # Let's add a new feature ('10073') from the mock download source dataset
    # to the source_dataset which will be contained in '188_18_1.tif'.
    new_vector_name = sorted(
        list(
            set(
                mock_download_source_connector.vectors_contained_in_raster(
                    "188_18_1.tif"
                )
            )
            - set(combiner.source_connector.vectors.index)
            - set(
                mock_download_source_connector.vectors_contained_in_raster(
                    "189_18_1.tif"
                )
            )
        )
    )[0]
    new_vector_row = mock_download_source_connector.vectors.loc[[new_vector_name]]
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
        new_vector_row[col] = class_probabilities[col]
    combiner.source_connector.add_to_vectors(new_vector_row)

    # Vector '10164' does not intersect any rasters in the source dataset.
    # Let's add the raster '188_17_1.tif' containing it from
    # the mock download source dataset
    # shutil.copy( # just create dummy raster in source dataset!!
    #     src=mock_download_source_connector.rasters_dir / '188_17_1.tif',
    #     dst=combiner.source_connector.rasters_dir
    # )
    new_raster_row = mock_download_source_connector.rasters.loc[["188_17_1.tif"]]
    combiner.source_connector.add_to_rasters(new_raster_row)
    combiner.source_connector.save()
    create_dummy_rasters(
        combiner.source_data_dir, raster_size=10980, raster_names=["188_17_1.tif"]
    )

    # make sure label for '188_18_1.tif' (which now contains a new
    # vector feature after updating) was updated
    label_creation_time = os.path.getctime(
        combiner.target_connector.labels_dir / "188_18_1.tif"
    )
    combiner.update()
    # since '188_18_1.tif' now contains a new vector feature
    # the label should have been updated
    new_label_creation_time = os.path.getctime(
        combiner.target_connector.labels_dir / "188_18_1.tif"
    )
    assert (
        new_label_creation_time > label_creation_time
    ), "label for '188_17_1.tif' which contains new feature wasn't updated"

    # make sure label for new raster '188_17_1.tif' was created
    assert set(os.listdir(combiner.target_connector.labels_dir)) == {
        "188_18_1.tif",
        "189_18_1.tif",
        "188_17_1.tif",
    }


if __name__ == "__main__":
    # tests need to be run in order
    test_combine_remove_seg_classes_categorical()
    test_combine_remove_seg_classes_soft_categorical()
    test_combine_remove_seg_classes_update()
