"""Unit (py)tests for rs_tools.cut.cut_dataset_iter_over_polygons.

WARNING! Needs to be updated to work with new associator. WARNING! Uses
real data in rstools folder on markov. Ideally, should use artificially
generated data.
"""

import os
import shutil
from pathlib import Path

import rasterio as rio

import rs_tools.img_polygon_associator as ipa
from rs_tools.cut.cut_dataset_iter_over_polygons import \
    new_tif_dataset_small_imgs_for_each_polygon as cut2small_imgs

# for test_new_dataset_one_small_img_for_each_polygon
SOURCE_DATA_DIR = Path.home() / Path("rstools/pytest_data_dirs/") / Path(
    "source")
NEW_TARGET_DATA_DIR = Path.home() / Path("rstools/pytest_data_dirs/") / Path(
    "target")

# for test_new_dataset_one_small_img_for_each_polygon
UPDATE_TARGET_DATA_DIR = Path.home() / Path("rstools/pytest_data_dirs/") / Path(
    "update_target")
COPY_UPDATE_TARGET_DATA_DIR = Path.home() / Path(
    "rstools/pytest_data_dirs/") / Path("update_target_copy")

# contains dataset/associator the way it should be split/cut for test_new_dataset_one_small_img_for_each_polygon and test_new_dataset_one_small_img_for_each_polygon
CORRECT_TARGET_DIR = Path.home() / Path("rstools/pytest_data_dirs/") / Path(
    "correct")

# for test_not_centered_cutting
NOT_CENTERED_SOURCE_DATA_DIR = Path.home() / Path(
    "rstools/pytest_data_dirs/") / Path("source_centered_false_test")
# (path of) data dir that will be created
NOT_CENTERED_TARGET_DATA_DIR = Path.home() / Path(
    "rstools/pytest_data_dirs/") / Path("centered_false_target_copy")


def test_new_dataset_one_small_img_for_each_polygon():
    """Create a new dataset from an old one using
    cut.new_dataset_one_small_img_for_each_polygon.

    The source dataset contains both polygons without images and
    polygons that will be contained in the same image.
    """
    # data dir of source dataset
    source_data_dir = SOURCE_DATA_DIR
    # (path of) data dir that will be created
    target_data_dir = NEW_TARGET_DATA_DIR
    # contains dataset/associator the way it should be split/cut
    correct_target_dir = CORRECT_TARGET_DIR

    # delete target dir, so that we create from scratch.
    if target_data_dir.exists():
        shutil.rmtree(target_data_dir)

    source_assoc = ipa.ImgPolygonAssociator(data_dir=source_data_dir)
    correct_target_assoc = ipa.ImgPolygonAssociator(
        data_dir=correct_target_dir)

    # make labels
    source_assoc.make_missing_geotif_labels()

    # create new dataset in target_dir by cutting source dataset
    cut.new_tif_dataset_small_imgs_for_each_polygon(source_data_dir,
                                                    target_data_dir,
                                                    img_size=1024,
                                                    centered=True)

    target_assoc = ipa.ImgPolygonAssociator(data_dir=target_data_dir)

    # check equality of graphs of old and new assoc
    assert target_assoc._graph._graph_dict == correct_target_assoc._graph._graph_dict

    # check equality of raster_imgss
    assert (target_assoc.raster_imgs).equals(correct_target_assoc.raster_imgs)
    assert target_assoc.raster_imgs.crs == source_assoc.raster_imgs.crs

    # check equality of polygons_dfs
    assert (target_assoc.polygons_df).equals(correct_target_assoc.polygons_df)
    assert target_assoc.polygons_df.crs == correct_target_assoc.polygons_df.crs

    # check generated right number of files
    assert len(list((target_data_dir / 'images').iterdir())) == 3
    assert len(list((target_data_dir / 'labels').iterdir())) == 3


def update_dataset_from_iter_over_polygons_test():
    """Test cut.update_dataset_from_iter_over_polygons.

    The dataset we're updating is a copy of the correct target dataset used in the function above where we deleted polygons 730.6, 730.7, 730.8 (all contained in the image for 730.1), and 668 (which does not have an image) and deleted the image (and label) S2A_MSIL2A_20200403T143721_N0214_R096_T19KES_20200403T184818_730.tif, which contains polygon 730 (and doesn't contain or intersect any other polygons).
    """
    # data dir of source dataset
    source_data_dir = SOURCE_DATA_DIR
    # target data_dir
    target_data_dir = COPY_UPDATE_TARGET_DATA_DIR
    # contains dataset/associator the way it should be split/cut
    correct_target_dir = CORRECT_TARGET_DIR

    # copy target dataset in UPDATE_TARGET_DATA_DIR to target_data_dir (we don't want to use that dataset directly because updating will change it, but we want to keep the dataset so for future tests.
    # first, delete target dir
    if target_data_dir.exists():
        shutil.rmtree(target_data_dir)
    # copy
    shutil.copytree(UPDATE_TARGET_DATA_DIR, target_data_dir)

    source_assoc = ipa.ImgPolygonAssociator(data_dir=source_data_dir)
    correct_target_assoc = ipa.ImgPolygonAssociator(
        data_dir=correct_target_dir)

    # make labels
    source_assoc.make_missing_geotif_labels()

    # udpate dataset in target_dir by cutting source dataset
    cut.create_or_update_tif_dataset_from_iter_over_polygons(source_data_dir,
                                                             target_data_dir,
                                                             img_size=1024,
                                                             centered=True)

    target_assoc = ipa.ImgPolygonAssociator(data_dir=target_data_dir)

    # check equality of graphs of old and new assoc
    assert target_assoc._graph._graph_dict == correct_target_assoc._graph._graph_dict

    # check equality of raster_imgss
    assert (target_assoc.raster_imgs).equals(correct_target_assoc.raster_imgs)
    assert target_assoc.raster_imgs.crs == source_assoc.raster_imgs.crs

    # check equality of polygons_dfs
    assert (target_assoc.polygons_df).equals(correct_target_assoc.polygons_df)
    assert target_assoc.polygons_df.crs == correct_target_assoc.polygons_df.crs

    # check generated right number of files
    assert len(list((target_data_dir / 'images').iterdir())) == 3
    assert len(list((target_data_dir / 'labels').iterdir())) == 3


def test_not_centered_cutting():
    """Small_imgs_centered_around_polygons_splitter in cut_dataset_utils has a
    "centered" argument.

    If False, the window around the image being cut should have random
    offsets subject to the contraint that it should fully contain the
    polygon under consideration. This test is to make sure that this
    constraint is always satisfied.
    """

    # data dir of source dataset
    source_data_dir = NOT_CENTERED_SOURCE_DATA_DIR

    # this dataset contains only one image and polygon. We'll cut it repeatedly, and check that for each cut the sole resulting label fully contains the polygon by bincounting.

    # dict of target data_dirs
    target_data_dir = NOT_CENTERED_TARGET_DATA_DIR

    # repeatedly create new
    for n in range(4):
        # first, delete target dir
        if target_data_dir.exists():
            shutil.rmtree(target_data_dir)
        # create new dataset in target_dir by cutting source dataset
        cut.new_tif_dataset_small_imgs_for_each_polygon(source_data_dir,
                                                        target_data_dir,
                                                        img_size=1024,
                                                        centered=False)

        # bincount to check generated image fully contains polygon
        with rio.open(target_data_dir / Path('labels') / Path(
                'S2A_MSIL2A_20200403T143721_N0214_R096_T19KES_20200403T184818_730.tif'
        )) as src:

            raster = src.read(
                1
            )  # polygon 730 is a  object, so the non-default (non-zero) values of the label are 2.

            # count entries
            count_dict = {0: 0, 2: 0}

            for row in range(raster.shape[0]):
                for col in range(raster.shape[1]):
                    count_dict[raster[row, col]] += 1

            # compare with correct answer
            assert count_dict == {0: 904614, 2: 143962}
            #### Once I got {0: 904627, 2: 143949}? Somehow this can fail if the object is at the edge of the picture... ???


if __name__ == "__main__":
    test_new_dataset_one_small_img_for_each_polygon()
    update_dataset_from_iter_over_polygons_test()
    test_not_centered_cutting()
