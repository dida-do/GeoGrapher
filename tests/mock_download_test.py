"""
for testing:
    TODO: write test for download_mode 'bboxgrid' using large polygon


Mock AuBeSa S2 connector for testing the download function.

Virtually 'downloads' (no files operations are actually done) from a
dataset of images in a source directory.
"""

import logging
from pathlib import Path
import random
import warnings
import shutil

import geopandas as gpd

from geographer import Connector
from geographer.downloaders.downloader_for_features import ImgDownloaderForVectorFeatures
from geographer.testing.graph_df_compatibility import check_graph_vertices_counts
from geographer.testing.mock_download import MockDownloaderForSingleFeature, MockDownloadProcessor
from tests.utils import get_test_dir


MOCK_DOWNLOAD_SOURCE_DATA_DIR = "mock_download_source"

def test_mock_download():
    """Test mock download function"""
    test_dir = get_test_dir()
    data_dir = test_dir / "temp/mock_download_few_features"
    download_source_data_dir = test_dir / MOCK_DOWNLOAD_SOURCE_DATA_DIR

    source_connector = Connector.from_scratch(data_dir=download_source_data_dir)
    connector = Connector.from_scratch(data_dir=data_dir ,task_feature_classes=['object'])

    vector_features = gpd.read_file(test_dir / "geographer_download_test.geojson", driver="GeoJSON")
    vector_features.set_index("name", inplace=True)

    connector.add_to_vector_features(vector_features)
    source_connector.add_to_vector_features(vector_features)

    raster_imgs_path = get_test_dir() / 'mock_download_tiles.geojson'
    raster_imgs = gpd.read_file(raster_imgs_path)
    raster_imgs.set_index("Name", inplace=True)
    source_connector.add_to_raster_imgs(raster_imgs)

    download_processor = MockDownloadProcessor(source_connector=source_connector)
    downloader_for_single_feature = MockDownloaderForSingleFeature(
        source_connector=source_connector,
        probability_of_download_error=0.0,
        probability_img_already_downloaded=0.0,
    )
    downloader = ImgDownloaderForVectorFeatures(
        download_dir=data_dir / "download",
        downloader_for_single_feature=downloader_for_single_feature,
        download_processor=download_processor,
    )

    # warnings.filterwarnings("ignore")
    downloader.download(
        connector=connector,
        target_img_count=1,
        shuffle=False,
    )

    # The vector_features contain
    #     - 2 objects in Berlin (Reichstag and Brandenburg gate) that are very close to each other
    #     - 2 objects in Lisbon (Castelo de Sao Jorge and Praca Do Comercio) that are very close to each other.
    # Thus the s2_downloader should have downloaded two images, one for Berlin and one for Lisbon, each containing two objects.

    # Berlin
    assert len(connector.imgs_containing_vector_feature("berlin_reichstag")) == 1
    assert len(connector.imgs_containing_vector_feature("berlin_brandenburg_gate")) == 1
    assert connector.imgs_containing_vector_feature("berlin_reichstag") == connector.imgs_containing_vector_feature("berlin_brandenburg_gate")

    # Lisbon
    assert len(connector.imgs_containing_vector_feature("lisbon_castelo_de_sao_jorge")) == 1
    assert len(connector.imgs_containing_vector_feature("lisbon_praca_do_comercio")) == 1
    assert connector.imgs_containing_vector_feature("lisbon_castelo_de_sao_jorge") == connector.imgs_containing_vector_feature("lisbon_praca_do_comercio")

    # Now, attempt to download 2 raster images per vector featue
    downloader.download(
        connector=connector,
        target_img_count=2,
        shuffle=False,
    )

    # The tiling is such that the s2_downloader should have downloaded three
    # images, two for Berlin and one for Lisbon.

    # Berlin
    assert len(connector.imgs_containing_vector_feature("berlin_reichstag")) == 2
    assert len(connector.imgs_containing_vector_feature("berlin_brandenburg_gate")) == 2
    assert connector.imgs_containing_vector_feature("berlin_reichstag") == connector.imgs_containing_vector_feature("berlin_brandenburg_gate")

    # Lisbon
    assert len(connector.imgs_containing_vector_feature("lisbon_castelo_de_sao_jorge")) == 1
    assert len(connector.imgs_containing_vector_feature("lisbon_praca_do_comercio")) == 1
    assert connector.imgs_containing_vector_feature("lisbon_castelo_de_sao_jorge") == connector.imgs_containing_vector_feature("lisbon_praca_do_comercio")


    # clean up
    shutil.rmtree(data_dir)


def test_mock_download_many_features():
    """Test ImgDownloaderForVectorFeatures using mock downloads"""

    random.seed(74) #74

    download_source_data_dir = get_test_dir() / MOCK_DOWNLOAD_SOURCE_DATA_DIR
    source_connector = Connector.from_data_dir(download_source_data_dir)

    data_dir = get_test_dir() / "temp/mock_download"
    connector = source_connector.empty_connector_same_format(data_dir=data_dir)
    connector.add_to_vector_features(source_connector.vector_features)

    download_processor = MockDownloadProcessor(source_connector=source_connector)
    downloader_for_single_feature = MockDownloaderForSingleFeature(source_connector=source_connector)
    downloader = ImgDownloaderForVectorFeatures(
        download_dir=data_dir / "download",
        downloader_for_single_feature=downloader_for_single_feature,
        download_processor=download_processor,
    )

    warnings.filterwarnings("ignore")
    downloader.download(
        connector=connector,
        target_img_count=1,
        shuffle=False,
    )

    assert connector.vector_features.img_count.value_counts().to_dict() == \
        {
            1: 247,
            2: 227, # lots of overlapping images
            3: 60,
        }

    downloader.download(
        connector=connector,
        target_img_count=8
    )

    assert all(download_processor.source_connector.vector_features.img_count.value_counts() == connector.vector_features.img_count.value_counts())
    assert check_graph_vertices_counts(connector)


if __name__ == "__main__":
    test_mock_download_many_features()
    test_mock_download()

