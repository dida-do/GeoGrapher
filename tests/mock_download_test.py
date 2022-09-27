"""
Test RasterDownloadProcessor using mock downloader.

Virtually 'downloads' (no files operations are actually done) from a
dataset of rasters in a source directory.

TODO: write test for download_mode 'bboxgrid' using large polygon
"""

import random
import shutil
import warnings

import geopandas as gpd
from utils import get_test_dir

from geographer import Connector
from geographer.downloaders.downloader_for_vectors import RasterDownloaderForVectors
from geographer.testing.graph_df_compatibility import check_graph_vertices_counts
from geographer.testing.mock_download import (
    MockDownloaderForSingleVector,
    MockDownloadProcessor,
)

MOCK_DOWNLOAD_SOURCE_DATA_DIR = "mock_download_source"


def test_mock_download():
    """Test mock download function."""
    test_dir = get_test_dir()
    data_dir = test_dir / "temp/mock_download_few_vectors"
    download_source_data_dir = test_dir / MOCK_DOWNLOAD_SOURCE_DATA_DIR

    source_connector = Connector.from_scratch(data_dir=download_source_data_dir)
    connector = Connector.from_scratch(
        data_dir=data_dir, task_vector_classes=["object"]
    )

    vectors = gpd.read_file(
        test_dir / "geographer_download_test.geojson", driver="GeoJSON"
    )
    vectors.set_index("name", inplace=True)

    connector.add_to_vectors(vectors)
    source_connector.add_to_vectors(vectors)

    rasters_path = get_test_dir() / "mock_download_tiles.geojson"
    rasters = gpd.read_file(rasters_path)
    rasters.set_index("Name", inplace=True)
    source_connector.add_to_rasters(rasters)

    download_processor = MockDownloadProcessor(source_connector=source_connector)
    downloader_for_single_vector = MockDownloaderForSingleVector(
        source_connector=source_connector,
        probability_of_download_error=0.0,
        probability_raster_already_downloaded=0.0,
    )
    downloader = RasterDownloaderForVectors(
        download_dir=data_dir / "download",
        downloader_for_single_vector=downloader_for_single_vector,
        download_processor=download_processor,
    )

    # warnings.filterwarnings("ignore")
    downloader.download(
        connector=connector,
        target_raster_count=1,
        shuffle=False,
    )

    # The vectors contain
    #     - 2 objects in Berlin (Reichstag and Brandenburg gate)
    #       that are very close to each other
    #     - 2 objects in Lisbon (Castelo de Sao Jorge and
    #       Praca Do Comercio) that are very close to each other.
    # Thus the s2_downloader should have downloaded two rasters,
    # one for Berlin and one for Lisbon, each containing two objects.

    # Berlin
    assert len(connector.rasters_containing_vector("berlin_reichstag")) == 1
    assert len(connector.rasters_containing_vector("berlin_brandenburg_gate")) == 1
    assert connector.rasters_containing_vector(
        "berlin_reichstag"
    ) == connector.rasters_containing_vector("berlin_brandenburg_gate")

    # Lisbon
    assert len(connector.rasters_containing_vector("lisbon_castelo_de_sao_jorge")) == 1
    assert len(connector.rasters_containing_vector("lisbon_praca_do_comercio")) == 1
    assert connector.rasters_containing_vector(
        "lisbon_castelo_de_sao_jorge"
    ) == connector.rasters_containing_vector("lisbon_praca_do_comercio")

    # Now, attempt to download 2 rasters per vector featue
    downloader.download(
        connector=connector,
        target_raster_count=2,
        shuffle=False,
    )

    # The tiling is such that the s2_downloader should have downloaded three
    # rasters, two for Berlin and one for Lisbon.

    # Berlin
    assert len(connector.rasters_containing_vector("berlin_reichstag")) == 2
    assert len(connector.rasters_containing_vector("berlin_brandenburg_gate")) == 2
    assert connector.rasters_containing_vector(
        "berlin_reichstag"
    ) == connector.rasters_containing_vector("berlin_brandenburg_gate")

    # Lisbon
    assert len(connector.rasters_containing_vector("lisbon_castelo_de_sao_jorge")) == 1
    assert len(connector.rasters_containing_vector("lisbon_praca_do_comercio")) == 1
    assert connector.rasters_containing_vector(
        "lisbon_castelo_de_sao_jorge"
    ) == connector.rasters_containing_vector("lisbon_praca_do_comercio")

    # clean up
    shutil.rmtree(data_dir)


def test_mock_download_many_vectors():
    """Test RasterDownloaderForVectors using mock downloads."""
    random.seed(74)  # 74

    download_source_data_dir = get_test_dir() / MOCK_DOWNLOAD_SOURCE_DATA_DIR
    source_connector = Connector.from_data_dir(download_source_data_dir)

    data_dir = get_test_dir() / "temp/mock_download"
    connector = source_connector.empty_connector_same_format(data_dir=data_dir)
    connector.add_to_vectors(source_connector.vectors)

    download_processor = MockDownloadProcessor(source_connector=source_connector)
    downloader_for_single_vector = MockDownloaderForSingleVector(
        source_connector=source_connector
    )
    downloader = RasterDownloaderForVectors(
        downloader_for_single_vector=downloader_for_single_vector,
        download_processor=download_processor,
    )

    warnings.filterwarnings("ignore")
    downloader.download(
        connector=connector,
        target_raster_count=1,
        shuffle=False,
    )

    assert connector.vectors.raster_count.value_counts().to_dict() == {
        1: 247,
        2: 227,  # lots of overlapping rasters
        3: 60,
    }

    downloader.download(connector=connector, target_raster_count=8)

    assert all(
        download_processor.source_connector.vectors.raster_count.value_counts()
        == connector.vectors.raster_count.value_counts()
    )
    assert check_graph_vertices_counts(connector)


if __name__ == "__main__":
    test_mock_download_many_vectors()
    test_mock_download()
