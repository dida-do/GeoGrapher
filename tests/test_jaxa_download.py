"""Manually triggered test of JAXA downloader.

Run by hand to test downloading JAXA data. Intentionally not discoverable
by pytest: Downloading JAXA rasters is slightly slow.
"""

import shutil

import geopandas as gpd
import pytest
from utils import get_test_dir

from geographer import Connector
from geographer.downloaders.downloader_for_vectors import RasterDownloaderForVectors
from geographer.downloaders.jaxa_download_processor import JAXADownloadProcessor
from geographer.downloaders.jaxa_downloader_for_single_vector import (
    JAXADownloaderForSingleVector,
)


# To run just this test, execute
# pytest -v -s tests/test_jaxa_download.py::test_jaxa_download
@pytest.mark.slow
def test_jaxa_download():
    """Test downloading JAXA data."""
    # noqa: D202
    """
    Create connector containing vectors but no rasters
    """
    test_dir = get_test_dir()
    data_dir = test_dir / "temp/download_jaxa_test"

    vectors = gpd.read_file(test_dir / "geographer_download_test.geojson")
    vectors.set_index("name", inplace=True)

    connector = Connector.from_scratch(
        data_dir=data_dir, task_vector_classes=["object"]
    )
    connector.add_to_vectors(vectors)
    """
    Test RasterDownloaderForVectors for Sentinel-2 data
    """
    jaxa_download_processor = JAXADownloadProcessor()
    jaxa_downloader_for_single_vector = JAXADownloaderForSingleVector()
    jaxa_downloader = RasterDownloaderForVectors(
        downloader_for_single_vector=jaxa_downloader_for_single_vector,
        download_processor=jaxa_download_processor,
    )
    """
    Download Sentinel-2 rasters
    """
    jaxa_downloader.download(
        connector=connector,
        downloader_params={
            "data_version": "1804",
            "download_mode": "bboxvertices",
        },
    )
    # The vectors contain
    #     - 2 objects in Berlin (Reichstag and Brandenburg gate)
    #       that are very close to each other
    #     - 2 objects in Lisbon (Castelo de Sao Jorge and
    #       Praca Do Comercio) that are very close to each other.
    # Thus the jaxa_downloader should have downloaded two rasters,
    # one for Berlin and one for Lisbon, each containing two objects.

    # Berlin
    assert len(connector.rasters_containing_vector("berlin_reichstag")) == 1
    assert len(connector.rasters_containing_vector("berlin_brandenburg_gate")) == 1
    assert connector.rasters_containing_vector("berlin_reichstag") == (
        connector.rasters_containing_vector("berlin_brandenburg_gate")
    )

    # Lisbon
    assert len(connector.rasters_containing_vector("lisbon_castelo_de_sao_jorge")) == 1
    assert len(connector.rasters_containing_vector("lisbon_praca_do_comercio")) == 1
    assert connector.rasters_containing_vector("lisbon_castelo_de_sao_jorge") == (
        connector.rasters_containing_vector("lisbon_praca_do_comercio")
    )
    """
    Clean up: delete downloads
    """
    shutil.rmtree(data_dir)


if __name__ == "__main__":
    test_jaxa_download()
