"""Manually triggered test of Sentinel-2 downloader.

Run by hand to test downloading Sentinel-2 data.

Intentionally not discoverable by pytest: Downloading Sentinel-2 rasters is slow
and uses a lot of disk space. Needs an .ini file with API credentials.
"""

import shutil
from datetime import date, timedelta

import geopandas as gpd
import pytest
from utils import get_test_dir

from geographer import Connector
from geographer.downloaders.downloader_for_vectors import RasterDownloaderForVectors
from geographer.downloaders.eodag_downloader_for_single_vector import (
    EodagDownloaderForSingleVector,
)
from geographer.downloaders.sentinel2_download_processor import Sentinel2SAFEProcessor


# To run just this test, execute
# pytest -v -s tests/test_eodag_s2_download.py::test_s2_download
@pytest.mark.slow
def test_s2_download():
    """Test downloading Sentinel-2 data."""
    # noqa: D202
    """
    Create connector containing vectors but no rasters
    """
    test_dir = get_test_dir()

    data_dir = test_dir / "temp/download_s2_test"

    vectors = gpd.read_file(test_dir / "geographer_download_test.geojson")
    vectors.set_index("name", inplace=True)

    connector = Connector.from_scratch(
        data_dir=data_dir, task_vector_classes=["object"]
    )
    connector.add_to_vectors(vectors)
    """
    Test RasterDownloaderForVectors for Sentinel-2 data
    """
    product_type = "S2_MSI_L2A"
    download_processor = Sentinel2SAFEProcessor()
    downloader_for_single_vector = EodagDownloaderForSingleVector()
    downloader = RasterDownloaderForVectors(
        downloader_for_single_vector=downloader_for_single_vector,
        download_processor=download_processor,
    )

    if "cop_dataspace" not in downloader_for_single_vector.eodag.available_providers(
        product_type=product_type
    ):
        raise RuntimeError(
            "'cop_dataspace' needs to be available as a provider. "
            "Probably the username and/or password are missing."
        )
    """
    Download Sentinel-2 rasters
    """
    downloader_params = {
        "search_kwargs": {
            "provider": "cop_dataspace",
            "productType": product_type,
            "start": (date.today() - timedelta(days=364)).strftime("%Y-%m-%d"),
            "end": date.today().strftime("%Y-%m-%d"),
        },
        "filter_online": True,
        "sort_by": ("cloudCover", "ASC"),
        "suffix_to_remove": ".SAFE",
    }
    processor_params = {
        "resolution": 10,
        "delete_safe": True,
    }
    downloader.download(
        connector=connector,
        downloader_params=downloader_params,
        processor_params=processor_params,
    )
    # The vectors contain
    #     - 2 objects in Berlin (Reichstag and Brandenburg gate)
    #       that are very close to each other
    #     - 2 objects in Lisbon (Castelo de Sao Jorge and
    #       Praca Do Comercio) that are very close to each other.
    # Thus the downloader should have downloaded two rasters,
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
    """
    Clean up: delete downloads
    """
    shutil.rmtree(data_dir)


if __name__ == "__main__":
    test_s2_download()
