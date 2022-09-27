"""
Manually triggered test of Sentinel-2 downloader.

Run by hand to test downloading Sentinel-2 data.

Intentionally not discoverable by pytest: Downloading Sentinel-2 rasters is slow
and uses a lot of disk space. Needs an .ini file with API credentials.

TODO: write test for download_mode 'bboxgrid' using large polygon
"""

import shutil

import geopandas as gpd
from utils import get_test_dir

from geographer import Connector
from geographer.downloaders.downloader_for_vectors import RasterDownloaderForVectors
from geographer.downloaders.sentinel2_download_processor import Sentinel2Processor
from geographer.downloaders.sentinel2_downloader_for_single_vector import (
    SentinelDownloaderForSingleVector,
)


def test_s2_download():
    """Test downloading Sentinel-2 data."""
    # noqa: D202
    """
    Create connector containing vectors but no rasters
    """
    test_dir = get_test_dir()

    data_dir = test_dir / "temp/download_s2_test"
    credentials_ini_path = test_dir / "download_s2_test_credentials.ini"
    assert (
        credentials_ini_path.is_file()
    ), f"Need credentials in {credentials_ini_path} to test sentinel download"

    vectors = gpd.read_file(
        test_dir / "geographer_download_test.geojson", driver="GeoJSON"
    )
    vectors.set_index("name", inplace=True)

    connector = Connector.from_scratch(
        data_dir=data_dir, task_vector_classes=["object"]
    )
    connector.add_to_vectors(vectors)

    """
    Test RasterDownloaderForVectors for Sentinel-2 data
    """
    s2_download_processor = Sentinel2Processor()
    s2_downloader_for_single_vector = SentinelDownloaderForSingleVector()
    s2_downloader = RasterDownloaderForVectors(
        downloader_for_single_vector=s2_downloader_for_single_vector,
        download_processor=s2_download_processor,
        kwarg_defaults={
            "producttype": "L2A",
            "resolution": 10,
            "max_percent_cloud_coverage": 10,
            "date": ("NOW-364DAYS", "NOW"),
            "area_relation": "Contains",
            "credentials": credentials_ini_path,
        },
    )

    """
    Download Sentinel-2 rasters
    """
    s2_downloader.download(connector=connector)
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

    """
    Clean up: delete downloads
    """
    shutil.rmtree(data_dir)


if __name__ == "__main__":
    test_s2_download()
