"""
TODO: write test for download_mode 'bboxgrid' using large polygon

Run by hand to test downloading Sentinel-2 data.

Intentionally not discoverable by pytest: Downloading Sentinel-2 images is slow
and uses a lot of disk space. Needs an .ini file with API credentials.
"""

from pathlib import Path
import shutil
import git
import geopandas as gpd
from pydantic import BaseModel

from geographer import Connector
from geographer.downloaders.downloader_for_features import ImgDownloaderForVectorFeatures
from geographer.downloaders.sentinel2_downloader_for_single_feature import SentinelDownloaderForSingleVectorFeature
from geographer.downloaders.sentinel2_download_processor import Sentinel2Processor
from utils import get_test_dir


def test_s2_download():
    """Test downloading Sentinel-2 data"""

    """
    Create connector containing vector_features but no raster images
    """
    test_dir = get_test_dir()

    data_dir = test_dir / "temp/download_s2_test"
    credentials_ini_path = test_dir / "download_s2_test_credentials.ini"
    assert credentials_ini_path.is_file(), f"Need credentials in {credentials_ini_path} to test sentinel download"

    vector_features = gpd.read_file(test_dir / "geographer_download_test.geojson", driver="GeoJSON")
    vector_features.set_index("name", inplace=True)

    connector = Connector.from_scratch(data_dir=data_dir ,task_feature_classes=['object'])
    connector.add_to_vector_features(vector_features)

    """
    Test ImgDownloaderForVectorFeatures for Sentinel-2 data
    """
    s2_download_processor = Sentinel2Processor()
    s2_downloader_for_single_feature = SentinelDownloaderForSingleVectorFeature()
    s2_downloader = ImgDownloaderForVectorFeatures(
        download_dir=data_dir / "download",
        downloader_for_single_feature=s2_downloader_for_single_feature,
        download_processor=s2_download_processor,
        kwarg_defaults={
            "producttype": "L2A",
            "resolution": 10,
            "max_percent_cloud_coverage": 10,
            "date": ("NOW-364DAYS", "NOW"),
            "area_relation": "Contains",
            "credentials_ini_path": credentials_ini_path,
        }
    )


    """
    Download Sentinel-2 images
    """
    s2_downloader.download(connector=connector)
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


    """
    Clean up: delete downloads
    """
    shutil.rmtree(data_dir)


if __name__ == "__main__":
    test_s2_download()