"""
Run by hand to test downloading JAXA data. Intentionally not discoverable
by pytest: Downloading JAXA images is slightly slow.
"""


from pathlib import Path
import shutil
import git
import geopandas as gpd
from pydantic import BaseModel

from geographer import Connector
from geographer.downloaders.downloader_for_features import ImgDownloaderForVectorFeatures
from geographer.downloaders.jaxa_downloader_for_single_feature import JAXADownloaderForSingleVectorFeature
from geographer.downloaders.jaxa_download_processor import JAXADownloadProcessor
from tests.utils import get_test_dir

def test_jaxa_download():
    """Test downloading JAXA data"""

    """
    Create connector containing vector_features but no raster images
    """
    test_dir = get_test_dir()
    data_dir = test_dir / "temp/download_jaxa_test"

    vector_features = gpd.read_file(test_dir / 'geographer_download_test.geojson', driver='GeoJSON')
    vector_features.set_index("name", inplace=True)

    connector = Connector.from_scratch(data_dir=data_dir ,task_feature_classes=['object'])
    connector.add_to_vector_features(vector_features)

    """
    Test ImgDownloaderForVectorFeatures for Sentinel-2 data
    """
    jaxa_download_processor = JAXADownloadProcessor()
    jaxa_downloader_for_single_feature = JAXADownloaderForSingleVectorFeature()
    jaxa_downloader = ImgDownloaderForVectorFeatures(
        download_dir=data_dir / "download",
        downloader_for_single_feature=jaxa_downloader_for_single_feature,
        download_processor=jaxa_download_processor,
        kwarg_defaults={
            "data_version": "1804",
            "download_mode": "bboxvertices",
        }
    )

    """
    Download Sentinel-2 images
    """
    jaxa_downloader.download(connector=connector)
    # The vector_features contain
    #     - 2 objects in Berlin (Reichstag and Brandenburg gate) that are very close to each other
    #     - 2 objects in Lisbon (Castelo de Sao Jorge and Praca Do Comercio) that are very close to each other.
    # Thus the jaxa_downloader should have downloaded two images, one for Berlin and one for Lisbon, each containing two objects.

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
    test_jaxa_download()