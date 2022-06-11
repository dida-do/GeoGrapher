"""
TODO: split up into smaller units (more simple nestings etc)

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


def test_save_load_nested_base_model():
    """Test saving and loading nested BaseModel"""

    # get repo working tree directory
    repo = git.Repo('.', search_parent_directories=True)
    repo_root = Path(repo.working_tree_dir)
    download_test_data_dir = repo_root / "tests/data/temp/download_s2_test"

    # define nested BaseModel
    s2_download_processor = Sentinel2Processor()
    s2_downloader_for_single_feature = SentinelDownloaderForSingleVectorFeature()
    s2_downloader = ImgDownloaderForVectorFeatures(
        download_dir=download_test_data_dir / "download",
        downloader_for_single_feature=s2_downloader_for_single_feature,
        download_processor=s2_download_processor,
        kwarg_defaults={ # further nesting: dictionary
            "producttype": "L2A",
            "resolution": 10,
            "max_percent_cloud_coverage": 10,
            "date": ("NOW-364DAYS", "NOW"),
            "area_relation": "Contains",
            "credentials_ini_path": download_test_data_dir / "credentials.ini",
            "additional_nested_dictionary":
                {"1":2, "3":4}, # keys must be strings here: https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings
            "some_list": [1,2,3,4],
        }
    )


    """
    Test save and load Sentinel-2 Downloader
    """
    # save
    s2_downloader_json_path = download_test_data_dir / "connector/s2_downloader.json"
    s2_downloader.save(s2_downloader_json_path)

    # load
    s2_downloader_from_json = ImgDownloaderForVectorFeatures.from_json_file(
        s2_downloader_json_path,
    )
    # make sure saving and loading again doesn't change anything
    assert s2_downloader_from_json == s2_downloader


if __name__ == "__main__":
    test_save_load_nested_base_model()