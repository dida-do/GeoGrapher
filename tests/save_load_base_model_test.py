"""Test saving/loading nested BaseModels.

TODO: split up into smaller units (more simple nestings etc)
"""

from pathlib import Path

import git

from geographer.downloaders.downloader_for_features import (
    ImgDownloaderForVectorFeatures,
)
from geographer.downloaders.sentinel2_download_processor import Sentinel2Processor
from geographer.downloaders.sentinel2_downloader_for_single_feature import (
    SentinelDownloaderForSingleVectorFeature,
)


def test_save_load_nested_base_model():
    """Test saving and loading nested BaseModel."""
    # get repo working tree directory
    repo = git.Repo(".", search_parent_directories=True)
    repo_root = Path(repo.working_tree_dir)
    download_test_data_dir = repo_root / "tests/data/temp/download_s2_test"

    # define nested BaseModel
    s2_download_processor = Sentinel2Processor()
    s2_downloader_for_single_feature = SentinelDownloaderForSingleVectorFeature()
    s2_downloader = ImgDownloaderForVectorFeatures(
        download_dir=download_test_data_dir / "download",
        downloader_for_single_feature=s2_downloader_for_single_feature,
        download_processor=s2_download_processor,
        kwarg_defaults={  # further nesting: dictionary
            "producttype": "L2A",
            "resolution": 10,
            "max_percent_cloud_coverage": 10,
            "date": ("NOW-364DAYS", "NOW"),
            "area_relation": "Contains",
            "credentials_ini_path": download_test_data_dir / "credentials.ini",
            # keys must be strings in the following dict, see
            # https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings  # noqa: E501
            "additional_nested_dictionary": {
                "1": 2,
                "3": 4,
            },
            "some_list": [1, 2, 3, 4],
        },
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
