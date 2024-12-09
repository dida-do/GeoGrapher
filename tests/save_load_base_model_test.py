"""Test saving/loading nested BaseModels."""

from pathlib import Path

import git

from geographer.downloaders.downloader_for_vectors import RasterDownloaderForVectors
from geographer.downloaders.eodag_downloader_for_single_vector import (
    EodagDownloaderForSingleVector,
)
from geographer.downloaders.sentinel2_download_processor import Sentinel2SAFEProcessor


def test_save_load_nested_base_model():
    """Test saving and loading nested BaseModel."""
    # get repo working tree directory
    repo = git.Repo(".", search_parent_directories=True)
    repo_root = Path(repo.working_tree_dir)
    download_test_data_dir = repo_root / "tests/data/temp/download_s2_test"

    # define nested BaseModel
    download_processor = Sentinel2SAFEProcessor(
        default_process_kwargs={
            "resolution": 10,
        },
    )
    downloader_for_single_vector = EodagDownloaderForSingleVector(
        default_params={
            # further nesting: dictionary
            "provider": "cop_dataspace",
            "productType": "S2_MSI_L2A",
            "start": "2021-03-01",
            "end": "2021-03-31",
            # keys must be strings in the following dict, see
            # https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings  # noqa: E501
            "additional_nested_dictionary": {
                "1": 2,
                "3": 4,
                "4": {
                    # One more layer of nesting
                    "5": None,
                },
            },
            "some_list": [1, 2, 3, 4],
        },
    )
    downloader = RasterDownloaderForVectors(
        download_dir=download_test_data_dir / "download",
        downloader_for_single_vector=downloader_for_single_vector,
        download_processor=download_processor,
    )
    """
    Test save and load Sentinel-2 Downloader
    """
    # save
    downloader_json_path = download_test_data_dir / "connector/s2_downloader.json"
    downloader.save(downloader_json_path)

    # load
    downloader_from_json = RasterDownloaderForVectors.from_json_file(
        downloader_json_path,
    )
    # make sure saving and loading again doesn't change anything
    assert downloader_from_json.model_dump() == downloader.model_dump()


if __name__ == "__main__":
    test_save_load_nested_base_model()
