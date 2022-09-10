"""Pytest fixtures."""

import shutil
from pathlib import Path

import pytest
from utils import create_dummy_imgs, delete_dummy_images, get_test_dir

from geographer import Connector

CUT_SOURCE_DATA_DIR_NAME = "cut_source"


@pytest.fixture(scope="session")
def dummy_cut_source_data_dir() -> Path:
    """Return cut source data dir containing dummy data.

    Dummy rasters are created before the pytest session starts and removed afterwards.
    """
    data_dir = get_test_dir() / CUT_SOURCE_DATA_DIR_NAME
    connector = Connector.from_data_dir(data_dir=data_dir)
    if not connector.images_dir.exists() or not connector.images_dir.glob("*.tif"):
        create_dummy_imgs(data_dir=data_dir, img_size=10980)
    yield data_dir
    delete_dummy_images(data_dir=data_dir)
    shutil.rmtree(get_test_dir() / "temp", ignore_errors=True)
