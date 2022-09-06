"""Pytest fixtures."""

import shutil

import pytest
from cut_every_img_to_grid_test import CUT_SOURCE_DATA_DIR_NAME
from utils import create_dummy_imgs, delete_dummy_images, get_test_dir

from geographer import Connector


@pytest.hookimpl()
def pytest_sessionstart(session):
    """Create dummy images before session starts."""
    data_dir = get_test_dir() / CUT_SOURCE_DATA_DIR_NAME
    connector = Connector.from_data_dir(data_dir=data_dir)
    if not connector.images_dir.exists() or not connector.images_dir.glob("*.tif"):
        create_dummy_imgs(data_dir=data_dir, img_size=10980)


@pytest.hookimpl()
def pytest_sessionfinish(session):
    """Delete dummy images after session ends."""
    data_dir = get_test_dir() / CUT_SOURCE_DATA_DIR_NAME
    delete_dummy_images(data_dir=data_dir)
    shutil.rmtree(get_test_dir() / "temp", ignore_errors=True)
