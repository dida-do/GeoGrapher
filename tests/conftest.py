"""Pytest fixtures."""

import shutil
from pathlib import Path

import pytest
from utils import create_dummy_rasters, delete_dummy_rasters, get_test_dir

from geographer import Connector

CUT_SOURCE_DATA_DIR_NAME = "cut_source"


@pytest.fixture(scope="session")
def dummy_cut_source_data_dir() -> Path:
    """Return cut source data dir containing dummy data.

    Dummy rasters are created before the pytest session starts and removed afterwards.
    """
    data_dir = get_test_dir() / CUT_SOURCE_DATA_DIR_NAME
    connector = Connector.from_data_dir(data_dir=data_dir)
    if not connector.rasters_dir.exists() or not connector.rasters_dir.glob("*.tif"):
        create_dummy_rasters(data_dir=data_dir, raster_size=10980)
    yield data_dir
    delete_dummy_rasters(data_dir=data_dir)
    shutil.rmtree(get_test_dir() / "temp", ignore_errors=True)
