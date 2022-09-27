"""Utils for testing."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

import git
import numpy as np
import rasterio as rio
from tqdm.auto import tqdm

from geographer import Connector
from geographer.utils.utils import transform_shapely_geometry


def get_test_dir():
    """Return directory containing test data."""
    repo = git.Repo(".", search_parent_directories=True)
    return Path(repo.working_tree_dir) / "tests/data"


def create_dummy_rasters(
    data_dir: Union[Path, str],
    raster_size: int,
    raster_names: Optional[list[str]] = None,
) -> None:
    """Create dummy rasters.

    Create dummy rasters for a dataset from the connector's
    rasters geodataframe.
    """
    connector = Connector.from_data_dir(data_dir)
    connector.rasters_dir.mkdir(parents=True, exist_ok=True)

    if raster_names is None:
        raster_names = connector.rasters.index.tolist()

    for raster_name, bbox_geom, epsg_code in tqdm(
        list(
            connector.rasters[["geometry", "orig_crs_epsg_code"]]
            .loc[raster_names]
            .itertuples()
        ),
        desc="Creating dummy rasters",
    ):

        raster_array = np.stack(
            [np.ones((raster_size, raster_size), dtype=np.uint8) * n for n in range(3)]
        )
        bbox_geom_in_raster_crs = transform_shapely_geometry(
            bbox_geom, from_epsg=connector.crs_epsg_code, to_epsg=epsg_code
        )
        transform = rio.transform.from_bounds(
            *bbox_geom_in_raster_crs.bounds, raster_size, raster_size
        )

        with rio.open(
            connector.rasters_dir / raster_name,
            "w",
            driver="GTiff",
            height=raster_size,
            width=raster_size,
            count=3,
            dtype=np.uint8,
            crs=f"EPSG:{epsg_code}",
            transform=transform,
        ) as dst:
            for idx in range(3):
                dst.write(raster_array[idx, :, :], idx + 1)


def delete_dummy_rasters(data_dir: Union[Path, str]) -> None:
    """Delete dummy raster data (rasters and segmentation labels) from dataset."""
    shutil.rmtree(data_dir / "rasters", ignore_errors=True)
    shutil.rmtree(data_dir / "labels", ignore_errors=True)
