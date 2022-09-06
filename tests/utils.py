"""Utils for testing."""

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


def create_dummy_imgs(
    data_dir: Union[Path, str], img_size: int, img_names: Optional[list[str]] = None
) -> None:
    """Create dummy images.

    Create dummy images for a dataset from the connector's
    raster_imgs geodataframe.
    """
    connector = Connector.from_data_dir(data_dir)
    connector.images_dir.mkdir(parents=True, exist_ok=True)

    if img_names is None:
        img_names = connector.raster_imgs.index.tolist()

    for img_name, bbox_geom, epsg_code in tqdm(
        list(
            connector.raster_imgs[["geometry", "orig_crs_epsg_code"]]
            .loc[img_names]
            .itertuples()
        ),
        desc="Creating dummy images",
    ):

        img_array = np.stack(
            [np.ones((img_size, img_size), dtype=np.uint8) * n for n in range(3)]
        )
        bbox_geom_in_img_crs = transform_shapely_geometry(
            bbox_geom, from_epsg=connector.crs_epsg_code, to_epsg=epsg_code
        )
        transform = rio.transform.from_bounds(
            *bbox_geom_in_img_crs.bounds, img_size, img_size
        )

        with rio.open(
            connector.images_dir / img_name,
            "w",
            driver="GTiff",
            height=img_size,
            width=img_size,
            count=3,
            dtype=np.uint8,
            crs=f"EPSG:{epsg_code}",
            transform=transform,
        ) as dst:
            for idx in range(3):
                dst.write(img_array[idx, :, :], idx + 1)


def delete_dummy_images(data_dir: Union[Path, str]) -> None:
    """Delete dummy image data (images and segmentation labels) from dataset."""
    shutil.rmtree(data_dir / "images", ignore_errors=True)
    shutil.rmtree(data_dir / "labels", ignore_errors=True)
