"""Utility functions for merging datasets."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from geographer.connector import Connector


def merge_datasets(
    source_data_dir: Path | str,
    target_data_dir: Path | str,
    delete_source: bool = True,
) -> None:
    """Merge datasets.

    Args:
        source_data_dir: data dir of source dataset
        target_data_dir: data dir of target dataset
        delete_source: Whether to delete source dataset after merging.
            Defaults to True.
    """
    source_connector = Connector.from_data_dir(source_data_dir)
    target_connector = Connector.from_data_dir(target_data_dir)

    # copy over raster_data_dirs
    for source_dir, target_dir in zip(
        source_connector.raster_data_dirs, target_connector.raster_data_dirs
    ):
        files_in_target_dir = {raster.name for raster in target_dir.iterdir()}
        pbar = tqdm(source_dir.iterdir())
        pbar.set_description(f"copying {str(source_dir.name)}")
        for raster_path in pbar:
            if raster_path.name not in files_in_target_dir:
                shutil.copy2(raster_path, target_dir)

    # merge/copy over downloads (e.g. safe_files)
    merge_dirs(str(source_connector.download_dir), str(target_connector.download_dir))

    target_connector.add_to_polygons_df(source_connector.polygons_df)
    target_connector.add_to_rasters(source_connector.rasters)
    target_connector.save()


# TODO rewrite using pathlib
def merge_dirs(root_src_dir: Path | str, root_dst_dir: Path | str) -> None:
    """Recursively merge two folders including subfolders.

    (Shamelessly copied from stackoverflow)

    Args:
        root_src_dir: root source directory
        root_dst_dir: root target directory
    """
    pbar = tqdm(os.walk(root_src_dir))
    for src_dir, dirs, files in pbar:
        pbar.set_description(str(src_dir))
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_dir)
