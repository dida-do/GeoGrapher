import os
import shutil
from pathlib import Path
from typing import Union

from tqdm.auto import tqdm

from rs_tools import Connector


def merge_datasets(source_data_dir: Union[Path, str],
                   target_data_dir: Union[Path, str],
                   delete_source: bool = True) -> None:
    """TODO.

    Args:
        source_data_dir (Union[Path, str]): [description]
        target_data_dir (Union[Path, str]): [description]
        delete_source (bool, optional): [description]. Defaults to True.
    """

    source_connector = Connector.from_data_dir(source_data_dir)
    target_connector = Connector.from_data_dir(target_data_dir)

    # copy over image_data_dirs
    for source_dir, target_dir in zip(source_connector.image_data_dirs,
                                      target_connector.image_data_dirs):
        files_in_target_dir = {img.name for img in target_dir.iterdir()}
        pbar = tqdm(source_dir.iterdir())
        pbar.set_description(f'copying {str(source_dir.name)}')
        for img_path in pbar:
            if img_path.name not in files_in_target_dir:
                shutil.copy2(img_path, target_dir)

    # merge/copy over downloads (e.g. safe_files)
    merge_dirs(str(source_connector.download_dir), str(target_connector.download_dir))

    target_connector.add_to_polygons_df(source_connector.polygons_df)
    target_connector.add_to_raster_imgs(source_connector.raster_imgs)
    target_connector.save()


# recursively merge two folders including subfolders
# shamelessly stolen from the internet to save time
# TODO rewrite using pathlib
def merge_dirs(root_src_dir: Union[Path, str],
               root_dst_dir: Union[Path, str]) -> None:

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
