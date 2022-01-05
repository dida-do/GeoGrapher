# move to rs_tools sentinel2-mixin

# !!!
# assert img.shape == (10980, 10980)

import itertools
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import time
from scipy.ndimage import zoom
import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.mask
from rstools.utils import create_logger
from rasterio.enums import Resampling
from shapely.geometry import box

NO_DATA_VAL = 0  # No data value for sentinel 2 L1C

logger = create_logger(__name__)


def safe_to_geotif_L2A(
        safe_root: Path,
        resolution: Union[str, int],
        upsample_lower_resolution: bool = True,
        outdir: Path = None,
        TCI: bool = True,
        requested_jp2_masks: List[str] = ["CLDPRB", "SNWPRB"],
        requested_gml_mask: List[Tuple[str]] = [("CLOUDS", "B00")],
        # upsampling_method_specifier: str = "bilinear"
        ) -> Tuple[Dict, Dict]:
    """
    Converts a .SAFE file with L2A sentinel-2 data to geotif and returns a dict 
    with the crs epsg code and a shapely polygon defined by the image bounds.

    ..note::
    
        - band structure of final geotif:
            if TCI: 1-3 TCI RGB
            else sorted(jps2_masks and bands (either only desired resolution or additionally upsampled)), gml_mask_order
        - jp2_masks are only available up to a resolution of 20 m, so for 10m the 20m mask ist taken
        - SNWPRB for snow masks

    Args:
        safe_root (Path): is the safe folder root
        resolution (Union[int,str]): the desired resolution
        upsample_lower_resolution (bool): Whether to include lower resolution bands and upsample them
        TCI (bool): whether to load the true color image
        requested_jp2_masks (List[str]): jp2 mask to load
        requested_gml_mask (List[Tuple[str]]): gml masks to load ([0] mask name as string, [1] band for which to get the mask)
        upsampling_method_specifier (str): method to upsample from lower resolution to higher. Options: (nearest,bilinear,cubic,average)

    Returns:
        Dict: tif crs and bounding rectangle
        Dict: Band names of tif

    """

    # DEBUG
    times = []
    starttime = (time.time(), 'start')
    times.append(starttime)

    #assert resolution is within available
    assert resolution in [10, 20, 60, "10", "20", "60"]

    #define output file
    out_file_parent_dir = outdir if (outdir
                                     and outdir.is_dir()) else safe_root.parent
    outfile = out_file_parent_dir / (safe_root.stem + ".tif")

    granule_dir = safe_root / "GRANULE"
    masks_dir = granule_dir / "{}/QI_DATA/".format(os.listdir(granule_dir)[0])

    # JP2 masks
    jp2_resolution = 20 if resolution in [10, "10"] else resolution
    jp2_mask_paths = list(
        filter(
            lambda file: any(mask_name in file.name
                             for mask_name in requested_jp2_masks) and
            f"{jp2_resolution}m" in file.name, masks_dir.glob("*.jp2")))

    # Paths for S2 Bands
    jp2_path_desired_resolution = granule_dir / "{}/IMG_DATA/R{}m/".format(
        os.listdir(granule_dir)[0], resolution)

    tci_path = next(
        filter(lambda path: path.stem.split("_")[-2] == "TCI",
               jp2_path_desired_resolution.glob("*.jp2")))

    #add missing higher res jps paths
    img_data_bands = list(
        filter(lambda path: path.stem.split("_")[-2] not in ["TCI"],
               jp2_path_desired_resolution.glob("*.jp2")))
    out_default_reader = rio.open(img_data_bands[0], driver="JP2OpenJPEG")

    # include lower resolution bands
    if upsample_lower_resolution:

        for higher_res in filter(lambda res: res > int(resolution),
                                 [10, 20, 60]):
            jp2_higher_res_path = granule_dir / "{}/IMG_DATA/R{}m/".format(
                os.listdir(granule_dir)[0], higher_res)

            img_data_bands += list(
                filter(
                    lambda path: path.stem.split("_")[-2] not in itertools.
                        chain(
                            map(
                                lambda path: path.stem.split("_")[-2],
                                img_data_bands
                            ),
                            ["TCI"]
                        ),
                    jp2_higher_res_path.glob("*.jp2")))

    # # if we have both B08 and B8A remove B8A
    # if {'B8A', 'B08'} <= {name.name.split("_")[-2] for name in img_data_bands}:
    #     img_data_bands = [path for path in img_data_bands if path.name.split("_")[-2] != 'B8A']

    #set up rasterio loaders
    bands_dict = OrderedDict()
    max_width = 0
    max_height = 0
    for file in img_data_bands + jp2_mask_paths:

        band = rio.open(file, driver="JP2OpenJPEG")
        max_width = max(max_width, band.width)
        max_height = max(max_height, band.height)
        bands_dict[file.stem.split("_")[-2]] = (
            band, int(file.stem.split("_")[-1][:2]))

    # sort bands_dict
    bands_dict = OrderedDict(sorted(bands_dict.items()))

    # paths for gml masks
    gml_mask_paths = list(
        filter(
            lambda path: tuple(path.stem.split("_")[-2:]) in
            requested_gml_mask, masks_dir.glob("*.gml")))
    gml_mask_paths_dict = {
        tuple(path.stem.split("_")[-2:]): path
        for path in gml_mask_paths
    }

    #reader for tci
    tci_band = rio.open(tci_path, driver="JP2OpenJPEG")

    #number of bands in final geotif
    count = len(img_data_bands + jp2_mask_paths + gml_mask_paths) + 3 * TCI

    # upsampling_function = {
    #     "bilinear": Resampling.bilinear,
    #     "cubic": Resampling.cubic,
    #     "nearest": Resampling.nearest,
    #     "average": Resampling.average
    # }.get(upsampling_method_specifier, None)

    # if upsampling_function is None:
    #     raise ValueError(
    #         f"Upsampling Specifier {upsampling_method_specifier} not understood"
    #     )

    # DEBUG
    timestamp = (time.time(), 'before_write')
    prev_time = times[-1][0]
    elapsed_time = timestamp[0] - prev_time
    print(f'{timestamp[1]}}: {elapsed_time}')
    times.append(timestamp)

    #write geotif
    tif_band_names = {}
    with rio.open(outfile,
                  "w",
                  driver="GTiff",
                  width=max_width,
                  height=max_width,
                  count=count,
                  crs=out_default_reader.crs,
                  transform=out_default_reader.transform,
                  dtype=out_default_reader.dtypes[0]) as dst:

        dst.nodata = NO_DATA_VAL

        #write gml masks
        for idx, (gml_name,
                  gml_path) in enumerate(gml_mask_paths_dict.items()):

            # DEBUG
            timestamp = (time.time(), f'before_generating_gml_{gml_name}')
            prev_time = times[-1][0]
            elapsed_time = timestamp[0] - prev_time
            print(f'{timestamp[1]}}: {elapsed_time}')
            times.append(timestamp)

            try:
                shapes = gpd.read_file(gml_path)["geometry"].values
                mask, _, _ = rasterio.mask.raster_geometry_mask(
                    out_default_reader, shapes, crop=False, invert=True)
            except ValueError:
                mask = np.full(shape=out_default_reader.read(1).shape,
                               fill_value=0.0,
                               dtype=np.uint16)

            band_idx = len(bands_dict) + 3 * TCI + idx + 1
            tif_band_names[band_idx] = "_".join(gml_name)

            # DEBUG
            timestamp = (time.time(), f'after_generating_gml_{gml_name}')
            prev_time = times[-1][0]
            elapsed_time = timestamp[0] - prev_time
            print(f'{timestamp[1]}}: {elapsed_time}')
            times.append(timestamp)

            dst.write(mask.astype(np.uint16), band_idx)

            # DEBUG
            timestamp = (time.time(), f'after_writing_gml_{gml_name}')
            prev_time = times[-1][0]
            elapsed_time = timestamp[0] - prev_time
            print(f'{timestamp[1]}}: {elapsed_time}')
            times.append(timestamp)

        #write jp2 bands
        for idx, (band_name, (dst_reader,
                              res)) in enumerate(bands_dict.items()):

            # if res != int(resolution):
            #     img = dst_reader.read(
            #         1,
            #         out_shape=(1, *out_default_reader.read(1).shape),
            #         resampling=upsampling_function)
            # else:
            #     img = dst_reader.read(1)

            # DEBUG
            timestamp = (time.time(), f'before_reading_jp2_{band_name}, res{res}')
            prev_time = times[-1][0]
            elapsed_time = timestamp[0] - prev_time
            print(f'{timestamp[1]}}: {elapsed_time}')
            times.append(timestamp)

            if res != int(resolution):

                assert res % int(resolution) == 0
                factor = res // int(resolution)

                img = dst_reader.read(1)
                img = zoom(img, factor, order=3)

                assert img.shape == (10980, 10980)

                # img = dst_reader.read(
                #     1,
                #     out_shape=(1, *out_default_reader.read(1).shape),
                #     resampling=upsampling_function)
            else:
                img = dst_reader.read(1)

            if not dst_reader.dtypes[0] == out_default_reader.dtypes[0]:
                img = (img * (65535.0 / 255.0)).astype(np.uint16)

            # DEBUG
            timestamp = (time.time(), f'after_reading_jp2_{band_name}, res{res}')
            prev_time = times[-1][0]
            elapsed_time = timestamp[0] - prev_time
            print(f'{timestamp[1]}}: {elapsed_time}')
            times.append(timestamp)

            band_idx = 3 * TCI + idx + 1
            tif_band_names[band_idx] = band_name
            dst.write(img, band_idx)
            dst_reader.close()

            # DEBUG
            timestamp = (time.time(), f'after_writing_jp2_{band_name}, res{res}')
            prev_time = times[-1][0]
            elapsed_time = timestamp[0] - prev_time
            print(f'{timestamp[1]}}: {elapsed_time}')
            times.append(timestamp)

        #write tci
        if TCI:
            for i in range(3):

                # DEBUG
                timestamp = (time.time(), f'before_reading_TCI_band_{i}')
                prev_time = times[-1][0]
                elapsed_time = timestamp[0] - prev_time
                print(f'{timestamp[1]}}: {elapsed_time}')
                times.append(timestamp)

                band_idx = i + 1
                img = (tci_band.read(band_idx) * (65535.0 / 255.0)).astype(
                    np.uint16)
                tif_band_names[band_idx] = f"tci_{band_idx}"

                # DEBUG
                timestamp = (time.time(), f'before_writing_TCI_band_{i}')
                prev_time = times[-1][0]
                elapsed_time = timestamp[0] - prev_time
                print(f'{timestamp[1]}}: {elapsed_time}')
                times.append(timestamp)

                dst.write(img, band_idx)

                # DEBUG
                timestamp = (time.time(), f'after_writing_TCI_band_{i}')
                prev_time = times[-1][0]
                elapsed_time = timestamp[0] - prev_time
                print(f'{timestamp[1]}}: {elapsed_time}')
                times.append(timestamp)

        # add tags and descriptions
        for band_idx, name in tif_band_names.items():
            dst.update_tags(band_idx, band_name=name)
            dst.set_band_description(bidx=band_idx, value=name)

        crs_epsg_code = dst.crs.to_epsg()
        img_bounding_rectangle = box(*dst.bounds)

    for t, action in enumerate(times):
        print(f'{action}: time {t}')

    return {
        'crs_epsg_code': crs_epsg_code,
        'img_bounding_rectangle': img_bounding_rectangle
    }

    # return {
    #     'crs_epsg_code': crs_epsg_code,
    #     'img_bounding_rectangle': img_bounding_rectangle
    # }, tif_band_names


if __name__ == "__main__":

    2
    # # name = "S2B_MSIL2A_20211020T110009_N0301_R094_T30STC_20211020T125659.SAFE"
    # name = "S2B_MSIL2A_20211120T035039_N0301_R104_T47PMS_20211120T060056.SAFE"
    # safe_path = Path(f'/home/rustam/rstools/data/segmentation/root4grid1024/downloads/safe_files/{name}')

    # SAFES_ROOT = Path(f'/home/rustam/rstools/data/segmentation/root4cutouts/downloads/')
    # # SAFES_ROOT_OLD = Path(f'/home/rustam/rstools/data/segmentation/root4cutouts/downloads/safe_files_OLD/')

    # import itertools

    # # for safe in itertools.chain(SAFES_ROOT.rglob(".SAFE"), SAFES_ROOT_OLD.rglob(".SAFE")):
    # for safe in SAFES_ROOT.rglob("*.SAFE"):
    #     print(safe)


    # safe_path = Path(f"/home/rustam/rstools/data/segmentation/rustam_testing_ground/{name}")
    # codes,names = safe_to_geotif_L2A(
    #     safe_root=safe_path,
    #     resolution=60)

    # codes = safe_to_geotif_L2A(
    #     safe_root=safe_path,
    #     resolution=10,
    #     outdir=Path(f"/home/rustam/rstools/data/segmentation/rustam_testing_ground/")
    # )

    # 123