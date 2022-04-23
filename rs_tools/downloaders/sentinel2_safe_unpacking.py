"""Unpack/convert sentinel-2 SAFE files to GeoTiffs."""

import itertools
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.mask
from rasterio.errors import RasterioIOError
from scipy.ndimage import zoom
from shapely.geometry import box

from rs_tools.utils.utils import create_logger

NO_DATA_VAL = 0  # No data value for sentinel 2 L1C

log = create_logger(__name__)


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
    """Converts a .SAFE file with L2A sentinel-2 data to geotif and returns a
    dict with the crs epsg code and a shapely polygon defined by the image
    bounds.

    Warning:
        The L2A band structure changed in October 2021, new products do not contain gml masks anymore. In this

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

    # assert resolution is within available
    assert resolution in [10, 20, 60, "10", "20", "60"]

    # define output file
    out_file_parent_dir = outdir if (outdir
                                     and outdir.is_dir()) else safe_root.parent
    outfile = out_file_parent_dir / (safe_root.stem + "_TEMP.tif")

    granule_dir = safe_root / "GRANULE"
    masks_dir = granule_dir / "{}/QI_DATA/".format(os.listdir(granule_dir)[0])

    # JP2 masks
    jp2_resolution = 20 if resolution in [10, "10"] else resolution
    jp2_mask_paths = list(
        filter(
            lambda file: any(mask_name in file.name
                             for mask_name in requested_jp2_masks) and
            f"{jp2_resolution}m" in file.name,
            masks_dir.glob("*.jp2"),
        ))

    # Paths for S2 Bands
    jp2_path_desired_resolution = granule_dir / "{}/IMG_DATA/R{}m/".format(
        os.listdir(granule_dir)[0], resolution)

    tci_path = next(
        filter(lambda path: path.stem.split("_")[-2] == "TCI",
               jp2_path_desired_resolution.glob("*.jp2")))

    # add missing higher res jps paths
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
                        map(lambda path: path.stem.split("_")[-2],
                            img_data_bands), ["TCI"]),
                    jp2_higher_res_path.glob("*.jp2"),
                ))

    # # if we have both B08 and B8A remove B8A
    # if {'B8A', 'B08'} <= {name.name.split("_")[-2] for name in img_data_bands}:
    #     img_data_bands = [path for path in img_data_bands if path.name.split("_")[-2] != 'B8A']

    # set up rasterio loaders
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
    # add invalid paths for missing gml masks (will result in zero bands later)
    for pair in requested_gml_mask:
        if pair not in gml_mask_paths_dict:
            gml_mask_paths_dict[pair] = Path(f"/path/to/nowhere")
            gml_mask_paths.append(Path(f"/path/to/nowhere"))

    # reader for tci
    tci_band = rio.open(tci_path, driver="JP2OpenJPEG")

    # number of bands in final geotif
    count = len(img_data_bands + jp2_mask_paths + gml_mask_paths) + 3 * TCI

    # write geotif
    tif_band_names = {}
    with rio.open(
            outfile,
            "w",
            driver="GTiff",
            width=max_width,
            height=max_width,
            count=count,
            crs=out_default_reader.crs,
            transform=out_default_reader.transform,
            dtype=out_default_reader.dtypes[0],
    ) as dst:

        dst.nodata = NO_DATA_VAL

        # write gml masks
        for idx, (gml_name,
                  gml_path) in enumerate(gml_mask_paths_dict.items()):

            try:
                assert gml_path.is_file()
                shapes = gpd.read_file(gml_path)["geometry"].values
                mask, _, _ = rasterio.mask.raster_geometry_mask(
                    out_default_reader, shapes, crop=False, invert=True)
            # in case mask is empty or does not exist:
            except (ValueError, AssertionError, RasterioIOError):
                log.info("Using all zero band for gml mask %s for %s",
                         gml_name, safe_root.name)
                mask = np.full(shape=out_default_reader.read(1).shape,
                               fill_value=0.0,
                               dtype=np.uint16)

            band_idx = len(bands_dict) + 3 * TCI + idx + 1
            tif_band_names[band_idx] = "_".join(gml_name)

            dst.write(mask.astype(np.uint16), band_idx)

        # write jp2 bands
        for idx, (band_name, (dst_reader,
                              res)) in enumerate(bands_dict.items()):

            if res != int(resolution):

                assert res % int(resolution) == 0
                factor = res // int(resolution)

                img = dst_reader.read(1)
                img = zoom(img, factor, order=3)

                assert img.shape == (10980, 10980)

            else:
                img = dst_reader.read(1)

            if not dst_reader.dtypes[0] == out_default_reader.dtypes[0]:
                img = (img * (65535.0 / 255.0)).astype(np.uint16)

            band_idx = 3 * TCI + idx + 1
            tif_band_names[band_idx] = band_name
            dst.write(img, band_idx)
            dst_reader.close()

        # write tci
        if TCI:
            for i in range(3):

                band_idx = i + 1
                img = (tci_band.read(band_idx) * (65535.0 / 255.0)).astype(
                    np.uint16)
                tif_band_names[band_idx] = f"tci_{band_idx}"

                dst.write(img, band_idx)

        # add tags and descriptions
        for band_idx, name in tif_band_names.items():
            dst.update_tags(band_idx, band_name=name)
            dst.set_band_description(bidx=band_idx, value=name)

        crs_epsg_code = dst.crs.to_epsg()
        img_bounding_rectangle = box(*dst.bounds)

    outfile.rename(out_file_parent_dir / (safe_root.stem + ".tif"))

    return {
        "crs_epsg_code": crs_epsg_code,
        "img_bounding_rectangle": img_bounding_rectangle
    }
