"""Unpack/convert sentinel-2 SAFE files to GeoTiffs."""

from __future__ import annotations

import itertools
import os
from collections import OrderedDict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.mask
from rasterio.errors import RasterioIOError
from scipy.ndimage import zoom
from shapely.geometry import box
from tqdm.auto import tqdm

from geographer.utils.utils import create_logger

NO_DATA_VAL = 0  # No data value for sentinel 2 L1C

log = create_logger(__name__)


def safe_to_geotif_L2A(
    safe_root: Path,
    resolution: str | int,
    upsample_lower_resolution: bool = True,
    outdir: Path = None,
    TCI: bool = True,
    requested_jp2_masks: list[str] = ["CLDPRB", "SNWPRB"],
    requested_gml_mask: list[tuple[str, str]] = [("CLOUDS", "B00")],
    nodata_val: int = NO_DATA_VAL,
) -> dict:
    """Convert a L2A-level Sentinel-2 .SAFE file to a GeoTIFF.

    The GeoTIFF contains raster bands derived from the .SAFE file, including:
    - True color composite (TCI) bands if requested.
    - JP2 masks (e.g., cloud or snow masks) at the desired resolution.
    - Additional GML masks if available.

    Warning:
        Sentinel-2 L2A products dated later than October 2021
        no longer include GML masks.

    Note:

        - The GeoTIFF bands are ordered as follows:

            1. **True Color Composite (TCI)** (optional):
                Red, Green, Blue (if ``TCI=True``).
            2. **Spectral Bands**: JP2 data bands at the target resolution,
                optionally including upsampled lower-resolution bands
                if ``upsample_lower_resolution=True``.
            3. **JP2 Masks**: Added in the order specified by ``requested_jp2_masks``
                (e.g., ``"CLDPRB"``, ``"SNWPRB"``). Masks are limited to a maximum
                resolution of 20m.
            4. **GML Masks**: Rasterized from ``requested_gml_mask``, with
                empty bands added for missing masks.

        - jp2_masks are only available up to a resolution of 20 m, so for 10m the 20m
            mask ist taken
        - ``"SNWPRB"`` for snow masks


    Args:
        safe_root:
            Path to the root directory of the .SAFE file.
        resolution:
            Desired resolution for the GeoTIFF (10, 20, or 60 meters).
        upsample_lower_resolution:
            If True, includes lower-resolution bands
            and upsamples them to match the target resolution. Defaults to True.
        outdir:
            Directory where the GeoTIFF will be saved. If None, saves the
            file in the parent directory of `safe_root`. Defaults to None.
        TCI:
            Whether to include true color raster bands (TCI).
            Defaults to True.
        requested_jp2_masks:
            List of JP2 masks to include in the output.
            Defaults to ["CLDPRB", "SNWPRB"].
        requested_gml_mask: List of GML masks to include. Each tuple contains
            the mask name (e.g., "CLOUDS") and the associated band (e.g., "B00").
            Defaults to [("CLOUDS", "B00")].
        nodata_val:
            Value to use for no-data areas in the GeoTIFF. Defaults to 0.

    Returns:
        dict: A dictionary containing:
            - `crs_epsg_code` (int):
                The EPSG code of the CRS.
            - `raster_bounding_rectangle` (shapely.geometry.Polygon):
                The bounding rectangle of the output GeoTIFF.

    Raises:
        AssertionError:
            If `resolution` is not one of the supported values (10, 20, 60).
        RasterioIOError:
            If there are issues reading or processing the JP2/GML files.
    """
    # assert resolution is within available
    assert resolution in [10, 20, 60, "10", "20", "60"]

    # define output file
    raster_name = safe_root.stem
    out_file_parent_dir = outdir if (outdir and outdir.is_dir()) else safe_root.parent
    outfile = out_file_parent_dir / (raster_name + "_TEMP.tif")

    granule_dir = safe_root / "GRANULE"
    masks_dir = granule_dir / "{}/QI_DATA/".format(os.listdir(granule_dir)[0])

    # JP2 masks
    jp2_resolution = 20 if resolution in [10, "10"] else resolution
    jp2_mask_paths = list(
        filter(
            lambda file: any(
                mask_name in file.name for mask_name in requested_jp2_masks
            )
            and f"{jp2_resolution}m" in file.name,
            masks_dir.glob("*.jp2"),
        )
    )

    # Paths for S2 Bands
    jp2_path_desired_resolution = granule_dir / "{}/IMG_DATA/R{}m/".format(
        os.listdir(granule_dir)[0], resolution
    )

    tci_path = next(
        filter(
            lambda path: path.stem.split("_")[-2] == "TCI",
            jp2_path_desired_resolution.glob("*.jp2"),
        )
    )

    # add missing higher res jps paths
    raster_data_bands = list(
        filter(
            lambda path: path.stem.split("_")[-2] not in ["TCI"],
            jp2_path_desired_resolution.glob("*.jp2"),
        )
    )
    out_default_reader = rio.open(raster_data_bands[0], driver="JP2OpenJPEG")

    # include lower resolution bands
    if upsample_lower_resolution:
        for higher_res in filter(lambda res: res > int(resolution), [10, 20, 60]):
            jp2_higher_res_path = granule_dir / "{}/IMG_DATA/R{}m/".format(
                os.listdir(granule_dir)[0], higher_res
            )

            raster_data_bands += list(
                filter(
                    lambda path: path.stem.split("_")[-2]
                    not in itertools.chain(
                        map(lambda path: path.stem.split("_")[-2], raster_data_bands),
                        ["TCI"],
                    ),
                    jp2_higher_res_path.glob("*.jp2"),
                )
            )

    # # if we have both B08 and B8A remove B8A
    # if {'B8A', 'B08'} <= {name.name.split("_")[-2] for name in raster_data_bands}:
    #     raster_data_bands = [path for path in raster_data_bands \
    #     if path.name.split("_")[-2] != 'B8A']

    # set up rasterio loaders
    bands_dict = OrderedDict()
    max_width = 0
    max_height = 0
    for file in raster_data_bands + jp2_mask_paths:
        band = rio.open(file, driver="JP2OpenJPEG")
        max_width = max(max_width, band.width)
        max_height = max(max_height, band.height)
        bands_dict[file.stem.split("_")[-2]] = (band, int(file.stem.split("_")[-1][:2]))

    # sort bands_dict
    bands_dict = OrderedDict(sorted(bands_dict.items()))

    # paths for gml masks
    gml_mask_paths = list(
        filter(
            lambda path: tuple(path.stem.split("_")[-2:]) in requested_gml_mask,
            masks_dir.glob("*.gml"),
        )
    )
    gml_mask_paths_dict = {
        tuple(path.stem.split("_")[-2:]): path for path in gml_mask_paths
    }
    # add invalid paths for missing gml masks (will result in zero bands later)
    for pair in requested_gml_mask:
        if pair not in gml_mask_paths_dict:
            gml_mask_paths_dict[pair] = Path("/path/to/nowhere")
            gml_mask_paths.append(Path("/path/to/nowhere"))

    # reader for tci
    tci_band = rio.open(tci_path, driver="JP2OpenJPEG")

    # number of bands in final geotif
    count = len(raster_data_bands + jp2_mask_paths + gml_mask_paths) + 3 * TCI

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
        dst.nodata = nodata_val

        with tqdm(total=count, desc=f"Extracting tif from {raster_name}.SAFE.") as pbar:

            # write gml masks
            for idx, (gml_name, gml_path) in enumerate(gml_mask_paths_dict.items()):
                try:
                    if not gml_path.is_file():
                        raise FileNotFoundError(
                            f"Can't find GML mask {gml_name} in expected location "
                            f"{gml_path.relative_to(safe_root)}"
                        )
                    shapes = gpd.read_file(gml_path)["geometry"].values
                    mask, _, _ = rasterio.mask.raster_geometry_mask(
                        out_default_reader, shapes, crop=False, invert=True
                    )
                # in case mask is empty or does not exist:
                except (ValueError, AssertionError, RasterioIOError, FileNotFoundError):
                    log.info(
                        "Using all zero band for gml mask %s for %s",
                        gml_name,
                        safe_root.name,
                    )
                    mask = np.full(
                        shape=out_default_reader.read(1).shape,
                        fill_value=0.0,
                        dtype=np.uint16,
                    )

                band_idx = len(bands_dict) + 3 * TCI + idx + 1
                tif_band_names[band_idx] = "_".join(gml_name)

                dst.write(mask.astype(np.uint16), band_idx)
                pbar.update(1)

            # write jp2 bands
            for idx, (band_name, (dst_reader, res)) in enumerate(bands_dict.items()):
                if res != int(resolution):
                    assert res % int(resolution) == 0
                    factor = res // int(resolution)

                    raster = dst_reader.read(1)
                    raster = zoom(raster, factor, order=3)

                    assert raster.shape == (10980, 10980)

                else:
                    raster = dst_reader.read(1)

                if not dst_reader.dtypes[0] == out_default_reader.dtypes[0]:
                    raster = (raster * (65535.0 / 255.0)).astype(np.uint16)

                band_idx = 3 * TCI + idx + 1
                tif_band_names[band_idx] = band_name
                dst.write(raster, band_idx)
                dst_reader.close()
                pbar.update(1)

            # write tci
            if TCI:
                for i in range(3):
                    band_idx = i + 1
                    raster = (tci_band.read(band_idx) * (65535.0 / 255.0)).astype(
                        np.uint16
                    )
                    tif_band_names[band_idx] = f"tci_{band_idx}"

                    dst.write(raster, band_idx)
                    pbar.update(1)

        # add tags and descriptions
        for band_idx, name in tif_band_names.items():
            dst.update_tags(band_idx, band_name=name)
            dst.set_band_description(bidx=band_idx, value=name)

        crs_epsg_code = dst.crs.to_epsg()
        raster_bounding_rectangle = box(*dst.bounds)

    outfile.rename(out_file_parent_dir / (raster_name + ".tif"))

    return {
        "crs_epsg_code": crs_epsg_code,
        "raster_bounding_rectangle": raster_bounding_rectangle,
    }
