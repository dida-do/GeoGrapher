"""RasterDownloadProcessor for Sentinel-2 data from Copernicus Sci-hub.

Should be easily extendable to Sentinel-1.
"""

import logging
import shutil
from pathlib import Path

from geographer.downloaders.base_download_processor import RasterDownloadProcessor
from geographer.downloaders.sentinel2_safe_unpacking import (
    NO_DATA_VAL,
    safe_to_geotif_L2A,
)
from geographer.utils.utils import transform_shapely_geometry

log = logging.getLogger(__name__)


# TODO Test with the 'creodias', 'onda', and 'sara' providers
# TODO (archive_depth 2).
# TODO Use provider's archive_depth to extend to archive_depth not
# TODO equal to 2 i.e. 'planetary_computer' (archive_depth 1).
class Sentinel2SAFEProcessor(RasterDownloadProcessor):
    """Processes downloads of L2A Sentinel-2 SAFE files."""

    def process(
        self,
        raster_name: str,
        download_dir: Path,
        rasters_dir: Path,
        return_bounds_in_crs_epsg_code: int,
        *,  # processor_params of RasterDownloaderForVectors.download start below
        resolution: int,
        delete_safe: bool,  # TODO better name, uniformly usable for all processors?
        file_suffix: str = ".SAFE",
        nodata_val: int = NO_DATA_VAL,
    ) -> dict:
        """Process Sentinel-2 download.

        Extract downloaded sentinel-2 zip file to a .SAFE directory, then
        process/convert to a GeoTiff raster, delete the zip file, put the
        GeoTiff raster in the right directory, and return information about the
        raster in a dict.

        Warning:
            Tested with the `cop_dataspace` eodag provider. It should also work with
            'creodias', 'onda', and 'sara', which have an `archive_depth` of 2.
            For providers with a different `archive_depth`, the processor may need
            adjustments to locate the SAFE file correctly based on the raster name.

        Args:
            raster_name:
                The name of the raster.
            download_dir:
                The dir containing the SAFE file to be processed.
            rasters_dir:
                The dir in which the .tif output file should be placed.
            return_bounds_in_crs_epsg_code:
                The EPSG of the CRS in which the bounds of the raster
                should be returned.
            resolution:
                The desired resolution of the output tif file.
            delete_safe:
                Whether to delete the SAFE file after extracting the tif file.
            file_suffix:
                Possible suffix by which the stem of the raster_name and the
                downloaded SAFE file to be processed differ. If used together
                with the `EodagDownloaderForSingleVector` for the 'cop_dataspace'
                provider and the `RasterDownloaderForVectors` and the
                `downloader_params` parameter dict of the
                `RasterDownloaderForVectors.download` method contains
                a `"suffix_to_remove: ".SAFE"` pair then the default value of
                ".SAFE" for the file_suffix will result in nicer tif names,
                e.g. S2B_MSIL2A_20231208T013039_N0509_R074_T54SUE_20231208T031743.tif
                instead of S2B_MSIL2A_20231208T013039_N0509_R074_T54SUE_20231208T031743.SAFE.tif.  # noqa
            nodata_val:
                The nodata value to fill. Defaults to 0.

        Returns:
            return_dict: Contains information about the downloaded product.
        """
        log.info("Processing %s to a .tif file. This might take a while..")

        safe_path = download_dir / raster_name.removesuffix(".tif")
        safe_path_with_suffix = safe_path.with_suffix(file_suffix)

        if safe_path.exists() and (not safe_path_with_suffix.exists()):
            pass  # Use safe_path
        elif safe_path_with_suffix.exists() and (not safe_path.exists()):
            safe_path = safe_path_with_suffix
        elif safe_path.exists() and safe_path_with_suffix.exists():
            msg = (
                "Both %s and %s exist in %s.\n"
                "Unable to resolve ambiguity in which file/dir to process."
            )
            log.error(msg, safe_path.name, safe_path_with_suffix.name, safe_path.parent)
            raise RuntimeError(
                msg % (safe_path.name, safe_path_with_suffix.name, safe_path.parent)
            )
        elif (not safe_path.exists()) and (not safe_path_with_suffix.exists()):
            msg = "Can't find SAFE file in expected location(s): %s"
            log.error(msg, safe_path)
            raise RuntimeError(msg % safe_path)

        conversion_dict = safe_to_geotif_L2A(
            safe_root=safe_path,
            resolution=resolution,
            outdir=rasters_dir,
            nodata_val=nodata_val,
        )

        if delete_safe:
            log.info("Deleting SAFE file: %s", safe_path)
            shutil.rmtree(safe_path, ignore_errors=True)

        orig_crs_epsg_code = int(conversion_dict["crs_epsg_code"])
        raster_bounding_rectangle_orig_crs = conversion_dict[
            "raster_bounding_rectangle"
        ]
        raster_bounding_rectangle = (
            transform_shapely_geometry(  # convert to standard crs
                raster_bounding_rectangle_orig_crs,
                from_epsg=orig_crs_epsg_code,
                to_epsg=return_bounds_in_crs_epsg_code,
            )
        )
        return {
            "raster_name": raster_name,
            "geometry": raster_bounding_rectangle,
            "orig_crs_epsg_code": orig_crs_epsg_code,
            "raster_processed?": True,
        }
