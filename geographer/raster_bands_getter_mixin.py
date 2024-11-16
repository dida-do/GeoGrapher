"""Mix-in that provides methods to get raster bands."""

from __future__ import annotations

from pathlib import Path

import rasterio as rio


class RasterBandsGetterMixIn:
    """Mix-in that provides methods to get raster bands."""

    def _get_bands_for_raster(
        self,
        bands: dict[str, list[int] | None] | None,
        source_raster_path: Path,
    ) -> list[int]:
        """Return bands indices to  be used in the target raster.

        Args:
            source_raster_path: path to source raster.
            bands: dict of band indices

        Raises:
            ValueError: If the optional bands dict is not None
            and the raster type (source_raster_path.parent.name)
            is not in the bands dict.

        Returns:
            band indices
        """
        raster_type = source_raster_path.parent.name
        if bands is None:
            return self._get_all_band_indices(source_raster_path)
        elif raster_type in bands and bands[raster_type] is None:
            return self._get_all_band_indices(source_raster_path)
        elif raster_type in bands and bands[raster_type] is not None:
            return bands[raster_type]  # type: ignore
        else:
            raise ValueError(f"Missing bands key: {raster_type}")

    def _get_all_band_indices(self, source_raster_path: Path) -> list[int]:
        """Return list of all band indices of source GeoTiff.

        Args:
            source_raster_path: path to source raster (or label etc)

        Returns:
            indices of all bands in GeoTiff
        """
        with rio.open(source_raster_path) as src:
            bands = list(range(1, src.count + 1))

        return bands
