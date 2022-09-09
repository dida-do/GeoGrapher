"""Mix-in that provides methods to get image bands."""

from pathlib import Path
from typing import Optional

import rasterio as rio


class ImgBandsGetterMixIn:
    """Mix-in that provides methods to get image bands."""

    def _get_bands_for_img(
        self,
        bands: Optional[dict[str, Optional[list[int]]]],
        source_img_path: Path,
    ) -> list[int]:
        """Return bands indices to  be used in the target image.

        Args:
            source_img_path: path to source image.
            bands: dict of band indices

        Raises:
            ValueError: If the optional bands dict is not None
            and the image type (source_img_path.parent.name) is not in the bands dict.

        Returns:
            band indices
        """
        img_type = source_img_path.parent.name
        if bands is None:
            return self._get_all_band_indices(source_img_path)
        elif img_type in bands and bands[img_type] is None:
            return self._get_all_band_indices(source_img_path)
        elif img_type in bands and bands[img_type] is not None:
            return bands[img_type]  # type: ignore
        else:
            raise ValueError(f"Missing bands key: {img_type}")

    def _get_all_band_indices(self, source_img_path: Path) -> list[int]:
        """Return list of all band indices of source GeoTiff.

        Args:
            source_img_path: path to source image (or label etc)

        Returns:
            indices of all bands in GeoTiff
        """
        with rio.open(source_img_path) as src:
            bands = list(range(1, src.count + 1))

        return bands
