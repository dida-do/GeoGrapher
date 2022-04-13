"""Mix-in that provides methods to get image bands"""

from pathlib import Path
from typing import Dict, List, Optional
import rasterio as rio


class ImgBandsGetterMixIn:
    """Mix-in that provides methods to get image bands"""

    def _get_bands_for_img(
        self,
        bands: Optional[Dict[str, Optional[List[int]]]],
        source_img_path: Path,
    ) -> List[int]:
        """
        Return list of bands indices in the source image to use in the target image.

        Args:
            source_img_path (Path): path to source image.
            bands (Optional[Dict[str, Optional[List[int]]]]): dict of band indices

        Raises:
            ValueError: If the optional bands dict is not None
            and the image type (source_img_path.parent.name) is not in the bands dict.

        Returns:
            List[int]: list of band indices
        """
        img_type = source_img_path.parent.name
        if bands is None:
            return self._get_all_band_indices(source_img_path)
        elif img_type in bands and bands[img_type] is None:
            return self._get_all_band_indices(source_img_path)
        elif img_type in bands and bands[img_type] is not None:
            return bands[img_type]
        else:
            raise ValueError(f"Missing bands key: {img_type}")

    def _get_all_band_indices(self, source_img_path: Path) -> List[int]:
        """Return list of all band indices of source GeoTiff.

        Args:
            source_img_path (Path): path to source image (or label etc)

        Returns:
            List[int]: list of indices of all bands in GeoTiff
        """

        with rio.open(source_img_path) as src:
            bands = list(range(1, src.count + 1))

        return bands
