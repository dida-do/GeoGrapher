"""Label maker for soft-categorical segmentation labels.

Soft-categorical are probabilistic multi-class labels.
"""

import logging

import numpy as np
import rasterio as rio
from rasterio.features import rasterize

from geographer.connector import Connector
from geographer.label_makers.seg_label_maker_base import SegLabelMaker
from geographer.utils.utils import transform_shapely_geometry

log = logging.getLogger(__name__)


class SegLabelMakerSoftCategorical(SegLabelMaker):
    """Label maker for soft-categorical segmentation labels.

    Soft-categorical are probabilistic multi-class labels.

    Assumes the connector's vectors contains for each segmentation class
    a "prob_seg_class<seg_class>" column containing the probabilities
    for that class.
    """

    add_background_band: bool

    @property
    def label_type(self):
        """Return label_type."""
        return "soft-categorical"

    def _make_label_for_raster(
        self,
        connector: Connector,
        raster_name: str,
    ) -> None:
        """Create (pixel) label for a raster.

        Args:
            connector : calling Connector
            raster_name: name of raster for which a label should be created


        Returns:
            None:
        """
        # paths
        raster_path = connector.rasters_dir / raster_name
        label_path = connector.labels_dir / raster_name

        # If the raster does not exist ...
        if not raster_path.is_file():
            # ... log error to file.
            log.error(
                "_make_geotif_label_soft_categorical: input raster %s does not exist!",
                raster_path,
            )

        # Else, if the label already exists ...
        elif label_path.is_file():
            # ... log error to file.
            log.error(
                "_make_geotif_label_soft_categorical: label %s already exists!",
                label_path,
            )

        # Else, ...
        else:
            label_bands_count = self._get_label_bands_count(connector)

            # ...open the raster, ...
            with rio.open(raster_path) as src:
                # Create profile for the label.
                profile = src.profile
                profile.update({"count": label_bands_count, "dtype": rio.float32})

                # Open the label ...
                with rio.open(label_path, "w+", **profile) as dst:
                    # ... and create one band in the label for each segmentation class.

                    # (if an implicit background band is to be included,
                    # it will go in band/channel 1.)
                    start_band = 1 if not self.add_background_band else 2

                    for count, seg_class in enumerate(
                        connector.task_vector_classes, start=start_band
                    ):
                        # To do that, first find (the df of)
                        # the geoms intersecting the raster ...
                        vectors_intersecting_raster_df = connector.vectors.loc[
                            connector.vectors_intersecting_raster(raster_name)
                        ]

                        # ... extract the geometries ...
                        vector_geoms_in_std_crs = list(
                            vectors_intersecting_raster_df["geometry"]
                        )

                        # ... and convert them to the crs of the source raster.
                        vector_geoms_in_src_crs = list(
                            map(
                                lambda geom: transform_shapely_geometry(
                                    geom,
                                    connector.vectors.crs.to_epsg(),
                                    src.crs.to_epsg(),
                                ),
                                vector_geoms_in_std_crs,
                            )
                        )

                        # Extract the class probabilities ...
                        class_probabilities = list(
                            vectors_intersecting_raster_df[f"prob_of_class_{seg_class}"]
                        )

                        # .. and combine with the geometries
                        # to a list of (geometry, value) pairs.
                        geom_value_pairs = list(
                            zip(vector_geoms_in_src_crs, class_probabilities)
                        )

                        # If there are no geoms of seg_type intersecting the raster ...
                        if len(vector_geoms_in_src_crs) == 0:
                            # ... the label raster is empty.
                            mask = np.zeros((src.height, src.width), dtype=np.uint8)
                        # Else, burn the values for those geoms into the band.
                        else:
                            mask = rasterize(
                                shapes=geom_value_pairs,
                                # or the other way around?
                                out_shape=(src.height, src.width),
                                fill=0.0,  #
                                transform=src.transform,
                                dtype=rio.float32,
                            )

                        # Write the band to the label file.
                        dst.write(mask, count)

                    # If the background is not included in the segmentation classes ...
                    if self.add_background_band:
                        # ... add background band.

                        non_background_band_indices = list(
                            range(
                                start_band,
                                2 + len(connector.task_vector_classes),
                            )
                        )

                        # The probability of a pixel belonging to
                        # the background is the complement of it
                        # belonging to some segmentation class.
                        background_band = 1 - np.add.reduce(
                            [
                                dst.read(band_index)
                                for band_index in non_background_band_indices
                            ]
                        )

                        dst.write(background_band, 1)

    def _get_label_bands_count(self, connector: Connector) -> bool:
        # If the background is not included in the segmentation classes (default) ...
        if self.add_background_band:
            # ... add a band for the implicit background segmentation class, ...
            label_bands_count = 1 + len(connector.task_vector_classes)

        # ... if the background *is* included, ...
        elif not self.add_background_band:
            # ... don't.
            label_bands_count = len(connector.task_vector_classes)

        return label_bands_count

    def _run_safety_checks(self, connector: Connector):
        """Run safety checks.

        Check existence of 'prob_of_class_<class name>' columns in
        connector.vectors.
        """
        # check required columns exist
        required_cols = {
            f"prob_of_class_{class_}" for class_ in connector.all_vector_classes
        }
        if not set(required_cols) <= set(connector.vectors.columns):
            missing_cols = set(required_cols) - set(connector.vectors.columns)
            raise ValueError(
                "connector.vectors.columns is missing required columns: "
                f"{', '.join(missing_cols)}"
            )

        # check no other columns will be mistaken for
        vector_classes_in_vectors = {
            col_name[(1 + len("prob_of_class")) :]  # noqa: E203
            for col_name in connector.vectors.columns
            if col_name.startswith("prob_of_class_")
        }
        if not vector_classes_in_vectors <= set(connector.all_vector_classes):
            log.warning(
                "Ignoring columns: %s. The corresponding classes "
                "are not in connector.all_vector_classes",
                vector_classes_in_vectors - set(connector.all_vector_classes),
            )
