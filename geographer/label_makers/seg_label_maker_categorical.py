"""Label maker for categorical segmentation labels."""

import logging

import numpy as np
import rasterio as rio
from geopandas import GeoDataFrame
from rasterio.features import rasterize

from geographer.connector import Connector
from geographer.label_makers.seg_label_maker_base import SegLabelMaker
from geographer.utils.utils import transform_shapely_geometry

log = logging.getLogger(__name__)


class SegLabelMakerCategorical(SegLabelMaker):
    """Label maker for categorical segmentation labels."""

    @property
    def label_type(self) -> str:
        """Return label type."""
        return "categorical"

    def _make_label_for_raster(self, connector: Connector, raster_name: str):
        """Create a categorical GeoTiff (pixel) label for a raster.

        Args:
            connector:
            raster_name: Name of raster for which a label should be created.

        Returns:
            None:
        """
        raster_path = connector.rasters_dir / raster_name
        label_path = connector.labels_dir / raster_name

        classes_to_ignore = {
            class_ for class_ in [connector.background_class] if class_ is not None
        }
        segmentation_classes = [
            class_
            for class_ in connector.task_vector_classes
            if class_ not in classes_to_ignore
        ]

        # If the raster does not exist ...
        if not raster_path.is_file():
            # ... log error to file.
            log.error(
                "SegLabelMakerCategorical: input raster %s does not exist!", raster_path
            )

        # Else, if the label already exists ...
        elif label_path.is_file():
            # ... log error to file.
            log.error("SegLabelMakerCategorical: label %s already exists!", label_path)

        # Else, ...
        else:
            # ...open the raster, ...
            with rio.open(raster_path) as src:
                profile = src.profile
                profile.update({"count": 1, "dtype": rio.uint8})

                # ... open the label ...
                with rio.open(
                    label_path,
                    "w",
                    # for writing single bit raster, see
                    # https://gis.stackexchange.com/questions/338410/rasterio-invalid-dtype-bool
                    # nbits=1,
                    **profile,
                ) as dst:
                    # ... create an empty band of zeros (background class) ...
                    label = np.zeros((src.height, src.width), dtype=np.uint8)

                    # and build up the shapes to be burnt in
                    shapes = []  # pairs of geometries and values to burn in

                    for count, seg_class in enumerate(segmentation_classes, start=1):
                        # To do that, first find (the df of) the geometries
                        # intersecting the raster ...
                        vectors_intersecting_raster: GeoDataFrame = (
                            connector.vectors.loc[
                                connector.vectors_intersecting_raster(raster_name)
                            ]
                        )

                        # ... then restrict to (the subdf of) geometries
                        # with the given class.
                        vectors_intersecting_raster_of_type: GeoDataFrame = (
                            vectors_intersecting_raster.loc[
                                vectors_intersecting_raster["type"] == seg_class
                            ]
                        )

                        # Extract those geometries ...
                        vector_geoms_in_std_crs = list(
                            vectors_intersecting_raster_of_type["geometry"]
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

                        shapes_for_seg_class = [
                            (vector_geom, count)
                            for vector_geom in vector_geoms_in_src_crs
                        ]

                        shapes += shapes_for_seg_class

                    # Burn the geomes into the label.
                    if len(shapes) != 0:
                        rasterize(
                            shapes=shapes,
                            out_shape=(
                                src.height,
                                src.width,
                            ),  # or the other way around?
                            fill=0,
                            merge_alg=rio.enums.MergeAlg.replace,
                            out=label,
                            transform=src.transform,
                            dtype=rio.uint8,
                        )

                    # Write label to file.
                    dst.write(label, 1)

    def _run_safety_checks(self, connector: Connector):
        """Run safety checks.

        Check existence of 'type' column in connector.vectors and make
        sure entries are allowed.
        """
        if "type" not in connector.vectors.columns:
            raise ValueError(
                "connector.vectors needs a 'type' column containing "
                "the ML task (e.g. segmentation or object detection) "
                "class of the geometries."
            )

        vector_classes_in_vectors = set(connector.vectors["type"].unique())
        if not vector_classes_in_vectors <= set(connector.all_vector_classes):
            unrecognized_classes = vector_classes_in_vectors - set(
                connector.all_vector_classes
            )
            raise ValueError(
                f"Unrecognized classes in connector.vectors: "
                f" {unrecognized_classes}."
            )
