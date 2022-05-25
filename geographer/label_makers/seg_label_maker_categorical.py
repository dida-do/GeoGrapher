"""Label maker for categorical segmentation labels"""

import logging
import numpy as np
import rasterio as rio

from geopandas import GeoDataFrame
from geographer.label_makers.seg_label_maker_base import SegLabelMaker
from geographer.utils.utils import transform_shapely_geometry
from geographer.connector import Connector

log = logging.getLogger(__name__)


class SegLabelMakerCategorical(SegLabelMaker):
    """
    Label maker that generates categorical segmentation labels
    from a connector's vector_features.
    """

    @property
    def label_type(self) -> str:
        """Return label type"""
        return 'categorical'

    def _make_label_for_img(self, connector: Connector, img_name: str):
        """Create a categorical GeoTiff (pixel) label for an image.

        Args:
            - connector (Connector):
            - img_name (str): Name of image for which a label should be created.
        Returns:
            - None:
        """

        img_path = connector.images_dir / img_name
        label_path = connector.labels_dir / img_name

        classes_to_ignore = {
            class_
            for class_ in [connector.background_class] if class_ is not None
        }
        segmentation_classes = [
            class_ for class_ in connector.task_feature_vector_classes
            if class_ not in classes_to_ignore
        ]

        # If the image does not exist ...
        if not img_path.is_file():

            # ... log error to file.
            log.error(
                "SegLabelMakerCategorical: input image %s does not exist!",
                img_path)

        # Else, if the label already exists ...
        elif label_path.is_file():

            # ... log error to file.
            log.error("SegLabelMakerCategorical: label %s already exists!",
                      label_path)

        # Else, ...
        else:

            # ...open the image, ...
            with rio.open(img_path) as src:

                profile = src.profile
                profile.update({"count": 1, "dtype": rio.uint8})

                # ... open the label ...
                with rio.open(
                        label_path,
                        'w',
                        # for writing single bit image,
                        # see https://gis.stackexchange.com/questions/338410/rasterio-invalid-dtype-bool
                        # nbits=1,
                        **profile) as dst:

                    # ... create an empty band of zeros (background class) ...
                    label = np.zeros((src.height, src.width), dtype=np.uint8)

                    # and build up the shapes to be burnt in
                    shapes = []  # pairs of geometries and values to burn in

                    for count, seg_class in enumerate(segmentation_classes,
                                                      start=1):

                        # To do that, first find (the df of) the geometries intersecting the image ...
                        features_intersecting_img: GeoDataFrame = connector.vector_features.loc[
                            connector.vector_features_intersecting_img(img_name)]

                        # ... then restrict to (the subdf of) geometries
                        # with the given class.
                        features_intersecting_img_of_type: GeoDataFrame = features_intersecting_img.loc[
                            features_intersecting_img['type'] == seg_class]

                        # Extract those geometries ...
                        feature_geoms_in_std_crs = list(
                            features_intersecting_img_of_type['geometry'])

                        # ... and convert them to the crs of the source image.
                        feature_geoms_in_src_crs = list(
                            map(
                                lambda geom: transform_shapely_geometry(
                                    geom,
                                    connector.vector_features.crs.to_epsg(),
                                    src.crs.to_epsg()),
                                feature_geoms_in_std_crs))

                        shapes_for_seg_class = [
                            (feature_geom, count)
                            for feature_geom in feature_geoms_in_src_crs
                        ]

                        shapes += shapes_for_seg_class

                    # Burn the geomes into the label.
                    if len(shapes) != 0:
                        rio.features.rasterize(
                            shapes=shapes,
                            out_shape=(src.height,
                                       src.width),  # or the other way around?
                            fill=0,
                            merge_alg=rio.enums.MergeAlg.replace,
                            out=label,
                            transform=src.transform,
                            dtype=rio.uint8)

                    # Write label to file.
                    dst.write(label, 1)

    def _run_safety_checks(self, connector: Connector):
        """Check existence of 'type' column in connector.vector_features and make sure entries are allowed."""

        if "type" not in connector.vector_features.columns:
            raise ValueError(
                "connector.vector_features needs a 'type' column containing the ML task (e.g. segmentation or object detection) class of the geometries"
            )

        feature_classes_in_vector_features = set(
            connector.vector_features["type"].unique())
        if not feature_classes_in_vector_features <= set(
                connector.all_feature_classes):
            raise ValueError(
                f"Unrecognized classes in connector.vector_features: {feature_classes_in_vector_features - set(self.all_feature_classes)}"
            )
