"""
Label maker for soft-categorical (i.e. probabilistic multi-class) segmentation labels.
"""

import logging

import numpy as np
import rasterio as rio

from rs_tools.label_makers.seg_label_maker_base import SegLabelMaker
from rs_tools.utils.utils import transform_shapely_geometry
from rs_tools.img_polygon_associator import ImgPolygonAssociator

log = logging.getLogger(__name__)


class SegLabelMakerSoftCategorical(SegLabelMaker):
    """
    Label maker for soft-categorical (i.e. probabilistic multi-class) segmentation labels.

    Assumes the associator's polygons_df contains for each segmentation class
    a "prob_seg_class<seg_class>" column containing the probabilities for that class.
    """

    add_background_band: bool

    @property
    def label_type(self):
        return 'soft-categorical'

    def _make_label_for_img(
        self,
        assoc: ImgPolygonAssociator,
        img_name: str,
    ) -> None:
        """Create a soft-categorical or onehot GeoTiff (pixel) label for an image.

        Args:
            assoc (ImgPolygonAssociator): calling ImgPolygonAssociator
            img_name (str): name of image for which a label should be created


        Returns:
            None:
        """

        # paths
        img_path = assoc.images_dir / img_name
        label_path = assoc.labels_dir / img_name

        # If the image does not exist ...
        if not img_path.is_file():

            # ... log error to file.
            log.error(
                "_make_geotif_label_soft_categorical: input image %s does not exist!",
                img_path)

        # Else, if the label already exists ...
        elif label_path.is_file():

            # ... log error to file.
            log.error(
                "_make_geotif_label_soft_categorical: label %s already exists!",
                label_path)

        # Else, ...
        else:

            label_bands_count = self._get_label_bands_count(assoc)

            # ...open the image, ...
            with rio.open(img_path) as src:

                # Create profile for the label.
                profile = src.profile
                profile.update({
                    "count": label_bands_count,
                    "dtype": rio.float32
                })

                # Open the label ...
                with rio.open(label_path, 'w+', **profile) as dst:

                    # ... and create one band in the label for each segmentation class.

                    # (if an implicit background band is to be included, it will go in band/channel 1.)
                    start_band = 1 if not self.add_background_band else 2

                    for count, seg_class in enumerate(
                            assoc.segmentation_classes, start=start_band):

                        # To do that, first find (the df of) the polygons intersecting the image ...
                        polygons_intersecting_img_df = assoc.polygons_df.loc[
                            assoc.polygons_intersecting_img(img_name)]

                        # Extract the polygon geometries of these polygons ...
                        polygon_geometries_in_std_crs = list(
                            polygons_intersecting_img_df['geometry'])

                        # ... and convert them to the crs of the source image.
                        polygon_geometries_in_src_crs = list(
                            map(
                                lambda geom: transform_shapely_geometry(
                                    geom, assoc.polygons_df.crs.to_epsg(),
                                    src.crs.to_epsg()),
                                polygon_geometries_in_std_crs))

                        # Extract the class probabilities ...
                        class_probabilities = list(
                            polygons_intersecting_img_df[
                                f"prob_seg_class_{seg_class}"])

                        # .. and combine with the polygon geometries
                        # to a list of (polygon, value) pairs.
                        polygon_value_pairs = list(
                            zip(polygon_geometries_in_src_crs,
                                class_probabilities))

                        # If there are no polygons of seg_type intersecting the image ...
                        if len(polygon_geometries_in_src_crs) == 0:
                            # ... the label raster is empty.
                            mask = np.zeros((src.height, src.width),
                                            dtype=np.uint8)
                        # Else, burn the values for those polygons into the band.
                        else:
                            mask = rio.features.rasterize(
                                shapes=polygon_value_pairs,
                                # or the other way around?
                                out_shape=(src.height, src.width),
                                fill=0.0,  #
                                transform=src.transform,
                                dtype=rio.float32)

                        # Write the band to the label file.
                        dst.write(mask, count)

                    # If the background is not included in the segmentation classes ...
                    if self.add_background_band:

                        # ... add background band.

                        non_background_band_indices = list(
                            range(start_band,
                                  2 + len(assoc.segmentation_classes)))

                        # The probability of a pixel belonging to
                        # the background is the complement of it
                        # belonging to some segmentation class.
                        background_band = 1 - np.add.reduce([
                            dst.read(band_index)
                            for band_index in non_background_band_indices
                        ])

                        dst.write(background_band, 1)

    def _get_label_bands_count(self, assoc: ImgPolygonAssociator) -> bool:

        # If the background is not included in the segmentation classes (default) ...
        if self.add_background_band:

            # ... add a band for the implicit background segmentation class, ...
            label_bands_count = 1 + len(assoc.segmentation_classes)

        # ... if the background *is* included, ...
        elif not self.add_background_band:

            # ... don't.
            label_bands_count = len(assoc.segmentation_classes)

        return label_bands_count

    def _run_safety_checks(self, assoc: ImgPolygonAssociator):
        """Check existence of 'prob_seg_class_<class name>' columns in assoc.polygons_df."""

        # check required columns exist
        required_cols = {
            f"prob_seg_class_{class_}"
            for class_ in assoc.all_polygon_classes
        }
        if not set(required_cols) <= set(assoc.polygons_df.columns):
            missing_cols = set(required_cols) - set(assoc.polygons_df.columns)
            raise ValueError(
                f"assoc.polygons_df.columns is missing required columns: {', '.join(missing_cols)}"
            )

        # check no other columns will be mistaken for
        polygon_classes_in_polygons_df = {
            col_name[15:]
            for col_name in assoc.polygons_df.columns
            if col_name.startswith("prob_seg_class_")
        }
        if not polygon_classes_in_polygons_df <= set(
                assoc.all_polygon_classes):
            log.warning(
                "Ignoring columns: %s. The corresponding classes are not in assoc.all_segmentation_classes",
                polygon_classes_in_polygons_df -
                set(assoc.all_polygon_classes))
