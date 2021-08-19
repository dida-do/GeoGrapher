"""Label maker for soft-categorical (i.e. probabilistic multi-class) labels."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import rasterio as rio

from rs_tools.utils.utils import transform_shapely_geometry
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator

log = logging.getLogger(__name__)


def _make_geotif_label_soft_categorical(assoc: ImgPolygonAssociator,
                                        img_name: str) -> None:
    """Create a soft categorical (i.e. probabilistic) GeoTiff pixel label for
    an image.

    Create a soft-categorical encoded (i.e. one channel per segmentation class,
    the at a given position in a given channel is the probability that the pixel
    at that position is classified as belonging to the corresponding segmentation
    class) GeoTiff raster label in the data directory's labels subdirectory for a
    GeoTiff image with image name img_name. Assumes the associator's polygons_df
    contains one column prob_seg_class_[segmentation_class] for each segmentation
    class containing the probability of that polygon belonging to the segmentation
    class.

    Args:
        - assoc (ImgPolygonAssociator): calling ImgPolygonAssociator.
        - img_name (str): Name of image for which a label should be created.
    Returns:
        - None:
    """

    img_path = Path(assoc.data_dir) / Path("images") / Path(img_name)

    label_path = Path(assoc.data_dir) / Path("labels") / Path(img_name)

    segmentation_classes = assoc._params_dict['segmentation_classes']

    # If the image does not exist ...
    if not img_path.is_file():

        # ... log error to file.
        log.error(
            f"_make_geotif_label_soft_categorical: input image {img_path} does not exist!"
        )

    # Else, if the label already exists ...
    elif label_path.is_file():

        # ... log error to file.
        log.error(
            f"_make_geotif_label_soft_categorical: label {label_path} already exists!"
        )

    # Else, ...
    else:
        # ...open the image, ...
        with rio.open(img_path) as src:

            # ... determine how many bands the label should have.
            # If the background is not included in the segmentation classes (default) ...
            if assoc._params_dict['add_background_band_in_labels'] is True:

                # ... add a band for the implicit background segmentation class, ...
                label_bands_count = 1 + len(segmentation_classes)

            # ... if the background *is* included, ...
            elif assoc._params_dict['background_in_seg_classes'] is False:

                # ... don't.
                label_bands_count = len(segmentation_classes)

            # If the add_background_band_in_labels value is neither True nor False ...
            else:

                # ... then it is nonsensical, so log an error.
                log.error(
                    f"Unknown background_in_seg_classes "
                    f"value: {assoc._params_dict['background_in_seg_classes']}."
                )

            # Create profile for the label.
            profile = src.profile
            profile.update({"count": label_bands_count, "dtype": rio.float32})

            # Open the label ...
            with rio.open(label_path, 'w+', **profile) as dst:

                # ... and create one band in the label for each segmentation class.
                for count, seg_class in enumerate(segmentation_classes,
                                                  start=1):

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
                    class_probabilities = list(polygons_intersecting_img_df[
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
                if assoc._params_dict['add_background_band_in_labels'] is True:

                    # ... add background band.

                    non_background_band_indices = list(
                        range(1, 1 + len(segmentation_classes)))

                    # The probability of a pixel belonging to the
                    # background is the complement of it belonging
                    # to some segmentation class.
                    background_band = 1 - np.add.reduce([
                        dst.read(band_index)
                        for band_index in non_background_band_indices
                    ])

                    dst.write(background_band, 1 + len(segmentation_classes))
