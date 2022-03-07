"""Label maker for categorical (pixel) labels."""
from __future__ import annotations
from logging import Logger
from pathlib import Path
import numpy as np
import rasterio as rio

from rs_tools.utils.utils import transform_shapely_geometry

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rs_tools.img_polygon_associator import ImgPolygonAssociator


def _make_geotif_label_categorical(
        assoc : ImgPolygonAssociator,
        img_name : str,
        logger : Logger
        ) -> None:
    """
    Create a categorical GeoTiff (pixel) label for an image.

    Args:
        - assoc (ImgPolygonAssociator): calling ImgPolygonAssociator.
        - img_name (str): Name of image for which a label should be created.
        - logger (Logger): logger of calling associator
    Returns:
        - None:
    """

    img_path = assoc.images_dir / img_name
    label_path = assoc.labels_dir / img_name

    classes_to_ignore = {class_ for class_ in {assoc.background_class} if class_ is not None}
    segmentation_classes = [class_ for class_ in assoc.segmentation_classes if class_ not in classes_to_ignore]

    # If the image does not exist ...
    if not img_path.is_file():

        # ... log error to file.
        logger.error(
            f"_make_geotif_label_categorical: input image {img_path} does not exist!"
        )

    # Else, if the label already exists ...
    elif label_path.is_file():

        # ... log error to file.
        logger.error(
            f"_make_geotif_label_categorical: label {label_path} already exists!"
        )

    # Else, ...
    else:

        # ...open the image, ...
        with rio.open(img_path) as src:

            profile = src.profile
            profile.update(
                {
                    "count": 1,
                    "dtype": rio.uint8
                }
            )

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
                shapes = [] # pairs of polygons and values to burn in

                for count, seg_class in enumerate(segmentation_classes, start=1):

                    # To do that, first find (the df of) the polygons intersecting the image ...
                    polygons_intersecting_img_df = assoc.polygons_df.loc[
                        assoc.polygons_intersecting_img(img_name)]

                    # ... then restrict to (the subdf of) polygons
                    # with the given segmentation class.
                    polygons_intersecting_img_df_of_type = polygons_intersecting_img_df.loc[
                        polygons_intersecting_img_df['type'] == seg_class]

                    # Extract the polygon geometries of these polygons ...
                    polygon_geometries_in_std_crs = list(
                        polygons_intersecting_img_df_of_type['geometry'])

                    # ... and convert them to the crs of the source image.
                    polygon_geometries_in_src_crs = list(
                        map(
                            lambda geom: transform_shapely_geometry(
                                geom, assoc.polygons_df.crs.to_epsg(),
                                src.crs.to_epsg()),
                            polygon_geometries_in_std_crs))

                    shapes_for_seg_class = [(polygon_geom, count) for polygon_geom in polygon_geometries_in_src_crs]

                    shapes += shapes_for_seg_class

                # Burn the polygon geometries into the label.
                if len(shapes) != 0:
                    rio.features.rasterize(
                        shapes=shapes,
                        out_shape=
                            (src.height,
                            src.width),  # or the other way around?
                        fill=0,
                        merge_alg=rio.enums.MergeAlg.replace,
                        out=label,
                        transform=src.transform,
                        dtype=rio.uint8)

                # Write label to file.
                dst.write(label, 1)
