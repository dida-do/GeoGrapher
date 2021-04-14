"""
Label maker for categorical labels.
"""
import numpy as np 
import rasterio as rio
from pathlib import Path

from rs_tools.utils.utils import transform_shapely_geometry

def _make_geotif_label_categorical(assoc, img_name, log):
    """
    Create a categorical GeoTiff pixel label for an image.

    Create a categorical GeoTiff pixel label (i.e. one channel images where each pixel is an integer corresponding to either the background or a segmentation class, 0 indicating the background class, and k=1,2, ... indicating the k-th entry (starting from 1) of the segmentation_classes parameter of the associator) in the data directory's labels subdirectory for the GeoTiff image img_name in the images subdirectory. 

    Args:
        - assoc: calling ImgPolygonAssociator.
        - img_name: The filename of the image in the dataset's images subdirectory. 
        - log: The logger of the calling associator.
    Returns: 
        - None:
    """

    img_path = Path(assoc.data_dir) / Path("images") / Path(img_name)

    label_path = Path(assoc.data_dir) / Path("labels") / Path(img_name)

    # If the image does not exist ...
    if not img_path.is_file():

        # ... log error to file.
        log.error(f"__make_geotif_label__: input image {img_path} does not exist!")

    # Else, if the label already exists ...
    elif label_path.is_file():

        # ... log error to file.
        log.error(f"__make_geotif_label__: label {label_path} already exists!")
    
    # Else, ...
    else:

        # ...open the image, ...
        with rio.open(img_path) as src:

            profile = src.profile
            profile.update({"count": 1, 
                            "dtype": rio.uint8})

            # ... open the label ...            
            with rio.open(label_path,
                            'w',
                            # nbits=1, # for writing single bit image, see https://gis.stackexchange.com/questions/338410/rasterio-invalid-dtype-bool
                            **profile) as dst:
                            
                # ... create an empty band of zeros (background class) ...
                label = np.zeros((src.height, src.width), dtype=np.uint8)

                # ... and fill in values for each segmentation class.
                for count, seg_class in enumerate(assoc.__params_dict__['segmentation_classes'], start=1):
                    
                    # To do that, first find (the df of) the polygons intersecting the image ...
                    polygons_intersecting_img_df = assoc.polygons_df.loc[assoc.polygons_intersecting_img(img_name)]
                    
                    # ... then restrict to (the subdf of) polygons with the given segmentation class. 
                    polygons_intersecting_img_df_of_type = polygons_intersecting_img_df.loc[polygons_intersecting_img_df['type'] == seg_class]

                    # Extract the polygon geometries of these polygons ...
                    polygon_geometries_in_std_crs = list(polygons_intersecting_img_df_of_type['geometry'])

                    # ... and convert them to the crs of the source image. 
                    polygon_geometries_in_src_crs = list(map(lambda geom: transform_shapely_geometry(geom, assoc.polygons_df.crs.to_epsg(), src.crs.to_epsg()), 
                                                    polygon_geometries_in_std_crs)) 

                    # Burn the polygon geometries into the label.
                    if len(polygon_geometries_in_src_crs) != 0:
                        rio.features.rasterize(shapes=polygon_geometries_in_src_crs, 
                                                out_shape=(src.height, src.width), # or the other way around?
                                                fill=0, 
                                                merge_alg=rio.enums.MergeAlg.add, # important!
                                                out=label,
                                                transform=src.transform, 
                                                default_value=count, # value to add for polygon
                                                dtype=rio.uint8) 

                # Write label to file.
                dst.write(label, 1)
