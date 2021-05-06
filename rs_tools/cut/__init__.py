"""
Functions to create or update datasets of GeoTiffs by cutting images by iterating over polygons.

new_dataset_one_small_img_for_each_polygon cuts square images surrounding each polygon. 
update_dataset_one_small_img_for_each_polygon updates a dataset created with 

new_dataset_one_small_img_for_each_polygon.

create_or_update_dataset_from_iter_over_polygons is a very general function to cut datasets by iterating over images.
"""

from rs_tools.cut.cut_dataset_iter_over_polygons import new_tif_dataset_small_imgs_for_each_polygon
from rs_tools.cut.cut_dataset_iter_over_polygons import update_tif_dataset_small_imgs_for_each_polygon
from rs_tools.cut.cut_dataset_iter_over_polygons import create_or_update_tif_dataset_from_iter_over_polygons