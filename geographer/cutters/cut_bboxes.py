"""
TODO: Include as method in RasterPolygonAssociator.

Functions to cut datasets of GeoTiffs (or update previously cut datasets)
by cutting each raster in the source dataset to a grid of rasters.
    - cut_dataset_raster_to_grid_of_rasters. Updates a dataset of
        GeoTiffs that was created with new_tif_dataset_raster2grid_rasters.
    - update_dataset_raster_to_grid_of_rasters: customizable general function
        to create or update datasets of GeoTiffs from existing ones
        by iterating over vector features.
"""

# yapf: disable

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from geopandas import GeoDataFrame

from geographer.cutters.single_raster_cutter_bbox import SingleRasterCutterFromBBoxes
from geographer.cutters.type_aliases import RasterSize

if TYPE_CHECKING:
    from geographer.raster_geom_associator import RasterPolygonAssociator

from geographer.cutters.cut_iter_over_rasters import (
    create_or_update_dataset_iter_over_rasters,
)
from geographer.cutters.raster_filter_predicates import AlwaysTrue

logger = logging.getLogger(__name__)

class DSCutterBBoxes:  # noqa: E302
    """Class for creating a dataset from bboxes and a source dataset."""

    def create_dataset_cut_bboxes(
            self,
            create_or_update: str,
            bounding_boxes: GeoDataFrame,
            source_assoc: RasterPolygonAssociator,
            target_data_dir: Union[str, Path],
            target_assoc: Optional[RasterPolygonAssociator] = None,
            new_raster_size: RasterSize = 512,
            raster_bands: Optional[list[int]] = None,
            label_bands: Optional[list[int]] = None) -> RasterPolygonAssociator:
        """TODO.

        Warning:
            TODO! update is not going to work because should be iter over
            (vector) geometries but uses iter over rasters.


        Args:
            source_data_dir: data directory (rasters, labels, associator) containing
                the GeoTiffs to be cut from.
            source_assoc: associator of dataset containing the GeoTiffs to be cut from.
            target_data_dir: path to data directory where the new dataset
                (rasters, labels, associator) will be created. If the directory
                does not exist it will be created.
            target_assoc: associator of target dataset.
            new_raster_size: size of new rasters (side length or (rows, col))
                for 'centered' and 'random' modes. Defaults to 512.
            raster_bands: list of bands to extract from source rasters.
                Defaults to None (i.e. all bands).
            label_bands:  list of bands to extract from source labels.
                Defaults to None (i.e. all bands).

        Returns:
            connector of new dataset in target_data_dir
        """
        target_data_dir = Path(target_data_dir)

        bbox_cutter = SingleRasterCutterFromBBoxes(
            source_assoc=source_assoc,
            target_rasters_dir=target_data_dir / 'rasters',
            target_labels_dir=target_data_dir / 'labels',
            new_raster_size=new_raster_size,
            bounding_boxes=bounding_boxes,
            raster_bands=raster_bands,
            label_bands=label_bands)
        always_true = AlwaysTrue()

        target_assoc = create_or_update_dataset_iter_over_rasters(
            create_or_update=create_or_update,
            source_assoc=source_assoc,
            target_data_dir=target_data_dir,
            target_assoc=target_assoc,
            raster_cutter=bbox_cutter,
            raster_filter_predicate=always_true)

        # throw out rasters with duplicate bboxes:
        # First, find a subset of rasters without duplicate bboxes ...
        rasters_to_keep: list[str] = []
        for count, raster_name in enumerate(target_assoc.rasters.index):
            raster_bbox = target_assoc.rasters.loc[raster_name, 'geometry']
            if {
                    raster_name_
                    for raster_name_ in rasters_to_keep
                    if raster_bbox.equals(target_assoc.rasters.loc[
                        raster_name_, 'geometry'])
            } == set():
                rasters_to_keep += [raster_name]
        # ... and delete the remaining rasters, which have duplicate bboxes
        rasters_to_delete = [
            raster_name for raster_name in target_assoc.rasters.index
            if raster_name not in rasters_to_keep
        ]
        target_assoc.drop_rasters(rasters_to_delete, remove_rasters_from_disk=True)

        # remember the cutting params.
        target_assoc._update_from_source_dataset_dict.update({
            'update_method':
            'update_dataset_rasters_around_every_polygon',
            'source_data_dir':
            source_assoc.rasters_dir.
            parent,  # Assuming standard data directory format
            'new_raster_size':
            new_raster_size,
            'raster_bands':
            raster_bands,
            'label_bands':
            label_bands,
        })
        target_assoc._params_dict['raster_size'] = new_raster_size
        target_assoc.save()

        return target_assoc
# yapf: enable
