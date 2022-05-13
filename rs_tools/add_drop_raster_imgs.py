from __future__ import annotations
import logging
from typing import Literal, Optional, Sequence, Union

import pandas as pd
from geopandas import GeoDataFrame
from rs_tools.graph.bipartite_graph_mixin import RASTER_IMGS_COLOR

from rs_tools.utils.connector_utils import _check_df_cols_agree
from rs_tools.utils.utils import concat_gdfs, deepcopy_gdf
from rs_tools.label_makers.label_maker_base import LabelMaker

log = logging.getLogger(__name__)



class AddDropRasterImgsMixIn:
    """Mix-in that implements methods to add and drop raster images."""

    def add_to_raster_imgs(self, new_raster_imgs: GeoDataFrame):
        """Add images to connector's ``raster_imgs`` attribute.

        Adds the new_raster_imgs to the connector's :ref:`raster_imgs` keeping track of
        which (vector) geometries are contained in which images.

        Args:
            new_raster_imgs (gdf.GeoDataFrame): GeoDataFrame of image information conforming to the connector's raster_imgs format
        """

        new_raster_imgs = deepcopy_gdf(
            new_raster_imgs)  #  don't want to modify argument

        duplicates = new_raster_imgs[new_raster_imgs.index.duplicated()]
        if len(duplicates) > 0:
            raise ValueError(
                f"new_raster_imgs contains rows with duplicate img_names: {duplicates.index.tolist()}"
            )

        if new_raster_imgs.geometry.isna().any():
            imgs_with_null_geoms: str = ', '.join(new_raster_imgs[new_raster_imgs.geometry.isna()].index)
            raise ValueError(
                f"new_raster_imgs contains rows with None geometries: {imgs_with_null_geoms}"
            )

        self._check_required_df_cols_exist(df=new_raster_imgs,
                                           df_name='new_raster_imgs',
                                           mode='raster_imgs')
        new_raster_imgs = self._get_df_in_crs(df=new_raster_imgs,
                                              df_name='new_raster_imgs',
                                              crs_epsg_code=self.crs_epsg_code)
        _check_df_cols_agree(df=new_raster_imgs,
                             df_name='new_raster_imgs',
                             self_df=self.raster_imgs,
                             self_df_name='self.raster_imgs')

        # go through all new imgs...
        for img_name in new_raster_imgs.index:

            # ... check if it is already in connector.
            if self._graph.exists_vertex(img_name, RASTER_IMGS_COLOR):

                # drop row from new_raster_imgs, so it won't be in self.raster_imgs twice after we concat new_raster_imgs to self.raster_imgs
                new_raster_imgs.drop(img_name, inplace=True)
                log.info(
                    "add_to_raster_imgs: dropping row for %s from input raster_imgs since an image with that name is already in the connector!",
                    img_name)

            else:

                # add new img vertex to the graph, add all connections to existing images,
                # and modify self.vector_features 'img_count' value
                img_bounding_rectangle = new_raster_imgs.loc[img_name,
                                                             'geometry']
                self._add_img_to_graph_modify_vector_features(
                    img_name, img_bounding_rectangle=img_bounding_rectangle)

        # append new_raster_imgs
        self.raster_imgs = concat_gdfs([self.raster_imgs, new_raster_imgs])
        #self.raster_imgs = self.raster_imgs.convert_dtypes()

    def drop_raster_imgs(
        self,
        img_names: Sequence[str],
        remove_imgs_from_disk: bool = True,
        label_maker: Optional[LabelMaker] = None,
    ):
        """Drop images from connector's ``raster_imgs`` attribute and from dataset.

        Remove rows from the connector's raster_imgs, delete the corresponding
        vertices in the graph, and delete the image from disk (unless
        remove_imgs_from_disk is set to False).

        Args:
            img_names (List[str]): img_names/ids of images to be dropped.
            remove_imgs_from_disk (bool): If true, delete images and labels
                from disk (if they exist). Defaults to True.
            label_maker (LabelMaker, optional): If given, will use label_makers
                delete_labels method. Defaults to None.
        """

        # make sure we don't interpret a string as a list of characters in the iteration below:
        if isinstance(img_names, str):
            img_names = [img_names]
        assert pd.api.types.is_list_like(img_names)

        # drop row from self.raster_imgs
        self.raster_imgs.drop(img_names, inplace=True)

        # remove all vertices from graph and modify vector_features if necessary
        for img_name in img_names:
            self._remove_img_from_graph_modify_vector_features(img_name)

        # remove imgs and labels from disk
        if remove_imgs_from_disk:
            if label_maker is not None:
                label_maker.delete_labels(self, img_names)
            for dir_ in self.image_data_dirs:
                for img_name in img_names:
                    (dir_ / img_name).unlink(missing_ok=True)
