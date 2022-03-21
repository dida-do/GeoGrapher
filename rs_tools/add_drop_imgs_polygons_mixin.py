import logging
from typing import Optional, Sequence, Union

import pandas as pd
from geopandas import GeoDataFrame

from rs_tools.utils.utils import deepcopy_gdf

# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)


class AddDropImgsPolygonsMixIn(object):
    """Mix-in that implements methods to add and drop polygons or images."""

    def add_to_polygons_df(self,
                           new_polygons_df: GeoDataFrame,
                           generate_labels: bool = False,
                           force_overwrite: bool = False):
        """Add (or overwrite) polygons in new_polygons_df to the associator
        (i.e. append to the associator's polygons_df) keeping track of which
        polygons are contained in which images.

        Args:
            new_polygons_df (GeoDataFrame): GeoDataFrame of polygons conforming to the associator's polygons_df format
            generate_labels (bool): Whether to generate new labels for images containing polygons that were added
            force_overwrite (bool): whether to overwrite existing rows for polygons, default is False
        """

        new_polygons_df = deepcopy_gdf(
            new_polygons_df)  #  don't modify argument
        new_polygons_df['img_count'] = 0

        duplicates = new_polygons_df[new_polygons_df.index.duplicated()]
        if len(duplicates) > 0:
            raise ValueError(
                f"new_polygons_df contains rows with duplicate polygon_names: {duplicates.index.tolist()}"
            )

        if len(new_polygons_df[new_polygons_df.geometry.isna()]) > 0:
            raise ValueError(
                f"new_polygons_df contains rows with None geometries: {', '.join(new_polygons_df[new_polygons_df.geometry.isna()].index)}"
            )

        self._standardize_df_crs(df=new_polygons_df, df_name='new_polygons_df')

        self._check_df_cols_index_name(df=new_polygons_df,
                                       df_name='new_polygons_df',
                                       self_df=self.polygons_df,
                                       self_df_name='self.polygons_df')

        self._check_classes_in_polygons_df_contained_in_all_classes(
            new_polygons_df, 'new_polygons_df')

        # For each new polygon...
        for polygon_name in new_polygons_df.index:

            # ... if it already is in the associator ...
            if self._graph.exists_vertex(
                    polygon_name,
                    'polygons'):  # or: polygon_name in self.polygons_df.index

                # ... if necessary. ...
                if force_overwrite == True:

                    # ... we overwrite the row in the associator's polygons_df ...
                    self.polygons_df.loc[polygon_name] = new_polygons_df.loc[
                        polygon_name].copy()

                    # ... and drop the row from new_polygons_df, so it won't be in self.polygons_df twice after we concatenate polygons_df to self.polygons_df. ...
                    new_polygons_df.drop(polygon_name, inplace=True)

                    # Then, we recalculate the connections. ...
                    self._remove_polygon_from_graph_modify_polygons_df(
                        polygon_name)
                    self._add_polygon_to_graph(polygon_name)

                # Else ...
                else:

                    # ... we drop the row from new_polygons_df...
                    new_polygons_df.drop(polygon_name, inplace=True)

                    log.info(
                        f"integrate_new_polygons_df: dropping row for {polygon_name} from input polygons_df since is already in the associator! (force_overwrite arg is set to{force_overwrite})"
                    )

            # If it is not in the associator ...
            else:

                # ... add a vertex for the new polygon to the graph and add all connections to existing images. ...
                self._add_polygon_to_graph(polygon_name,
                                           polygons_df=new_polygons_df)

        # Finally, append new_polygons_df to the associator's (self.)polygons_df.
        data_frames_list = [self.polygons_df, new_polygons_df]
        self.polygons_df = GeoDataFrame(pd.concat(data_frames_list),
                                        crs=data_frames_list[0].crs)
        #self.polygons_df = self.polygons_df.convert_dtypes()

        if generate_labels == True:
            imgs_w_new_polygons = [
                img_name for polygon_name in new_polygons_df.index
                for img_name in self.imgs_intersecting_polygon(polygon_name)
            ]
            for img_name in imgs_w_new_polygons:
                label_path = self.labels_dir / img_name
                if label_path.suffix != '.tif':
                    raise ValueError(f"Can only generate labels for geotifs.")
                else:
                    label_path.unlink(missing_ok=True)
            self.make_missing_labels(img_names=imgs_w_new_polygons)

    def add_to_imgs_df(self, new_imgs_df: GeoDataFrame):
        """Add image data in new_imgs_df to the associator keeping track of
        which polygons are contained in which images.

        Args:
            new_imgs_df (gdf.GeoDataFrame): GeoDataFrame of image information conforming to the associator's imgs_df format
        """

        new_imgs_df = deepcopy_gdf(
            new_imgs_df)  #  don't want to modify argument

        duplicates = new_imgs_df[new_imgs_df.index.duplicated()]
        if len(duplicates) > 0:
            raise ValueError(
                f"new_imgs_df contains rows with duplicate img_names: {duplicates.index.tolist()}"
            )

        if len(new_imgs_df[new_imgs_df.geometry.isna()]) > 0:
            raise ValueError(
                f"new_imgs_df contains rows with None geometries: {', '.join(new_imgs_df[new_imgs_df.geometry.isna()].index)}"
            )

        self._standardize_df_crs(df=new_imgs_df, df_name='new_imgs_df')

        self._check_df_cols_index_name(df=new_imgs_df,
                                       df_name='new_imgs_df',
                                       self_df=self.imgs_df,
                                       self_df_name='self.imgs_df')

        # go through all new imgs...
        for img_name in new_imgs_df.index:

            # ... check if it is already in associator.
            if self._graph.exists_vertex(img_name, 'imgs'):

                # drop row from new_imgs_df, so it won't be in self.imgs_df twice after we concat new_imgs_df to self.imgs_df
                new_imgs_df.drop(img_name, inplace=True)
                log.info(
                    f"integrate_new_imgs_df: dropping row for {img_name} from input imgs_df since an image with that name is already in the associator!"
                )

            else:

                # add new img vertex to the graph, add all connections to existing images,
                # and modify self.polygons_df 'img_count' value
                img_bounding_rectangle = new_imgs_df.loc[img_name, 'geometry']
                self._add_img_to_graph_modify_polygons_df(
                    img_name, img_bounding_rectangle=img_bounding_rectangle)

        # append new_imgs_df
        data_frames_list = [self.imgs_df, new_imgs_df]
        self.imgs_df = GeoDataFrame(pd.concat(data_frames_list),
                                    crs=data_frames_list[0].crs)
        #self.imgs_df = self.imgs_df.convert_dtypes()

    def drop_polygons(self,
                      polygon_names: Sequence[Union[str, int]],
                      recompute_labels: bool = True):
        """Drop polygons from associator (i.e. remove rows from the
        associator's polygons_df)

        Args:
            polygon_names (Sequence[str]): polygon_names/identifiers of polygons to be dropped.
            recompute_labels (bool): whether to recompute labels for imgs intersecting the dropped polygons.
        """

        # make sure we don't interpret a string as a list of characters in the iteration below:
        if isinstance(polygon_names, (str, int)):
            polygon_names = [polygon_names]
        assert pd.api.types.is_list_like(polygon_names)

        names_of_imgs_with_labels_to_recompute = set()

        # remove the polygon vertices (along with their edges)
        for polygon_name in polygon_names:
            names_of_imgs_with_labels_to_recompute.update(
                set(self.imgs_intersecting_polygon(polygon_name)))
            self._graph.delete_vertex(polygon_name,
                                      'polygons',
                                      force_delete_with_edges=True)

        # drop row from self.polygons_df
        self.polygons_df.drop(polygon_names, inplace=True)

        # recompute labels
        if recompute_labels is True:
            names_of_imgs_with_labels_to_recompute = list(
                names_of_imgs_with_labels_to_recompute)
            self.delete_labels(names_of_imgs_with_labels_to_recompute)
            self.make_labels(names_of_imgs_with_labels_to_recompute)

    def drop_imgs(self,
                  img_names: Sequence[str],
                  remove_imgs_from_disk: bool = True):
        """Drop images from associator and dataset, i.e. remove rows from the
        associator's imgs_df, delete the corresponding vertices in the graph,
        and delete the image from disk (unless remove_imgs_from_disk is set to
        False).

        Args:
            img_names (List[str]): img_names/ids of images to be dropped.
            remove_imgs_from_disk (bool): If true, delete images and labels from disk (if they exist).
        """

        # make sure we don't interpret a string as a list of characters in the iteration below:
        if type(img_names) == str:
            img_names = [img_names]
        assert pd.api.types.is_list_like(img_names)

        # drop row from self.imgs_df
        self.imgs_df.drop(img_names, inplace=True)

        # remove all vertices from graph and modify polygons_df if necessary
        for img_name in img_names:
            self._remove_img_from_graph_modify_polygons_df(img_name)

        # remove imgs and labels from disk
        if remove_imgs_from_disk == True:
            for dir in self.image_data_dirs:
                for img_name in img_names:
                    (dir / img_name).unlink(missing_ok=True)

    def _standardize_df_crs(self, df: GeoDataFrame, df_name: str):

        df = df.to_crs(epsg=self.crs_epsg_code)

    def _check_df_cols_index_name(self, df: GeoDataFrame, df_name: str,
                                  self_df: GeoDataFrame,
                                  self_df_name: str) -> bool:
        """Check if column and index names of df1 and df2 agree."""

        required_cols = self._get_required_df_cols(self_df_name.split(".")[1])

        if not set(required_cols) <= set(df.columns):

            missing_cols = set(required_cols) - set(df.columns)
            raise ValueError(
                f"{df_name} is missing required columns: {', '.join(missing_cols)}"
            )

        if set(df.columns) != set(self_df.columns) and len(self_df) > 0:

            df1_cols_not_in_df2 = set(df.columns) - set(self_df.columns)
            df2_cols_not_in_df1 = set(self_df.columns) - set(df.columns)

            if df1_cols_not_in_df2 != {}:
                log.debug(
                    f"columns that are in {df_name} but not in {self_df_name}: {df1_cols_not_in_df2}"
                )
            if df2_cols_not_in_df1 != {}:
                log.debug(
                    f"columns that are in {self_df_name} but not in {df_name}: {df2_cols_not_in_df1}"
                )

            log.debug(f"columns of {df_name} and {df_name} don't agree.")

        if df.index.name != self_df.index.name:
            raise ValueError(
                f"Index names for {df_name} and {self_df_name} disagree: {df.index.name} and {self_df.index.name}"
            )
