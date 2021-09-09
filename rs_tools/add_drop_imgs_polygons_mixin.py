from typing import Sequence
import logging
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
            new_polygons_df : GeoDataFrame, 
            generate_labels : bool=False, 
            force_overwrite : bool=False):
        """
        Add (or overwrite) polygons in new_polygons_df to the associator (i.e. append to the associator's polygons_df) keeping track of which polygons are contained in which images.

        Args:
            new_polygons_df (GeoDataFrame): GeoDataFrame of polygons conforming to the associator's polygons_df format
            recreate_labels (bool): Whether to generate new labels for images containing polygons that were added
            force_overwrite (bool): whether to overwrite existing rows for polygons, default is False
        """        

        new_polygons_df = deepcopy_gdf(new_polygons_df) #  don't modify argument

        self._standardize_df_crs(
            df=new_polygons_df, 
            df_name='new_polygons_df')

        self._compare_df_col_index_names(
            df=new_polygons_df, 
            df_name='new_polygons_df', 
            self_df=self.polygons_df, 
            self_df_name='self.polygons_df' 
        )

        new_polygons_df['img_count'] = 0

        # For each new polygon...
        for polygon_name in new_polygons_df.index:
            
            # ... if it already is in the associator ...
            if self._graph.exists_vertex(polygon_name, 'polygons'): # or: polygon_name in self.polygons_df.index
                
                # ... if necessary. ...
                if force_overwrite == True:
                    
                    # ... we overwrite the row in the associator's polygons_df ...
                    self.polygons_df.loc[polygon_name] = new_polygons_df.loc[polygon_name].copy()
                    
                    # ... and drop the row from new_polygons_df, so it won't be in self.polygons_df twice after we concatenate polygons_df to self.polygons_df. ...
                    new_polygons_df.drop(polygon_name, inplace=True) 
                    
                    # Then, we recalculate the connections. ...
                    self._remove_polygon_from_graph_modify_polygons_df(polygon_name)
                    self._add_polygon_to_graph(polygon_name)
                
                # Else ...
                else:
                    
                    # ... we drop the row from new_polygons_df...
                    new_polygons_df.drop(polygon_name, inplace=True)

                    log.info(f"integrate_new_polygons_df: dropping row for {polygon_name} from input polygons_df since is already in the associator! (force_overwrite arg is set to{force_overwrite})")

            # If it is not in the associator ...
            else:
                
                # ... add a vertex for the new polygon to the graph and add all connections to existing images. ...
                self._add_polygon_to_graph(polygon_name, polygons_df=new_polygons_df)

        # Finally, append new_polygons_df to the associator's (self.)polygons_df.
        data_frames_list = [self.polygons_df, new_polygons_df]
        self.polygons_df = GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)

        if generate_labels == True:
            imgs_w_new_polygons = [img_name for polygon_name in new_polygons_df.index for img_name in self.imgs_intersecting_polygon(polygon_name)]
            for img_name in imgs_w_new_polygons:
                label_path = self.labels_dir / img_name
                if label_path.suffix != '.tif':
                    raise ValueError(f"Can only generate labels for geotifs.")
                else:
                    label_path.unlink(missing_ok=True)
            self.make_missing_labels(img_names=imgs_w_new_polygons)


    def add_to_imgs_df(self, new_imgs_df : GeoDataFrame):
        """
        Add image data in new_imgs_df to the associator keeping track of which polygons are contained in which images.

        Args:
            new_imgs_df (gdf.GeoDataFrame): GeoDataFrame of image information conforming to the associator's imgs_df format
        """        

        new_imgs_df = deepcopy_gdf(new_imgs_df) #  don't want to modify argument

        self._standardize_df_crs(
            df=new_imgs_df, 
            df_name='new_imgs_df')

        self._compare_df_col_index_names(
            df=new_imgs_df, 
            df_name='new_imgs_df', 
            self_df=self.imgs_df, 
            self_df_name='self.imgs_df')

        # go through all new imgs...
        for img_name in new_imgs_df.index:
            
            # ... check if it is already in associator.
            if self._graph.exists_vertex(img_name, 'imgs'): 
                
                # drop row from new_imgs_df, so it won't be in self.imgs_df twice after we concat new_imgs_df to self.imgs_df
                new_imgs_df.drop(img_name, inplace=True) 
                log.info(f"integrate_new_imgs_df: dropping row for {img_name} from input imgs_df since an image with that name is already in the associator!")
                
            else:
                
                # add new img vertex to the graph, add all connections to existing images, 
                # and modify self.polygons_df 'img_count' value
                img_bounding_rectangle=new_imgs_df.loc[img_name, 'geometry']
                self._add_img_to_graph_modify_polygons_df(
                    img_name, 
                    img_bounding_rectangle=img_bounding_rectangle)

        # append new_imgs_df
        data_frames_list = [self.imgs_df, new_imgs_df]
        self.imgs_df = GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)


    def drop_polygons(self, polygon_names : Sequence[str]):
        """
        Drop polygons from associator (i.e. remove rows from the associator's polygons_df)

        Args:
            polygon_names (Sequence[str]): polygon_names/identifiers of polygons to be dropped.
        """

        # make sure we don't interpret a string as a list of characters in the iteration below:
        if type(polygon_names) == str:
            polygon_names = [polygon_names]
        assert pd.api.types.is_list_like(polygon_names)

        # remove the polygon vertices (along with their edges) 
        for polygon_name in polygon_names:
            self._graph.delete_vertex(polygon_name, 'polygons', force_delete_with_edges=True)
            
        # drop row from self.polygons_df
        self.polygons_df.drop(polygon_names, inplace=True)


    def drop_imgs(self, 
        img_names : Sequence[str], 
        remove_imgs_from_disk : bool=True):
        """
        Drop images from associator and dataset, i.e. remove rows from the associator's imgs_df, delete the corresponding vertices in the graph, and delete the image from disk (unless remove_imgs_from_disk is set to False).

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
        if remove_imgs_from_disk==True:
            for dir in self.image_data_dirs: 
                for img_name in img_names:
                    (dir / img_name).unlink(missing_ok=True)

    @staticmethod
    def _compare_df_col_index_names(
            df : GeoDataFrame,
            df_name : str, 
            self_df : GeoDataFrame,
            self_df_name : str 
            ) -> bool:
        """Check if column and index names of df1 and df2 agree. Raise a ValueError if they don't. """
        
        if set(df.columns) != set(self_df.columns):

            df1_cols_not_in_df2 = set(df.columns) - set(self_df.columns)
            df2_cols_not_in_df1 = set(self_df.columns) - set(df.columns)

            if df1_cols_not_in_df2 != {}:
                log.error(f"columns that are in {df_name} but not in {self_df_name}: {df1_cols_not_in_df2}")
            if df2_cols_not_in_df1 != {}:
                log.error(f"columns that are in {self_df_name} but not in {df_name}: {df2_cols_not_in_df1}")
            
            raise ValueError(f"columns of {df_name} and {df_name} don't agree.")

        if df.index.name != self_df.index.name:
            raise ValueError(f"Index names for {df_name} and {self_df_name} disagree: {df.index.name} and {self_df_name.index.name}")
