"""
TODO: parallelize _add_img_to_graph_modify_polygons_df and _add_polygon_to_graph!

"""

from typing import List, Sequence, Optional, Union
import logging
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, polygon
from rs_tools.utils.utils import deepcopy_gdf
from rs_tools.graph import BipartiteGraph

# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)


class ImgPolygonAssociatorBase(object):
    """Mix-in that implements the public and private interface to the internal bipartite graph."""

    def have_img_for_polygon(self, polygon_name: str) -> bool:
        """
        Return whether there is an image in the dataset fully containing the polygon.

        Args:
            polygon_name (str): Name of polygon

        Returns:
            bool: `True` if there is an image in the dataset fully containing the polygon, False otherwise.
        """

        return self.polygons_df.loc[polygon_name, 'img_count'] > 0


    def rectangle_bounding_img(self, img_name: str) -> Polygon:
        """
        Return the shapely polygon of the rectangle bounding the image in coordinates in the associator's (standard) crs.

        Args:
            img_name (str): the img_name/identifier of the image

        Returns:
            Polygon: shapely polygon giving the bounds of the image in the standard crs of the associator
        """

        return self.imgs_df.loc[img_name, 'geometry']


    def polygons_intersecting_img(self, img_name : Union[str, List[str]]) -> List[str]:
        """
        Given an image or a list of images, return the list of (the names of) all polygons
        which have non-empty intersection with it.

        Args:
            img_name (str, or List[str]): name/id of image or list names/ids

        Returns:
            list of strs of polygon_names/ids of all polygons in associator which have non-empty intersection with the image(s)
        """

        if isinstance(img_name, str):
            img_names = [img_name]
        else:
            img_names = img_name

        polygon_names = []

        for img_name in img_names:
            try:
                polygon_names += self._graph.vertices_opposite(vertex=img_name, vertex_color='imgs')
            except KeyError:
                raise ValueError(f"Unknown image: {img_name}")

        return polygon_names

    def imgs_intersecting_polygon(self,
            polygon_name: Union[str, List[str]],
            mode : str = 'names'
            ) -> List[str]:
        """
        Given a polygon (or list of polygons), return a list of the names or paths of all images
        which have non-empty intersection with the polygon(s).

        Args:
            polygon_name (str): name/id (or list) of polygon(s)
            mode (str): One of 'names' or 'paths'. In the former case the image names are returned in the latter case paths to the images. Defaults to 'names'.

        Returns:
            list of str: list of the polygon_names/identifiers of all polygons in associator with non-empty intersection with the image.
        """

        if isinstance(polygon_name, str):
            polygon_names = [polygon_name]
        else:
            polygon_names = polygon_name

        img_names = []

        for polygon_name in polygon_names:
            try:
                img_names += self._graph.vertices_opposite(vertex=polygon_name, vertex_color='polygons')
            except KeyError:
                raise ValueError(f"Unknown polygon: {polygon_name}")

        if mode == 'names':
            answer = img_names
        elif mode == 'paths':
            answer = list(
                        map(
                            lambda img_name: self.images_dir / img_name,
                            img_names
                        )
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return answer


    def polygons_contained_in_img(self, img_name: Union[str, List[str]]) -> List[str]:
        """
        Given an image, return an iterator of the names of all polygons
        which it fully contains.

        Args:
            img_name (str): name/id of image or list of names/ids of images

        Returns:
            list of str: list of the polygon_names/identifiers of all polygons in associator contained in the image(s).
        """

        if isinstance(img_name, str):
            img_names = [img_name]
        else:
            img_names = img_name

        polygon_names = []

        for img_name in img_names:
            try:
                polygon_names += self._graph.vertices_opposite(vertex=img_name, vertex_color='imgs', edge_data='contains')
            except KeyError:
                raise ValueError(f"Unknown image: {img_name}")

        return polygon_names

    def imgs_containing_polygon(self, 
            polygon_name: Union[str, List[str]],
            mode : str = 'names', 
            ) -> List[str]:
        """
        Given a polygon (or a list of polygons), return a list of the names or paths of all images in which the polygon(s) is/are fully contained.

        Args:
            polygon_name (str): polygon name/id (or list)
            mode (str): One of 'names' or 'paths'. In the former case the image names are returned in the latter case paths to the images. Defaults to 'names'.

        Returns:
            List[str]: list of the img_names/identifiers of all images in associator containing the polygon(s)
        """

        if isinstance(polygon_name, str):
            polygon_names = [polygon_name]
        else:
            polygon_names = polygon_name

        img_names = []

        for polygon_name in polygon_names:
            try:
                img_names = self._graph.vertices_opposite(vertex=polygon_name, vertex_color='polygons', edge_data='contains')
            except KeyError:
                raise ValueError(f"Unknown polygon: {polygon_name}")

        if mode == 'names':
            answer = img_names
        elif mode == 'paths':
            answer = list(
                        map(
                            lambda img_name: self.images_dir / img_name,
                            img_names
                        )
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return answer


    def does_img_contain_polygon(self, img_name: str, polygon_name: str) -> bool:
        """
        Args:
            img_name (str): Name of image
            polygon_name (str): name of polygon

        Returns:
            bool: True or False depending on whether the image contains the polygon or not
        """

        return polygon_name in self.polygons_contained_in_img(img_name)


    def is_polygon_contained_in_img(self, polygon_name: str, img_name: str) -> bool:
        """
        Args:
            img_name (str): Name of image
            polygon_name (str): name of polygon

        Returns:
            bool: True or False depending on whether the polygon contains the image or not
        """

        return self.does_img_contain_polygon(img_name, polygon_name)


    def does_img_intersect_polygon(self, img_name: str, polygon_name: str) -> bool:
        """
        Args:
            img_name (str): Name of image
            polygon_name (str): name of polygon

        Returns:
            bool: True or False depending on whether the image intersects the polygon or not
        """

        return (polygon_name in self.polygons_intersecting_img(img_name))


    def does_polygon_intersect_img(self, polygon_name: str, img_name: str) -> bool:
        """
        Args:
            img_name (str): Name of image
            polygon_name (str): name of polygon

        Returns:
            bool: True or False depending on whether the polygon intersects the image or not
        """

        return self.does_img_intersect_polygon(img_name, polygon_name)


    def _connect_img_to_polygon(self,
            img_name : str,
            polygon_name : str,
            contains_or_intersects : Optional[str] = None,
            polygons_df : Optional[GeoDataFrame] = None,
            img_bounding_rectangle : Optional[Polygon] = None,
            graph : Optional[BipartiteGraph] = None,
            do_safety_check: bool = True):
        """
        Connect an image to a polygon in the graph.

        Remember (i.e. create a connection in the graph) whether the image fully contains or just has non-empty intersection with the polygon, i.e. add an edge of the approriate type between the image and the polygon.

        Args:
            img_name (str): Name of image to connect
            polygon_name (str): Name of polygon to connect
            contains_or_intersects (optional, str): Optional connection criteria
            polygons_df (optional, gdf.GeoDataFrame): Optional polygon dataframe
            img_bounding_rectangle (optional, Polygon): polygon decribing image footprint
            graph (optional, BipartiteGraph): optional bipartied graph
            ignore_safety_check (bool): whether to check contains_or_intersects relation
        """

        if contains_or_intersects not in {'contains', 'intersects', None}:
            raise ValueError(f"contains_or_intersects should be one of 'contains' or 'intersects' or None, is {contains_or_intersects}")

        # default polygons_df
        if polygons_df is None:
            polygons_df=self.polygons_df

        # default graph
        if graph is None:
            graph = self._graph

        # default img_bounding_rectangle
        if img_bounding_rectangle is None:
            img_bounding_rectangle = self.imgs_df.loc[img_name, 'geometry']

        # get containment relation if not given
        if contains_or_intersects is None:

            polygon_geometry = polygons_df.loc[polygon_name, 'geometry']

            non_empty_intersection = polygon_geometry.intersects(img_bounding_rectangle)
            if not non_empty_intersection:
                log.info(f"_connect_img_to_polygon: not connecting, sinceimg  {img_name} and polygon {polygon_name} do not overlap.")
            else:
                contains_or_intersects = 'contains' if img_bounding_rectangle.contains(polygon_geometry) else 'intersects'

        elif do_safety_check:
            polygon_geometry = polygons_df.loc[polygon_name, 'geometry']
            assert img_bounding_rectangle.intersects(polygon_geometry)
            assert contains_or_intersects == 'contains' if img_bounding_rectangle.contains(polygon_geometry) else 'intersects'

        graph.add_edge(img_name, 'imgs', polygon_name, contains_or_intersects)

        # if the polygon is fully contained in the image increment the image counter in self.polygons_df
        if contains_or_intersects == 'contains':
            polygons_df.loc[polygon_name, 'img_count'] += 1


    def _add_polygon_to_graph(self,
            polygon_name : str,
            polygons_df : Optional[GeoDataFrame] = None):
        """
        Connects a polygon to those images in self.imgs_df with which it has non-empty intersection.

        Args:
            polygon_name (str): name/id of polygon to add
            polygons_df (GeoDataFrame, optional): Defaults to None (i.e. self.polygons_df).
        """

        # default polygons_df
        if polygons_df is None:
            polygons_df = self.polygons_df

        # add vertex if one does not yet exist
        if not self._graph.exists_vertex(polygon_name, 'polygons'):
            self._graph.add_vertex(polygon_name, 'polygons')

        # raise an exception if the polygon already has connections
        if list(self._graph.vertices_opposite(polygon_name, 'polygons')) != []:
            log.warning(f"_add_polygon_to_graph: !!!Warning (connect_polygon): polygon {polygon_name} already has connections! Probably _add_polygon_to_graph is being used wrongly. Check your code!")

        polygon_geometry = polygons_df.geometry.loc[polygon_name]

        # # REFACTORED
        # # go through all images and connect if intersection is non-empty
        # intersecting_imgs = set()
        # containing_imgs = set()
        # for img_name, img_bounding_rectangle in self.imgs_df.loc[:, ['geometry']].itertuples():
        #     if img_bounding_rectangle.intersects(polygon_geometry):
        #         contains_or_intersects = 'contains' if img_bounding_rectangle.contains(polygon_geometry) else 'intersects'
        #         # DEBUG
        #         if contains_or_intersects == 'contains':
        #             containing_imgs.add(img_name)
        #         elif contains_or_intersects == 'intersects':
        #             intersecting_imgs.add(img_name)

        #         self._connect_img_to_polygon(img_name, polygon_name, contains_or_intersects, polygons_df=polygons_df, do_safety_check=False)

        # determine intersecting and containing imgs
        intersection_mask = self.imgs_df.geometry.intersects(polygon_geometry)
        containment_mask = self.imgs_df.loc[intersection_mask].geometry.contains(polygon_geometry)

        intersecting_imgs = set(self.imgs_df.loc[intersection_mask].index)
        containing_imgs = set(self.imgs_df.loc[intersection_mask].loc[containment_mask].index)

        # add edges in graph
        for img_name in containing_imgs:
            self._connect_img_to_polygon(
                img_name,
                polygon_name,
                'contains',
                polygons_df=polygons_df,
                do_safety_check=False)
        for img_name in intersecting_imgs - containing_imgs:
            self._connect_img_to_polygon(
                img_name,
                polygon_name,
                'intersects',
                polygons_df=polygons_df,
                do_safety_check=False)


    def _add_img_to_graph_modify_polygons_df(self,
            img_name: str,
            img_bounding_rectangle: Optional[Polygon]=None,
            polygons_df: Optional[GeoDataFrame]=None,
            graph: Optional[BipartiteGraph]=None):
        """
        Create a vertex in the graph for the image if one does not yet exist and connect it to all polygons in the graph while modifying the img_counts in polygons_df where appropriate. The default values None for polygons_df and graph will be interpreted as self.polygons_df and self.graph. If img_bounding_rectangle is None, we assume we can get it from self. If the image already exists and already has connections a warning will be logged. 

        Args:
            img_name (str): Name of image to add
            img_bounding_rectangle (optional, Polygon): polygon decribing image footprint
            polygons_df (optional, gdf.GeoDataFrame): Optional polygons dataframe
            graph (optional, BipartiteGraph): optional bipartied graph
        """

        # default polygons_df
        if polygons_df is None:
            polygons_df = self.polygons_df

        # default graph:
        if graph is None:
            graph = self._graph

        # default img_bounding_rectangle
        if img_bounding_rectangle is None:
            img_bounding_rectangle = self.imgs_df.geometry.loc[img_name]

        # add vertex if it does not yet exist
        if not graph.exists_vertex(img_name, 'imgs'):
            graph.add_vertex(img_name, 'imgs')

        # check if img already has connections
        if list(graph.vertices_opposite(img_name, 'imgs')) != []:
            log.warning(f"!!!Warning (connect_img): image {img_name} already has connections!")

        # # REFACTORED
        # # # go through all polygons in polygons_df and connect by an edge if the polygon and img intersect

        # intersecting_polygons = set()
        # contained_polygons = set()
        # for polygon_name, polygon_geometry in polygons_df.loc[:, ['geometry']].itertuples():
        #     if img_bounding_rectangle.intersects(polygon_geometry):
        #         contains_or_intersects = 'contains' if img_bounding_rectangle.contains(polygon_geometry) else 'intersects'
        #         # DEBUG
        #         if contains_or_intersects == 'contains':
        #             contained_polygons.add(polygon_name)
        #         elif contains_or_intersects == 'intersects':
        #             intersecting_polygons.add(polygon_name)


        #         self._connect_img_to_polygon(
        #             img_name=img_name,
        #             polygon_name=polygon_name,
        #             contains_or_intersects=contains_or_intersects,
        #             polygons_df=polygons_df,
        #             img_bounding_rectangle=img_bounding_rectangle,
        #             graph=graph,
        #             do_safety_check=False)

        # # determine intersecting and containing imgs
        intersection_mask = self.polygons_df.geometry.intersects(img_bounding_rectangle)
        containment_mask = self.polygons_df.loc[intersection_mask].geometry.within(img_bounding_rectangle)

        intersecting_polygons = set(self.polygons_df.loc[intersection_mask].index)
        contained_polygons = set(self.polygons_df.loc[intersection_mask].loc[containment_mask].index)

        # add edges in graph
        for polygon_name in contained_polygons:
            self._connect_img_to_polygon(
                img_name,
                polygon_name,
                'contains',
                polygons_df=polygons_df,
                img_bounding_rectangle=img_bounding_rectangle,
                graph=graph,
                do_safety_check=False)
        for polygon_name in intersecting_polygons - contained_polygons:
            self._connect_img_to_polygon(
                img_name,
                polygon_name,
                'intersects',
                polygons_df=polygons_df,
                img_bounding_rectangle=img_bounding_rectangle,
                graph=graph,
                do_safety_check=False)


    def _remove_polygon_from_graph_modify_polygons_df(self,
            polygon_name : str,
            set_img_count_to_zero : bool = True):
        """
        Removes a polygon from the graph (i.e. removes the vertex and all incident edges) and (if set_img_count_to_zero == True) sets the polygons_df field 'img_count' to 0.

        Args:
            polygon_name (str): polygon name/id
            set_img_count_to_zero (bool): Whether to set img_count to 0.        
        """

        self._graph.delete_vertex(polygon_name, 'polygons', force_delete_with_edges=True)

        if set_img_count_to_zero==True:
            self.polygons_df.loc[polygon_name, 'img_count' ] = 0


    def _remove_img_from_graph_modify_polygons_df(self, img_name: str):
        """
        Removes an img from the graph (i.e. removes the vertex and all incident edges) and modifies the polygons_df fields 'img_count' for the polygons contained in the image.

        Args:
            img_name (str): name/id of image to remove
        """

        for polygon_name in self.polygons_contained_in_img(img_name):
            self.polygons_df.loc[polygon_name, 'img_count'] -= 1

        self._graph.delete_vertex(img_name, 'imgs', force_delete_with_edges=True)
