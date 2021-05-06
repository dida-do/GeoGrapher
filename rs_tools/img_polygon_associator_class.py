"""
Abstract class defining the structure of the ImgPolygonAssociator.
"""

class ImgPolygonAssociatorClass:

    def rectangle_bounding_img(self, img_name):
        raise NotImplementedError

    def have_img_for_polygon(self, polygon_name):
        raise NotImplementedError

    def polygons_intersecting_img(self, img_name):
        raise NotImplementedError

    def imgs_intersecting_polygon(self, polygon_name):
        raise NotImplementedError

    def polygons_contained_in_img(self, img_name):
        raise NotImplementedError

    def imgs_containing_polygon(self, polygon_name):
        raise NotImplementedError

    def does_img_contain_polygon(self, img_name, polygon_name):
        raise NotImplementedError

    def is_polygon_contained_in_img(self, polygon_name, img_name):
        raise NotImplementedError

    def does_img_intersect_polygon(self, img_name, polygon_name):
        raise NotImplementedError

    def does_polygon_intersect_img(self, polygon_name, img_name):
        raise NotImplementedError

    def integrate_new_polygons_df(self, new_polygons_df, force_overwrite=False):
        raise NotImplementedError

    def integrate_new_imgs_df(self, new_imgs_df):
        raise NotImplementedError

    def drop_polygons(self, polygon_names):
        raise NotImplementedError

    def drop_imgs(self, img_names):
        raise NotImplementedError

    def download_missing_imgs_for_polygons_df(self, polygons_df, **kwargs):
        raise NotImplementedError
