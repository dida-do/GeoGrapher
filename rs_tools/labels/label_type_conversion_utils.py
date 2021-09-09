from geopandas import GeoDataFrame
from rs_tools.utils.utils import deepcopy_gdf


def convert_polygons_df_soft_cat_to_cat(polygons_df: GeoDataFrame) -> GeoDataFrame:
    """Take a polygons_df in soft-categorical format and return a copy converted to categorical format"""

    new_polygons_df = deepcopy_gdf(polygons_df)

    # make 'type' column
    new_polygons_df['type'] = new_polygons_df[[col for col in new_polygons_df.columns if col[:15] == 'prob_seg_class_']].idxmax(axis='columns').apply(lambda x: x[15:])
    
    # drop 'prob_seg_class_[seg_class]' cols
    new_polygons_df.drop([col for col in new_polygons_df.columns if col[:15] == 'prob_seg_class_'], axis='columns', inplace=True)

    return new_polygons_df