TOC: 

1) Folder structure/overview of contents (outdated)
2) Definitions


Folder structure/overview of contents (outdated)

    (public API:)
    * Connector - class to organize/handle datasets    
    * convert_assoc_dataset_tif2numpy - create a new dataset/associator from an old one by converting GeoTiffs to numpy .npy files.
    * rasters_from_tif_dir - create an rasters from a folder of rasters. Useful if the associator data has been lost for some reason. 
    * empty_assoc_same_format_as - creates an empty associator of the same format (i.e. index and column names and types of the associator's rasters and polygons_df GeoDataFrames)
    * empty_polygons_df - Create an empty polygons_df.
    * empty_rasters - Create an empty rasters. 

    - cut. Functions to 'cut' a dataset/associator, i.e. create a new dataset/associator by cutting each raster into small rasters.
        Public API:
        * new_dataset_one_small_raster_for_each_polygon. Creates a new _target dataset_ from a _source dataset_ by cutting one small square raster around each polygon.
        * update_dataset_one_small_raster_for_each_polygon. Updates a _target dataset_ from an updated (i.e. containing new polygons) source dataset.
        
    - utils. Various functions that are of use when handling datsets.
        * convert_assoc_dataset_tif2numpy. Converts a dataset/associator of GeoTiffs to .npys.
        * empty_rasters. Creates an empty rasters to be used by a newly created associator.

    - graph. Code that defines a bipartite graph. Needed internally by the associator class. Since the whole idea of the associator is that one never has to explicitly manipulate the graph you can ignore this unless you are trying to change the internal logic of the associator.

    - tests. Pytest test suite.


Definitions:

    - *Polygon*: vector polygon defining either an object of interest belonging to a segmentation class, or .... FINISH

    - *Label* vs *(Segmentation) Mask* vs pixel label

    - *Dataset*: A remote sensing segmentation _dataset_ consists of (vector) polygons demarkating the areas of interest/segmentation targets (e.g. mines, solar panels, or objects), rasters, segmentation (raster) labels generated from the polygons, tabular information about the rasters and polygons, as well as the information which polygons are contained in or intersect with which rasters. This data is organized in a _data directory_ (see next bullet point):

    - *Data directory* A dataset is contained in a _data directory_. A data directory is a directiry containing subfolders named rasters and labels containing remote sensing rasters and labels, usually in GeoTiff or .npy format. The information which polygons are contained in or intersect which rasters (implicitly defining a _bipartite graph_ - WRITE SOMETHING ABOUT WHY WE CARE ABOUT THIS) as well as the tabular information about the rasters and polygon labels is contained in an _associator_ which is the data structure organizing a dataset. Each instance of the associator class corresponds to a data directory, and therefore to a dataset. The two main attributes of the associator are the GeoDataFrames rasters and polygons_df containing the tabular information about the rasters and polygons. Any manipulations of these dataframes that change the implicit graph structure defined by the relations which polygons are contained in or intersects which rasters (i.e. adding or deleting rasters or polygons, or changing the polygon geometries) !MUST! be done using associator methods rather than by direct manipulation of the dataframes. The associator will then make sure that the implicit graph structure reflects the contents of the dataset/dataframes. In addition to the rasters and labels, the data directory will contain the rasters and polygons_df as well as other information needed by the associator.

    - A *soft-categorical* (pixel) label is one for which there are channels for each segmentation class
    and the value at a given position in a given channel is the probability that the pixel
    at that position is classified as belonging to the corresponding segmentation
    class.

    - A *categorical* GeoTiff pixel label (i.e. one channel rasters
    where each pixel is an integer corresponding to either the background
    or a segmentation class, 0 indicating the background class, and k=1,2, ...
    indicating the k-th entry (starting from 1) of the segmentation_classes
    parameter of the associator) in the data directory's labels subdirectory
    for the GeoTiff raster raster_name in the rasters subdirectory.

    - A *onehot* label is a soft-categorical one for which all probabilities are 0 or 1.


