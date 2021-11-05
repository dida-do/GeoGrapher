# remote-sensing-tools: the ImgPolygonAssociator

Organize, build up, and handle (e.g. generate labels, cut, convert GeoTiffs to numpy files) remote sensing segmentation datasets consisting of polygons, images and segmentation labels while always keeping track of which polygons are contained in or intersect which images and vice versa.

# How to contribute
Please let me know if you use this repo! If you want to contribute you can add functionality, install sphinx, write a Makefile, create a requirements.txt, review code, clean up code, write documentation, submit issues, or just let me know what needs improving or what you found confusing.

# Installation
At the moment, you have to clone the git repo and run `pip install -e .`. 
Whenever you get an import error you'l have to import the missing libraries into your
environment by hand. Sorry, no (working) requirements.txt or Makefile for now (feel free to change this!).

# Documentation
Please help me out by installing sphinx!

# Basic usage

Everything is built around the associator class.
An associator organizes a segmentation dataset consisting of images and
segmentation labels (raster data) and polygons (vector data). 
The associator itself consists of two geopandas GeoDataFrames
polygons_df and imgs that contain tabular information about
the polygons and images. 

    assoc.imgs_df
(returns geodataframe)

    assoc.polygons_df
(returns geodataframe)

The graph can be queried with the `imgs_containing_polygon`, `imgs_intersecting_polygon`, `polygons_contained_in_img`, `polygons_intersecting_img` methods:

    assoc.imgs_containing_polygon(polygon_name)
(returns list of images containing polygon)

## Creating an associator from scratch

    from rs_tools import ImgPolygonAssociator as IPA
    assoc = IPA.from_scratch(
        data_dir=<DATA_DIR>,
        segmentation_classes = ['seg_class1', 'seg_class2'],
        label_type='categorical')

## Initializing an existing associator

    assoc = IPA.from_data_dir(data_dir=<DATA_DIR>)

## Adding or deleting polygons or images to polygons_df or imgs_df

NEVER use any other methods to add/drop polygons or images
to the imgs_df or polygons_df dataframes! You'll mess up the graph.

    assoc.add_to_polygons_df(new_polygons_df)
    assoc.drop_polygons(list_of_polygons)
and similarly for images.

## Downloading images (basic usage)

One can download either Sentinel-2 or JAXA DEM data. Easily extendable to other data sources.

    assoc.download_imgs(
        polygon_names=optional_list_of_polygon_names_you_want_to_download_imgs_for,
        add_labels=True,
        downloader='sentinel2',
        producttype='L2A',
        max_percent_cloud_coverage=10,
        resolution=10, 
        date=(“NOW-1DAY”, “NOW”),
        area_relation='Contains')
All the downloader (`'sentinel-2'` or `'jaxa'`) specific arguments will be remembered and used as defaults after the first time the method is called. When dealing with 'large' polygons consider using the
`filter_out_polygons_contained_in_union_of_intersecting_imgs` argument (see docstring).

## Create new datasets from existing ones by cutting the images and labels (basic usage)

    assoc.cut_every_img_to_grid(
            target_data_dir = TARGET_DATA_DIR,
            new_img_size = 1024)

    assoc.cut_imgs_around_every_polygon(
            target_data_dir = TARGET_DATA_DIR,
            new_img_size = 1024
            target_img_count = 1,
            mode = 'random')
If these methods are not exactly what you need you can roll your own 
using the general purpose higher order methods `create_or_update_dataset_iter_over_polygons` or `create_or_update_dataset_iter_over_imgs`.

## Create a new dataset from an existing one by combining or removing segmentation classes

Suppose the associator assoc's dataset has the following segmentation classes:

'ct', 'ht', 'wr', 'h', 'pt', 'ig', 'bg'

Then,

    assoc.create_dataset_by_combining_or_removing_seg_classes(
        target_data_dir=TARGET_DATA_DIR,
        seg_classes=[['ct', 'ht'], 'wr', ['h']])
will create a new dataset in TARGET_DATA_DIR in which only the polygons belonging
to the 'ct', 'ht', 'wr', and 'h' (or having a non-zero probability of belonging to these
classes if the labels are soft-categorical) will be retained, and the classes 'ct' and 'ht' will be combined to a class 'ct+ht'. 

## Creating a dataset of .npy files from an existing dataset of GeoTiffs

    assoc.convert_dataset_from_tif_to_npy(target_data_dir)

## Updating an existing dataset created from a source dataset

**Warning: Hasn't been tested at all!**

    assoc.update()

## TODO: Other methods:

converting label types, 



