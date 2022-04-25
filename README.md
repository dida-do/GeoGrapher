# GeoGrapher

The GeoGrapher library can organize, build up, and handle (e.g. generate labels, cut, convert GeoTiffs
to numpy files) *object-centric* remote sensing computer vision datasets
consisting of vector features, raster images and ML task-specific labels (e.g.
segmentation pixel-labels or object-detection bounding-box labels) while always
keeping track of the containment or intersection relations between vector
features and raster images.

## What is GeoGrapher good for? Object-centric remote sensing
Most remote sensing tasks are ‘area-centric’ in that they are defined by an
area of interest for which data is acquired before being labeled. Typically,
the objects as individual entities are not really of much interest. An
‘object-centric’ remote sensing computer vision task is one in which the
objects do not necessarily lie in a predefined area of interest and are labeled
first before data is acquired and/or are of individual interest.

## What else is GeoGrapher good for? Highly configurable cutting of remote sensing datasets

The library can also "cut" datasets according to highly configurable "queries". As an example, say we want to segment buildings belonging to different building types (so this a multiclass segmentation problem) and that we have a ‘raw’ source dataset of 100km by 100km Sentinel-2 GeoTiff images. Suppose we want to create a new dataset of smaller numpy images that we can use for training a neural network and for some reason we want 3 distinct small images for each building belonging to some segmentation class building_type_1 two of which should be from before 2012 (oversampling of rare classes) but only 1 image for each building of segmentation class building_type_1 with no restriction on the date the image was taken and no images for any of the other building types (e.g. bridges). The cutter classes defined in the repo can then cut a new dataset from the source dataset according to this specification.

# Installation
TODO: fix before open-sourcing! At the moment, you have to clone the git repo
and run `pip install -e .`. Whenever you get an import error you'l have to
import the missing libraries into your environment by hand. Sorry, no (working)
requirements.txt or Makefile for now (feel free to change this!).

# Documentation
TODO: remove before open-sourcing! Please help me out by installing sphinx!

# Getting started
## The Connector Class

The library is built around the Connector class which connects vector features
and raster images by keeping track of the bipartite graph defined by the
containment or intersection relationships between vector features and raster
images. The most important attributes are the raster_imgs and vector_features
GeoDataFrames. These contain the geometries of the vector features and the
bounding boxes of the images respectively along with other tabular information
(e.g. classes for the machine learning task for the vector features or
meta-data).

    connector.raster_imgs
(returns geodataframe)

    connector.vector_features
(returns geodataframe)

The `img_count` column in `connector.vector_features` contains the number of
images in `raster_imgs` that fully contain the vector feature.

The graph can be queried with the `imgs_containing_vector_feature`,
`imgs_intersecting_vector_feature`, `vector_features_contained_in_img`,
`vector_features_intersecting_img` methods:

    connector.imgs_containing_vector_feature(feature_name)
(returns list of images containing vector feature)

The actual images can be found in:

    connector.images_dir
(returns `pathlib.Paths` to images, usually data_dir / 'images')

## Creating an (empty) connector from scratch

    from rs_tools import Connector
    connector = Connector.from_scratch(
        data_dir=<DATA_DIR>)

## TODO: Move to 'Multiclass Vision Tasks' section
        task_feature_classes=['building', 'bridge'])
The optional `task_feature_classes` argument defines the allowable classes for a
multi-class computer vision task (e.g. segmentation).

## Saving a connector
The `save` method will save the associator to `data_dir`.

    connector.save()
Unfortunately geopanpdas can not save empty GeoDataFrames.

## Initializing an existing connector

    connector = Connector.from_data_dir(data_dir=<DATA_DIR>)

## Adding or dropping vector features

Add or drop vector features to/from the connector:

    connector.add_to_vector_features(new_vector_features)
    connector.drop_vector_features(list_of_vector_features)

The names of the `new_vector_features` in the GeoDataFrame's index need to be unique. If one supplies the optional `label_maker` argument any labels in the target dataset that need to be updated (e.g. if the source dataset contains new vector features intersecting already existing images in the target dataset) will be recreated.


**NEVER add/drop vector features or images by modifying the vector_features
(or raster_imgs) attributes by hand! Use the connector methods described here
to ensure the graph encoding the containment/intersectiob relations is always
correct.**

TODO: There has to be a `'type'` column specifying each vector features label
class.

## Adding or dropping raster images

Add or drop raster images to/from the connector:

    connector.add_to_raster_imgs(new_raster_imgs)
    connector.drop_raster_imgs(list_of_raster_img_names)

To add/drop raster images, just replace `vector_features` with `raster_imgs` in
the methods.  Again, if one supplies the optional `label_maker` argument any labels in the target dataset that need to be updated (e.g. if the source dataset contains new vector features intersecting already existing images in the target dataset) will be recreated.

The connector only knows about the `raster_imgs` GeoDataFrame,
not whether the images actually exist in the `connector.images_dir` directory.
You can use the `raster_imgs_from_tif_dir` function in `utils/utils.py` to
create a `new_raster_imgs` GeoDataFrame from a directory of GeoTiffs you can
add to the connector.

**NEVER add/drop vector features or images by modifying the vector_features
(or raster_imgs) attributes by hand! Use the connector methods described here
to ensure the graph encoding the containment/intersectiob relations is always
correct.**

## Downloading raster images

The ImgDownloaderForVectorFeatures can download images targeting a number of images per vector feature. It can interface with different download sources (Copernicu Open Access Hub for Sentinel imagery, JAXA for ALOS DEM data), and can easily be extended to other data sources by writing custom `DownloaderForSingleSingleVectorFeature` and `Processor` classes.

    from rs_tools.downloaders import (
        ImgDownloaderForVectorFeatures,
        SentinelDownloaderForSingleVectorFeature,
        Sentinel2Processor)
    downloader = ImgDownloaderForVectorFeatures(
        download_dir=<DOWNLOAD_DIR>,
        downloader_for_single_feature=SentinelDownloaderForSingleVectorFeature(),
        download_processor=Sentinel2Processor(),
    )
    downloader.download(
        connector=my_connector,
        feature_names=optional_list_of_vector_feature_names,
        target_img_count=2,
        producttype='L2A',
        max_percent_cloud_coverage=10,
        resolution=10,
        date=(“NOW-10DAYS”, “NOW”),
        area_relation='Contains'))

The image count per vector feature is updated after every download, so that unnecessary downloads and an imbalance in the dataset due to clustering of nearby vector features are avoided.

One can supply default values for dataset/data source specific `download` arguments (e.g. producttype, max_percent_cloud_coverage) in the `ImgDownloaderForVectorFeatures`'s `kwarg_defaults` argument. The downloaders are serializable:

    downloader.save()
(Will save the downloader to the connector's data_dir)

## Creating labels

The `LabelMaker` classes create labels from the vector features for computer vision tasks. Currently, label creation for multiclass segmentation tasks for both 'categorical' and 'soft-categorical' (i.e. there is a probability distribution over the classes for each vector feature) label-types are supported.

    from rs_tools.label_makers import SegLabelMakerCategorical
    label_maker = SegLabelMakerCategorical()
    label_maker.make_labels(
        connector=<your_connector>,
        img_names=<optional image names>
    )

Depending on the label-type the LabelMakers assume the connector's `vector_features` GeoDataFrame has certain columns. For the categorical case this is a "type" column containing the classes of the vector features and for the soft-categorical case there should be prob_seg_class<CLASS_NAME> columns containing the probabilities of a vector feature belonging to a class.

## Basic Cutting: Cutting every image to a grid of images

To cut every image in the dataset to a grid of images:

    from rs_tools.cutters import DSCutterEveryImgToGrid
    cutter = DSCutterEveryImgToGrid(
        new_img_size=512,
        source_data_dir=<SOURCE_DATA_DIR>,
        target_data_dir=<TARGET_DATA_DIR>,
        name=<OPTIONAL_NAME_FOR_SAVING>)
    cutter.cut()

To update the `target_data_dir` after the the `source_data_dir` has grown since it was cut:

    cutter.update()

## Basic Cutting: Cutting images around vector features

To cut images around vector features (e.g. create 512x512 pixel cutouts around vector features from 10980x10980 Sentinel-2 tiles.)

    from rs_tools.cutters import DSCutterImgsAroundEveryFeature
    cutter = DSCutterImgsAroundEveryFeature(
        source_data_dir=<SOURCE_DATA_DIR>,
        target_data_dir=<TARGET_DATA_DIR>,
        name=<OPTIONAL_NAME_FOR_SAVING>
        new_img_size: Optional[ImgSize]
        new_img_size=512,
        target_img_count=2,
        mode: "random")
    cutter.cut()

Again, to update the `target_data_dir` after the the `source_data_dir` has grown (more images or vector features) since it was cut one can use:

    cutter.update()

## Advanced Cutting: Iterating Over Vector Features or Images

One can use the `DSCutterIterOverFeatures` and `DSCutterIterOverImgs` classes for more customized cutting of datasets. The `DSCutterIterOverFeatures` method iterates over vector features in the source dataset, filters them according to a FeatureFilterPredicate, selects images containing the feature according to an ImgSelector, cuts each selected image using a SingleImageCutter, and updates the target connector after each image cut. Plug in your favorite components (you can easily write your own if you need to) to customize the cutting process.

    from rs_tools.cutters import DSCutterIterOverFeatures

    single_img_cutter: SingleImgCutter = <single_image_cutter>
    img_selector: ImgSelector = <image_selector>",
    feature_filter_predicate: FeatureFilterPredicate = <predicate_to_filter_vector_features>

    cutter = DSCutterIterOverFeatures(
        source_data_dir=<SOURCE_DATA_DIR>,
        target_data_dir=<TARGET_DATA_DIR>,
        name=<OPTIONAL_NAME_FOR_SAVING>,
    )
    cutter.cut()

Again, to update the `target_data_dir` after the the `source_data_dir` has grown (more images or vector features) since it was cut one can use:

    cutter.update()

If one supplies the optional `label_maker` argument the update method will recreate the labels in the target dataset that need to be updated (e.g. if the source dataset contains new vector features intersecting already existing images in the target dataset).

## Combining and/or Removing ML Task Vector Feature Classes
TODO: to be written

## Converting a dataset of GeoTiffs to .npy
TODO: to be written
## TODO: other stuff
TODO: to be written

# How to contribute
Please send me feedback if you use the library! This is an open-source project,
so feel free to contribute in any other way (e.g. submitting code) as well.


