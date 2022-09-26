Cutting Datasets: Basic
#######################

The ``DSCutter`` classes are used for :term:`cutting` datasets.
 GeoGrapher has two general customizable
``DSCutter`` classes: :class:`geographer.cutter`. There are two helper
functions that return ``DSCutter`` s customized for the following two common
use cases:

- :ref:`cutting_every_img_to_a_grid`
- :ref:`cutting_images_around_vector_features`

.. _cutting_every_img_to_a_grid:

Cutting Every Image To A Grid of Images
=======================================

To create a new dataset in ``target_data_dir`` from a source dataset in
``source_data_dir`` by cutting every image in the dataset to a grid of
images use the :func:`geographer.cutters.get_cutter_every_img_to_grid`
function::

    from geographer.cutters import get_cutter_every_img_to_grid
    cutter = get_cutter_every_img_to_grid(
        new_img_size=512,
        source_data_dir=<SOURCE_DATA_DIR>,
        target_data_dir=<TARGET_DATA_DIR>,
        name=<OPTIONAL_NAME_FOR_SAVING>)
    cutter.cut()

The :func:`geographer.cutters.get_cutter_every_img_to_grid`
function returns a :class:`geographer.cutters.DSCutterIterOverImgs` instance.
The :meth:`cut` method will save the cutter to a json file in
``connector.connector_dir / <NAME>.json``.
To update the target dataset after the source dataset has grown, first read the json file
and then run :meth:`update`::

    from geographer.cutters import DSCutterIterOverImgs
    dataset_cutter = DSCutterIterOverImgs.from_json_file(<path/to/saved.json>)
    dataset_cutter.update()

.. warning::

    The ``update`` method assumes that that no vector_features or raster
    images that remain in the target dataset have been removed from the
    source dataset.

.. _cutting_images_around_vector_features:

Cutting Images Around Vector Features
====================================================

Cutting images around vector features (e.g. create 512 x 512 pixel
cutouts around vector features from 10980 x 10980 Sentinel-2 tiles)::

    from geographer.cutters import get_cutter_imgs_around_every_feature
    cutter = get_cutter_imgs_around_every_feature(
        source_data_dir=<SOURCE_DATA_DIR>,
        target_data_dir=<TARGET_DATA_DIR>,
        name=<OPTIONAL_NAME_FOR_SAVING>
        new_img_size: Optional[ImgSize]
        new_img_size=512,
        target_img_count=2,
        mode: "random")
    cutter.cut()

The :func:`geographer.cutters.get_cutter_imgs_around_every_feature`
function returns a :class:`geographer.cutters.DSCutterIterOverFeatures` instance.
The :meth:`cut` method will save the cutter to a json file in
``connector.connector_dir / <NAME>.json``.
To update the target dataset after the source dataset has grown, first read the json file
and then run :meth:`update`::

    from geographer.cutters import DSCutterIterOverFeatures
    dataset_cutter = DSCutterIterOverFeatures.from_json_file(<path/to/saved.json>)
    dataset_cutter.update()

.. warning::

    The ``update`` method assumes that that no vector_features or raster images that remain in the target dataset have been removed from the source dataset.

