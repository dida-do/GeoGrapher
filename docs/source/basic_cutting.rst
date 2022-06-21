Cutting Datasets: Basic
#######################

.. todo::

    - make it clear we are creating new datasets from old ones and not operating in place.
    - other options.
    - saving and loading etc.
    - add entry for 'cutting' in gloassary

Use the ``DSCutter`` classes to create a new dataset from an existing source dataset by cutting the source dataset.
GeoGrapher has two general customizable ``DSCutter`` classes: :class:`geographer.cutter`.
There are two helper functions that return ``DSCutter``s customized for the following two common use cases:

- :ref:`cutting_every_img_to_a_grid`
- :ref:`cutting_images_around_vector_features`

.. _cutting_every_img_to_a_grid:

Cutting Every Image To A Grid of Images
=======================================

To create a new dataset in ``target_data_dir`` from a source dataset in ``source_data_dir`` by cutting every image in the dataset to a grid of images use the :func:`geographer.cutters.get_cutter_every_img_to_grid` function::

    from geographer.cutters import get_cutter_every_img_to_grid
    cutter = get_cutter_every_img_to_grid(
        new_img_size=512,
        source_data_dir=<SOURCE_DATA_DIR>,
        target_data_dir=<TARGET_DATA_DIR>,
        name=<OPTIONAL_NAME_FOR_SAVING>)
    cutter.cut()

Updating the ``target_data_dir`` after the ``source_data_dir`` has
grown since it was cut::

    cutter.update()

.. warning::

    The ``update`` method assumes that that no vector_features or raster images that remain in the target dataset have been removed from the source dataset.

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

Updating the ``target_data_dir`` after the ``source_data_dir``
has grown (more images or vector features) since it was cut::

    cutter.update()

.. warning::

    The ``update`` method assumes that that no vector_features or raster images that remain in the target dataset have been removed from the source dataset.

