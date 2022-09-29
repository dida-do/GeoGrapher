Cutting datasets: basic
#######################

The ``DSCutter`` classes are used for :term:`cutting` datasets.
 GeoGrapher has two general customizable
``DSCutter`` classes: :class:`geographer.cutter`. There are two helper
functions that return ``DSCutter`` s customized for the following two common
use cases:

- :ref:`cutting_every_raster_to_a_grid`
- :ref:`cutting_rasters_around_vectors`

.. _cutting_every_raster_to_a_grid:

Cutting every raster to a grid of rasters
=========================================

To create a new dataset in ``target_data_dir`` from a source dataset in
``source_data_dir`` by cutting every raster in the dataset to a grid of
rasters use the :func:`geographer.cutters.get_cutter_every_raster_to_grid`
function::

    from geographer.cutters import get_cutter_every_raster_to_grid
    cutter = get_cutter_every_raster_to_grid(
        new_raster_size=512,
        source_data_dir=<SOURCE_DATA_DIR>,
        target_data_dir=<TARGET_DATA_DIR>,
        name=<OPTIONAL_NAME_FOR_SAVING>)
    cutter.cut()

The :func:`geographer.cutters.get_cutter_every_raster_to_grid`
function returns a :class:`geographer.cutters.DSCutterIterOverRasters` instance.
The :meth:`cut` method will save the cutter to a json file in
``connector.connector_dir / <NAME>.json``.
To update the target dataset after the source dataset has grown, first read the json file
and then run :meth:`update`::

    from geographer.cutters import DSCutterIterOverRasters
    dataset_cutter = DSCutterIterOverRasters.from_json_file(<path/to/saved.json>)
    dataset_cutter.update()

.. warning::

    The ``update`` method assumes that that no vectors or raster
    rasters that remain in the target dataset have been removed from the
    source dataset.

.. _cutting_rasters_around_vectors:

Cutting rasters around vectors
====================================================

Cutting rasters around vector features (e.g. create 512 x 512 pixel
cutouts around vector features from 10980 x 10980 Sentinel-2 tiles)::

    from geographer.cutters import get_cutter_rasters_around_every_vector
    cutter = get_cutter_rasters_around_every_vector(
        source_data_dir=<SOURCE_DATA_DIR>,
        target_data_dir=<TARGET_DATA_DIR>,
        name=<OPTIONAL_NAME_FOR_SAVING>
        new_raster_size: Optional[RasterSize]
        new_raster_size=512,
        target_raster_count=2,
        mode: "random")
    cutter.cut()

The :func:`geographer.cutters.get_cutter_rasters_around_every_vector`
function returns a :class:`geographer.cutters.DSCutterIterOverVectors` instance.
The :meth:`cut` method will save the cutter to a json file in
``connector.connector_dir / <NAME>.json``.
To update the target dataset after the source dataset has grown, first read the json file
and then run :meth:`update`::

    from geographer.cutters import DSCutterIterOverVectors
    dataset_cutter = DSCutterIterOverVectors.from_json_file(<path/to/saved.json>)
    dataset_cutter.update()

.. warning::

    The ``update`` method assumes that that no vectors or rasters that remain in the target dataset have been removed from the source dataset.

