Cutting datasets: advanced
##########################

GeoGrapher provides two general templates for cutting datasets:

- iterating over vector features using
  :class:`geographer.cutters.DSCutterIterOverVectors`
- iterating over rasters using
  :class:`geographer.cutters.DSCutterIterOverRasters`

As described below, both depend on various components.
Choosing different components allows for customization.

.. note::

    All ``DSCutters`` operate on two datasets: a source and a target dataset.
    At the moment in-place operations are not supported.

Iterating over vector vectors
++++++++++++++++++++++++++++++

Desription
~~~~~~~~~~

Cutting of datasets by iterating over vector features is accomplished by the
:class:`geographer.cutters.DSCutterIterOverVectors` class. An instance of this
class is initialized with the following arguments:

    - ``name``: A name used for saving the ``DSCutter``.
    - ``source_data_dir``: The source data directory.
    - ``target_data_dir``: The target data directory.
    - ``bands``: An optional dict containing the bands to be selected.
      See :ref:`bands_dict1`.
    - ``vector_filter_predicate``: A ``VectorFilterPredicate`` used
      for filtering the vector features.
    - ``raster_selector``: A ``RasterSelector`` for selecting the rasters
      in the source dataset to create cutouts from.
    - ``raster_cutter``: A ``SingleRasterCutter`` for cutting the selected rasters.
    - ``label_maker``: An optional ``LabelMaker`` (see :doc:`label_makers`)
      for generating labels for the cutouts.

The ``cut`` method of ``DSCutterIterOverVectors`` creates a new dataset
(the *target_dataset*) and then calls the ``update`` method. The ``update``
method does the following:

- Add all vector features from the source dataset to the target dataset.
- Iterate over the vector features. In each iteration,
    - use the ``vector_filter_predicate`` to decide whether to create one
      or more new cutouts in the target dataset for the vector feature,
    - select one or several rasters for the vector feature using the ``raster_selector``,
    - create cutouts from the selected rasters using the ``raster_cutter``,
    - update the ``cut_rasters`` dict, which contains vector features as keys
      and for each vector feature a list of rasters in the source dataset
      from which cutouts were created for the vector feature.
- Save the ``DSCutterIterOverVectors`` to a ``<name>.json`` file
  in the target connector's ``connector_dir``.

Example
~~~~~~~

Defining a ``vector_filter``
-------------------------------------

Assume our source dataset contains vector features from around the world and that
the source dataset's vector features have a 'climate_zone' attribute (i.e.
a 'climate_zone' column in the source connector's ``vectors`` attribute).
Suppose you only want to cut vector features which are located in an area of interest
and whose 'climate_zone' is 'tropical'. You can use the ``GeomFilterRowCondition`` as
our ``vector_filter``::

    from geographer.vector_filter_predicate import GeomFilterRowCondition

    # Polygon describing area of interest
    aoi: shapely.geometry.Polygon = ...

    def row_series_predicate(series: GeoSeries):
        return series.loc['geometry'].within(aoi) \
               and series['climate_zone'] == 'tropical'

    my_vector_filter = FilterVectorByRowCondition(
        row_series_predicate=row_series_predicate,
        mode='source_connector'
    )

Defining a raster_cutter
-------------------------

To create cutouts around for each vector features, with the bounding boxes of the
cutout chosen at random subject to the constraint that it contains the vector
feature, use the
``SingleRasterCutterAroundVector``::

    from geographer.cutters import SingleRasterCutterAroundVector

    my_raster_cutter = SingleRasterCutterAroundVector(
        mode="random",
        new_raster_size=512,
    )

If a vector feature is too large to be contained in a cutout of size 512, a grid
of several cutouts jointly containing the vector feature will be cut.

Defining a ``raster_selector``
------------------------------

Suppose that for a vector feature you want to randomly select any two rasters in
the source dataset containing the vector features. This can be achieved with::

    from geographer.cutters.raster_selector import RandomRasterSelector
    my_raster_selector = RandomRasterSelector(target_raster_count=2)

.. note::

    When updating, the ``RandomRasterSelector`` will only consider rasters
    not previously cut for a vector feature.

Defining a ``label_maker`` (recommended)
----------------------------------------

If your datasets include labels you should define the optional ``label_maker``::

    from geographer.label_makers import SegLabelMakerCategorical
    my_label_maker = SegLabelMakerCategorical()

See :doc:`label_makers` for more details on making labels.

.. _bands_dict1:

Defining a ``bands`` dict (optional)
------------------------------------

.. warning::

    Be careful about the different indexing conventions used in rasterio
    (first index is 1) and numpy (indices start at 0). The cutting methods
    on GeoTiffs operate on GeoTiffs, for which ``GeoGrapher`` uses rasterio,
    so the rasterio indexing convention should be followed.

You can select the bands to extract from the source dataset using the optional
``bands`` argument. ``bands`` should contain the ``Connector`` classes raster
data directory attribute names as keys (e.g. 'rasters_dir' and, for segmentation
problems, 'labels_dir') and a list of bands to extract::

    bands = {
        'rasters_dir': [1,2,3],
        'labels_dir': [1]
    }

If ``bands`` is not given or a key is missing, all bands will be extracted.

Putting it all together: cutting
---------------------------------

::

    from geographer.cutters import DSCutterIterOverVectors

    dataset_cutter = DSCutterIterOverVectors(
        name="my_cutter",
        source_data_dir=<PATH/TO/SOURCE/DATA_DIR>,
        target_data_dir=<PATH/TO/TARGET/DATA_DIR>,
        bands=my_bands,
        vector_filter_predicate=my_vector_filter_predicate,
        raster_selector=my_raster_selector,
        raster_cutter=my_raster_cutter,
        label_maker=my_label_maker
    )

    dataset_cutter.cut()

After cutting, the ``DSCutterIterOverVectors`` will automatically be saved as
``target_connector.connector_dir / <name>.json``.

Updating the target dataset:
----------------------------

To update the target dataset after the source dataset has grown, use the
following::

    from geographer.cutters import DSCutterIterOverVectors
    dataset_cutter = DSCutterIterOverVectors.from_json_file(<path/to/saved.json>)
    dataset_cutter.update()

.. note::

    To unpack the JSON representation, the :meth:`from_json_file` method needs
    a symbol table mapping the class names to the class constructors. To convert
    a json representation of custom classes you wrote yourself, you'll need to
    extend the symbol table using the optional `constructor_symbol_table` argument.

Iterating over rasters
++++++++++++++++++++++

Description
~~~~~~~~~~~

Cutting of datasets by iterating over rasters is accomplished by the
:class:`geographer.cutters.DSCutterIterOverRasters` class.
An instance is initialized with the following arguments:

    - ``name``: A name used for saving the ``DSCutter``.
    - ``source_data_dir``: The source data directory.
    - ``target_data_dir``: The target data directory.
    - ``bands``: An optional dict containing the bands to be selected.
       See :ref:`bands_dict2`.
    - ``raster_filter_predicate``: A ``RasterFilterPredicate`` used for selecting
      rasters from which cutouts are to be cut.
    - ``raster_cutter``: A ``SingleRasterCutter`` for cutting the rasters.
    - An optional ``LabelMaker`` (see :doc:`label_makers`) for generating
      labels for the cutouts.

The ``cut`` method of ``DSCutterIterOverVectors`` creates a new dataset
(the *target_dataset*) and then calls the ``update`` method.
The ``update`` method does the following:

- Add all vector features from the source dataset to the target dataset.
- Iterate over the rasters. In each iteration:
    - use the ``raster_filter_predicate`` to decide whether to create one
      or more new cutouts in the target dataset for the vector feature,
    - create cutouts from the the selected rasters using the ``raster_cutter``,
    - record from which rasters in the source dataset cutouts were created
      in the ``cut_rasters`` list,
- Save the ``DSCutterIterOverRasters`` as a ``<name>.json`` file in the
  target connector's ``connector_dir``.

Example
~~~~~~~

Defining a ``raster_filter_predicate``
--------------------------------------

Suppose you want to select rasters that

- were taken between 10am and 4pm, and
- contain at least 3 vector features.

You can write a custom ``RasterFilterPredicate`` to do this::

    from geographer.cutters import RasterFilterPredicate

    class MyRasterFilterPredicate(RasterFilterPredicate):
        def __call__(
            self,
            raster_name: str,
            target_assoc: Connector,
            new_raster_dict: dict,
            source_assoc: Connector,
            cut_rasters: List[str],
        ) -> bool:

        local_timestamp: str = rasters.loc[raster_name, 'local_timestamp']
        local_time = datetime.strptime(
            local_timestamp,
            '%m/%d/%y %H:%M:%S'
        ).time()
        local_time_within_window = local_time >= datetime.time(10)\
            and local_time <= datetime.time(16)

        vector_count = len(
            source_assoc.vectors_contained_in_raster(raster_name)
        )

        return local_time_within_window and vector_count >= 3

    my_raster_filter_predicate = MyRasterFilterPredicate()

Defining a raster_cutter
-------------------------

Suppose you want to cut every selected raster to a grid of rasters.
You can use the ``SingleRasterCutterToGrid``
(:class:`geographer.cutters.single_raster_cutter_grid.SingleRasterCutterToGrid`)
to do this::

    from geographer.cutters.single_raster_cutter_grid import SingleRasterCutterToGrid
    my_raster_cutter = SingleRasterCutterToGrid(new_raster_size=512)

Defining a ``label_maker`` (recommended)
----------------------------------------

If your datasets include labels you should define the optional ``label_maker``::

    from geographer.label_makers import SegLabelMakerCategorical
    my_label_maker = SegLabelMakerCategorical()

See :doc:`label_makers` for more details on making labels.

.. _bands_dict2:

Defining a ``bands`` dict (optional)
------------------------------------

This is done as in the case of iterating over rasters, see :ref:`bands_dict1`.

Putting it all together: cutting
---------------------------------

::

    from geographer.cutters import DSCutterIterOverRasters

    dataset_cutter = DSCutterIterOverRasters(
        name="my_cutter",
        source_data_dir=<PATH/TO/SOURCE/DATA_DIR>,
        target_data_dir=<PATH/TO/TARGET/DATA_DIR>,
        bands=my_bands,
        raster_filter_predicate=my_raster_filter_predicate,
        raster_cutter=my_raster_cutter,
        label_maker=my_label_maker
    )

    dataset_cutter.cut()

After cutting, the ``DSCutterIterOverRasters`` will automatically be
saved to ``target_connector.connector_dir / <name>.json``.

Updating the target dataset:
----------------------------

Updating the target dataset after the source dataset has grown::

    from geographer.cutters import DSCutterIterOverRasters

    dataset_cutter = DSCutterIterOverRasters.from_json_file(
        <path/to/saved.json>
    )

    dataset_cutter.update()

.. note::

    To unpack the json representation, the :meth:`from_json_file` method needs
    a symbol table mapping the class names to the class constructors. To convert
    a JSON representation of custom classes you wrote yourself, you'll need to
    extend the symbol table using the optional `constructor_symbol_table` argument.
