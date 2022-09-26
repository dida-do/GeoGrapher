Cutting Datasets: Advanced
##########################

GeoGrapher provides two general templates for cutting datasets:

- iterating over vector features using
  :class:`geographer.cutters.DSCutterIterOverFeatures`
- iterating over raster images using
  :class:`geographer.cutters.DSCutterIterOverImgs`

As described below, both depend on various components.
Choosing different components allows for customization.

.. note::

    All ``DSCutters`` operate on two datasets: a source and a target dataset.
    At the moment in place operations are not supported.

Iterating Over Vector Features
++++++++++++++++++++++++++++++

Desription
~~~~~~~~~~

Cutting of datasets by iterating over vector features is accomplished by the
``DSCutterIterOverFeatures`` class (:class:`geographer.cutters.DSCutterIterOverFeatures`).
A ``DSCutterIterOverFeatures`` is initialized with:

    - ``name``: a name used for saving the ``DSCutter``.
    - ``source_data_dir``: the source data directory
    - ``target_data_dir``: the target data directory
    - ``bands``: an optional dict containing the bands to be selected.
      See :ref:`bands_dict1`.
    - ``feature_filter_predicate``: a ``FeatureFilterPredicate`` used
      for filtering the vector features
    - ``img_selector``: a ``ImgSelector`` for selecting the images
      in the source dataset to create cutouts from
    - ``img_cutter``: a ``SingleImgCutter`` for cutting the selected images
    - ``label_maker``: an optional ``LabelMaker`` (see :doc:`label_makers`)
      for generating labels for the cutouts.

The ``cut`` method of ``DSCutterIterOverFeatures`` creates a new dataset
(the *target_dataset*) and then calls the ``update`` method. The ``update``
method does the following:

- Add all vector features from the source dataset to the target dataset.
- Iterate over the vector features. In each iteration:
    - use the ``feature_filter_predicate`` to decide whether to create one
      or more new cutouts in the target dataset for the vector feature
    - select one or several images for the vector feature using the ``img_selector``
    - create cutouts from the selected images using the ``img_cutter``
    - update the ``cut_imgs`` dict, which contains vector features as keys
      and for each vector feature a list of images in the source dataset
      from which cutouts were created for the vector feature
- save the ``DSCutterIterOverFeatures`` to a ``<name>.json`` file
  in the target connector's ``connector_dir``

Example
~~~~~~~

Defining a ``vector_feature_filter``
-------------------------------------

Assume our source dataset contains vector features from around the world and that
the source dataset's vector features have a 'climate_zone' attribute (i.e.
a 'climate_zone' column in the source connector's ``vector_features`` attribute).
Suppose you only want to cut vector features which are located in an area of interest
and whose 'climate_zone' is 'tropical'. You can use the ``GeomFilterRowCondition`` as
our ``vector_feature_filter``::

    from geographer.feature_filter_predicate import GeomFilterRowCondition

    aoi: shapely.geometry.Polygon = ... # polygon describing area of interest
    def row_series_predicate(series: GeoSeries):
        return series.loc['geometry'].within(aoi) and series['climate_zone'] == 'tropical'
    my_vector_feature_filter = FilterVectorFeatureByRowCondition(
        row_series_predicate=row_series_predicate,
        mode='source_connector'
    )

Defining an img_cutter
----------------------

To create cutouts around for each vector features with the bounding boxes of the
cutout chosen at random subject to the constraint that it contains the vector
feature use the
``SingleImgCutterAroundFeature``::

    from geographer.cutters import SingleImgCutterAroundFeature
    my_img_cutter = SingleImgCutterAroundFeature(
        mode="random",
        new_img_size=512,
    )

If a vector feature is too large to be contained in a cutout of size 512, a grid
of several cutouts jointly containing the vector feature will be cut.

Defining an ``img_selector``
-----------------------------

Suppose for a vector feature you want to randomly select any two images
in the source dataset containing the vector features::

    from geographer.cutters.img_selector import RandomImgSelector
    my_img_selector = RandomImgSelector(target_img_count=2)

.. note::

    When updating, the ``RandomImgSelector`` will only consider images
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
``bands`` argument. ``bands`` should contain the ``Connector`` classes image
data directory attribute names as keys (e.g. 'images_dir' and, for segmentation
problems, 'labels_dir') and a list of bands to extract::

    bands = {
        'images_dir': [1,2,3],
        'labels_dir': [1]
    }

If ``bands`` is not given or a key is missing, all bands will be extracted.

Putting It All Together: Cutting
---------------------------------

::

    from geographer.cutters import DSCutterIterOverFeatures
    dataset_cutter = DSCutterIterOverFeatures(
        name="my_cutter",
        source_data_dir=<PATH/TO/SOURCE/DATA_DIR>,
        target_data_dir=<PATH/TO/TARGET/DATA_DIR>,
        bands=my_bands,
        feature_filter_predicate=my_feature_filter_predicate,
        img_selector=my_img_selector,
        img_cutter=my_img_cutter,
        label_maker=my_label_maker
    )
    dataset_cutter.cut()

After cutting, the ``DSCutterIterOverFeatures`` will automatically be saved to
``target_connector.connector_dir / <name>.json``.

Updating The Target Dataset:
----------------------------

Updating the target dataset after the source dataset has grown::

    from geographer.cutters import DSCutterIterOverFeatures
    dataset_cutter = DSCutterIterOverFeatures.from_json_file(<path/to/saved.json>)
    dataset_cutter.update()

.. note::

    To unpack the json representation, the :meth:`from_json_file` method needs
    a symbol table mapping the class names to the class constructors. To convert
    a json representation of custom classes you wrote yourself, you'll need to
    extend the symbol table using the optional `constructor_symbol_table` argument.

Iterating Over Raster Images
++++++++++++++++++++++++++++

Description
~~~~~~~~~~~

Cutting of datasets by iterating over raster images is accomplished by the
``DSCutterIterOverImgs`` class (:class:`geographer.cutters.DSCutterIterOverImgs`).
A ``DSCutterIterOverImgs`` is initialized with:

    - ``name``: a name used for saving the ``DSCutter``.
    - ``source_data_dir``: the source data directory
    - ``target_data_dir``: the target data directory
    - ``bands``: an optional dict containing the bands to be selected.
      See :ref:`bands_dict2`.
    - ``img_filter_predicate``: a ``ImgFilterPredicate`` used for selecting
      raster images from which cutouts are to be cut
    - ``img_cutter``: a ``SingleImgCutter`` for cutting the raster images
    - an optional ``LabelMaker`` (see :ref:`here <label_makers>`) for
      generating labels for the cutouts.

The ``cut`` method of ``DSCutterIterOverFeatures`` creates a new dataset
(the *target_dataset*) and then calls the ``update`` method.
The ``update`` method does the following:

- Add all vector features from the source dataset to the target dataset.
- Iterate over the raster images. In each iteration:
    - use the ``img_filter_predicate`` to decide whether to create one
      or more new cutouts in the target dataset for the vector feature
    - create cutouts from the the selected images using the ``img_cutter``
    - record from which images in the source dataset cutouts were created
      in the ``cut_imgs`` list
- save the ``DSCutterIterOverImages`` as a ``<name>.json`` file in the
target connector's ``connector_dir``

Example
~~~~~~~

Defining a ``img_filter_predicate``
-----------------------------------

Suppose you want to select images that
- were taken between 10am and 4pm
- and contain at least 3 vector features.
You can write a custom ``ImgFilterPredicate`` to do this::

    from geographer.cutters import ImgFilterPredicate

    class MyImgFilterPredicate(ImgFilterPredicate):
        def __call__(
            self,
            img_name: str,
            target_assoc: Connector,
            new_img_dict: dict,
            source_assoc: Connector,
            cut_imgs: List[str],
        ) -> bool:

        local_timestamp: str = raster_imgs.loc[img_name, 'local_timestamp']
        local_time = datetime.strptime(
            local_timestamp,
            '%m/%d/%y %H:%M:%S'
        ).time()
        local_time_within_window = local_time >= datetime.time(10)\
            and local_time <= datetime.time(16)

        vector_feature_count = len(
            source_assoc.vector_features_contained_in_img(img_name)
        )

        return local_time_within_window and vector_feature_count >= 3

    my_img_filter_predicate = MyImgFilterPredicate()

Defining an img_cutter
----------------------

Suppose you want to cut every selected image to a grid of images.
You can use the ``SingleImgCutterToGrid``
(:class:`geographer.cutters.single_img_cutter_grid.SingleImgCutterToGrid`)
to do this::

    from geographer.cutters.single_img_cutter_grid import SingleImgCutterToGrid
    my_img_cutter = SingleImgCutterToGrid(new_img_size=512)

Defining a ``label_maker`` (recommended)
----------------------------------------

If your datasets include labels you should define the optional ``label_maker``::

    from geographer.label_makers import SegLabelMakerCategorical
    my_label_maker = SegLabelMakerCategorical()

See :doc:`label_makers` for more details on making labels.

.. _bands_dict2:

Defining a ``bands`` dict (optional)
------------------------------------

This is done as in the case of iterating over raster images, see :ref:`bands_dict1`.

Putting It All Together: Cutting
---------------------------------

::

    from geographer.cutters import DSCutterIterOverImgs
    dataset_cutter = DSCutterIterOverImgs(
        name="my_cutter",
        source_data_dir=<PATH/TO/SOURCE/DATA_DIR>,
        target_data_dir=<PATH/TO/TARGET/DATA_DIR>,
        bands=my_bands,
        img_filter_predicate=my_img_filter_predicate,
        img_cutter=my_img_cutter,
        label_maker=my_label_maker
    )
    dataset_cutter.cut()

After cutting, the ``DSCutterIterOverImgs`` will automatically be
saved to ``target_connector.connector_dir / <name>.json``.

Updating The Target Dataset:
----------------------------

Updating the target dataset after the source dataset has grown::

    from geographer.cutters import DSCutterIterOverImgs
    dataset_cutter = DSCutterIterOverImgs.from_json_file(<path/to/saved.json>)
    dataset_cutter.update()

.. note::

    To unpack the json representation, the :meth:`from_json_file` method needs
    a symbol table mapping the class names to the class constructors. To convert
    a json representation of custom classes you wrote yourself, you'll need to
    extend the symbol table using the optional `constructor_symbol_table` argument.
