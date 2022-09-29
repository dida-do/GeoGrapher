Conversion
##########

.. note::

    All converters operate on two datasets: a source and a target dataset. At the moment in place operations are not supported.

Combining/removing task vector classes
++++++++++++++++++++++++++++++++++++++++++++++++

Assume your dataset has ``task_vector_classes`` given by
``['class1', 'class2', 'class3', 'class4']``. Suppose you want to combine
``'class1'`` and ``'class2'`` to a new class ``'new_class_name1'``, rename
``class3`` to ``'new_class_name2'``, drop all vector features belonging to
``'class4'``, and remove all rasters not containing and of the new classes
``'new_class_name1'`` and  ``'new_class_name2'``. This can be accomplished
using the ``DSConverterCombineRemoveClasses`` as follows (see :doc:`here <label_makers>`
for creating a label maker). ::

    from geographer.converters import DSConverterCombineRemoveClasses

    converter = DSConverterCombineRemoveClasses(
        name="combine_remove_classes",
        source_data_dir=<PATH/TO/SOURCE/DATA_DIR>,
        target_data_dir=<PATH/TO/TARGET/DATA_DIR>,
        seg_classes=[['class1', 'class2'], 'class3'],
        new_seg_classes=['new_class_name1', 'new_class_name2'],
        label_maker=label_maker,
        remove_rasters=True
        )
    converter.convert()
    converter.save(<PATH/TO/SOURCE/DATA_DIR/<name>.JSON>)

Updating the target dataset after the source dataset has grown::

    converter = DSConverterCombineRemoveClasses.from_json_file(
        <PATH/TO/SOURCE/DATA_DIR/<name>.JSON>
    )
    converter.update()

Converting task vector class types
+++++++++++++++++++++++++++++++++++++

Soft-categorical to categorical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converting a dataset from soft-categorical to categorical vector features
and labels using ``DSConverterCombineRemoveClasses`` (see :doc:`here <label_makers>`
for how to create a label maker to pass as a ``label_maker`` argument)::

    from geographer.converters import DSConverterCombineRemoveClasses

    converter = DSConverterSoftCatToCat(
        name="convert_soft_to_cat",
        source_data_dir=<PATH/TO/SOURCE/DATA_DIR>,
        target_data_dir=<PATH/TO/TARGET/DATA_DIR>,
        label_maker=label_maker,
    )
    converter.convert()
    converter.save(<PATH/TO/SOURCE/DATA_DIR/<name>.JSON>)

Updating the target dataset after the source dataset has grown::

    converter = DSConverterCombineRemoveClasses.from_json_file(
        <PATH/TO/SOURCE/DATA_DIR/<name>.JSON>
    )
    converter.update()

GeoTiff To .npy
+++++++++++++++

Converting a dataset from GeoTiff to .npy::

    from geographer.converters import DSConverterGeoTiffToNpy

    converter = DSConverterGeoTiffToNpy(
        name="convert_soft_to_cat",
        source_data_dir=<PATH/TO/SOURCE/DATA_DIR>,
        target_data_dir=<PATH/TO/TARGET/DATA_DIR>,
    )
    converter.convert()
    converter.save(<PATH/TO/SOURCE/DATA_DIR/<name>.JSON>)

Updating the target dataset after the source dataset has grown::

    converter = DSConverterCombineRemoveClasses.from_json_file(
        <PATH/TO/SOURCE/DATA_DIR/<name>.JSON>
    )
    converter.update()




