Conversion
##########

.. todo::

    rewrite/refactor for general vision tasks instead of just for segmentation tasks

.. note::

    All converters operate on two datasets: a source and a target dataset. At the moment in place operations are not supported.

Combining And/Or Removing Vector Feature Classes
++++++++++++++++++++++++++++++++++++++++++++++++

.. todo::

    - label type vs vector feature class type
    - label_maker arg

Assume your dataset has ``task_vector_feature_classes`` given by ``['class1', 'class2', 'class3', 'class4']``. Suppose you want to combine  ``'class1'`` and ``'class2'`` to a new class ``'new_class_name1'``, rename ``class3`` to ``'new_class_name2'``, drop all vector features belonging to ``'class4'``, and remove all images not containing and of the new classes ``'new_class_name1'`` and  ``'new_class_name2'``. This can be accomplished using the ``DSConverterCombineRemoveClasses`` as follows::

    from rs_tools.converters import DSConverterCombineRemoveClasses
    converter = DSConverterCombineRemoveClasses(
        name="combine_remove_classes",
        source_data_dir=<PATH/TO/SOURCE/DATA_DIR>,
        target_data_dir=<PATH/TO/TARGET/DATA_DIR>,
        seg_classes=[['class1', 'class2'], 'class3'],
        new_seg_classes=['new_class_name1', 'new_class_name2'],
        remove_imgs=True
        )
    converter.convert()
    converter.save(<PATH/TO/SOURCE/DATA_DIR/<name>.JSON>)

Updating the target dataset after the source dataset has grown::

    converter = DSConverterCombineRemoveClasses.from_json_file(<PATH/TO/SOURCE/DATA_DIR/<name>.JSON>)
    converter.update()

Converting Vector Feature Class Types
+++++++++++++++++++++++++++++++++++++

Soft-Categorical To Categorical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converting a segmentation dataset from soft-categorical to categorical vector features and labels using ``DSConverterCombineRemoveClasses``::

    from rs_tools.converters import DSConverterCombineRemoveClasses
    converter = DSConverterSoftCatToCat(
        name="convert_soft_to_cat",
        source_data_dir=<PATH/TO/SOURCE/DATA_DIR>,
        target_data_dir=<PATH/TO/TARGET/DATA_DIR>,
    )
    converter.convert()
    converter.save(<PATH/TO/SOURCE/DATA_DIR/<name>.JSON>)

Updating the target dataset after the source dataset has grown::

    converter = DSConverterCombineRemoveClasses.from_json_file(<PATH/TO/SOURCE/DATA_DIR/<name>.JSON>)
    converter.update()

GeoTiff To .npy
+++++++++++++++

Converting a dataset from GeoTiff to .npy::

    from rs_tools.converters import DSConverterGeoTiffToNpy
    converter = DSConverterGeoTiffToNpy(
        name="convert_soft_to_cat",
        source_data_dir=<PATH/TO/SOURCE/DATA_DIR>,
        target_data_dir=<PATH/TO/TARGET/DATA_DIR>,
    )
    converter.convert()
    converter.save(<PATH/TO/SOURCE/DATA_DIR/<name>.JSON>)

Updating the target dataset after the source dataset has grown::

    converter = DSConverterCombineRemoveClasses.from_json_file(<PATH/TO/SOURCE/DATA_DIR/<name>.JSON>)
    converter.update()




