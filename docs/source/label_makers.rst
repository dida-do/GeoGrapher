Making labels
#############

.. todo::

    - implicit background class
    - explain/mention label_type

Use the ``LabelMaker`` classes to create labels from the vector features for computer vision tasks. Currently, label creation for multiclass segmentation tasks for both 'categorical' and 'soft-categorical' (i.e. there is a probability distribution over the classes for each vector feature) label-types are supported.

Making Segmentation Labels
++++++++++++++++++++++++++

Categorical Segmentation Labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating categorical segmentation labels encoded for each pixel as an integer corresponding to the class::

    from rs_tools.label_makers import SegLabelMakerCategorical
    label_maker = SegLabelMakerCategorical()
    label_maker.make_labels(
        connector=<your_connector>,
        img_names=<optional image names>
    )

.. note::

    To use a ``SegLabelMakerCategorical`` with a connector, the connector's ``vector_features`` GeoDataFrame needs to have a ``'type'`` column containing the classes the features belong to.

Soft-Categorical Segmentation Labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Soft-categorical* labels are labels in which for each vector feature (i.e. polygon in the segmentation case) there is a probability distribution for which of the segmentation classes the vector feature belongs to. In the segmentation labels, the probability will be encoded in a class dimension/axis, i.e. a label have dimensions HxWxC where H, W are the height and width of the corresponding image and C is the number of segmentation classes.

Creating soft-categorical segmentation labels::

    from rs_tools.label_makers import SegLabelMakerSoftCategorical
    label_maker = SegLabelMakerSoftCategorical()
    label_maker.make_labels(
        connector=<your_connector>,
        img_names=<optional image names>
    )

.. note::

    To use a ``SegLabelMakerSoftCategorical`` with a connector, there needs to be a ``prob_seg_class_<segmentation_class_name>`` column in the connector's ``vector_features`` GeoDataFrame for each segmentation class in ``connector.ml_task_classes`` containing the probability that the features belong to the class.

Other Vision Tasks or Label Types
+++++++++++++++++++++++++++++++++

Feel free to submit a feature request or submit a pull request with ``LabelMaker`` s for other computer vision tasks or labels types.


