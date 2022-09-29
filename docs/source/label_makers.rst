Making labels
#############

Use the ``LabelMaker`` classes to create labels from the vector features
for computer vision tasks. Currently, label creation for multiclass segmentation
tasks for both 'categorical' and 'soft-categorical' (i.e. there is a probability
distribution over the classes for each vector feature) label-types are supported.

Task vector classes and class types
++++++++++++++++++++++++++++++++++++++

GeoGrapher is designed for *multiclass* vision tasks.

.. note::

    GeoGrapher should be easily extendable to multi-label vision tasks. Submit an
    issue, feature request, or pull request to extend GeoGrapher if you want!

Task vector classes
~~~~~~~~~~~~~~~~~~~~~~

The classes the ``vectors`` may belong to for a machine learning task
are set in

- the ``task_vector_classes`` argument of the ``Connector`` class's ``from_scratch``
  class constructor method
- a ``Connector``'s ``task_vector_classes`` attribute.

In some applications it is useful to have vector features that belong to a background
class different from any of the machine learning task classes. A optional background
class can be set in

- the ``background_class`` argument of the ``Connector`` class's ``from_scratch`` class constructor method
- the ``background_class`` attribute of a ``Connector``.

To add other non-``task_vector_classes``, subclass the ``Connector`` and modify the ``_non_task_vector_classes`` class variable appropriately. A ``Connector``'s ``all_vector_classes`` attribute returns a list of all classes the vector features may belong to.

.. _vector_class_types

Vector class types
~~~~~~~~~~~~~~~~~~

Currently, GeoGrapher's LabelMakers support two formats (*vector feature class types*)
in which classes can be assigned to vector features:

- *categorical*: The class information is contained in a ``'type'`` column in
  ``vectors``
- *soft-categorical*: Probabilistic/soft class information is contained in
  ``prob_of_class_<class_name>`` columns (one for each class) containing the
  probabilities that the vector features belong to a given class.

Making segmentation labels
++++++++++++++++++++++++++

Categorical segmentation labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating categorical segmentation labels encoded for each pixel as an integer
corresponding to the class::

    from geographer.label_makers import SegLabelMakerCategorical
    label_maker = SegLabelMakerCategorical()
    label_maker.make_labels(
        connector=<your_connector>,
        raster_names=<optional raster names>
    )

.. note::

    To use a ``SegLabelMakerCategorical`` with a connector, the connector's
    ``vectors`` GeoDataFrame needs to have a ``'type'`` column containing
    the classes the features belong to.

Soft-categorical segmentation labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Soft-categorical* labels are labels in which for each vector feature
(i.e. polygon in the segmentation case) there is a probability distribution
for which of the segmentation classes the vector feature belongs to. In the
segmentation labels, the probability will be encoded in a class dimension/axis,
i.e. a label have dimensions HxWxC where H, W are the height and width of
the corresponding raster and C is the number of segmentation classes.

Creating soft-categorical segmentation labels::

    from geographer.label_makers import SegLabelMakerSoftCategorical
    label_maker = SegLabelMakerSoftCategorical()
    label_maker.make_labels(
        connector=<your_connector>,
        raster_names=<optional raster names>
    )

.. note::

    To use a ``SegLabelMakerSoftCategorical`` with a connector, there needs to
    be a ``prob_of_class_<segmentation_class_name>`` column in the connector's
    ``vectors`` GeoDataFrame for each segmentation class in
    ``connector.ml_task_classes`` containing the probability that the features
    belong to the class.

Other vision tasks or label types
+++++++++++++++++++++++++++++++++

Feel free to submit a feature request or submit a pull request with ``LabelMaker``
s for other computer vision tasks or labels types.


