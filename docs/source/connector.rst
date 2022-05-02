###################
The Connector Class
###################

.. todo::

    "The dataset class connecting vector features and raster images" renders as a section instead of a subtitle.

.. todo::

    explain relation between ``raster_imgs`` GeoDataFrame and actual images on disk

.. todo::
    
    - task classes and multiclass vision tasks
    - label_types

.. todo::

    use reStructedText python domain directives: \:class\:, \:method\: etc

GeoGrapher is built around the ``Connector`` class.

The ``Connector`` class representing a remote sensing computer vision dataset composed of vector features and raster images is at the core of the GeoGrapher library. A ``Connector`` connects the features and images by a bipartite graph encoding the containment or intersection relationships between them and is a container for tabular information about the features and images as well as for metadata about the dataset.

The ``vector_features`` and ``raster_imgs`` attributes
++++++++++++++++++++++++++++++++++++++++++++++++++++++

A ``Connector``'s most important attributes are the ``vector_features`` and ``raster_imgs`` GeoDataFrames. These contain the geometries of the vector features and the bounding boxes of the images respectively along with other tabular information (e.g. classes for the machine learning task for the vector features or meta-data)::

    connector.vector_features
    # (returns geodataframe)

    connector.raster_imgs
    # (returns geodataframe)

.. todo::

    insert return values as image or text (e.g. using rst's ``.. ipython::`` directive)

.. todo::
    
    insert \:ref\: to explanation of ML task classes

The `img_count` column in `connector.vector_features` will automatically contain the number of images in `raster_imgs` that fully contain the vector feature.

Querying the graph
++++++++++++++++++

The graph can be queried with the ``imgs_containing_vector_feature``,
``imgs_intersecting_vector_feature``, ``vector_features_contained_in_img``,
``vector_features_intersecting_img`` methods::

    connector.imgs_containing_vector_feature(feature_name)
    # (returns list of images containing vector feature)

``Attrs``: Further attributes
+++++++++++++++++++++++++

The ``attrs`` attribute is a dictionary for custom attributes that can contain e.g. metadata about the dataset::

    connector.attrs['some_field'] = some_value

    connector.attrs
    # (returns dictionary)

Location of images on disk
++++++++++++++++++++++++++

The ``images_dir`` attribute should point to the directory containing the images::

    connector.images_dir
    # (returns ``pathlib.Path`` to images, usually data_dir / 'images')

Creating and loading Connectors
+++++++++++++++++++++++++++++++

Creating an empty connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. todo::

    This looks like regular text, not like a subsubsection title.

To create an empty connector use the ``from_scratch`` class method::

    from rs_tools import Connector
    connector = Connector.from_scratch(
        data_dir=<DATA_DIR>)

The created connector will be empty, i.e. the ``vector_feates`` and ``raster_imgs`` attributes will be empty GeoDataFrames.

Initializing an existing connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To initialize an existing connector you can use either the ``from_data_dir`` or ``from_paths`` class method::

    connector = Connector.from_data_dir(data_dir=<DATA_DIR>)

The ``from_paths`` class method allows you to work with datasets that do not ::

    connector = Connector.from_paths(
        connector_dir=<CONNECTOR_DIR>,
        images_dir=<IMAGES_DIR>))

Saving a connector
~~~~~~~~~~~~~~~~~~

Use the ``save`` method to save the connector::

    connector.save()

This will save the connector's components (``vector_features``, ``raster_imgs``, the graph, and the ``attrs``) to the ``connector``'s ``connector_dir``.

.. note:: 
    
    Unfortunately, since geopanpdas can not save empty GeoDataFrames both the ``vector_features`` and ``raster_imgs`` GeoDataFrames need to be non-empty to save a connector.

Adding or dropping vector features
++++++++++++++++++++++++++++++++++

Adding or dropping vector features to/from a connector::

    connector.add_to_vector_features(new_vector_features)
    # (concatenates the new_vector_features to connector.vector_features and updates the graph)
    connector.drop_vector_features(list_of_vector_features)
    # (concatenates the new_raster_imgs to connector.raster_imgs and updates the graph)

The names of the ``new_vector_features`` in the GeoDataFrame's index must be unique. You can supply an optional ``label_maker`` argument to automatically update the labels to reflect the added or dropped features (i.e. modify the labels of any images intersecting added or dropped features).

.. important::
    
    Always use the ``add_to_vector_features`` and ``drop_vector_features`` methods to add or drop vector features to/from a connector or to modify the geometries of the ``vector_features`` in a way that would change the containment/intersection relations! If you directly manipulate the ``vector_features`` GeoDataFrame the graph encoding the relations will not be updated and therefore incorrect.

Adding or dropping raster images
++++++++++++++++++++++++++++++++


Adding or dropping raster images to/from the connector::

    connector.add_to_raster_imgs(new_raster_imgs)
    connector.drop_raster_imgs(list_of_raster_img_names)

As with adding or dropping vector features, you can supply an optional ``label_maker`` argument to automatically update the labels to reflect the added or dropped images.

.. note ::
    
    The connector only knows about the ``raster_imgs`` GeoDataFrame, not
    whether the images actually exist in the ``connector.images_dir``
    directory.  You can use the ``raster_imgs_from_tif_dir`` function in
    ``utils/utils.py`` to create a ``new_raster_imgs`` GeoDataFrame from a
    directory of GeoTiffs you can add to the connector.
