Cluster rasters
###############

.. todo::

    explain clsuters defined by, preclustering methods

To get a list of the raster clusters that need to be respected in the
train/validation split to avoid data leakage use the
:func:`geographer.utils.cluster_rasters.get_raster_clusters` function::

.. note::

    If you just naively split your rasters into a train and a validation set
    there might be data leakage. Some vector features might intersect
    several rasters. Also, rasters can overlap and there might be vector
    features in the overlaps. The clusters returned by
    :func:`geographer.utils.cluster_rasters.get_raster_clusters`
    are the minimal clusters of rasters that need to be consistently assigned
    to the train or validation splits to avoid data leakage.

    from geographer.utils.cluster_rasters.= import get_raster_clusters
    clusters : List[Set[str]] = get_raster_clusters(
        connector=connector,
        clusters_defined_by='rasters_that_share_vector_features',
        preclustering_method='y then x-axis')