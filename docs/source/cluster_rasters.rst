Cluster rasters
###############

.. todo::

    explain clsuters defined by, preclustering methods

To avoid data leakage the train validation split should respect
rasters that cluster together::

    clusters : List[Set[str]] = get_raster_clusters(
        connector=connector,
        clusters_defined_by='rasters_that_share_vector_features',
        preclustering_method='y then x-axis')