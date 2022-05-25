Downloading Raster Images
#########################

To download images for vector features use ``ImgDownloaderForVectorFeatures``. 

By plugging in different ``DownloaderForSingleVectorFeature`` and ``Processor`` components it can interface with different sources of remote sensing imagery. Currently, it can interface with the Copernicus Open Access Hub for Sentinel-2 imagery, and JAXA for ALOS DEM (digital elevation model) data, and can easily be extended to other data sources by writing custom ``DownloaderForSingleSingleVectorFeature`` and ``Processor`` classes.

Example usage
+++++++++++++

Example usage::

    from geographer.downloaders import (
        ImgDownloaderForVectorFeatures,
        SentinelDownloaderForSingleVectorFeature,
        Sentinel2Processor
    )
    downloader_for_single_feature=SentinelDownloaderForSingleVectorFeature()
    download_processor=Sentinel2Processor()
    downloader = ImgDownloaderForVectorFeatures(
        download_dir=<DOWNLOAD_DIR>,
        downloader_for_single_feature=downloader_for_single_feature,
        download_processor=download_processor,
    )
    downloader.download(
        connector=my_connector,
        feature_names=optional_list_of_vector_feature_names,
        target_img_count=2,
        producttype='L2A',
        max_percent_cloud_coverage=10,
        resolution=10,
        date=(“NOW-10DAYS”, “NOW”),
        area_relation='Contains'
    )

The image counts for all vector features are updated after every download, so that unnecessary downloads and an imbalance in the dataset due to clustering of nearby vector features are avoided.

You can supply default values for dataset/data source specific ``download`` arguments (e.g. ``producttype``, ``max_percent_cloud_coverage`` for the ``SentinelDownloaderForSingleVectorFeature``) in the ``ImgDownloaderForVectorFeatures``'s ``kwarg_defaults`` arguments dict, so that one doesn't have to pass them by hand to the ``download`` method, for example::
    
        downloader = ImgDownloaderForVectorFeatures(
            download_dir=<DOWNLOAD_DIR>,
            downloader_for_single_feature=SentinelDownloaderForSingleVectorFeature(),
            download_processor=Sentinel2Processor(),
            kwarg_defaults={
                'max_percent_cloud_coverage' = 10,
                'producttype': L2A,
                'resolution': 10,
                'date': (“NOW-10DAYS”, “NOW”),
                'area_relation': 'Contains'})
        downloader.download(
            connector=my_connector,
            feature_names=optional_list_of_vector_feature_names,
            target_img_count=2)

Data sources
++++++++++++

The ``DownloaderForSingleSingleVectorFeature`` class interfaces with the raster image data source's API and the ``Processor`` class processes downloaded files to GeoTiffs. 

Sentinel-2
~~~~~~~~~~

For *Sentinel-2* data, use the ``SentinelDownloaderForSingleVectorFeature`` to download images from the Copernicus Open Access Hub and the ``Sentinel2Processor``.

Sentinel-1
~~~~~~~~~~

The ``SentinelDownloaderForSingleVectorFeature`` should work with slight modifications for downloading Sentinel-1 data from Copernicus Open Access Hub as well. Feel free to submit a pull request for this feature.

JAXA DEM data
~~~~~~~~~~~~~

For *JAXA* DEM (digital elevation model) data use ``JAXADownloaderForSingleVectorFeature`` and ``JAXADownloadProcessor``.

Other sources for remote sensing imagery:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass ``DownloaderForSingleSingleVectorFeature`` and ``Processor`` to interface with other API's for remote sensing data.

.. todo::

    describe the Sentinel and JAXA 

Saving and loading a downloader
+++++++++++++++++++++++++++++++

Serializing a ``ImgDownloaderForVectorFeatures`` and all its components as a json file::

    downloader.save(<PATH_TO_JSON>)

Loading a saved ``ImgDownloaderForVectorFeatures`` from a saved json file::

    downloader = ImgDownloaderForVectorFeatures.from_json_file(<PATH_TO_JSON>)
