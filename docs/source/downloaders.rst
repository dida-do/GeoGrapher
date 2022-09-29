Downloading rasters
###################

To download rasters for vector features use ``RasterDownloaderForVectors``. 

By plugging in different ``DownloaderForSingleVector`` and ``Processor``
components it can interface with different sources of remote sensing rasters.
Currently, it can interface with the Copernicus Open Access Hub for Sentinel-2
rasters, and JAXA for ALOS DEM (digital elevation model) data, and can easily
be extended to other data sources by writing custom
``DownloaderForSingleSingleVector`` and ``Processor`` classes.

Example usage
+++++++++++++

Example usage::

    from geographer.downloaders import (
        RasterDownloaderForVectors,
        SentinelDownloaderForSingleVector,
        Sentinel2Processor
    )
    downloader_for_single_vector=SentinelDownloaderForSingleVector()
    download_processor=Sentinel2Processor()
    downloader = RasterDownloaderForVectors(
        downloader_for_single_vector=downloader_for_single_vector,
        download_processor=download_processor,
    )
    downloader.download(
        connector=my_connector,
        vector_names=optional_list_of_vector_names,
        target_raster_count=2,
        producttype='L2A',
        max_percent_cloud_coverage=10,
        resolution=10,
        date=(“NOW-10DAYS”, “NOW”),
        area_relation='Contains'
    )

The raster counts for all vector features are updated after every download,
so that unnecessary downloads and an imbalance in the dataset due to clustering
of nearby vector features are avoided.

You can supply default values for dataset/data source specific ``download``
arguments (e.g. ``producttype``, ``max_percent_cloud_coverage`` for the
``SentinelDownloaderForSingleVector``) in the
``RasterDownloaderForVectors``'s ``kwarg_defaults`` arguments dict,
so that one doesn't have to pass them by hand to the ``download`` method,
for example::
    
        downloader = RasterDownloaderForVectors(
            download_dir=<DOWNLOAD_DIR>,
            downloader_for_single_vector=SentinelDownloaderForSingleVector(),
            download_processor=Sentinel2Processor(),
            kwarg_defaults={
                'max_percent_cloud_coverage' = 10,
                'producttype': L2A,
                'resolution': 10,
                'date': (“NOW-10DAYS”, “NOW”),
                'area_relation': 'Contains'})
        downloader.download(
            connector=my_connector,
            vector_names=optional_list_of_vector_names,
            target_raster_count=2)

Data sources
++++++++++++

The ``DownloaderForSingleSingleVector`` class interfaces with the raster
raster data source's API and the ``Processor`` class processes downloaded files
to GeoTiffs. 

Sentinel-2
~~~~~~~~~~

For *Sentinel-2* data, use the ``SentinelDownloaderForSingleVector``
to download rasters from the Copernicus Open Access Hub and the ``Sentinel2Processor``.

Sentinel-1
~~~~~~~~~~

The ``SentinelDownloaderForSingleVector`` should work with slight modifications
for downloading Sentinel-1 data from Copernicus Open Access Hub as well. Feel free to
submit a pull request for this feature.

JAXA DEM data
~~~~~~~~~~~~~

For *JAXA* DEM (digital elevation model) data use ``JAXADownloaderForSingleVector``
and ``JAXADownloadProcessor``.

Other sources for remote sensing rasters:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass ``DownloaderForSingleSingleVector`` and ``Processor`` to interface with
other API's for remote sensing data.

Saving and loading a downloader
+++++++++++++++++++++++++++++++

Serializing a ``RasterDownloaderForVectors`` and all its components as a json file::

    downloader.save(<PATH_TO_JSON>)

Loading a saved ``RasterDownloaderForVectors`` from a saved json file::

    downloader = RasterDownloaderForVectors.from_json_file(<PATH_TO_JSON>)
