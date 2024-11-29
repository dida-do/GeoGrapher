Downloading rasters
###################

To download rasters for vector features use ``RasterDownloaderForVectors``. 

A ``RasterDownloaderForVectors`` requires components that implement the
abstract base classes ``DownloaderForSingleVector`` and ``Processor``.

    - ``DownloaderForSingleVector`` defines how to search for and download
        a raster for a single vector from a provider.
    - ``Processor`` how to process the downloaded file into a GeoTiff raster.

Available Implementations
+++++++++++++++++++++++++

Currently, there are two concrete implementations of ``DownloaderForSingleVector``:

    - ``EodagDownloaderForSingleVector``: Based on
        the excellent `"eodag" <EODAG_>`_ package, this implementation supports downloading
        over 50 product types from than 10 providers.
    - ``JAXADownloaderForSingleVector``: Designed for downloading DEM (digital elevation model)
        data from the `"JAXA ALOS" <JAXA_ALOS_>`_ mission.

Additionally, there are two concrete implementations of the ``Processor`` class:

    - ``Sentinel2SAFEProcessor``: Processes Level-2A Sentinel-2 SAFE files.
    - ``JAXADownloadProcessor``: Processes JAXA DEM data.

To use the ``RasterDownloaderForVectors`` for a new provider or product type
you only need to write custom implementations of ``DownloaderForSingleVector``
or ``Processor``.

.. _EODAG: http://supertech.csail.mit.edu/papers/steal.pdf
.. _JAXA_ALOS: https://www.eorc.jaxa.jp/ALOS/en/index_e.htm

Example usage
+++++++++++++

Example usage:

.. code-block:: python

    from geographer.downloaders import (
        RasterDownloaderForVectors,
        EodagDownloaderForSingleVector,
        Sentinel2SAFEProcessor,
    )
    download_processor = Sentinel2SAFEProcessor()
    downloader_for_single_vector = EodagDownloaderForSingleVector()
    downloader = RasterDownloaderForVectors(
        downloader_for_single_vector=downloader_for_single_vector,
        download_processor=download_processor,
    )

    # Parameters needed by the EodagDownloaderForSingleVector.download method
    downloader_kwargs = {
        "search_kwargs": {  # Keyword arguments for the eodag search_all method
            "provider": "cop_dataspace",  # Download from copernicus dataspace
            "productType": "S2_MSI_L2A",  # Search for Sentinel-2 L2A products
            "start": "2024-11-01",
            "end": "2024-12-01",
        },
        "filter_online": True,  # Filter out products that are not online
        "sort_by": ("cloudCover", "ASC"),  # Prioritize search results with less cloud cover
        "suffix_to_remove": ".SAFE",  # Will strip .SAFE from the stem of the tif file names
    }
    # Parameters needed by the Sentinel2SAFEProcessor
    processor_kwargs = {
        "resolution": 10,  # Extract all 10m resolution bands
        "delete_safe": True,  # Delete the SAFE file after extracting a .tif file
    }

    downloader.download(
        connector=my_connector,
        vector_names=optional_list_of_vector_names,
        target_raster_count=2,
        downloader_kwargs=downloader_kwargs,  # Only needed the first time downloader.download is called
        processor_kwargs=processor_kwargs,    # Only needed the first time downoader.download is called
    )

The raster counts for all vector features are updated after every download,
so that unnecessary downloads and an imbalance in the dataset due to clustering
of nearby vector features are avoided.

You can supply default values for dataset/data source specific ``download``
arguments (e.g. ``producttype``, ``max_percent_cloud_coverage`` for the
``SentinelDownloaderForSingleVector``) in the
``RasterDownloaderForVectors``'s ``kwarg_defaults`` arguments dict,
so that one doesn't have to pass them by hand to the ``download`` method,
for example:

.. code-block:: python
    
        downloader = RasterDownloaderForVectors(
            download_dir=<DOWNLOAD_DIR>,
            downloader_for_single_vector=SentinelDownloaderForSingleVector(),
            download_processor=Sentinel2Processor(),
            kwarg_defaults={
                'max_percent_cloud_coverage' = 10,
                'producttype': L2A,
                'resolution': 10,
                'date': ('NOW-10DAYS', 'NOW'),
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
