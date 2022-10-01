############
Installation
############

This package has two external dependencies:

- Python 3.8 or newer.
- The geopandas and rasterio libraries might depend on GDAL base C libraries.

See `here for the geopandas instructions <https://geopandas.org/en/stable/getting_started/install.html#dependencies>`_
and `here for the rasterio instructions <https://pypi.org/project/rasterio/>`_
for instructions on installing these. On many Linux distros, you can use the
the `gdal-devel` or `libgdal-dev` packages.

The package itself (and its Python dependencies) can be installed with::

    pip install GeoGrapher

To create the documentation locally in a development installation,
the pandoc external dependency is needed as well. See
`here <https://pandoc.org/installing.html>`_
for installation instructions.
