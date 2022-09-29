############
Installation
############

This package has two external dependencies:
- Python 3.8 or newer.
- The geopandas and rasterio libraries might depend on GDAL base C libraries.
See [https://geopandas.org/en/stable/getting_started/install.html#dependencies](https://geopandas.org/en/stable/getting_started/install.html#dependencies)
and [https://pypi.org/project/rasterio/](https://pypi.org/project/rasterio/)
for instructions on installing these. On many Linux distros, you can use the
the `gdal-devel` or `libgdal-dev` packages.

The package itself (and its Python dependencies) can be installed with

```
pip install GeoGrapher
```

To create the documentation locally in a development installation,
the pandoc external dependency is needed as well. See
[https://pandoc.org/installing.html](https://pandoc.org/installing.html)
for installation instructions.
