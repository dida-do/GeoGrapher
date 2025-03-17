<p align="center">
  <img src="docs/source/_static/GeoGrapher.png" alt="GeoGrapher Logo" width="70"><br>
</p>
<p align="center">
  <a href="https://firstdonoharm.dev/version/3/0/cl-eco-mil.html">
    <img src="https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-CL-ECO-MIL&labelColor=5e2751&color=bc8c3d" alt="Hippocratic License">
  </a>
</p>

# GeoGrapher

*GeoGrapher* is a Python library for building _object-centric_
machine learning datasets from remote sensing data—starting with a
list of objects (vector features such as locations or polygonal
segmentation masks) rather than a specific area. At its core,
GeoGrapher leverages a bipartite graph structure to efficiently
associate objects with relevant geospatial data, solving common
challenges in object-centric dataset construction. It provides
flexible, customizable cutting of large remote sensing rasters,
ensuring precise data extraction.

For a deeper dive, check out the [blog post](https://dida.do/blog/geographer-an-open-source-python-library-for-building-object-centric-machine-learning-datasets-from-remote-sensing-data)
 or explore the [documentation](https://geographer.readthedocs.io/en/latest/).

# Installation
This package has two external dependencies:
- Python 3.9 or newer.
- The geopandas and rasterio libraries might depend on GDAL base C libraries.
See [https://geopandas.org/en/stable/getting_started/install.html#dependencies](https://geopandas.org/en/stable/getting_started/install.html#dependencies)
and [https://pypi.org/project/rasterio/](https://pypi.org/project/rasterio/)
for instructions on installing these. On many Linux distros, you can use the
the `gdal-devel` or `libgdal-dev` packages.

The package itself (and its Python dependencies) can be installed with

```
pip install GeoGrapher
```

# Getting started
Here are several options for getting started:
- Try out the tutorial notebooks in the [notebooks directory](https://github.com/dida-do/GeoGrapher/tree/main/notebooks),
- Read the [documentation](https://geographer.readthedocs.io/en/latest/).

# Development
To create a local dev installation, clone the repo, `cd` into the repo folder,
and use the commands

```
python -m venv geographer-env
source geographer-env/bin/activate
make -B venv
```

to create a virtual environment, activate it, and create an editable local installation
of the package and its dependencies.

Use the `make format`, `make lint`, `make test`, and `make docs` commands to format,
lint, or test the code or to create the documentation locally.

To create the documentation locally, you will need to install
the pandoc external dependency following the instructions here
[https://pandoc.org/installing.html](https://pandoc.org/installing.html).

# Contributing to GeoGrapher
You can contribute by giving feedback, opening an issue, submitting
feature requests or bug reports, or submitting pull requests.

# Show your support
Give this project a star if you like it!

# Contact
Contact the maintainer at [@rustamdantia](https://github.com/rustamdantia).

# License
This project is licensed under a version of the Hippocratic License that in particular prohibits use for military activites. You can find a copy of the license [here](LICENSE).

[![Hippocratic License HL3-CL-ECO-MIL](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-CL-ECO-MIL&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/cl-eco-mil.html)
