# GeoGrapher

*GeoGrapher* is a Python library for building and handling remote sensing
computer vision datasets assembled from vector features and raster images.
It connects the features and images by a bipartite graph that keeps track of
the containment and intersection relations between them making it particularly
suited for *object-centric* vision tasks. GeoGrapher also provides highly
general and customizable dataset cutting functionality as well as other utility
functions.

# Installation

This package has two external dependencies:
- Python 3.X or newer [where X is to be determined, possibly X=7]
- GDAL development files.  On most Linux distros, this is provided by
  the `gdal-devel` or `libgdal-dev` packages.

The package itself (and its Python dependencies) can be installed with

```
pip install GeoGrapher
```

To install the package for development work, clone this repo and run

```
make [TODO]
```

# Documentation
To create html html documentation in docs/build/html, run
```
make docs
```

# Getting started
Anything here?

# Contributing to GeoGrapher

You can contribute by giving feedback, opening an issue, submitting feature
requests or bug reports, or submitting code. To contribute code to GeoGrapher,
follow these steps:

- Fork this repository.
- Create a branch: git checkout -b <branch_name>.
- Make your changes and commit them: git commit -m '<commit_message>'
- Push to the original branch: git push origin <project_name>/<location>
- Create the pull request.

Alternatively see the GitHub documentation on creating a pull request.

# Show your support
Give this project a star if you like it!

# Contact
Contact the maintainer at [@rustamdantia](https://github.com/rustamdantia).

# License
This project is Apache licensed, see [here](LICENSE).
