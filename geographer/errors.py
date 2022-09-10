"""Custom Error classes."""


class Error(Exception):
    """Base class for exceptions in the connector class."""

    pass


class ImgAlreadyExistsError(Error):
    """Image already exists in dataset."""

    pass


class NoImgsForVectorFeatureFoundError(Error):
    """No rasters found or none could be downloaded."""

    pass


class ImgDownloadError(Error):
    """Error occurs while downloading raster."""

    pass
