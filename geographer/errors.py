"""Custom Error classes."""


class Error(Exception):
    """Base class for exceptions in the connector class."""

    pass


class RasterAlreadyExistsError(Error):
    """Raster already exists in dataset."""

    pass


class NoRastersForVectorFoundError(Error):
    """No rasters found or none could be downloaded."""

    pass


class RasterDownloadError(Error):
    """Error occurs while downloading raster."""

    pass
