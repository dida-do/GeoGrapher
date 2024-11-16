"""Custom Error classes."""


class GeoGrapherError(Exception):
    """Base class for exceptions."""

    pass


class RasterAlreadyExistsError(GeoGrapherError):
    """Raster already exists in dataset."""

    pass


class NoRastersForVectorFoundError(GeoGrapherError):
    """No rasters found or none could be downloaded."""

    pass


class RasterDownloadError(GeoGrapherError):
    """Error occurs while downloading raster."""

    pass
