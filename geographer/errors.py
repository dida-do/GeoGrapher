class Error(Exception):
    """Base class for exceptions in the connector class."""

    pass


class ImgAlreadyExistsError(Error):
    """Error raised when an attempt was made at downloading an image that has
    already been downloaded to the connector's dataset."""

    pass


class NoImgsForVectorFeatureFoundError(Error):
    """Error raised by an connector's downloader to indicate no images could
    be found or downloaded."""

    pass


class ImgDownloadError(Error):
    """Error raised by an connector's downloader if an error occurs while
    downloading."""

    pass
