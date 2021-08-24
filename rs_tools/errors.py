class Error(Exception):
    """Base class for exceptions in the assoc module"""
    
    pass

class ImgAlreadyExistsError(Error):
    """
    Error raised when an attempt was made at downloading an image that has already been downloaded to the associator's dataset.
    """

    pass

class NoImgsForPolygonFoundError(Error):
    """
    Error raised by associator's __download_imgs_for_polygon__ method to indicate no images could be found or downloaded.
    """

    pass

class ImgDownloadError(Error):
    """
    Error raised by associator's __download_imgs_for_polygon__ if an error occurs while downloading. 
    """

    pass
