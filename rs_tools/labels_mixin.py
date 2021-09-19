import logging
from typing import List, Optional, Callable
from tqdm import tqdm
from rs_tools.labels.make_geotif_label_categorical import _make_geotif_label_categorical
from rs_tools.labels.make_geotif_label_soft_categorical_onehot import _make_geotif_label_onehot, _make_geotif_label_soft_categorical

# logger
log = logging.getLogger(__name__)

# log level (e.g. 'DEBUG')
# log.setLevel(logging.DEBUG)

LABEL_MAKERS = {
    'soft-categorical' : _make_geotif_label_soft_categorical, 
    'categorical' : _make_geotif_label_categorical, 
    'onehot' : _make_geotif_label_onehot
}


class LabelsMixIn(object):   
    """Mix-in that implements a method to generate labels. """     

    def make_missing_labels(self, 
            img_names : Optional[List[str]]=None):
        """
        Creates pixel labels for all images without a label. 

        Currently only works for GeoTiffs. 

        Args:
            img_names (List[str], optional): list of image names to create labels. Defaults to None (i.e. all images without a label).
        """
        self._compare_existing_imgs_to_imgs_df()

        log.info("\nCreating missing labels.\n")

        # Make sure the labels_dir exists.
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        existing_images = {img_path.name for img_path in self.images_dir.iterdir() if img_path.is_file()}

        if img_names is None:  # Find images without labels
            existing_labels = {img_path.name for img_path in self.labels_dir.iterdir() if img_path.is_file()}
            img_names = existing_images - existing_labels
        elif not set(img_names) <= existing_images:
            raise FileNotFoundError(f"Can't make labels for missing images: {existing_images - img_names}")

        try:
            label_maker = self._get_label_maker(self.label_type)
        except KeyError as e:
            log.exception(f"Unknown label_type: {self.label_type}")
            raise e

        for img_name in tqdm(img_names):
            label_maker(
                assoc=self,
                img_name=img_name, 
                logger=log)


    def _check_label_type(self, label_type : str):
        """Check if label_type is allowed."""
        if not label_type in LABEL_MAKERS.keys():
            raise ValueError(f"Unknown label_type: {label_type}")


    def _get_label_maker(self, label_type : str) -> Callable:
        """Return label maker for label_type"""
        return LABEL_MAKERS[label_type]


    def _compare_existing_imgs_to_imgs_df(self):
        """
        Safety check that compares the set of images in the images_dir with the set of images in self.imgs_df

        Raises: 
            Exception if there are images in the dataset's images subdirectory that are not in self.imgs_df.
        """

        # Find the set of existing images in the dataset, ...
        existing_images = {img_path.name for img_path in self.images_dir.iterdir() if img_path.is_file()}

        # ... then if the set of images is a strict subset of the images in imgs_df ...
        if existing_images < set(self.imgs_df.index):
            
            # ... log a warning 
            log.warning(f"There images in self.imgs_df that are not in the images_dir {self.images_dir}.")

        # ... and if it is not a subset, ...
        if not existing_images <= set(self.imgs_df.index):
            
            # ... log an error...
            message = f"There are images in the dataset's images subdirectory that are not in self.imgs_df."
            log.error(message)

            # ...and raise an exception.
            raise Exception(message)

        