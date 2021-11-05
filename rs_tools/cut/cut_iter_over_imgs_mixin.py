"""
Mixin that implements a general-purpose higher order function to create
or update datasets of GeoTiffs from existing ones by iterating over images.
"""

from __future__ import annotations
from typing import Union, List, Any, Optional, TYPE_CHECKING
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from geopandas import GeoDataFrame

if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator
from rs_tools.global_constants import DATA_DIR_SUBDIRS
from rs_tools.cut.single_img_cutter_base import SingleImgCutter
from rs_tools.cut.img_filter_predicates import ImgFilterPredicate
from rs_tools.cut.img_filter_predicates import AlwaysTrue as AlwaysTrueImgs

logger = logging.getLogger(__name__)

class CreateDSCutIterOverImgsMixIn(object):

    def create_or_update_dataset_iter_over_imgs(
            self,
            create_or_update : str,
            img_cutter : SingleImgCutter,
            source_data_dir : Optional[Union[str, Path]] = None,
            target_data_dir : Optional[Union[str, Path]] = None,
            img_filter_predicate : ImgFilterPredicate = AlwaysTrueImgs()
            ) -> ImgPolygonAssociator:
        """
        Higher order method to create or update a data set of GeoTiffs
        by iterating over images in the source dataset.

        Create or update a data set of GeoTiffs (images, labels, and associator)
        in target_data_dir from the data set of GeoTiffs in source_data_dir by
        iterating over the images in the source dataset/associator that have
        not been cut to images in the target_data_dir (i.e. all images if the
        target dataset doesn not exist yet), filtering the images
        using the img_filter_predicate, and then cutting using an img_cutter.

        Warning:
            Make sure this does exactly what you want when updating an existing data_dir (e.g. if new polygons have been addded to the source_data_dir that overlap with existing labels in the target_data_dir these labels will not be updated. This should be fixed!). It might be safer to just recut the source_data_dir.

        Args:
            create_or_update (str) : One of 'create' or 'update'.
            source_data_dir (Union[str, Path]): data directory (images, labels,
                associator) containing the GeoTiffs to be cut from.
                Use only if updating a dataset.
            target_data_dir (Union[str, Path]): data directory of target
                dataset. Use only if creating a dataset.
            img_cutter (SingleImgCutter): single image cutter used to cut
                selected images in the source dataset.
            img_filter_predicate (ImgFilterPredicate, optional):
                predicate to filter images. Defaults to AlwaysTrue().
        """

        if not create_or_update in {'create', 'update'}:
            raise ValueError(f"Unknown create_or_update arg {create_or_update}, should be one of 'create', 'update'.")

        if create_or_update == 'create':

            # Check args
            if source_data_dir is not None:
                raise ValueError(f"TODOTODO")
            if target_data_dir is None:
                raise ValueError("TODOTODO")

            # Create source_assoc
            source_assoc = self

            # Create target assoc, ...
            target_assoc = self.empty_assoc_same_format_as(target_data_dir)
            target_assoc._update_from_source_dataset_dict['cut_imgs'] = []

            # ..., image data dirs, ...
            for dir in target_assoc.image_data_dirs:
                dir.mkdir(parents=True, exist_ok=True)

            # ... and the associator dir.
            target_assoc.assoc_dir.mkdir(parents=True, exist_ok=True)

            # Make sure no associator files already exist.
            if list(target_assoc.assoc_dir.iterdir()) != []:
                raise Exception(f"The assoc_dir in {target_assoc.assoc_dir} should be empty!")

        elif create_or_update == 'update':

            # Check args
            if target_data_dir is not None:
                raise ValueError(f"TODOTODO")
            if source_data_dir is None:
                raise ValueError("TODOTODO")

            target_assoc = self
            source_assoc = self.__class__.from_data_dir(source_data_dir)

        # Remember information to determine for which images to generate new labels
        imgs_in_target_dataset_before_update = set(target_assoc.imgs_df.index)
        added_polygons = []  # updated as we iterate

        # dict to temporarily store information which will be appended to target_assoc's imgs_df after cutting
        new_imgs_dict = {index_or_col_name: [] for index_or_col_name in [source_assoc.imgs_df.index.name] + list(source_assoc.imgs_df.columns)}

        target_assoc.add_to_polygons_df(source_assoc.polygons_df)

        imgs_in_src_that_have_been_cut = target_assoc._update_from_source_dataset_dict['cut_imgs']
        mask_imgs_in_src_that_have_not_been_cut = ~source_assoc.imgs_df.index.isin(imgs_in_src_that_have_been_cut)
        names_of_imgs_in_src_that_have_not_been_cut = source_assoc.imgs_df.loc[mask_imgs_in_src_that_have_not_been_cut].index

        # Iterate over all images in source dataset that have not been cut.
        for img_name in tqdm(names_of_imgs_in_src_that_have_not_been_cut, desc='Cutting dataset: '):

            # If filter condition is satisfied, (if not, don't do anything) ...
            if img_filter_predicate(
                    img_name,
                    target_assoc=target_assoc,
                    new_img_dict=new_imgs_dict,
                    source_assoc=source_assoc):

                # ... cut the images (and their labels) and remember information to be appended to target_assoc imgs_df in return dict
                imgs_from_single_cut_dict = img_cutter(
                                                img_name=img_name,
                                                new_polygons_df=target_assoc.polygons_df,
                                                new_graph=target_assoc._graph)

                # Make sure img_cutter returned dict with same keys as needed by new_imgs_dict.
                assert set(imgs_from_single_cut_dict.keys()) == set(target_assoc.imgs_df.columns) | {target_assoc.imgs_df.index.name}, f"dict returned by img_cutter doesn't contain the same keys as needed by new_imgs_dict!"

                # Accumulate information for the new imgs in new_imgs_dict.
                for key in new_imgs_dict.keys():
                            new_imgs_dict[key] += (imgs_from_single_cut_dict[key])

                new_img_names = imgs_from_single_cut_dict[source_assoc.imgs_df.index.name]
                img_bounding_rectangles = imgs_from_single_cut_dict['geometry']
                for new_img_name, img_bounding_rectangle in zip(new_img_names, img_bounding_rectangles):

                    # Update target_assoc._update_from_source_dataset_dict
                    target_assoc._update_from_source_dataset_dict['cut_imgs'].append(img_name)

                    # Update graph and modify polygons_df in target_assoc
                    target_assoc._add_img_to_graph_modify_polygons_df(
                        img_name=new_img_name,
                        img_bounding_rectangle=img_bounding_rectangle)

        # Extract accumulated information about the imgs we've created in the target dataset into a dataframe...
        new_imgs_df = GeoDataFrame(
                        new_imgs_dict,
                        crs=target_assoc.imgs_df.crs)
        new_imgs_df.set_index(target_assoc.imgs_df.index.name, inplace=True)

        # ... and append it to self.imgs_df.
        data_frames_list = [target_assoc.imgs_df, new_imgs_df]
        target_assoc.imgs_df = GeoDataFrame(pd.concat(data_frames_list), crs=data_frames_list[0].crs)

        # For those images that existed before the update and now intersect with newly added polygons ...
        imgs_w_new_polygons = [img_name for polygon_name in added_polygons for img_name in target_assoc.imgs_intersecting_polygon(polygon_name) if img_name in imgs_in_target_dataset_before_update]
        # Delete the old labels (since they won't show the new polygons)...
        for img_name in imgs_w_new_polygons:
            label_path = target_assoc.labels_dir / img_name
            label_path.unlink(missing_ok=True)
        # ... and generate new ones.
        target_assoc.make_labels(img_names=imgs_w_new_polygons)

        # Finally, save associator to disk.
        target_assoc.save()

        return target_assoc