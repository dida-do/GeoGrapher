"""
Create a new dataset from an existing one
by combining and/or removing segmentation classes.

TODO: TEST!
"""

import logging
import shutil
from typing import List, Optional, Union

import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from pydantic import Field
from tqdm.auto import tqdm
from rs_tools.creator_from_source_dataset_base import DSCreatorFromSource
from rs_tools.global_constants import VECTOR_DATA_INDEX_NAME
from rs_tools.utils.utils import concat_gdfs, deepcopy_gdf
from rs_tools import ImgPolygonAssociator

log = logging.Logger(__name__)


class DSConverterCombineRemoveClasses(DSCreatorFromSource):
    """
    Create a new dataset from an existing one by combining and/or removing segmentation classes
    """

    seg_classes: List[Union[str, List[str]]] = Field(
        description="Classes to keep and combine. See docstring.")
    new_seg_classes: Optional[List[str]] = Field(
        default=None, description="Names of new classes")
    class_separator: str = Field(
        default="+", description="Separator used when combining class names.")
    new_background_class: Optional[str] = Field(
        default=None, description="Class to be set as new background class")
    remove_imgs: bool = Field(
        default=True,
        description=
        "Whether to remove images not containing new classes from disk")

    def _create(self) -> ImgPolygonAssociator:
        return self._create_or_update()

    def _update(self) -> ImgPolygonAssociator:
        return self._create_or_update()

    def _create_or_update(self):
        """Create a new dataset/associator from an existing one by combining
        and/or removing segmentation classes. Works for both categorical and
        soft-categorical label types.

        Warning:
            Will only add images and (vector) geometries from the source dataset, which is assumed to have grown in size. Deletions in the source dataset will not be inherited.

        Args:
            source_data_dir (pathlib.Path or str): data_dir of source dataset/associator
            target_data_dir (pathlib.Path or str, optional): data_dir of target dataset/associator. If None (default value), will convert in place, i.e. overwrite source dataset and associator of tifs.
            seg_classes: (List[Union[List[str], str]]) segmentation classes in existing dataset/associator to be kept and combined in new dataset/associator. E.g. [['ct', 'ht'], 'wr', ['h']] will combine the 'ct' and 'ht' classes, and also keep the 'wr' and 'h' classes. Along with the regular segmentation classes one may also use the background class here.
            new_seg_class_names: (Optional[List[str]]) optional list of names of new segmentation classes corresponding to seg_classes. Defaults to joining the names of existing using the class_separator (which defaults to class_separator).
            class_separator: (str) used if the new_seg_class_names argument is not provided to join the names of existing segmentation classes that are to be kept. Defaults to class_separator.
            new_background_class (Optional[str]): optional new background class, defaults to None, i.e. old background class
            remove_imgs: (bool). If True, remove images not containing geometries of the segmentation classes to be kept.

        Returns:
            The ImgPolygonAssociator of the new dataset.

        Note:
            For the purposes of this function the background classes will be treated as regular segmentation classes. In particular, if you do not include them in the seg_classes argument, (vector) geometries of the background class will be lost.
        """

        # Determine classes
        seg_classes = list(  # convert strings in seg_classes to singleton lists
            map(lambda x: x if isinstance(x, list) else [x], self.seg_classes))
        classes_to_keep = [
            class_ for list_of_classes in seg_classes
            for class_ in list_of_classes
        ]
        new_seg_classes = self._get_new_seg_classes(seg_classes)

        self._run_safety_checks(classes_to_keep, new_seg_classes)

        # Set information about background ...
        if self.new_background_class is not None:
            self.target_assoc.background_class = self.new_background_class
        elif self.source_assoc.background_class not in new_seg_classes:
            self.target_assoc.attrs['background_class'] = None
        # ... and segmentation classes in self.target_assoc.
        self.target_assoc.segmentation_classes = [
            class_ for class_ in new_seg_classes
            if class_ != self.target_assoc.background_class
        ]

        geoms_from_source_df = self._combine_or_remove_seg_classes_from_vector_data(
            # label_type=self.source_assoc.label_type,
            # vector_data=self.source_assoc.vector_data,
            seg_classes=seg_classes,
            new_seg_classes=new_seg_classes,
        )
        # all_geom_classes=self.source_assoc.all_geom_classes)

        # need this later
        geoms_to_add_to_target_dataset = set(
            geoms_from_source_df.index) - set(
                self.target_assoc.vector_data.index)

        # THINK ABOUT THIS!!!!
        # if we are creating a new soft-categorical dataset adjust columns of empty self.target_assoc.vector_data
        if len(self.target_assoc.vector_data
               ) == 0 and self.target_assoc.label_type == 'soft-categorical':
            empty_vector_data_with_corrected_columns = self._combine_or_remove_seg_classes_from_vector_data(
                # label_type=self.target_assoc.label_type,
                # vector_data=self.target_assoc.vector_data,
                seg_classes=seg_classes,
                new_seg_classes=new_seg_classes,
                # all_geom_classes=self.source_assoc.
                # all_geom_classes  # self, since we already set classes in self.target_assoc
            )
            self.target_assoc.vector_data = empty_vector_data_with_corrected_columns

        self.target_assoc.add_to_vector_data(geoms_from_source_df)

        # Determine which images to copy to target dataset
        imgs_in_target_dataset_before_addings_imgs_from_source_dataset = {
            img_path.name
            for img_path in self.target_assoc.images_dir.iterdir()
        }
        imgs_in_source_images_dir = {
            img_path.name
            for img_path in self.source_assoc.images_dir.iterdir()
        }
        if self.remove_imgs:
            imgs_in_source_that_should_be_in_target = {
                # all images in the source dataset ...
                img_name
                for img_name in imgs_in_source_images_dir
                # ... that intersect with the (vector) geometries that will be kept.
                if set(self.source_assoc.geoms_intersecting_img(img_name))
                & set(geoms_from_source_df.index) != set()
            }
        else:
            imgs_in_source_that_should_be_in_target = imgs_in_source_images_dir
        imgs_to_copy_to_target_dataset = imgs_in_source_that_should_be_in_target - imgs_in_target_dataset_before_addings_imgs_from_source_dataset

        # Copy those images
        for img_name in tqdm(imgs_to_copy_to_target_dataset):
            source_img_path = self.source_assoc.images_dir / img_name
            target_img_path = self.target_assoc.images_dir / img_name
            shutil.copyfile(source_img_path, target_img_path)

        # add images to self.target_assoc
        df_of_imgs_to_add_to_target_dataset = self.source_assoc.img_data.loc[
            list(imgs_to_copy_to_target_dataset)]
        self.target_assoc.add_to_img_data(df_of_imgs_to_add_to_target_dataset)

        # Determine labels to delete:
        # For each image that already existed in the target dataset ...
        for img_name in imgs_in_target_dataset_before_addings_imgs_from_source_dataset:
            # ... if among the (vector) geometries intersecting it in the target dataset ...
            geoms_intersecting_img = set(
                self.target_assoc.geoms_intersecting_img(img_name))
            # ... there is a *new* (vector) geometry ...
            if geoms_intersecting_img & geoms_to_add_to_target_dataset != set(
            ):
                # ... then we need to update the label for it, so we delete the current label.
                (self.target_assoc.labels_dir /
                 img_name).unlink(missing_ok=True)

        # make labels
        self.target_assoc.make_labels()

        # remember original type
        if self.target_assoc.label_type == 'categorical':
            self.target_assoc.vector_data.loc[
                geoms_to_add_to_target_dataset,
                'orig_type'] = self.source_assoc.vector_data.loc[
                    geoms_to_add_to_target_dataset, 'type']

        self.target_assoc.save()
        self.save()
        log.info("new_cat_assoc_remove_classes: done!")

        return self.target_assoc

    def _get_new_seg_classes(self, seg_classes: List[str]) -> List[str]:
        # new_seg_classes
        if self.new_seg_classes is None:
            new_seg_classes = list(map(self.class_separator.join, seg_classes))
        else:
            new_seg_classes = self.new_seg_classes
            assert len(new_seg_classes) == len(set(
                new_seg_classes)), "new_seg_class_names need to be distinct!"
            assert len(new_seg_classes) == len(
                seg_classes
            ), "there should be as many new_seg_class_names as there are seg_classes!"
        return new_seg_classes

    def _run_safety_checks(self, classes_to_keep: List[str],
                           new_seg_classes: List[str]):

        if not set(classes_to_keep) <= set(
                self.source_assoc.all_classes):
            classes_not_in_source_dataset = set(classes_to_keep) - set(
                self.source_assoc.all_classes)
            raise ValueError(
                f"The following classes are not in self.source_assoc.all_geom_classes: {classes_not_in_source_dataset}"
            )
        if not len(classes_to_keep) == len(set(classes_to_keep)):
            raise ValueError(
                "a segmentation class in the source dataset can only be in at most one of the new segmentation classes"
            )

        if self.new_background_class is not None and self.new_background_class not in new_seg_classes:
            raise ValueError(
                f"new_background_class not in {self.new_seg_classes}")

    def _combine_or_remove_seg_classes_from_vector_data(
        self,
        # label_type: str,
        # vector_data: GeoDataFrame,
        seg_classes: List[Union[str, List[str]]],
        new_seg_classes: List[str],
        # all_geom_classes: List[str],
    ) -> GeoDataFrame:
        """
        Args:
            label_type (str): [description]
            vector_data (GeoDataFrame): [description]
            seg_classes (List[str]):
            new_seg_classes (List[str]):
            all_geom_classes (List[str]):

        Returns:
            GeoDataFrame: [description]
        """
        if self.source_assoc.label_type not in {
                'categorical', 'soft-categorical'
        }:
            raise ValueError(
                f"Unknown label_type: {self.source_assoc.label_type}")

        vector_data = deepcopy_gdf(self.source_assoc.vector_data)

        classes_to_keep = [
            class_ for list_of_classes in seg_classes
            for class_ in list_of_classes
        ]

        if self.source_assoc.label_type == 'categorical':

            def get_new_class(class_: str) -> str:
                for count, classes_ in enumerate(seg_classes):
                    if class_ in classes_:
                        return new_seg_classes[count]

            # keep only (vector) geometries belonging to segmentation we want to keep
            vector_data = vector_data.loc[vector_data['type'].apply(
                lambda class_: class_ in classes_to_keep)]
            # rename to new classes
            vector_data.loc[:,
                            'type'] = vector_data['type'].apply(get_new_class)

        elif self.source_assoc.label_type == 'soft-categorical':

            def prob_seg_class_names(classes: List[str]) -> List[str]:
                answer = list(
                    map(lambda class_: f"prob_seg_class_{class_}", classes))
                return answer

            # drop cols of classes we don't want to keep
            classes_to_drop = [
                class_ for class_ in self.source_assoc.all_classes
                if class_ not in classes_to_keep
            ]
            cols_to_drop = prob_seg_class_names(classes_to_drop)
            vector_data = vector_data.drop(columns=cols_to_drop)

            # create temporary dataframe to avoid column name conflicts when renaming/deleting etc
            temp_vector_data = pd.DataFrame()
            temp_vector_data.index.name = vector_data.index.name

            # for each row/(vector) geometry find sum of probabilities for the remaining segmentation classes
            cols_with_probs_of_remaining_classes = prob_seg_class_names(
                classes_to_keep)
            sum_of_probs_of_remaining_classes = pd.DataFrame(
                vector_data[cols_with_probs_of_remaining_classes].sum(axis=1),
                columns=['sum'],
                index=vector_data.index)
            rows_where_sum_is_zero = (
                sum_of_probs_of_remaining_classes['sum'] == 0)

            # remove rows/(vector) geometries which do not belong to remaining classes
            vector_data = vector_data.loc[~rows_where_sum_is_zero]
            sum_of_probs_of_remaining_classes = sum_of_probs_of_remaining_classes.loc[
                ~rows_where_sum_is_zero]

            # renormalize probabilities to sum to 1
            vector_data.loc[:,
                            cols_with_probs_of_remaining_classes] = vector_data[
                                cols_with_probs_of_remaining_classes].div(
                                    sum_of_probs_of_remaining_classes['sum'],
                                    axis=0)

            # combine probabilities of new_classes and drop old classes
            for classes_of_new_seg_class, new_seg_class_name in zip(
                    seg_classes, new_seg_classes):
                cols_of_probs_to_be_added = prob_seg_class_names(
                    classes_of_new_seg_class)
                temp_vector_data[
                    f"prob_seg_class_{new_seg_class_name}"] = vector_data[
                        cols_of_probs_to_be_added].sum(axis=1)
                vector_data = vector_data.drop(
                    columns=cols_of_probs_to_be_added)

            # add new columns
            vector_data = concat_gdfs
            vector_data = pd.concat([vector_data, temp_vector_data],
                                    axis=1)  # column axis
            vector_data.index.name = VECTOR_DATA_INDEX_NAME

            # Recompute most likely type column.
            vector_data["most_likely_class"] = vector_data[
                temp_vector_data.columns].apply(lambda s: ",".join(
                    map(lambda col_name: col_name[15:], s[
                        (s == s.max()) & (s != 0)].index.values)),
                                                axis=1)

        return vector_data
