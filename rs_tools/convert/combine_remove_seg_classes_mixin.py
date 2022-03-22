"""
TODO: IS THIS DONE? MAYBE TEST AGAIN...

Mix-in that implements creating a new dataset from an existing one
by combining and/or removing segmentation classes.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator

from rs_tools.utils.utils import deepcopy_gdf

log = logging.Logger(__name__)


class CreateDSCombineRemoveSegClassesMixIn(object):

    def create_dataset_by_combining_or_removing_seg_classes(
            self,
            seg_classes: List[Union[str, List[str]]],
            target_data_dir: Union[Path, str],
            new_seg_classes: Optional[List[str]] = None,
            class_separator: str = "+",
            new_background_class: Optional[str] = None,
            remove_imgs: bool = True) -> ImgPolygonAssociator:
        """Create a new dataset by combining and/or removing segmentation
        classes. Works for categorical and soft-categorical label types.

        Args:
            seg_classes (List[Union[str, List[str]]]): [description]
            target_data_dir (Union[Path, str]): [description]
            new_seg_classes (Optional[List[str]], optional): [description]. Defaults to None.
            class_separator (str, optional): [description]. Defaults to "+".
            new_background_class (Optional[str], optional): [description]. Defaults to None.
            remove_imgs (bool, optional): [description]. Defaults to True.

        Returns:
            ImgPolygonAssociator: associator of created dataset

        Example:

            Suppose the associator assoc's dataset has the following segmentation classes:

            'ct', 'ht', 'wr', 'h', 'pt', 'ig', 'bg'

            Then,

            >>> assoc.create_dataset_by_combining_or_removing_seg_classes(
                    target_data_dir=TARGET_DATA_DIR,
                    seg_classes=[['ct', 'ht'], 'wr', ['h']])
            will create a new dataset in TARGET_DATA_DIR in which only the polygons belonging
            to the 'ct', 'ht', 'wr', and 'h' (or having a non-zero probability of belonging to these
            classes if the labels are soft-categorical) will be retained, and the classes 'ct' and 'ht' will be combined to a class 'ct+ht'. If the label type is soft-categorical, the new dataset will still retain any existing prob_seg_class columns for the background class, though all the entries will in this case be zero, since neither 'bg' nor 'ig' were in the seg_classes argument.

        Warning:
            Make sure this works as desired for the edge cases as regards the ignore and background classes!
        """

        target_assoc = self._create_or_update_dataset_by_combining_or_removing_seg_classes(
            create_or_update='create',
            seg_classes=seg_classes,
            target_data_dir=target_data_dir,
            new_seg_classes=new_seg_classes,
            class_separator=class_separator,
            new_background_class=new_background_class,
            remove_imgs=remove_imgs)

        return target_assoc

    def _update_dataset_by_combining_or_removing_seg_classes(
            self) -> ImgPolygonAssociator:
        """Update a dataset created using create_dataset_by_combining_or_removi
        ng_seg_classes_from_existing_dataset when the source dataset has become
        larger.

        Warning:
            Will only add images and polygons from the source dataset, which is assumed to have grown in size. Deletions in the source dataset will not be inherited.
        """

        seg_classes = self._update_from_source_dataset_dict['seg_classes']
        source_data_dir = self._update_from_source_dataset_dict[
            'source_data_dir']
        new_seg_classes = self._update_from_source_dataset_dict[
            'new_seg_classes']
        class_separator = self._update_from_source_dataset_dict[
            'class_separator']
        new_background_class = self._update_from_source_dataset_dict[
            'new_background_class']
        remove_imgs = self._update_from_source_dataset_dict['remove_imgs']

        source_assoc = self.__class__.from_data_dir(source_data_dir)

        source_assoc._create_or_update_dataset_by_combining_or_removing_seg_classes(
            create_or_update='update',
            seg_classes=seg_classes,
            target_assoc=self,
            new_seg_classes=new_seg_classes,
            class_separator=class_separator,
            new_background_class=new_background_class,
            remove_imgs=remove_imgs)

    def _create_or_update_dataset_by_combining_or_removing_seg_classes(
            self,
            create_or_update: str,
            seg_classes: List[Union[str, List[str]]],
            target_data_dir: Optional[Union[Path, str]] = None,
            target_assoc: Optional[ImgPolygonAssociator] = None,
            new_seg_classes: Optional[List[str]] = None,
            class_separator: str = "+",
            new_background_class: Optional[str] = None,
            remove_imgs: bool = True) -> ImgPolygonAssociator:
        """Create a new dataset/associator from an existing one by combining
        and/or removing segmentation classes. Works for both categorical and
        soft-categorical label types.

        Warning:
            Will only add images and polygons from the source dataset, which is assumed to have grown in size. Deletions in the source dataset will not be inherited.

        Args:
            source_data_dir (pathlib.Path or str): data_dir of source dataset/associator
            target_data_dir (pathlib.Path or str, optional): data_dir of target dataset/associator. If None (default value), will convert in place, i.e. overwrite source dataset and associator of tifs.
            seg_classes: (List[Union[List[str], str]]) segmentation classes in existing dataset/associator to be kept and combined in new dataset/associator. E.g. [['ct', 'ht'], 'wr', ['h']] will combine the 'ct' and 'ht' classes, and also keep the 'wr' and 'h' classes. Along with the regular segmentation classes one may also use the background class here.
            new_seg_class_names: (Optional[List[str]]) optional list of names of new segmentation classes corresponding to seg_classes. Defaults to joining the names of existing using the class_separator (which defaults to class_separator).
            class_separator: (str) used if the new_seg_class_names argument is not provided to join the names of existing segmentation classes that are to be kept. Defaults to class_separator.
            new_background_class (Optional[str]): optional new background class, defaults to None, i.e. old background class
            remove_imgs: (bool). If True, remove images not containing polygons of the segmentation classes to be kept.

        Returns:
            The ImgPolygonAssociator of the new dataset.

        Note:
            For the purposes of this function the background classes will be treated as regular segmentation classes. In particular, if you do not include them in the seg_classes argument, polygons of the background class will be lost.
        """

        # argument consistency checks
        if create_or_update not in {'create', 'update'}:
            raise ValueError(
                f"create_or_update argument must be one of 'create' or 'update'"
            )

        # load target associator
        if create_or_update == 'update':

            if target_assoc is None:
                raise ValueError("TODO")
            if target_data_dir is not None:
                raise ValueError("TODO")

        elif create_or_update == 'create':

            if target_assoc is not None:
                raise ValueError("TODO")
            if target_data_dir is None:
                raise ValueError("TODO")

            target_assoc = self.empty_assoc_same_format_as(target_data_dir)
            # Note might have to adjust the columns of target_assoc.polygons_df. We will do this later, as we need to define some arguments first.

            # Create image data dirs ...
            for dir in target_assoc.image_data_dirs:
                dir.mkdir(parents=True, exist_ok=True)
                if list(dir.iterdir()) != []:
                    raise Exception(f"{dir} should be empty!")
            # ... and the associator dir.
            target_assoc.assoc_dir.mkdir(parents=True, exist_ok=True)

            # Make sure no associator files already exist.
            if list(target_assoc.assoc_dir.iterdir()) != []:
                raise Exception(
                    f"The assoc_dir in {target_assoc.assoc_dir} should be empty!"
                )

        seg_classes = list(  # convert strings in seg_classes to singleton lists
            map(lambda x: x if isinstance(x, list) else [x], seg_classes))
        classes_to_keep = [
            class_ for list_of_classes in seg_classes
            for class_ in list_of_classes
        ]

        if not set(classes_to_keep) <= set(self.all_polygon_classes):
            classes_not_in_source_dataset = set(classes_to_keep) - set(
                self.all_polygon_classes)
            raise ValueError(
                "The following classes are not in self.all_polygon_classes: {classes_not_in_source_dataset}"
            )
        if not len(classes_to_keep) == len(set(classes_to_keep)):
            raise ValueError(
                "a segmentation class in the source dataset can only be in at most one of the new segmentation classes"
            )

        # new_seg_classes
        if new_seg_classes is None:
            new_seg_classes = list(
                map(lambda x: class_separator.join(x), seg_classes))
        else:
            assert len(new_seg_classes) == len(set(
                new_seg_classes)), "new_seg_class_names need to be distinct!"
            assert len(new_seg_classes) == len(
                seg_classes
            ), "there should be as many new_seg_class_names as there are seg_classes!"

        if new_background_class is not None:
            assert new_background_class in new_seg_classes

        # Set information about background ...
        if new_background_class is not None:
            target_assoc._params_dict[
                'background_class'] = new_background_class
        elif self._params_dict['background_class'] not in new_seg_classes:
            target_assoc._params_dict['background_class'] = None
        # ... and segmentation classes in target_assoc.
        target_assoc._params_dict['segmentation_classes'] = [
            class_ for class_ in new_seg_classes
            if class_ not in {target_assoc._params_dict['background_class']}
        ]

        polygons_from_source_df = self._combine_or_remove_seg_classes_from_polygons_df(
            label_type=self.label_type,
            polygons_df=self.polygons_df,
            seg_classes=seg_classes,
            new_seg_classes=new_seg_classes,
            all_polygon_classes=self.all_polygon_classes)

        # need this later
        polygons_to_add_to_target_dataset = set(
            polygons_from_source_df.index) - set(
                target_assoc.polygons_df.index)

        # if we are creating a new soft-categorical dataset adjust columns of empty target_assoc.polygons_df
        if create_or_update == 'create' and target_assoc.label_type == 'soft-categorical':
            empty_polygons_df_with_corrected_columns = self._combine_or_remove_seg_classes_from_polygons_df(
                label_type=target_assoc.label_type,
                polygons_df=target_assoc.polygons_df,
                seg_classes=seg_classes,
                new_seg_classes=new_seg_classes,
                all_polygon_classes=self.
                all_polygon_classes  # self, since we already set classes in target_assoc
            )
            target_assoc.polygons_df = empty_polygons_df_with_corrected_columns

        target_assoc.add_to_polygons_df(polygons_from_source_df)

        # Determine which images to copy to target dataset
        imgs_in_target_dataset_before_addings_imgs_from_source_dataset = {
            img_path.name
            for img_path in target_assoc.images_dir.iterdir()
        }
        imgs_in_source_images_dir = {
            img_path.name
            for img_path in self.images_dir.iterdir()
        }
        if remove_imgs:
            imgs_in_source_that_should_be_in_target = {
                # all images in the source dataset ...
                img_name
                for img_name in imgs_in_source_images_dir
                # ... that intersect with the polygons that will be kept.
                if set(self.polygons_intersecting_img(img_name))
                & set(polygons_from_source_df.index) != set()
            }
        else:
            imgs_in_source_that_should_be_in_target = imgs_in_source_images_dir
        imgs_to_copy_to_target_dataset = imgs_in_source_that_should_be_in_target - imgs_in_target_dataset_before_addings_imgs_from_source_dataset

        # Copy those images
        for img_name in tqdm(imgs_to_copy_to_target_dataset):
            source_img_path = self.images_dir / img_name
            target_img_path = target_assoc.images_dir / img_name
            shutil.copyfile(source_img_path, target_img_path)

        # add images to target_assoc
        df_of_imgs_to_add_to_target_dataset = self.imgs_df.loc[list(
            imgs_to_copy_to_target_dataset)]
        target_assoc.add_to_imgs_df(df_of_imgs_to_add_to_target_dataset)

        # Determine labels to delete:
        # For each image that already existed in the target dataset ...
        for img_name in imgs_in_target_dataset_before_addings_imgs_from_source_dataset:
            # ... if among the polygons intersecting it in the target dataset ...
            polygons_intersecting_img = set(
                target_assoc.polygons_intersecting_img(img_name))
            # ... there is a *new* polygon ...
            if polygons_intersecting_img & polygons_to_add_to_target_dataset != set(
            ):
                # ... then we need to update the label for it, so we delete the current label.
                (target_assoc.labels_dir / img_name).unlink(missing_ok=True)

        # make labels
        target_assoc.make_labels()

        # remember original type
        if target_assoc.label_type == 'categorical':
            target_assoc.polygons_df.loc[polygons_to_add_to_target_dataset,
                                         'orig_type'] = self.polygons_df.loc[
                                             polygons_to_add_to_target_dataset,
                                             'type']

        target_assoc._update_from_source_dataset_dict.update({
            'update_method':
            '_update_dataset_by_combining_or_removing_seg_classes',
            'seg_classes':
            seg_classes,
            'source_data_dir':
            self.images_dir.parent,
            'new_seg_classes':
            new_seg_classes,
            'class_separator':
            class_separator,
            'new_background_class':
            new_background_class,
            'remove_imgs':
            remove_imgs
        })

        # save associator
        target_assoc.save()

        log.info(f"new_cat_assoc_remove_classes: done!")

        if create_or_update == 'create':
            return target_assoc

    def _combine_or_remove_seg_classes_from_polygons_df(
        self,
        label_type: str,
        polygons_df: GeoDataFrame,
        seg_classes: List[Union[str, List[str]]],
        new_seg_classes: List[str],
        all_polygon_classes: List[str],
    ) -> GeoDataFrame:
        """
        Args:
            label_type (str): [description]
            polygons_df (GeoDataFrame): [description]
            seg_classes (List[str]):
            new_seg_classes (List[str]):
            all_polygon_classes (List[str]):

        Returns:
            GeoDataFrame: [description]
        """
        if label_type not in {'categorical', 'soft-categorical'}:
            raise ValueError(f"Unknown label_type: {label_type}")

        polygons_df = deepcopy_gdf(polygons_df)

        classes_to_keep = [
            class_ for list_of_classes in seg_classes
            for class_ in list_of_classes
        ]

        if label_type == 'categorical':

            def get_new_class(class_: str) -> str:
                for count, classes_ in enumerate(seg_classes):
                    if class_ in classes_:
                        return new_seg_classes[count]

            # keep only polygons belonging to segmentation we want to keep
            polygons_df = polygons_df.loc[polygons_df['type'].apply(
                lambda class_: class_ in classes_to_keep)]
            # rename to new classes
            polygons_df.loc[:,
                            'type'] = polygons_df['type'].apply(get_new_class)

        elif label_type == 'soft-categorical':

            def prob_seg_class_names(classes: List[str]) -> List[str]:
                answer = list(
                    map(lambda class_: f"prob_seg_class_{class_}", classes))
                return answer

            # drop cols of classes we don't want to keep
            classes_to_drop = [
                class_ for class_ in all_polygon_classes
                if class_ not in classes_to_keep
            ]
            cols_to_drop = prob_seg_class_names(classes_to_drop)
            polygons_df = polygons_df.drop(columns=cols_to_drop)

            # create temporary dataframe to avoid column name conflicts when renaming/deleting etc
            temp_polygons_df = pd.DataFrame()
            temp_polygons_df.index.name = polygons_df.index.name

            # for each row/polygon find sum of probabilities for the remaining segmentation classes
            cols_with_probs_of_remaining_classes = prob_seg_class_names(
                classes_to_keep)
            sum_of_probs_of_remaining_classes = pd.DataFrame(
                polygons_df[cols_with_probs_of_remaining_classes].sum(axis=1),
                columns=['sum'],
                index=polygons_df.index)
            rows_where_sum_is_zero = (
                sum_of_probs_of_remaining_classes['sum'] == 0)

            # remove rows/polygons which do not belong to remaining classes
            polygons_df = polygons_df.loc[~rows_where_sum_is_zero]
            sum_of_probs_of_remaining_classes = sum_of_probs_of_remaining_classes.loc[
                ~rows_where_sum_is_zero]

            # renormalize probabilities to sum to 1
            polygons_df.loc[:,
                            cols_with_probs_of_remaining_classes] = polygons_df[
                                cols_with_probs_of_remaining_classes].div(
                                    sum_of_probs_of_remaining_classes['sum'],
                                    axis=0)

            # combine probabilities of new_classes and drop old classes
            for classes_of_new_seg_class, new_seg_class_name in zip(
                    seg_classes, new_seg_classes):
                cols_of_probs_to_be_added = prob_seg_class_names(
                    classes_of_new_seg_class)
                temp_polygons_df[
                    f"prob_seg_class_{new_seg_class_name}"] = polygons_df[
                        cols_of_probs_to_be_added].sum(axis=1)
                polygons_df = polygons_df.drop(
                    columns=cols_of_probs_to_be_added)

            # add new columns
            polygons_df = pd.concat([polygons_df, temp_polygons_df],
                                    axis=1)  # column axis

            # Recompute most likely type column.
            polygons_df["most_likely_class"] = polygons_df[
                temp_polygons_df.columns].apply(lambda s: ",".join(
                    map(lambda col_name: col_name[15:], s[
                        (s == s.max()) & (s != 0)].index.values)),
                                                axis=1)

        return polygons_df
