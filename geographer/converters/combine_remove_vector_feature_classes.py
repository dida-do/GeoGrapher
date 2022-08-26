"""Create a new dataset from an existing one by combining and/or removing
vector feature classes."""

from __future__ import annotations

import logging
import shutil
from codecs import ignore_errors
from typing import Optional, Union

import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from pydantic import Field
from tqdm.auto import tqdm

from geographer import Connector
from geographer.creator_from_source_dataset_base import DSCreatorFromSource
from geographer.global_constants import VECTOR_FEATURES_INDEX_NAME
from geographer.label_makers.label_maker_base import LabelMaker
from geographer.utils.utils import concat_gdfs, deepcopy_gdf

log = logging.Logger(__name__)


class DSConverterCombineRemoveClasses(DSCreatorFromSource):
    """Create a new dataset from an existing one by combining and/or removing
    vector feature classes."""

    classes: list[Union[str, list[str]]] = Field(
        description="Classes to keep and combine. See docstring."
    )
    new_class_names: Optional[list[str]] = Field(
        default=None, description="Names of new classes"
    )
    class_separator: str = Field(
        default="+", description="Separator used when combining class names."
    )
    new_background_class: Optional[str] = Field(
        default=None, description="Class to be set as new background class"
    )
    remove_imgs: bool = Field(
        default=True,
        description="Whether to remove images not containing new classes from disk",
    )
    label_maker: Optional[LabelMaker] = Field(
        default=None, description="Optional LabelMaker. If given, will update labels."
    )

    def _create(self):
        self._create_or_update()

    def _update(self):
        self._create_or_update()

    def _create_or_update(self):
        """Create a new dataset/connector from an existing one by combining
        and/or removing vector feature classes. Works for both categorical and
        soft-categorical label types.

        Warning:
            Will only add images and vector features from the source dataset, which is
            assumed to have grown in size. Deletions in the source dataset will not be
            inherited.

        Args:
            source_data_dir: data_dir of source dataset/connector
            target_data_dir: data_dir of target dataset/connector. If None (default
                value), will convert in place, i.e. overwrite source dataset and
                connector of tifs.
            classes: vector feature classes in existing dataset/connector to be kept
                and combined in new dataset/connector. E.g. [['ct', 'ht'], 'wr', ['h']]
                will combine the 'ct' and 'ht' classes, and also keep the 'wr' and 'h'
                classes. Along with the regular vector feature classes one may also use
                the background class here.
            new_class_names: optional list of names of new vector feature classes
                corresponding to classes. Defaults to joining the names of existing
                using the class_separator (which defaults to class_separator).
            class_separator: used if the new_class_names argument is not provided
                to join the names of existing vector feature classes that are to be
                kept. Defaults to class_separator.
            new_background_class: optional new background class, defaults to None,
                i.e. old background class
            remove_imgs: If True, remove images not containing vector features
                belonging to the vector feature classes to be kept.

        Returns:
            The Connector of the new dataset.

        Note:
            For the purposes of this function the background classes will be treated
            as regular vector feature classes. In particular, if you do not include
            them in the classes argument, vector features of the background class will
            be lost.
        """

        # Determine classes
        classes = list(  # convert strings in classes to singleton lists
            map(lambda x: x if isinstance(x, list) else [x], self.classes)
        )
        classes_to_keep = [
            class_ for list_of_classes in classes for class_ in list_of_classes
        ]
        new_class_names = self._get_new_class_names(classes)

        self._run_safety_checks(classes_to_keep, new_class_names)

        # Set information about background ...
        if self.new_background_class is not None:
            self.target_connector.background_class = self.new_background_class
        elif self.source_connector.background_class not in new_class_names:
            self.target_connector.attrs["background_class"] = None
        # ... and vector feature classes in self.target_connector.
        self.target_connector.task_vector_feature_classes = [
            class_
            for class_ in new_class_names
            if class_ != self.target_connector.background_class
        ]

        features_from_source_df = self._combine_or_remove_classes_from_vector_features(
            label_type=self.source_connector.label_type,
            vector_features=self.source_connector.vector_features,
            all_source_vector_feature_classes=self.source_connector.all_vector_feature_classes,
            classes=classes,
            new_class_names=new_class_names,
        )

        # need this later
        features_to_add_to_target_dataset = set(features_from_source_df.index) - set(
            self.target_connector.vector_features.index
        )

        # THINK ABOUT THIS!!!!
        # if we are creating a new soft-categorical dataset adjust columns
        # of empty self.target_connector.vector_features
        if (
            len(self.target_connector.vector_features) == 0
            and self.target_connector.label_type == "soft-categorical"
        ):
            empty_vector_features_with_corrected_columns = self._combine_or_remove_classes_from_vector_features(
                label_type="soft-categorical",
                vector_features=self.target_connector.vector_features,
                all_source_vector_feature_classes=self.source_connector.all_vector_feature_classes,
                classes=classes,
                new_class_names=new_class_names,
            )
            self.target_connector.vector_features = (
                empty_vector_features_with_corrected_columns
            )

        self.target_connector.add_to_vector_features(
            features_from_source_df.loc[list(features_to_add_to_target_dataset)]
        )

        # Determine which images to copy to target dataset
        imgs_in_target_dataset_before_addings_imgs_from_source_dataset = (
            {img_path.name for img_path in self.target_connector.images_dir.iterdir()}
            if self.target_connector.images_dir.exists()
            else set()
        )
        imgs_in_source_images_dir = {
            img_path.name for img_path in self.source_connector.images_dir.iterdir()
        }
        if self.remove_imgs:
            imgs_in_source_that_should_be_in_target = {
                # all images in the source dataset ...
                img_name
                for img_name in self.source_connector.raster_imgs.index
                # ... that intersect with the vector features that will be kept.
                if (
                    not set(
                        self.source_connector.vector_features_intersecting_img(img_name)
                    ).isdisjoint(features_from_source_df.index)
                )
                and (self.source_connector.images_dir / img_name).exists()
            }
        else:
            imgs_in_source_that_should_be_in_target = imgs_in_source_images_dir
        imgs_to_copy_to_target_dataset = (
            imgs_in_source_that_should_be_in_target
            - imgs_in_target_dataset_before_addings_imgs_from_source_dataset
        )

        # Copy those images
        self.target_connector.images_dir.mkdir(parents=True, exist_ok=True)
        for img_name in tqdm(
            imgs_to_copy_to_target_dataset, desc="Copying raster images"
        ):
            source_img_path = self.source_connector.images_dir / img_name
            target_img_path = self.target_connector.images_dir / img_name
            shutil.copyfile(source_img_path, target_img_path)

        # add images to self.target_connector
        df_of_imgs_to_add_to_target_dataset = self.source_connector.raster_imgs.loc[
            list(imgs_to_copy_to_target_dataset)
        ]
        self.target_connector.add_to_raster_imgs(df_of_imgs_to_add_to_target_dataset)

        if self.label_maker is not None:

            # Determine labels to delete:
            # For each image that already existed in the target dataset ...
            for (
                img_name
            ) in imgs_in_target_dataset_before_addings_imgs_from_source_dataset:
                # ... if among the vector features intersecting it in the target dataset ...
                vector_features_intersecting_img = set(
                    self.target_connector.vector_features_intersecting_img(img_name)
                )
                # ... there is a *new* (vector) geometry ...
                if (
                    vector_features_intersecting_img & features_to_add_to_target_dataset
                    != set()
                ):
                    # ... then we need to update the label for it, so we delete the current label.
                    self.label_maker.delete_labels(
                        connector=self.target_connector, img_names=[img_name]
                    )

            # make labels
            self.label_maker.make_labels(connector=self.target_connector)

        # remember original type
        if self.target_connector.label_type == "categorical":
            self.target_connector.vector_features.loc[
                features_to_add_to_target_dataset, "orig_type"
            ] = self.source_connector.vector_features.loc[
                features_to_add_to_target_dataset, "type"
            ]

        return self.target_connector

    def _get_new_class_names(self, classes: list[str]) -> list[str]:
        # new_class_names
        if self.new_class_names is None:
            new_class_names = list(map(self.class_separator.join, classes))
        else:
            new_class_names = self.new_class_names
            assert len(new_class_names) == len(
                set(new_class_names)
            ), "new_class_names need to be distinct!"
            assert len(new_class_names) == len(
                classes
            ), "there should be as many new_class_names as there are classes!"
        return new_class_names

    def _run_safety_checks(
        self, classes_to_keep: list[str], new_class_names: list[str]
    ):

        if not set(classes_to_keep) <= set(
            self.source_connector.all_vector_feature_classes
        ):
            classes_not_in_source_dataset = set(classes_to_keep) - set(
                self.source_connector.all_vector_feature_classes
            )
            raise ValueError(
                f"The following classes are not in self.source_connector.all_vector_feature_classes: {classes_not_in_source_dataset}"
            )
        if not len(classes_to_keep) == len(set(classes_to_keep)):
            raise ValueError(
                "a vector feature class in the source dataset can only be in at most one of the new classes"
            )

        if (
            self.new_background_class is not None
            and self.new_background_class not in new_class_names
        ):
            raise ValueError(f"new_background_class not in {self.new_class_names}")

    def _combine_or_remove_classes_from_vector_features(
        self,
        label_type: str,
        vector_features: GeoDataFrame,
        classes: list[Union[str, list[str]]],
        new_class_names: list[str],
        all_source_vector_feature_classes: list[str],
    ) -> GeoDataFrame:
        """
        Args:
            label_type: [description]
            vector_features: [description]
            classes:
            new_class_names:

        Returns:
            GeoDataFrame: [description]
        """
        if label_type not in {"categorical", "soft-categorical"}:
            raise ValueError(f"Unknown label_type: {label_type}")

        vector_features = deepcopy_gdf(vector_features)

        classes_to_keep = [
            class_ for list_of_classes in classes for class_ in list_of_classes
        ]

        if label_type == "categorical":

            def get_new_class(class_: str) -> str:
                for count, classes_ in enumerate(classes):
                    if class_ in classes_:
                        return new_class_names[count]

            # keep only vector features belonging to vector feature we want to keep
            vector_features = vector_features.loc[
                vector_features["type"].apply(lambda class_: class_ in classes_to_keep)
            ]
            # rename to new classes
            vector_features.loc[:, "type"] = vector_features["type"].apply(
                get_new_class
            )

        elif label_type == "soft-categorical":

            def prob_of_class_names(classes: list[str]) -> list[str]:
                answer = list(map(lambda class_: f"prob_of_class_{class_}", classes))
                return answer

            # drop cols of classes we don't want to keep
            classes_to_drop = [
                class_
                for class_ in all_source_vector_feature_classes
                if class_ not in classes_to_keep
            ]
            cols_to_drop = prob_of_class_names(classes_to_drop)
            vector_features = vector_features.drop(columns=cols_to_drop)

            # create temporary dataframe to avoid column name conflicts
            # when renaming/deleting etc
            temp_vector_features = pd.DataFrame()
            temp_vector_features.index.name = vector_features.index.name

            # for each row/(vector) geometry find sum of probabilities
            # for the remaining vector feature classes
            cols_with_probs_of_remaining_classes = prob_of_class_names(classes_to_keep)
            sum_of_probs_of_remaining_classes = pd.DataFrame(
                vector_features[cols_with_probs_of_remaining_classes].sum(axis=1),
                columns=["sum"],
                index=vector_features.index,
            )
            rows_where_sum_is_zero = sum_of_probs_of_remaining_classes["sum"] == 0

            # remove rows/vector features which do not belong to remaining classes
            vector_features = vector_features.loc[~rows_where_sum_is_zero]
            sum_of_probs_of_remaining_classes = sum_of_probs_of_remaining_classes.loc[
                ~rows_where_sum_is_zero
            ]

            # renormalize probabilities to sum to 1
            vector_features.loc[
                :, cols_with_probs_of_remaining_classes
            ] = vector_features[cols_with_probs_of_remaining_classes].div(
                sum_of_probs_of_remaining_classes["sum"], axis=0
            )

            # combine probabilities of new_classes and drop old classes
            for classes_of_new_class, new_class_name in zip(classes, new_class_names):
                cols_of_probs_to_be_added = prob_of_class_names(classes_of_new_class)
                temp_vector_features[
                    f"prob_of_class_{new_class_name}"
                ] = vector_features[cols_of_probs_to_be_added].sum(axis=1)
                vector_features = vector_features.drop(
                    columns=cols_of_probs_to_be_added
                )

            # add new columns
            vector_features = GeoDataFrame(
                pd.concat(
                    [vector_features, temp_vector_features], axis=1  # column axis
                ),
                crs=vector_features.crs,
            )
            vector_features.index.name = VECTOR_FEATURES_INDEX_NAME

            # Recompute most likely type column.
            vector_features["most_likely_class"] = vector_features[
                temp_vector_features.columns
            ].apply(
                lambda s: ",".join(
                    map(
                        lambda col_name: col_name[15:],
                        s[(s == s.max()) & (s != 0)].index.values,
                    )
                ),
                axis=1,
            )

        return vector_features
