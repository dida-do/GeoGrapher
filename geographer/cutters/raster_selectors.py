"""Callable classes for selecting a sublist from a list of rasters.

Used by cutting functions.
"""

from __future__ import annotations

import collections
import random
from abc import abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from geographer.connector import Connector


class RasterSelector(collections.abc.Callable, BaseModel):
    """ABC for selecting from a list of rasters.

    Subclasses are used by DSCutterIterOverVectors.

    Subclasses should implement a __call__method that has the arguments
    and behavior given below.
    """

    @abstractmethod
    def __call__(
        self,
        raster_names_list: list[str],
        target_connector: Connector,
        new_rasters_dict: dict,
        source_connector: Connector,
        cut_rasters: dict[str, list[str]],
        **kwargs: Any,
    ) -> list[str]:
        """Select rasters to create cutouts from a list of rasters.

        Args:
            raster_names_list: list of rasters to be selected from
            target_connector: connector of target dataset.
            new_rasters_dict: dict with keys index or column names of
                target_connector.rasters and values lists of entries correspondong
                to rasters
            source_connector: connector of source dataset that new rasters are being cut
                out from
            cut_rasters: dict containing for each raster in the target dataset
                the list of rasters in the source from which cutouts have been
                created for it
            kwargs: Optional keyword arguments

        Returns:
            sublist of raster_names_list

        Note:
            - Override to subclass. If raster_names_list is empty an empty list
            should be returned.

            - The new_vectors and new_graph arguments contain all the
            information available to decide which rasters to select. They should not be
            modified by this method.

            It should be possible for the returned sublist to depend on all the
            information in the source and target connectors. The RasterSelector used by
            the cutting function
            in geographer.cut.cut_iter_over_vectors. This function does not
            concatenate the information about the new rasters that have been cut to the
            target_connector.rasters until after all vector features have been
            iterated over. We want to use the vector features filter
            predicate _during_ this iteration, so we allow the call function to also
            depend on a new_rasters_dict argument which contains the information about
            the new rasters that have been cut. Unlike the target_connector.rasters,
            the target_connector.vectors and graph are updated during the
            iteration. One should thus think of the target_connector and
            new_rasters_dict arguments together as the actual the target connector
            argument.
        """


class RandomRasterSelector(RasterSelector):
    """RasterSelector that randomly selects randomly from a list of rasters."""

    target_raster_count: int = 1

    def __call__(
        self,
        vector_name: str | int,
        raster_names_list: list[str],
        target_connector: Connector,
        new_rasters_dict: dict,
        source_connector: Connector,
        cut_rasters: dict[str, list[str]],
        **kwargs: Any,
    ) -> list[str]:
        """Randomly select rasters from a list of rasters.

        Select target_raster_count - #{raster_count of vector feature
        in target_connector} rasters (or if not possible less) from
        raster_names_list.
        """
        containing_rasters_in_target = set(
            target_connector.rasters_containing_vector(vector_name)
        )
        # if the raster_size is smaller than the vector_feature polygon,
        # the result of `rasters_containing_vector` will always be zero.
        # So in this case we consider the cut_rasters.
        cut_rasters_in_source = {
            raster
            for raster in cut_rasters[vector_name]
            # if none of the containing_rasters_in_target were generated from raster
            if not any(
                {
                    raster_.startswith(Path(raster).stem)
                    for raster_ in containing_rasters_in_target
                }
            )
        }

        num_rasters_already_there = len(containing_rasters_in_target) + len(
            cut_rasters_in_source
        )
        target_num_rasters_to_sample = (
            self.target_raster_count - num_rasters_already_there
        )

        # can only sample a non-negative number of rasters
        target_num_rasters_to_sample = max(0, target_num_rasters_to_sample)

        # can only sample from raster_names_list
        num_rasters_to_sample = min(
            len(raster_names_list), target_num_rasters_to_sample
        )

        return random.sample(raster_names_list, num_rasters_to_sample)


random_raster_selector = RandomRasterSelector()
