from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from rs_tools import ImgPolygonAssociator


class CreateDSCutIterBaseMixIn:
    """
    Mixin that implements methods common to both
    CreateDSCutIterOverPolygonsMixIn and CreateDSCutIterOverImgsMixIn
    """

    def _get_source_and_target_assocs(self, create_or_update: str,
                                      source_data_dir: Path,
                                      target_data_dir: Path) -> Tuple[ImgPolygonAssociator, ImgPolygonAssociator]:
        """Return source and target associators"""

        if not create_or_update in {'create', 'update'}:
            raise ValueError(
                f"Unknown create_or_update arg {create_or_update}, should be one of 'create', 'update'."
            )

        if create_or_update == 'create':
            # Check args
            if source_data_dir is not None:
                raise ValueError(
                    f"create mode: source_data_dir needs to be None")
            if target_data_dir is None:
                raise ValueError(f"create mode: need target_data_dir")

            # Create source_assoc
            source_assoc = self

            # Create target assoc, ...
            target_assoc = self.empty_assoc_same_format_as(target_data_dir)
            target_assoc._update_from_source_dataset_dict[
                'cut_imgs'] = defaultdict(list)

            # ..., image data dirs, ...
            for dir in target_assoc.image_data_dirs:
                dir.mkdir(parents=True, exist_ok=True)

            # ... and the associator dir.
            target_assoc.assoc_dir.mkdir(parents=True, exist_ok=True)

            # Make sure no associator files already exist.
            if list(target_assoc.assoc_dir.iterdir()) != []:
                raise Exception(
                    f"The assoc_dir in {target_assoc.assoc_dir} should be empty!"
                )
