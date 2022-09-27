"""Test segmentation label creation.

Test SegLabelMakerCategorical and SegLabelMakerSoftCategorical.
"""

from utils import get_test_dir

from geographer import Connector
from geographer.label_makers import (
    SegLabelMakerCategorical,
    SegLabelMakerSoftCategorical,
)


def test_label_maker_categorical_seg():
    """Test SegLabelMakerCategorical."""
    data_dir = get_test_dir() / "cut_source"
    connector = Connector.from_data_dir(data_dir)

    label_maker = SegLabelMakerCategorical()
    label_maker.delete_labels(connector)
    label_maker.make_labels(
        connector=connector,
    )


def test_label_maker_soft_categorical_seg():
    """Test SegLabelMakerSoftCategorical."""
    data_dir = get_test_dir() / "cut_source"
    connector = Connector.from_data_dir(data_dir)
    class_names = connector.all_vector_classes
    assert len(class_names) == 1
    class_name = class_names[0]
    connector.vectors[f"prob_of_class_{class_name}"] = 1.0

    label_maker = SegLabelMakerSoftCategorical(add_background_band=True)
    label_maker.delete_labels(connector)
    label_maker.make_labels(
        connector=connector,
    )


if __name__ == "__main__":
    test_label_maker_categorical_seg()
    test_label_maker_soft_categorical_seg()
