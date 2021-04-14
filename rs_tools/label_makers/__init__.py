"""
Label makers for different types of labels.

__make_geotif_label_categorical__: make categorical label.
__make_geotif_label_onehot__: make one-hot encoded categorical label.
__make_geotif_label_soft_categorical__: make soft categorical (i.e. probabilistic) label.
"""

from rs_tools.label_makers.make_geotif_label_categorical import _make_geotif_label_categorical
from rs_tools.label_makers.make_geotif_label_onehot import _make_geotif_label_onehot
from rs_tools.label_makers.make_geotif_label_soft_categorical import _make_geotif_label_soft_categorical