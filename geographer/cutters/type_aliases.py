"""Type aliases."""

from __future__ import annotations

from typing import Optional, Tuple, Union  # noqa: I001,I005

# Tuple instead of tuple because pydantic needs old-style
# type declarations for python 3.8
RasterSize = Optional[Union[int, Tuple[int, int]]]
