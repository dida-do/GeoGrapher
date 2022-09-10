"""Type aliases."""

from __future__ import annotations

from typing import Optional, Union, Tuple

# Tuple instead of tuple because pydantic needs old-style
# type declarations for python 3.8
ImgSize = Optional[Union[int, Tuple[int, int]]]
