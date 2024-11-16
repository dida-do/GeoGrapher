"""Contains mix-in class to save and load BaseModels."""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from importlib import import_module
from inspect import getmro, isabstract, isclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from geographer.base_model_dict_conversion.base_model_dict_conversion_functional import (  # noqa: E501
    eval_nested_base_model_dict,
    get_nested_base_model_dict,
)

logger = logging.getLogger(__name__)


class SaveAndLoadBaseModelMixIn:
    """Mix-in class to save and load BaseModels."""

    @abstractmethod
    def save(self):
        """Save instance to file."""
        pass

    def _save(self, json_file_path: Path | str) -> None:
        """Save to json_file."""
        # Use to implement save method with file_path determined by use case
        json_file_path = Path(json_file_path)
        if json_file_path.suffix != ".json":
            raise ValueError("Need file path to .json file")
        json_file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(json_file_path, "w", encoding="utf-8") as file:
            base_model_dict = get_nested_base_model_dict(self)
            json.dump(base_model_dict, file, ensure_ascii=False, indent=4)

    @classmethod
    def from_json_file(
        cls,
        json_file_path: Path | str,
        constructor_symbol_table: dict[str, Any] | None = None,
    ) -> Any:
        """Load and return saved BaseModel."""
        if constructor_symbol_table is None:
            constructor_symbol_table = {}

        # add all classes in geographer inherited from BaseModel
        # to constructor symbol table
        geographer_dir = Path(__file__).resolve().parent.parent
        for py_file_path in geographer_dir.rglob("*.py"):
            py_file_path_no_suffix = str(py_file_path.with_suffix(""))
            idx = py_file_path_no_suffix.rfind("geographer")
            module_import_str = ".".join(Path(py_file_path_no_suffix[idx:]).parts)

            # import the module and iterate through its attributes
            try:
                module = import_module(module_import_str)
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)

                    if (
                        isclass(attribute)
                        and attribute_name not in constructor_symbol_table
                        and BaseModel in getmro(attribute)
                        and not isabstract(attribute)
                    ):
                        constructor_symbol_table[attribute_name] = attribute
            except Exception:
                logger.debug(f"Error when importing {module_import_str}")

        # open json and load
        with open(json_file_path) as file:
            saved_base_model_dict = json.load(file)
            loaded_base_model = eval_nested_base_model_dict(
                saved_base_model_dict, constructor_symbol_table
            )
            return loaded_base_model
