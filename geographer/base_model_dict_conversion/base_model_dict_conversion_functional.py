"""Utils for conversion of BaseModels to nested dicts.

The nested dicts keep track of class constructors and are used for
serializing BaseModels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from geographer.utils.utils import removeprefix


def get_nested_base_model_dict(
    base_model_obj_or_dict: BaseModel | dict | Any,
) -> dict:
    """Return nested dict for BaseModel or dict.

    Return nested dict for nested BaseModel containing fields and
    BaseModel constructors or of dict.
    """
    if isinstance(base_model_obj_or_dict, dict):
        dict_ = base_model_obj_or_dict
        dict_items = base_model_obj_or_dict.items()
    elif isinstance(base_model_obj_or_dict, BaseModel):
        dict_ = base_model_obj_or_dict.model_dump()
        dict_items = base_model_obj_or_dict

    dict_or_base_model_fields_dict = {
        add_escape_str(key): get_nested_base_model_dict(val)
        for key, val in dict_items
        if isinstance(val, (dict, BaseModel))
        and key in dict_.keys()  # to avoid excluded fields for BaseModels
    }
    path_fields_dict = {
        add_escape_str(key): {"constructor_Path": str(val)}
        for key, val in dict_items
        if isinstance(val, Path)
        and key in dict_.keys()  # to avoid excluded fields for BaseModels
    }
    tuple_fields_dict = {
        add_escape_str(key): {
            "constructor_tuple": (
                get_nested_base_model_dict(val)
                if isinstance(val, (BaseModel, dict))
                else val
            )
        }
        for key, val in dict_items
        if isinstance(val, tuple)
        and key in dict_.keys()  # to avoid excluded fields for BaseModels
    }
    remaining_fields_dict = {
        add_escape_str(key): val
        for key, val in dict_items
        if key in dict_.keys()  # to avoid excluded fields for BaseModels
        and not isinstance(val, (BaseModel, Path, dict, tuple))
    }

    if isinstance(base_model_obj_or_dict, dict):
        result = {
            **remaining_fields_dict,
            **dict_or_base_model_fields_dict,
            **path_fields_dict,
            **tuple_fields_dict,
        }
    elif isinstance(base_model_obj_or_dict, BaseModel):
        result = {
            f"constructor_{type(base_model_obj_or_dict).__name__}": {
                **remaining_fields_dict,
                **dict_or_base_model_fields_dict,
                **path_fields_dict,
                **tuple_fields_dict,
            }
        }

    return result


def get_nested_dict(obj: BaseModel | dict | Any) -> dict | Any:
    """Return nested dict if obj is a BaseModel or dict else return obj."""


def eval_nested_base_model_dict(
    dict_or_field_value: dict | Any,
    constructor_symbol_table: dict[str, Any] | None = None,
) -> BaseModel | Any:
    """Evaluate nested BaseModel dict (or field contents).

    Args:
        dict_or_field_value: nested base model dict or field value
        constructor_symbol_table: symbol table of constructors. Defaults to None.

    Returns:
        BaseModel or component
    """
    if not isinstance(dict_or_field_value, dict):
        return dict_or_field_value
    elif is_path_constructor_dict(dict_or_field_value):
        return Path(get_path_constructor_arg(dict_or_field_value))
    elif is_tuple_constructor_dict(dict_or_field_value):
        tuple_components = [
            eval_nested_base_model_dict(val, constructor_symbol_table)
            for val in get_tuple_constructor_args(dict_or_field_value)
        ]
        return tuple(tuple_components)
    elif is_base_model_constructor_dict(dict_or_field_value):
        constructor = get_base_model_constructor(
            dict_or_field_value, constructor_symbol_table=constructor_symbol_table
        )
        constructor_args_dict = eval_nested_base_model_dict(
            get_base_model_constructor_kwargs_dict(dict_or_field_value),
            constructor_symbol_table,
        )
        return constructor(**constructor_args_dict)
    else:
        return {
            remove_escape_str(key): eval_nested_base_model_dict(
                val, constructor_symbol_table
            )
            for key, val in dict_or_field_value.items()
        }


def is_path_constructor_dict(dict_: dict) -> bool:
    """Return True if and only if dict_ encodes a pathlib.Path."""
    return (
        isinstance(dict_, dict)
        and len(dict_) == 1
        and list(dict_.keys())[0] == "constructor_Path"
    )


def get_path_constructor_arg(dict_: dict) -> str:
    """Return argument string for Path constructor."""
    key = list(dict_.keys())[0]
    assert isinstance(dict_[key], str)
    return dict_[key]


def is_tuple_constructor_dict(dict_: dict) -> bool:
    """Return True if and only if dict_ encodes a tuple."""
    return (
        isinstance(dict_, dict)
        and len(dict_) == 1
        and list(dict_.keys())[0] == "constructor_tuple"
    )


def get_tuple_constructor_args(dict_: dict) -> list:
    """Return list of tuple components."""
    key = list(dict_.keys())[0]
    assert isinstance(dict_[key], (list, tuple))
    return dict_[key]


def is_base_model_constructor_dict(dict_: dict) -> bool:
    """Return True if and only if the dict_ encodes a BaseModel instance."""
    keys = list(dict_.keys())
    if keys:
        key = keys[0]
    return (
        isinstance(dict_, dict)
        and len(dict_) == 1
        and isinstance(key, str)
        and key.startswith("constructor_")
        and not removeprefix(key, "constructor_").startswith("constructor_")
    )


def get_base_model_constructor(
    dict_: dict, constructor_symbol_table: dict[str, Any] | None = None
) -> BaseModel:
    """Return constructor corresponding to encoded BaseModel.

    Args:
        dict_ (dict): nested base model dict
        constructor_symbol_table (dict[str, Any] | None, optional): optional symbol
            table of constructors. Defaults to None.

    Returns:
        constrctor of BaseModel
    """
    symbol_table = globals()
    if constructor_symbol_table is not None:
        symbol_table.update(constructor_symbol_table)
    constructor_name = removeprefix(list(dict_.keys())[0], "constructor_")
    return symbol_table[constructor_name]


def get_base_model_constructor_kwargs_dict(dict_: dict) -> dict:
    """Return kwargs for constructor corresponding to encoded BaseModel."""
    key = list(dict_.keys())[0]
    assert isinstance(dict_[key], dict)
    return dict_[key]


def add_escape_str(key: Any) -> Any:
    """Increment by 1 the number of 'constructor_' prefixes."""
    if isinstance(key, str) and key.startswith("constructor_"):
        return "constructor_" + key
    else:
        return key


def remove_escape_str(key: Any) -> Any:
    """Decrement by 1 the number of 'constructor_' prefixes."""
    if isinstance(key, str) and key.startswith("constructor_"):
        return removeprefix(key, "constructor_")
    else:
        return key
