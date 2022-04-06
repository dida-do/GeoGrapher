"""
Util functions to convert BaseModels to nested dicts
keeping track of class constructors. Used for serializing BaseModels
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel


def get_nested_base_model_dict(base_model_obj: BaseModel) -> dict:
    """Return nested dict of nested BaseModel containing fields and BaseModel constructors"""
    non_base_model_fields_dict = {
        add_escape_str(key): val
        for key, val in base_model_obj
        if key in base_model_obj.dict().keys()  # to avoid excluded fields
        and not isinstance(val, (BaseModel, Path))
    }
    base_model_fields_dict = {
        add_escape_str(key): get_nested_base_model_dict(val)
        for key, val in base_model_obj if isinstance(val, BaseModel)
        and key in base_model_obj.dict().keys()  # to avoid excluded fields
    }
    path_fields_dict = {
        add_escape_str(key): {
            'constructor_Path': str(val)
        }
        for key, val in base_model_obj if isinstance(val, Path)
        and key in base_model_obj.dict().keys()  # to avoid excluded fields
    }
    return {
        f'constructor_{type(base_model_obj).__name__}':
        non_base_model_fields_dict | base_model_fields_dict | path_fields_dict
    }


def eval_nested_base_model_dict(
    dict_or_field_value: Union[BaseModel, Any],
    constructor_symbol_table: Optional[Dict[str, Any]] = None,
) -> Union[BaseModel, Any]:
    """Evaluate nested BaseModel dict (or field contents)

    Args:
        dict_or_field_value (Union[BaseModel, Any]): nested base model dict or field value
        constructor_symbol_table (Optional[Dict[str, Any]], optional): symbol table of constructors. Defaults to None.

    Returns:
        Union[BaseModel, Any]: _description_
    """
    if not isinstance(dict_or_field_value, dict):
        return dict_or_field_value
    elif is_path_constructor_dict(dict_or_field_value):
        return Path(get_path_constructor_arg(dict_or_field_value))
    elif is_base_model_constructor_dict(dict_or_field_value):
        constructor = get_base_model_constructor(
            dict_or_field_value,
            constructor_symbol_table=constructor_symbol_table)
        constructor_args_dict = eval_nested_base_model_dict(
            get_base_model_constructor_kwargs_dict(dict_or_field_value),
            constructor_symbol_table)
        return constructor(**constructor_args_dict)
    else:
        return {
            remove_escape_str(key):
            eval_nested_base_model_dict(val, constructor_symbol_table)
            for key, val in dict_or_field_value.items()
        }


def is_path_constructor_dict(dict_: dict) -> bool:
    """Return True if and only if dict_ encodes a pathlib.Path"""
    return isinstance(dict_, dict) and len(dict_) == 1 and list(
        dict_.keys())[0] == 'constructor_Path'


def get_path_constructor_arg(dict_: dict) -> str:
    """Return argument string for Path constructor"""
    key = list(dict_.keys())[0]
    assert isinstance(dict_[key], str)
    return dict_[key]


def is_base_model_constructor_dict(dict_: dict) -> bool:
    """Return True if and only if the dict_ encodes a BaseModel instance"""
    keys = list(dict_.keys())
    if keys:
        key = keys[0]
    return isinstance(dict_, dict) and len(dict_) == 1 and isinstance(
        key, str) and key.startswith('constructor_') and not key.removeprefix(
            'constructor_').startswith('constructor_')


def get_base_model_constructor(
        dict_: dict,
        constructor_symbol_table: Optional[Dict[str, Any]] = None) -> bool:
    """
    Return constructor corresponding to encoded BaseModel

    Args:
        dict_ (dict): nested base model dict
        constructor_symbol_table (Optional[Dict[str, Any]], optional): optional symbol table of constructors. Defaults to None.

    Returns:
        bool: _description_
    """
    symbol_table = globals() | (constructor_symbol_table if
                                constructor_symbol_table is not None else {})
    constructor_name = list(dict_.keys())[0].removeprefix('constructor_')
    return symbol_table[constructor_name]


def get_base_model_constructor_kwargs_dict(dict_: dict) -> dict:
    """Return kwargs for constructor corresponding to encoded BaseModel"""
    key = list(dict_.keys())[0]
    assert isinstance(dict_[key], dict)
    return dict_[key]


def add_escape_str(key: Any) -> Any:
    """Increment by 1 the number of 'constructor_' prefixes"""
    if isinstance(key, str) and key.startswith('constructor_'):
        return 'constructor_' + key
    else:
        return key


def remove_escape_str(key: Any) -> Any:
    """Decrement by 1 the number of 'constructor_' prefixes"""
    if isinstance(key, str) and key.startswith('constructor_'):
        return key.removeprefix('constructor_')
    else:
        return key


########################################################
from abc import ABC, abstractmethod


class BaseClassLevel1(ABC, BaseModel):

    a: int
    b: int
    p: Path

    @abstractmethod
    def do_something(self):
        pass


class Level1_A(BaseClassLevel1):

    def do_something(self):
        print("Level1_A: do_something")


class Level1_B(BaseClassLevel1):

    c: int

    def do_something(self):
        print("Level1_B: do_something")


class BaseClassLevel0(ABC, BaseModel):

    obj_level1: BaseClassLevel1
    obj_level2: BaseClassLevel1
    var: int

    @abstractmethod
    def fun(self, x):
        pass


class Level0_A(BaseClassLevel0):

    def fun(self):
        print("Level0_A")


class Level0_B(BaseClassLevel0):

    def fun(self):
        print("Level0_B")
