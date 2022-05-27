from abc import abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path
import json

from geographer.base_model_dict_conversion.base_model_dict_conversion_functional import eval_nested_base_model_dict, get_nested_base_model_dict


class SaveAndLoadBaseModelMixIn:
    """
    Mix-in class to save and load BaseModels.
    """

    @abstractmethod
    def save(self):
        pass

    def _save(self, json_file_path: Union[str, Path]) -> None:
        """Save to json_file"""
        #Use to implement save method with file_path determined by use case
        json_file_path = Path(json_file_path)
        if json_file_path.suffix != '.json':
            raise ValueError("Need file path to .json file")
        json_file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(json_file_path, 'w', encoding='utf-8') as file:
            base_model_dict = get_nested_base_model_dict(self)
            json.dump(base_model_dict, file, ensure_ascii=False, indent=4)

    @classmethod
    def from_json_file(
        cls,
        json_file_path: Union[Path, str],
        constructor_symbol_table: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Load and return saved BaseModel"""
        with open(json_file_path) as file:
            saved_base_model_dict = json.load(file)
            loaded_base_model = eval_nested_base_model_dict(
                saved_base_model_dict, constructor_symbol_table)
            return loaded_base_model