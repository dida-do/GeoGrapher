from typing import Any, Dict, Optional, Union
from pathlib import Path
import json

from rs_tools.base_model_dict_conversion.base_model_dict_conversion_functional import eval_nested_base_model_dict, get_nested_base_model_dict


class SaveAndLoadBaseModelMixIn:
    """
    Mix-in class to save and load BaseModels.

    Assumes the BaseModel has a data_dir field.
    """

    def save(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        if file_path.suffix != '.json':
            raise ValueError("Need file path to .json file")
        with open(file_path, 'w') as file:
            base_model_dict = get_nested_base_model_dict(self)
            json.dump(base_model_dict, file)

    @classmethod
    def from_file(
        cls,
        json_file_path: Union[Path, str],
        constructor_symbol_table: Optional[Dict[str, Any]] = None,
    ) -> Any:

        with open(json_file_path) as file:
            saved_base_model_dict = json.load(file)
            loaded_base_model = eval_nested_base_model_dict(
                saved_base_model_dict, constructor_symbol_table)
            return loaded_base_model