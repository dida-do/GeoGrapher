"""Test saving/loading nested BaseModels."""

from pathlib import Path

import git
from pydantic import BaseModel

from geographer.base_model_dict_conversion.save_load_base_model_mixin import (
    SaveAndLoadBaseModelMixIn,
)


class InnermostBaseModel(BaseModel):
    """Innermost BaseModel."""

    int_value: int
    str_value: str


class NestedBaseModel(BaseModel):
    """Nested BaseModel."""

    dict_value: dict
    innermost_base_model: InnermostBaseModel


class OutermostBaseModel(BaseModel, SaveAndLoadBaseModelMixIn):
    """Outermost BaseModel."""

    nested_base_model: NestedBaseModel

    json_path: Path

    def save(self):
        """Save the model."""
        self._save(self.json_path)


def test_save_load_nested_base_model():
    """Test saving and loading nested BaseModel."""
    repo = git.Repo(".", search_parent_directories=True)
    repo_root = Path(repo.working_tree_dir)
    temp_dir = repo_root / "tests/data/temp/"
    outermost_base_model_json_path = temp_dir / "outermost_base_model.json"

    # Define a nested model
    outermost_base_model = OutermostBaseModel(
        nested_base_model=NestedBaseModel(
            dict_value={
                "a": 1,
                "b": {
                    "c": None,
                },
            },
            innermost_base_model=InnermostBaseModel(
                int_value=2,
                str_value="str_value",
            ),
        ),
        json_path=outermost_base_model_json_path,
    )

    """
    Test saving and loading a nested BaseModel
    """
    # save
    outermost_base_model.save()

    # load
    outermost_base_model_from_json = OutermostBaseModel.from_json_file(
        outermost_base_model_json_path,
        constructor_symbol_table={
            "InnermostBaseModel": InnermostBaseModel,
            "NestedBaseModel": NestedBaseModel,
            "OutermostBaseModel": OutermostBaseModel,
        },
    )
    # make sure saving and loading again doesn't change anything
    assert (
        outermost_base_model_from_json.model_dump() == outermost_base_model.model_dump()
    )


if __name__ == "__main__":
    test_save_load_nested_base_model()
