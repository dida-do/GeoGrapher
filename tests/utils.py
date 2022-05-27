from pathlib import Path
import git

def get_test_dir():
    """Return directory containing test data"""

    repo = git.Repo('.', search_parent_directories=True)
    return Path(repo.working_tree_dir) / "tests/data"
