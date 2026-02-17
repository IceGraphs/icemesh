"""A test file for config.py."""
import pytest

from icemesh.config import Config
import yaml


@pytest.fixture
def config_parameters():
    return {
        "data": {
            "root_dir": "."
        },
        "model": {
            "k": 4
        }
    }

@pytest.fixture
def config_file(tmp_path, config_parameters):
    """Return a config file."""
    # change a value
    config_parameters["model"]["k"] = 3
    filename = tmp_path / "icemesh_config.yaml"
    with open(filename , "w") as file:
        yaml.dump(config_parameters, file, sort_keys=False)
    return filename

def test_from_yaml(config_file):
    """Test Config.from_yaml."""
    config = Config.from_yaml(config_file)
    assert config.data["root_dir"] == "."
    assert config.model["k"] == 3
