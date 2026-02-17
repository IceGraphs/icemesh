"""
Configuration parameters for icemesh.
"""

from pathlib import Path
import yaml

class Config:
    """
    Base class for configuration parameters.
    """
    data: dict
    # root_dir: Path = "."
    model: dict
    # k: int = 0

    def __init__(self, **config_dict):
        self.__dict__.update(config_dict)

    @classmethod
    def from_yaml(cls, config_file: str):
        """
        Read config from a config.yaml file.

        If key is not found in config.yaml, the default value is used.
        """
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file {config_file} not found.")

        with open(config_file, "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exception:
                raise SyntaxError(f"Error parsing config file {config_file}.") from exception
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Read configs from a dict."""
        if not isinstance(config_dict, dict):
            raise TypeError("Input must be a dictionary.")
        return cls(**config_dict)
