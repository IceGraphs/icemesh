"""
Configuration parameters for icemesh.
"""

from pathlib import Path
from pydantic import BaseModel
import yaml

class DataConfig(BaseModel):
    root_dir: Path                  = "."    # All other data paths are relative to this root directory.
    wavi_outputs_subdir: Path       = "WAVI_simulations/outputs"
    wavi_checkpoints_subdir: Path   = "WAVI_simulations/checkpoints"
    wavi_simulation: str            = ""
    mesh_subdir: Path               = "preprocessed_datasets"
    mesh_filename: str              = ""

class ModelConfig(BaseModel):
    delta_time: float               = 1.0     # Time delta between files.   'dt' in orig notebooks.
    history_size: int               = 0       # Additional timesteps besides current to include in input.   'k' in orig notebooks.
    
class Config(BaseModel):
    """Class for icemesh configuration parameters."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    
    @classmethod
    def from_dict(cls, config_dict):
        """Read configs from a dict."""
        if not isinstance(config_dict, dict):
            raise TypeError("Input must be a dictionary.")
        return cls(**config_dict)

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
