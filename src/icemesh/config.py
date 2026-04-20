"""Configuration parameters for icemesh."""

from pathlib import Path
import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    root_dir: Path = "."  # All other data paths are relative to this root directory.
    wavi_outputs_subdir: Path = "WAVI_simulations/outputs"
    wavi_checkpoints_subdir: Path = "WAVI_simulations/checkpoints"
    mesh_subdir: Path = "preprocessed_datasets"


class ModelConfig(BaseModel):
    delta_time: float = 1.0  # Time delta between files used for finite-difference targets.   #TODO(tvl) rem Note: 'dt' in orig notebooks.
    history_size: int = 0  # Additional timesteps besides current to include in input.   #TODO(tvl) rem Note: 'k' in original notebooks.
    use_node_types: bool = False # If True, compute and return one-hot node-type encodings.
    num_node_types: int = 3
    node_type_interior: int = 0
    node_type_left_no_slip: int = 1
    node_type_free_slip: int = 2

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
        """Read config from a config.yaml file.

        If key is not found in config.yaml, the default value is used.
        """
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file {config_file} not found.")

        with open(config_file) as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exception:
                raise SyntaxError(f"Error parsing config file {config_file}.") from exception
        return cls(**config_dict)
