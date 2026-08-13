"""Configuration parameters for icemesh."""

from pathlib import Path
import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    root_dir: Path = "."  # All other data paths are relative to this root directory.

    # Regular grid.
    wavi_trajectories_subdir: Path = "WAVI_simulations/trajectories"
    wavi_checkpoints_subdir: Path = "WAVI_simulations/checkpoints"
    
    # Irregular mesh.
    mesh_subdir: Path = "preprocessed_datasets"
    mesh_gt_subdir: Path = "preprocessed_datasets"


class ModelConfig(BaseModel):
    delta_time: float = 1.0  # Time delta between files used for finite-difference targets.   #TODO(tvl) rem Note: 'dt' in orig notebooks.
    history_size: int = 0  # Additional timesteps besides current to include in input.   #TODO(tvl) rem Note: 'k' in original notebooks.
    future_size: int = 0
    use_node_types: bool = False
    num_node_types: int = 3
    node_type_interior: int = 0
    node_type_free_slip: int = 1
    node_type_left_no_slip: int = 2
    selected_features_x: list = []
    minimum_thickness: float = 50.0
    delaunay_edge_factor: float = 1.5
    file_prefix: str = ""
    coordinate_units: str = "m"
    time_units: str = "years"
    overwrite: bool = False


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
