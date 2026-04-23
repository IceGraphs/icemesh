"""A test file for wavi_to_mesh.py."""

from pathlib import Path
import pytest
import yaml
from icemesh.wavi_to_mesh import wavi_to_mesh, read_pt_file

# TODO rem tmp imports
import copy
import subprocess


# @pytest.fixture
# def config_parameters():
#     return {
#         "data": {
#             "root_dir": ".",
#         },
#         "model": {
#             "history_size": 4,
#         },
#     }


# @pytest.fixture
# def config_file(tmp_path, config_parameters):
#     """Return a config file."""
#     # change a value
#     config_parameters["model"]["history_size"] = 3
#     filename = tmp_path / "icemesh_config.yaml"
#     with open(filename, "w") as file:
#         yaml.dump(config_parameters, file, sort_keys=False)
#     return filename


# def test_from_yaml(config_file):
#     """Test Config.from_yaml."""
#     config = Config.from_yaml(config_file)
#     assert config.data.root_dir == Path()
#     assert config.model.history_size == 3





# TODO(tvl) remove: testing...
if __name__ == "__main__":
    print("exec main")
    config = Config.from_yaml("tests/wavi_to_mesh_config.yaml")

    config_single = copy.deepcopy(config)
    config_single.model.use_node_types = True
    os_filename = (
        "rw_lhs_from_SMB0.29_gT3.12e-03__T100_traj001_incl_node_enc.pt"
        if config_single.model.use_node_types
        else "rw_lhs_from_SMB0.29_gT3.12e-03__T100_traj001.pt"
    )
    os_out = config.data.root_dir / config.data.mesh_subdir / os_filename
    if os_out.exists():
        os_out.unlink()

    one_simulation(config_single, "rw_lhs_from_SMB0.29_gT3.12e-03__T100_traj001")
    # transform_single_grid_to_mesh(config) <- TODO(tvl) rename function to something like this...
    if False:
        multi_simulation(config)
    if False:
        read_pt_file(config, "rw_lhs_from_SMB0.29_gT3.12e-03__T100_traj001.pt")

    process = [
        "./compare.sh",
        config.data.root_dir / "preprocessed_datasets_test",
        config.data.root_dir / "preprocessed_datasets",
    ]
    subprocess.run(process)