"""A test file for wavi_to_mesh.py."""

from pathlib import Path
import pytest
import yaml
from icemesh.config import Config
from icemesh.wavi_to_mesh import wavi_to_mesh, read_pt_file

# TODO rem tmp imports
#import copy
import subprocess


@pytest.fixture
def config():
    return Config.from_yaml("tests/wavi_to_mesh_config.yaml")


def test_one_simulation(config):
    """Converting one simulation from WAVI to mesh."""
    
    wavi_simulation = "rw_lhs_from_SMB0.29_gT3.12e-03__T100_traj001"
    mesh_filename = f"{wavi_simulation}.pt"
    mesh_path = config.data.root_dir / config.data.mesh_subdir / mesh_filename
    mesh_gt_path = config.data.root_dir / config.data.mesh_gt_subdir / mesh_filename
    if mesh_path.exists():
        mesh_path.unlink()

    wavi_to_mesh(config, wavi_simulation)

    # Check that all relevant files exist.
    assert config.data.root_dir.exists()
    assert mesh_path.exists()
    assert mesh_path.stat().st_size > 0
    assert mesh_gt_path.exists()
    assert mesh_gt_path.stat().st_size > 0

    # Compare the output to ground truth data.
    command = ["./tests/compare.sh", mesh_path, mesh_gt_path]
    process = subprocess.run(command, capture_output=True)
    assert process.returncode == 0


def test_one_simulation_node_types(config):
    """Converting one simulation from WAVI to mesh."""
    
    config.model.use_node_types = True

    wavi_simulation = "rw_lhs_from_SMB0.29_gT3.12e-03__T100_traj001"
    mesh_filename = f"{wavi_simulation}_incl_node_enc.pt"
    mesh_path = config.data.root_dir / config.data.mesh_subdir / mesh_filename
    mesh_gt_path = config.data.root_dir / config.data.mesh_gt_subdir / mesh_filename
    if mesh_path.exists():
        mesh_path.unlink()

    wavi_to_mesh(config, wavi_simulation)

    # Check that all relevant files exist.
    assert config.data.root_dir.exists()
    assert mesh_path.exists()
    assert mesh_path.stat().st_size > 0
    assert mesh_gt_path.exists()
    assert mesh_gt_path.stat().st_size > 0

    # Compare the output to ground truth data.
    command = ["./tests/compare.sh", mesh_path, mesh_gt_path]
    process = subprocess.run(command, capture_output=True)
    assert process.returncode == 0


def test_multi_simulation(config):
    """Converting one simulation from WAVI to mesh."""
    
    mesh_dir = config.data.root_dir / config.data.mesh_subdir
    mesh_gt_dir = config.data.root_dir / config.data.mesh_gt_subdir
    mesh_glob = "*"
    if mesh_dir.exists():
        for mesh_path in mesh_dir.glob(mesh_glob):
            mesh_path.unlink()

    wavi_to_mesh(config)

    # Check that all relevant files exist.
    assert config.data.root_dir.exists()
    assert mesh_dir.exists()
    assert mesh_gt_dir.exists()

    # Compare the output to ground truth data.
    command = ["./tests/compare.sh", mesh_dir, mesh_gt_dir]
    process = subprocess.run(command, capture_output=True)
    assert process.returncode == 0


def test_read_pt_file(config):
    """Converting one simulation from WAVI to mesh."""
    
    mesh_filename = "rw_lhs_from_SMB0.29_gT3.12e-03__T100_traj001.pt"
    mesh_path = config.data.root_dir / config.data.mesh_subdir / mesh_filename
    
    read_pt_file(config, mesh_path)
    