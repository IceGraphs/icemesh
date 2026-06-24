"""A test file for wavi_to_mesh.py."""

import pytest
import subprocess
from icemesh import Config, wavi_to_mesh, read_pt_file
from pathlib import Path


@pytest.fixture
def config():
    return Config.from_yaml("tests/wavi_to_mesh_config.yaml")


def compare_output(test_path: Path, gt_path: Path) -> int:
    command = ["./tests/compare.sh", test_path, gt_path]
    process = subprocess.run(command, capture_output=True)
    return process.returncode


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
    returncode = compare_output(test_path=mesh_path, gt_path=mesh_gt_path)
    assert returncode == 0


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
    returncode = compare_output(test_path=mesh_path, gt_path=mesh_gt_path)
    assert returncode == 0


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
    returncode = compare_output(test_path=mesh_dir, gt_path=mesh_gt_dir)
    assert returncode == 0


def test_read_pt_file(config):
    """Converting one simulation from WAVI to mesh."""
    
    mesh_filename = "rw_lhs_from_SMB0.29_gT3.12e-03__T100_traj001.pt"
    mesh_path = config.data.root_dir / config.data.mesh_subdir / mesh_filename
    
    read_pt_file(config, mesh_path)
