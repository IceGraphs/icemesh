"""Integration tests for converting WAVI trajectories to NetCDF datasets."""

from pathlib import Path

import pytest
import xarray as xr

from icemesh import Config, read_netcdf_file, wavi_to_mesh


@pytest.fixture
def config() -> Config:
    """Load the test configuration and verify that external data is available."""
    return Config.from_yaml("tests/wavi_to_mesh_config.yaml")


@pytest.fixture
def wavi_simulation() -> str:
    # A single known WAVI simulation used by the individual conversion tests.
    return "rwnr_from_SMB0.30_gT1.00e-03__T100_traj001"


def delete_output(config: Config, wavi_simulation: str = ""):
    # Converting WAVI to mesh files will skip files that already exist.
    # Delete this output to force the functions to run.
    mesh_dir = config.data.root_dir / config.data.mesh_subdir
    if wavi_simulation:
        mesh_path = mesh_dir / f"{wavi_simulation}.pt"
        if mesh_path.exists():
            mesh_path.unlink()
    else:
        mesh_glob = "*"
        if mesh_dir.exists():
            for mesh_path in mesh_dir.glob(mesh_glob):
                mesh_path.unlink()


def compare_output(test_path: Path, gt_path: Path) -> None:
    """Compare a generated NetCDF dataset with its reference dataset."""
    assert test_path.exists(), f"Generated dataset not found: {test_path}"
    assert gt_path.exists(), f"Reference dataset not found: {gt_path}"

    # Compare the contents of two NetCDF files.
    opened = False
    with (
        xr.open_dataset(test_path) as result,
        xr.open_dataset(gt_path) as reference,
    ):
        opened = True
        xr.testing.assert_allclose(result, reference)
    assert opened, f"Failed to open generated or reference dataset: {test_path}, {gt_path}"


def test_one_simulation(config: Config, wavi_simulation: str) -> None:
    """Convert and validate one simulation without node-type features."""
    config.model.use_node_types = False

    # Clean up existing output.
    delete_output(config, wavi_simulation)

    output_paths = wavi_to_mesh(config, wavi_simulation)
    assert output_paths, "No output dataset was generated."

    # Compare the generated file to the reference file with the same name.
    mesh_path = output_paths[0]
    mesh_gt_path = config.data.root_dir / config.data.mesh_gt_subdir / mesh_path.name
    compare_output(mesh_path, mesh_gt_path)


def test_one_simulation_node_types(config: Config, wavi_simulation: str) -> None:
    """Convert and validate one simulation with node-type features."""
    config.model.use_node_types = True

    # Clean up existing output.
    delete_output(config, wavi_simulation)

    # Use a separate filename to avoid overwriting the output without node types.
    config.model.file_prefix = "8km_node_types"

    output_paths = wavi_to_mesh(config, wavi_simulation)
    assert output_paths, "No output dataset was generated."

    # Compare the generated file to the reference file with the same name.
    mesh_path = output_paths[0]
    mesh_gt_path = config.data.root_dir / config.data.mesh_gt_subdir / mesh_path.name
    compare_output(mesh_path, mesh_gt_path)


def test_multi_simulation(config: Config) -> None:
    """Convert all available simulations and validate their outputs."""

    # Clean up existing output.
    delete_output(config)

    output_paths = wavi_to_mesh(config)

    assert output_paths, "No output datasets were generated."

    # Compare every generated dataset with the corresponding reference dataset.
    for mesh_path in output_paths:
        mesh_gt_path = config.data.root_dir / config.data.mesh_gt_subdir / mesh_path.name
        compare_output(mesh_path, mesh_gt_path)


def test_read_netcdf_file(config: Config, wavi_simulation: str) -> None:
    """Verify that a generated NetCDF dataset can be read."""
    output_paths = wavi_to_mesh(config, wavi_simulation)
    assert output_paths, "No output dataset was generated."

    # read_netcdf_file expects a filename relative to mesh_subdir.
    dataset = read_netcdf_file(config, output_paths[0].name)
    assert dataset is not None
