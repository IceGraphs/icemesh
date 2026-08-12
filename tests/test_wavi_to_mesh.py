"""Integration tests for converting WAVI trajectories to NetCDF datasets."""

from pathlib import Path

import pytest
import xarray as xr

from icemesh import Config, read_netcdf_file, wavi_to_mesh

# A single known WAVI simulation used by the individual conversion tests.
SIMULATION = "rwnr_from_SMB0.30_gT1.00e-03__T100_traj001"


@pytest.fixture
def config() -> Config:
    """Load the test configuration and verify that external data is available."""
    test_config = Config.from_yaml("tests/wavi_to_mesh_config.yaml")

    # The WAVI trajectories are stored locally and are not included in GitHub.
    wavi_dir = (
        test_config.data.root_dir
        / test_config.data.wavi_trajectories_subdir
    )

    # Skip these integration tests on systems without the external WAVI data.
    if not wavi_dir.is_dir():
        pytest.skip(f"External WAVI test data is unavailable: {wavi_dir}")

    return test_config


def compare_output(test_path: Path, gt_path: Path) -> None:
    """Compare a generated NetCDF dataset with its reference dataset."""
    assert test_path.exists(), f"Generated dataset not found: {test_path}"
    assert gt_path.exists(), f"Reference dataset not found: {gt_path}"

    # Compare the contents rather than the raw NetCDF file bytes.
    with (
        xr.open_dataset(test_path) as result,
        xr.open_dataset(gt_path) as reference,
    ):
        xr.testing.assert_allclose(result, reference)


def test_one_simulation(config: Config) -> None:
    """Convert and validate one simulation without node-type features."""
    config.model.use_node_types = False

    output_paths = wavi_to_mesh(config, SIMULATION)
    assert output_paths, "No output dataset was generated."

    mesh_path = output_paths[0]

    # The reference file must have the same filename as the generated file.
    mesh_gt_path = (
        config.data.root_dir
        / config.data.mesh_gt_subdir
        / mesh_path.name
    )

    compare_output(mesh_path, mesh_gt_path)


def test_one_simulation_node_types(config: Config) -> None:
    """Convert and validate one simulation with node-type features."""
    config.model.use_node_types = True

    # Use a separate filename to avoid overwriting the output without node types.
    config.model.file_prefix = "8km_node_types"

    output_paths = wavi_to_mesh(config, SIMULATION)
    assert output_paths, "No output dataset was generated."

    mesh_path = output_paths[0]
    mesh_gt_path = (
        config.data.root_dir
        / config.data.mesh_gt_subdir
        / mesh_path.name
    )

    compare_output(mesh_path, mesh_gt_path)


def test_multi_simulation(config: Config) -> None:
    """Convert all available simulations and validate their outputs."""
    output_paths = wavi_to_mesh(config)

    assert output_paths, "No output datasets were generated."

    # Compare every generated dataset with the corresponding reference dataset.
    for mesh_path in output_paths:
        mesh_gt_path = (
            config.data.root_dir
            / config.data.mesh_gt_subdir
            / mesh_path.name
        )
        compare_output(mesh_path, mesh_gt_path)


def test_read_netcdf_file(config: Config) -> None:
    """Verify that a generated NetCDF dataset can be read."""
    output_paths = wavi_to_mesh(config, SIMULATION)
    assert output_paths, "No output dataset was generated."

    # read_netcdf_file expects a filename relative to mesh_subdir.
    dataset = read_netcdf_file(config, output_paths[0].name)

    assert dataset is not None
    dataset.close()