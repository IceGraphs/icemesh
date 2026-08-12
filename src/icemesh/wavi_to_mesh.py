"""Convert time-ordered WAVI output into NetCDF graph trajectories.

Each input trajectory is a directory of WAVI ``.jld2`` snapshots on a fixed
regular grid. The preprocessing pipeline reads all snapshots, removes nodes
that reach the configured minimum thickness, reconstructs one static Delaunay
mesh and bundles graph samples through time.

Model inputs contain user-selected physical features and optional node
encodings, with historical context length of choice. Targets always contain 
one-step derivatives of ``u``, ``v`` and ``h``. Future physical states and 
prescribed forcing variables (accumulation and basal-melt) fields are retained 
for rollout supervision. One compressed NetCDF file stores the temporal tensors, 
static topology, coordinates, feature metadata and filtering diagnostics for 
each trajectory.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from scipy.spatial import Delaunay, QhullError, cKDTree
from torch_geometric.data import Data
from tqdm import tqdm

from icemesh.config import Config


# Stable complete-feature convention before model-specific selection
FEATURE_NAMES = {
    0: "u",
    1: "v",
    2: "us",
    3: "vs",
    4: "ub",
    5: "vb",
    6: "h",
    7: "b",
    8: "s",
    9: "dhdt",
    10: "accumulation",
    11: "basal_melt",
    12: "grounded_frac",
    13: "av_speed",
    14: "bed_speed",
    15: "weertman_c",
    16: "haf",
    17: "dsdh",
    18: "shelf_strain",
    19: "beta",
    20: "beta_eff",
    21: "tau_bed",
    22: "eta_av",
    23: "quad_f1",
    24: "quad_f2",
    25: "mask",
    26: "param_smb",
    27: "param_gt",
    28: "param_dt",
    29: "node_type_interior",
    30: "node_type_free_slip",
    31: "node_type_left_no_slip",
    32: "mesh_boundary",
}

# Fixed prediction targets and prescribed rollout forcing fields
TARGET_FEATURE_INDICES = [0, 1, 6]
TARGET_FEATURE_NAMES = ["u", "v", "h"]
FORCING_FEATURE_INDICES = [10, 11]
FORCING_FEATURE_NAMES = ["accumulation", "basal_melt"]
THICKNESS_FEATURE_INDEX = 6
MESH_BOUNDARY_FEATURE_INDEX = 32

# Default model input when no explicit selection is provided
DEFAULT_SELECTED_FEATURES = [
    0,
    1,
    6,
    7,
    8,
    10,
    11,
    16,
    29,
    30,
    31,
    32,
]


def _config_section(config: Any, *names: str) -> Any:
    """Return the first available named section from a configuration object.

    The aliases retain compatibility with older configurations that used
    ``directories`` and ``settings`` instead of ``data`` and ``model``.
    """
    for name in names:
        section = getattr(config, name, None)
        if section is not None:
            return section
    raise AttributeError(f"Configuration has none of these sections: {names}")


def _setting(section: Any, *names: str, default: Any = None) -> Any:
    """Return the first available setting or a supplied default value."""
    for name in names:
        if hasattr(section, name):
            return getattr(section, name)
    return default


def natural_sort_key(path: Path) -> list[Any]:
    """Create a key that sorts numbered timestep files chronologically.

    Numeric filename fragments are compared as integers, so timestep 10 sorts
    after timestep 9 rather than after timestep 1.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def validate_selected_features(
    selected_features_x: Sequence[int],
    use_node_types: bool,
) -> list[int]:
    """Validate and normalize complete-feature indices used in ``Data.x``.

    Node encodings at indices 29 to 32 are removed automatically when
    ``use_node_types`` is false. Remaining indices must be unique and available
    in the complete feature convention.

    Args:
        selected_features_x: Complete-feature indices in the desired
            model-input order.
        use_node_types: Whether physical node types and the reconstructed
            boundary are added.

    Returns:
        Validated indices in final model-input order.
    """
    if selected_features_x is None:
        raise ValueError("selected_features_x must contain at least one index")

    selected = [int(index) for index in selected_features_x]

    # Exclude optional node encodings when node types are disabled
    if not use_node_types:
        node_encoding_indices = {29, 30, 31, 32}
        selected = [
            index
            for index in selected
            if index not in node_encoding_indices
        ]

    if not selected:
        raise ValueError("selected_features_x must contain at least one index")

    if len(set(selected)) != len(selected):
        raise ValueError("selected_features_x contains duplicate indices")

    available = set(range(29))

    if use_node_types:
        available.update((29, 30, 31, 32))

    invalid = [index for index in selected if index not in available]

    if invalid:
        raise IndexError(
            f"Unavailable selected feature indices: {invalid}. "
            f"Available indices: {sorted(available)}"
        )

    return selected


def print_selected_features(selected_features_x: Sequence[int]) -> None:
    """Print input, target and forcing feature conventions for inspection."""
    print("\nSelected input features for x:")
    for new_index, original_index in enumerate(selected_features_x):
        print(
            f"  x[{new_index}] = original feature {original_index}: "
            f"{FEATURE_NAMES[original_index]}"
        )

    print("\nFixed targets for y and future_states:")
    for target_index, original_index in enumerate(TARGET_FEATURE_INDICES):
        print(
            f"  target[{target_index}] = original feature {original_index}: "
            f"{FEATURE_NAMES[original_index]}"
        )

    print("\nFuture prescribed forcing fields:")
    for forcing_index, original_index in enumerate(FORCING_FEATURE_INDICES):
        print(
            f"  forcing[{forcing_index}] = original feature {original_index}: "
            f"{FEATURE_NAMES[original_index]}"
        )


def _grid_to_mesh(
    model_config: Any,
    jld2_file: h5py.File,
    dt: float,
) -> tuple[np.ndarray, torch.Tensor]:
    """Convert one WAVI grid snapshot to coordinates and complete features.

    Physical fields are flattened in the stable order defined by
    ``FEATURE_NAMES``. When node types are enabled, three one-hot physical
    boundary channels are appended as indices 29 to 31. The reconstructed mesh
    boundary is added later because it is only known after filtering and
    remeshing.

    Args:
        model_config: Validated preprocessing settings.
        jld2_file: Open WAVI JLD2 timestep file.
        dt: Time interval between consecutive snapshots.

    Returns:
        Node coordinates with shape ``[nodes, 2]`` and complete node features
        with shape ``[nodes, features]``.
    """
    x_coords = jld2_file["x"][0, :]
    y_coords = jld2_file["y"][:, 0]
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    mesh_pos = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    u = jld2_file["u"][:, :]
    use_node_types = bool(_setting(model_config, "use_node_types", default=False))

    node_type_oh = None
    if use_node_types:
        n_rows, n_cols = u.shape
        num_node_types = int(_setting(model_config, "num_node_types", default=3))
        node_type_interior = int(
            _setting(model_config, "node_type_interior", default=0)
        )
        node_type_free_slip = int(
            _setting(model_config, "node_type_free_slip", default=1)
        )
        node_type_left_no_slip = int(
            _setting(model_config, "node_type_left_no_slip", default=2)
        )

        node_ids = {
            node_type_interior,
            node_type_free_slip,
            node_type_left_no_slip,
        }
        if len(node_ids) != 3 or min(node_ids) < 0 or max(node_ids) >= num_node_types:
            raise ValueError(
                "The three node-type identifiers must be distinct values in "
                f"[0, {num_node_types - 1}]"
            )

        node_type_2d = np.full(
            (n_rows, n_cols), node_type_interior, dtype=np.int64
        )
        node_type_2d[0, :] = node_type_free_slip
        node_type_2d[-1, :] = node_type_free_slip
        node_type_2d[:, 0] = node_type_left_no_slip
        node_type = torch.as_tensor(node_type_2d.ravel(), dtype=torch.long)
        node_type_oh = F.one_hot(
            node_type, num_classes=num_node_types
        ).float()

    def vector(first: str, second: str) -> torch.Tensor:
        """Combine two grid fields as one two-component node feature."""
        values = np.column_stack(
            [jld2_file[first][:, :].ravel(), jld2_file[second][:, :].ravel()]
        )
        return torch.as_tensor(values, dtype=torch.float32)

    def column(name: str) -> torch.Tensor:
        """Flatten one scalar grid field as a node-feature column."""
        values = jld2_file[name][:, :].ravel()
        return torch.as_tensor(values, dtype=torch.float32).unsqueeze(-1)

    h_shape = jld2_file["h"].shape

    def optional_column(name: str) -> torch.Tensor:
        """Load an optional scalar field or return a zero-filled column."""
        if name in jld2_file:
            return column(name)
        return torch.zeros((int(np.prod(h_shape)), 1), dtype=torch.float32)

    dt_column = torch.full(
        (int(np.prod(h_shape)), 1), float(dt), dtype=torch.float32
    )

    feature_tensors = [
        vector("u", "v"),
        vector("us", "vs"),
        vector("ub", "vb"),
        column("h"),
        column("b"),
        column("s"),
        column("dhdt"),
        column("accumulation"),
        column("basal_melt"),
        column("grounded_frac"),
        column("av_speed"),
        column("bed_speed"),
        column("weertman_c"),
        column("haf"),
        column("dsdh"),
        column("shelf_strain"),
        column("β"),
        column("βeff"),
        column("τbed"),
        column("ηav"),
        column("quad_f1"),
        column("quad_f2"),
        column("mask"),
        optional_column("param_smb"),
        optional_column("param_gt"),
        dt_column,
    ]

    if node_type_oh is not None:
        feature_tensors.append(node_type_oh)

    return mesh_pos, torch.cat(feature_tensors, dim=-1).float()


def determine_automatic_thickness_value(
    all_feature_snapshots: Sequence[torch.Tensor],
    maximum_value: float = 100.0,
) -> tuple[float, float | None]:
    """Detect an integer-valued numerical thickness floor.

    This compatibility helper is not used by the configured preprocessing
    pipeline, which filters with ``model.minimum_thickness`` directly.
    """
    if not all_feature_snapshots:
        raise ValueError("No feature snapshots were provided")

    minimum_thickness = min(
        float(features[:, THICKNESS_FEATURE_INDEX].min().item())
        for features in all_feature_snapshots
    )
    use_for_filtering = (
        np.isfinite(minimum_thickness)
        and minimum_thickness.is_integer()
        and minimum_thickness <= maximum_value
    )
    return minimum_thickness, minimum_thickness if use_for_filtering else None


def find_nodes_reaching_thickness_value(
    all_feature_snapshots: Sequence[torch.Tensor],
    thickness_value: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find nodes reaching a thickness value anywhere in the trajectory.

    A node detected at any timestep is removed from every timestep, preserving
    constant node order and topology throughout the processed trajectory.

    Returns:
        A Boolean removal mask over original nodes and the first detection
        timestep per node, with ``-1`` for retained nodes.
    """
    if not all_feature_snapshots:
        raise ValueError("No feature snapshots were provided")

    n_nodes = all_feature_snapshots[0].shape[0]
    remove_mask = torch.zeros(n_nodes, dtype=torch.bool)
    first_hit_timestep = torch.full((n_nodes,), -1, dtype=torch.long)
    if thickness_value is None:
        return remove_mask, first_hit_timestep

    for timestep, features in enumerate(all_feature_snapshots):
        hit = (
            features[:, THICKNESS_FEATURE_INDEX] == thickness_value
        ).detach().cpu()
        newly_detected = hit & ~remove_mask
        first_hit_timestep[newly_detected] = timestep
        remove_mask |= hit
    return remove_mask, first_hit_timestep


def estimate_mesh_spacing(mesh_pos: torch.Tensor) -> float:
    """Estimate regular-grid spacing from median nearest-neighbor distance."""
    positions = mesh_pos[:, :2].detach().cpu().numpy()
    if positions.shape[0] < 2:
        raise ValueError("At least two nodes are needed to estimate mesh spacing")

    distances, _ = cKDTree(positions).query(positions, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if nearest.size == 0:
        raise ValueError("Could not determine a positive mesh spacing")
    return float(np.median(nearest))


def _triangles_to_edges(triangles: torch.Tensor) -> torch.Tensor:
    """Convert triangular faces to unique bidirectional graph edges.

    Each undirected mesh edge is retained once and then stored in both
    directions for message passing. The result has shape ``[2, edges]``.
    """
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            "triangles must have shape [num_triangles, 3], received "
            f"{tuple(triangles.shape)}"
        )

    triangle_array = triangles.detach().cpu().numpy()
    undirected = np.concatenate(
        [
            triangle_array[:, [0, 1]],
            triangle_array[:, [1, 2]],
            triangle_array[:, [2, 0]],
        ],
        axis=0,
    )
    undirected = np.unique(np.sort(undirected, axis=1), axis=0)
    directed = np.concatenate([undirected, undirected[:, ::-1]], axis=0)
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    return torch.as_tensor(directed.T, dtype=torch.long)


def _build_edge_attributes(
    mesh_pos: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """Build ``dx``, ``dy`` and distance for every directed graph edge."""
    relative_position = mesh_pos[edge_index[0], :2] - mesh_pos[edge_index[1], :2]
    edge_distance = torch.linalg.vector_norm(
        relative_position, dim=1, keepdim=True
    )
    return torch.cat([relative_position, edge_distance], dim=-1).float()


def _build_delaunay_topology(
    kept_mesh_pos: torch.Tensor,
    maximum_edge_length: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Remesh retained nodes and reject triangles with excessively long edges.

    Limiting triangle edges prevents Delaunay triangulation from reconnecting
    distant nodes across holes introduced by thickness filtering.

    Returns:
        Bidirectional connectivity, geometric edge attributes, triangle
        connectivity and the number of isolated retained nodes.
    """
    positions = kept_mesh_pos[:, :2].detach().cpu().numpy()
    n_nodes = positions.shape[0]
    if n_nodes < 3:
        raise ValueError(
            f"Delaunay triangulation needs at least 3 nodes; {n_nodes} remain"
        )

    try:
        triangles = Delaunay(positions).simplices.astype(np.int64)
    except QhullError as exc:
        raise ValueError(
            "Delaunay remeshing failed; retained coordinates may be collinear "
            "or geometrically degenerate"
        ) from exc

    p0 = positions[triangles[:, 0]]
    p1 = positions[triangles[:, 1]]
    p2 = positions[triangles[:, 2]]
    longest = np.maximum.reduce(
        [
            np.linalg.norm(p0 - p1, axis=1),
            np.linalg.norm(p1 - p2, axis=1),
            np.linalg.norm(p2 - p0, axis=1),
        ]
    )
    triangles = triangles[longest <= maximum_edge_length]
    if triangles.shape[0] == 0:
        raise ValueError(
            "No triangles remained after applying maximum edge length "
            f"{maximum_edge_length:.6g}; increase delaunay_edge_factor"
        )

    triangles_tensor = torch.as_tensor(triangles, dtype=torch.long)
    edge_index = _triangles_to_edges(triangles_tensor)
    edge_attr = _build_edge_attributes(kept_mesh_pos, edge_index)
    face = triangles_tensor.T.contiguous()

    if int(face.min().item()) < 0 or int(face.max().item()) >= n_nodes:
        raise ValueError("Remeshed face contains an invalid node index")

    node_degree = torch.bincount(edge_index[0], minlength=n_nodes)
    n_isolated_nodes = int((node_degree == 0).sum().item())
    return edge_index, edge_attr, face, n_isolated_nodes


def _build_mesh_boundary_feature(
    face: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Identify nodes on the boundary of the reconstructed triangular mesh.

    An edge belongs to the mesh boundary when it occurs in exactly one face.
    Every node touching such an edge receives one in the returned binary
    feature channel.
    """
    if face.ndim != 2 or face.shape[0] != 3:
        raise ValueError(
            "face must have shape [3, number_of_triangles], received "
            f"{tuple(face.shape)}"
        )

    triangles = face.T.contiguous()
    edges = torch.cat(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        dim=0,
    )
    edges = torch.sort(edges, dim=1).values
    unique_edges, counts = torch.unique(edges, dim=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    boundary_mask = torch.zeros(num_nodes, dtype=torch.bool, device=face.device)
    if boundary_edges.numel() > 0:
        boundary_mask[torch.unique(boundary_edges.reshape(-1))] = True
    return boundary_mask.float().unsqueeze(-1)


def _build_bundled_dataset(
    model_config: Any,
    simulation_dir: Path,
    trajectory_id: str | None = None,
    print_filter_summary: bool = True,
    show_timestep_progress: bool = True,
) -> tuple[list[Data], dict[str, Any]]:
    """Convert one complete WAVI trajectory to temporally bundled graphs.

    Every raw snapshot is loaded before filtering. Nodes that equal the
    configured minimum thickness during any snapshot are removed from all
    snapshots. The retained coordinates are remeshed once, producing static
    topology shared by every temporal sample.

    For current timestep ``t``, ``x`` contains the interval
    ``[t-history_size, ..., t]``. ``y`` contains the forward-difference
    derivative from ``t`` to ``t+1``. ``future_states`` and
    ``future_forcings`` contain the next ``future_size`` timesteps.

    Args:
        model_config: Temporal, feature, filtering and remeshing settings.
        simulation_dir: Directory containing one trajectory's timestep files.
        trajectory_id: Identifier stored in every output sample.
        print_filter_summary: Whether to print feature and remeshing
            diagnostics.
        show_timestep_progress: Whether to display progress while reading
            snapshots.

    Returns:
        PyTorch Geometric samples and trajectory-wide filtering, remeshing and
        source-file diagnostics.
    """
    simulation_dir = Path(simulation_dir)
    past_steps = int(
        _setting(model_config, "history_size", "past_steps", default=0)
    )
    future_steps = int(
        _setting(model_config, "future_size", "future_steps", default=0)
    )
    dt = float(_setting(model_config, "delta_time", "dt", default=1.0))
    use_node_types = bool(
        _setting(model_config, "use_node_types", default=False)
    )
    use_thickness_filter = bool(
        _setting(model_config, "use_thickness_filter", default=True)
    )
    minimum_thickness = float(
        _setting(model_config, "minimum_thickness", default=50.0)
    )
    filter_thickness = minimum_thickness if use_thickness_filter else None
    delaunay_edge_factor = float(
        _setting(model_config, "delaunay_edge_factor", default=1.5)
    )
    selected_features_x = _setting(
        model_config,
        "selected_features_x",
        default=DEFAULT_SELECTED_FEATURES,
    )
    selected_features_x = validate_selected_features(
        selected_features_x, use_node_types
    )

    if past_steps < 0 or future_steps < 0:
        raise ValueError("history_size and future_size must be non-negative")
    if dt <= 0:
        raise ValueError(f"delta_time must be positive, received {dt}")
    if minimum_thickness < 0:
        raise ValueError("minimum_thickness must be non-negative")
    if delaunay_edge_factor <= 0:
        raise ValueError("delaunay_edge_factor must be positive")
    if not simulation_dir.exists():
        raise FileNotFoundError(f"Trajectory folder does not exist:\n{simulation_dir}")
    if not simulation_dir.is_dir():
        raise NotADirectoryError(f"Expected a trajectory directory:\n{simulation_dir}")

    if trajectory_id is None:
        trajectory_id = simulation_dir.name
    if print_filter_summary:
        print_selected_features(selected_features_x)

    files = sorted(
        [
            path
            for path in simulation_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".jld2"
        ],
        key=natural_sort_key,
    )
    if not files:
        raise ValueError(
            f"No .jld2 files found in:\n{simulation_dir}"
        )

    timesteps = len(files)
    minimum_timesteps = past_steps + max(1, future_steps) + 1
    if timesteps < minimum_timesteps:
        raise ValueError(
            f"Trajectory has {timesteps} timesteps; at least {minimum_timesteps} "
            f"are required for history_size={past_steps} and "
            f"future_size={future_steps}"
        )

    all_feature_snapshots: list[torch.Tensor] = []
    reference_mesh_pos: torch.Tensor | None = None
    file_iterator = (
        tqdm(files, desc=f"Parsing {simulation_dir.name[:25]}", leave=False)
        if show_timestep_progress
        else files
    )

    for timestep, file_path in enumerate(file_iterator):
        try:
            with h5py.File(file_path, "r") as jld2_file:
                mesh_pos, all_features = _grid_to_mesh(
                    model_config=model_config,
                    jld2_file=jld2_file,
                    dt=dt,
                )
        except OSError as exc:
            raise OSError(f"Could not read timestep file:\n{file_path}") from exc

        mesh_pos_tensor = torch.as_tensor(mesh_pos, dtype=torch.float32)
        if reference_mesh_pos is None:
            reference_mesh_pos = mesh_pos_tensor
        else:
            if mesh_pos_tensor.shape != reference_mesh_pos.shape:
                raise ValueError(
                    f"Mesh shape changed at timestep {timestep}: "
                    f"{tuple(reference_mesh_pos.shape)} to "
                    f"{tuple(mesh_pos_tensor.shape)}"
                )
            if not torch.allclose(
                mesh_pos_tensor, reference_mesh_pos, rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    f"Node coordinates or ordering changed at timestep {timestep}"
                )

        if all_features.shape[0] != reference_mesh_pos.shape[0]:
            raise ValueError(
                f"Feature and mesh node counts differ in {file_path.name}"
            )
        all_feature_snapshots.append(all_features)

    assert reference_mesh_pos is not None
    minimum_original_thickness = min(
        float(features[:, THICKNESS_FEATURE_INDEX].min().item())
        for features in all_feature_snapshots
    )

    remove_mask, first_hit_timestep = find_nodes_reaching_thickness_value(
        all_feature_snapshots,
        filter_thickness,
    )
    keep_indices = torch.nonzero(~remove_mask, as_tuple=False).flatten()
    removed_indices = torch.nonzero(remove_mask, as_tuple=False).flatten()
    n_original_nodes = int(reference_mesh_pos.shape[0])
    n_removed_nodes = int(removed_indices.numel())
    n_kept_nodes = int(keep_indices.numel())
    if n_kept_nodes < 3:
        raise ValueError(
            f"Only {n_kept_nodes} nodes remain after filtering; at least 3 are required"
        )

    filtered_snapshots = [
        features.index_select(0, keep_indices)
        for features in all_feature_snapshots
    ]
    kept_mesh_pos = reference_mesh_pos.index_select(0, keep_indices)
    mesh_spacing = estimate_mesh_spacing(reference_mesh_pos)
    maximum_edge_length = delaunay_edge_factor * mesh_spacing
    edge_index, edge_attr, face, n_isolated_nodes = _build_delaunay_topology(
        kept_mesh_pos, maximum_edge_length
    )

    if use_node_types:
        boundary_feature = _build_mesh_boundary_feature(face, n_kept_nodes)
        filtered_snapshots = [
            torch.cat([features, boundary_feature.to(features)], dim=1)
            for features in filtered_snapshots
        ]
        n_mesh_boundary_nodes = int(boundary_feature.sum().item())
    else:
        n_mesh_boundary_nodes = 0

    removed_node_indices = [int(index) for index in removed_indices.tolist()]
    filter_info = {
        "trajectory_id": trajectory_id,
        "n_timesteps": timesteps,
        "source_file_names": [path.name for path in files],
        "n_original_nodes": n_original_nodes,
        "n_kept_nodes": n_kept_nodes,
        "n_removed_nodes": n_removed_nodes,
        "percentage_nodes_removed": (
            100.0 * n_removed_nodes / max(n_original_nodes, 1)
        ),
        "removed_node_indices": removed_node_indices,
        "kept_node_indices": [int(index) for index in keep_indices.tolist()],
        "first_hit_timestep": {
            index: int(first_hit_timestep[index].item())
            for index in removed_node_indices
        },
        "minimum_original_thickness": minimum_original_thickness,
        "use_thickness_filter": use_thickness_filter,
        "minimum_thickness": filter_thickness,
        "estimated_mesh_spacing": mesh_spacing,
        "delaunay_edge_factor": delaunay_edge_factor,
        "maximum_edge_length": maximum_edge_length,
        "n_edges_after": int(edge_index.shape[1]),
        "n_triangles_after": int(face.shape[1]),
        "n_isolated_nodes_after": n_isolated_nodes,
        "n_mesh_boundary_nodes": n_mesh_boundary_nodes,
    }

    if print_filter_summary:
        print(
            f"\nMinimum original ice thickness: "
            f"{minimum_original_thickness:.6g}"
        )
        if use_thickness_filter:
            print(f"Configured filtering thickness: {minimum_thickness:g}")
        else:
            print("Thickness filtering: disabled")
        print(
            f"Nodes: {n_original_nodes} to {n_kept_nodes}; "
            f"{n_removed_nodes} removed "
            f"({filter_info['percentage_nodes_removed']:.2f}%)"
        )
        print(f"Estimated mesh spacing: {mesh_spacing:.6g}")
        print(f"Maximum retained triangle edge: {maximum_edge_length:.6g}")
        print(f"Directed edges after remeshing: {edge_index.shape[1]}")
        print(f"Triangles after remeshing: {face.shape[1]}")
        print(f"Isolated retained nodes: {n_isolated_nodes}")
        print(f"Remeshed boundary nodes: {n_mesh_boundary_nodes}")

    selected_tensor = torch.as_tensor(selected_features_x, dtype=torch.long)
    target_tensor = torch.as_tensor(TARGET_FEATURE_INDICES, dtype=torch.long)
    forcing_tensor = torch.as_tensor(FORCING_FEATURE_INDICES, dtype=torch.long)
    bundled_dataset: list[Data] = []
    required_future_steps = max(1, future_steps)

    for time_index in range(past_steps, timesteps - required_future_steps):
        x = torch.stack(
            [
                filtered_snapshots[history_index].index_select(
                    1, selected_tensor
                )
                for history_index in range(
                    time_index - past_steps, time_index + 1
                )
            ],
            dim=1,
        ).float()
        current_state = filtered_snapshots[time_index].index_select(
            1, target_tensor
        )
        next_state = filtered_snapshots[time_index + 1].index_select(
            1, target_tensor
        )
        y = ((next_state - current_state) / dt).float()

        if future_steps > 0:
            future_states = torch.stack(
                [
                    filtered_snapshots[time_index + offset].index_select(
                        1, target_tensor
                    )
                    for offset in range(1, future_steps + 1)
                ],
                dim=1,
            ).float()
            future_forcings = torch.stack(
                [
                    filtered_snapshots[time_index + offset].index_select(
                        1, forcing_tensor
                    )
                    for offset in range(1, future_steps + 1)
                ],
                dim=1,
            ).float()
        else:
            future_states = torch.empty(
                (n_kept_nodes, 0, len(TARGET_FEATURE_INDICES)),
                dtype=torch.float32,
            )
            future_forcings = torch.empty(
                (n_kept_nodes, 0, len(FORCING_FEATURE_INDICES)),
                dtype=torch.float32,
            )

        bundled_dataset.append(
            Data(
                x=x,
                y=y,
                future_states=future_states,
                future_forcings=future_forcings,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mesh_pos=kept_mesh_pos,
                face=face,
                time_index=time_index,
                trajectory_id=trajectory_id,
            )
        )

    return bundled_dataset, filter_info


def _json_safe(value: Any) -> Any:
    """Convert nested NumPy and PyTorch values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _choose_netcdf_engine() -> str:
    """Select the best available xarray NetCDF backend.

    ``netCDF4`` is preferred, followed by ``h5netcdf``. The SciPy backend is a
    final uncompressed fallback with limited support for empty dimensions.
    """
    if importlib.util.find_spec("netCDF4") is not None:
        return "netcdf4"
    if importlib.util.find_spec("h5netcdf") is not None:
        return "h5netcdf"
    warnings.warn(
        "Neither netCDF4 nor h5netcdf is installed; falling back to scipy "
        "NetCDF-3 without compression",
        RuntimeWarning,
    )
    return "scipy"


def _validate_static_graph_fields(bundled_dataset: Sequence[Data]) -> None:
    """Verify that all temporal samples share one mesh and topology."""
    first_sample = bundled_dataset[0]
    for sample_index, sample in enumerate(bundled_dataset[1:], start=1):
        for field_name in ("edge_index", "edge_attr", "mesh_pos", "face"):
            first_value = getattr(first_sample, field_name)
            current_value = getattr(sample, field_name)
            if first_value.shape != current_value.shape or not torch.equal(
                first_value, current_value
            ):
                raise ValueError(
                    f"Static graph field '{field_name}' changes at sample "
                    f"{sample_index}"
                )


def _derive_original_node_indices(
    filter_info: dict[str, Any] | None,
    number_of_kept_nodes: int,
) -> np.ndarray:
    """Map retained graph nodes to indices in the original regular grid."""
    if filter_info is None:
        return np.arange(number_of_kept_nodes, dtype=np.int32)
    kept = filter_info.get("kept_node_indices")
    if kept is None:
        n_original = int(
            filter_info.get("n_original_nodes", number_of_kept_nodes)
        )
        removed = {
            int(index) for index in filter_info.get("removed_node_indices", [])
        }
        kept = [index for index in range(n_original) if index not in removed]
    result = np.asarray(kept, dtype=np.int32)
    if result.shape != (number_of_kept_nodes,):
        raise ValueError("Original-node mapping does not match graph node count")
    return result


def save_bundled_dataset_to_netcdf(
    bundled_dataset: Sequence[Data],
    output_path: str | Path,
    selected_features_x: Sequence[int],
    dt: float,
    filter_info: dict[str, Any] | None = None,
    coordinate_units: str = "m",
    time_units: str = "years",
    overwrite: bool = False,
) -> xr.Dataset:
    """Save one processed graph trajectory as structured NetCDF.

    Temporal graph tensors are stacked along ``sample``. Static coordinates,
    edges and triangular faces are stored once. Global attributes record the
    feature convention, temporal setup and filtering diagnostics. NetCDF-4
    backends use compression level 4 with byte shuffling.

    Args:
        bundled_dataset: Temporal samples produced by
            ``_build_bundled_dataset``.
        output_path: Destination path ending in ``.nc``.
        selected_features_x: Complete-feature indices represented by the
            input tensor.
        dt: Time interval between consecutive snapshots.
        filter_info: Optional trajectory-wide filtering and remeshing
            metadata.
        coordinate_units: Coordinate unit label written to NetCDF metadata.
        time_units: Time unit label written to NetCDF metadata.
        overwrite: Whether an existing destination may be replaced.

    Returns:
        In-memory representation of the written NetCDF dataset.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".nc":
        raise ValueError("NetCDF output_path must end in '.nc'")
    if not bundled_dataset:
        raise ValueError("Cannot save an empty bundled dataset")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists:\n{output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_static_graph_fields(bundled_dataset)
    first_sample = bundled_dataset[0]

    input_x = torch.stack(
        [sample.x.detach().cpu() for sample in bundled_dataset]
    ).numpy().astype(np.float32, copy=False)
    target_y = torch.stack(
        [sample.y.detach().cpu() for sample in bundled_dataset]
    ).numpy().astype(np.float32, copy=False)
    future_states = torch.stack(
        [sample.future_states.detach().cpu() for sample in bundled_dataset]
    ).numpy().astype(np.float32, copy=False)
    future_forcings = torch.stack(
        [sample.future_forcings.detach().cpu() for sample in bundled_dataset]
    ).numpy().astype(np.float32, copy=False)
    time_index = np.asarray(
        [int(sample.time_index) for sample in bundled_dataset], dtype=np.int32
    )

    trajectory_ids = {str(sample.trajectory_id) for sample in bundled_dataset}
    if len(trajectory_ids) != 1:
        raise ValueError(
            "All samples in one NetCDF file must share one trajectory_id; "
            f"found {sorted(trajectory_ids)}"
        )
    trajectory_id = next(iter(trajectory_ids))

    mesh_pos = first_sample.mesh_pos.detach().cpu().numpy().astype(np.float32)
    directed_edge_index = (
        first_sample.edge_index.detach().cpu().T.contiguous().numpy().astype(np.int32)
    )
    edge_attr = first_sample.edge_attr.detach().cpu().numpy().astype(np.float32)
    face_nodes = (
        first_sample.face.detach().cpu().T.contiguous().numpy().astype(np.int32)
    )

    n_samples, n_nodes, history_length, n_input_features = input_x.shape
    n_future_steps = future_states.shape[2]
    selected_features_x = [int(index) for index in selected_features_x]
    if len(selected_features_x) != n_input_features:
        raise ValueError("selected_features_x does not match stored x features")
    if target_y.shape != (n_samples, n_nodes, len(TARGET_FEATURE_INDICES)):
        raise ValueError(f"Unexpected y shape: {target_y.shape}")
    if future_states.shape != (
        n_samples,
        n_nodes,
        n_future_steps,
        len(TARGET_FEATURE_INDICES),
    ):
        raise ValueError(f"Unexpected future_states shape: {future_states.shape}")
    if future_forcings.shape != (
        n_samples,
        n_nodes,
        n_future_steps,
        len(FORCING_FEATURE_INDICES),
    ):
        raise ValueError(
            f"Unexpected future_forcings shape: {future_forcings.shape}"
        )

    edge_nodes = np.unique(
        np.sort(directed_edge_index, axis=1), axis=0
    ).astype(np.int32, copy=False)
    original_node_index = _derive_original_node_indices(filter_info, n_nodes)
    past_steps = history_length - 1
    selected_names = [FEATURE_NAMES[index] for index in selected_features_x]

    dataset = xr.Dataset(
        data_vars={
            "x": (
                ("sample", "node", "history", "input_feature"),
                input_x,
            ),
            "y": (("sample", "node", "target_feature"), target_y),
            "future_states": (
                ("sample", "node", "future_step", "target_feature"),
                future_states,
            ),
            "future_forcings": (
                ("sample", "node", "future_step", "forcing_feature"),
                future_forcings,
            ),
            "time_index": (("sample",), time_index),
            "model_time": (("sample",), time_index.astype(np.float64) * dt),
            "mesh_x": (("node",), mesh_pos[:, 0]),
            "mesh_y": (("node",), mesh_pos[:, 1]),
            "original_node_index": (("node",), original_node_index),
            "mesh2d": ((), np.int8(0)),
            "face_nodes": (("face", "face_vertex"), face_nodes),
            "edge_nodes": (("mesh_edge", "edge_endpoint"), edge_nodes),
            "directed_edge_index": (
                ("directed_edge", "edge_endpoint"),
                directed_edge_index,
            ),
            "edge_attr": (("directed_edge", "edge_feature"), edge_attr),
        },
        coords={
            "sample": np.arange(n_samples, dtype=np.int32),
            "node": np.arange(n_nodes, dtype=np.int32),
            "history": np.arange(-past_steps, 1, dtype=np.int32),
            "input_feature": np.arange(n_input_features, dtype=np.int32),
            "target_feature": np.arange(
                len(TARGET_FEATURE_INDICES), dtype=np.int32
            ),
            "forcing_feature": np.arange(
                len(FORCING_FEATURE_INDICES), dtype=np.int32
            ),
            "future_step": np.arange(1, n_future_steps + 1, dtype=np.int32),
            "face": np.arange(face_nodes.shape[0], dtype=np.int32),
            "face_vertex": np.arange(3, dtype=np.int32),
            "mesh_edge": np.arange(edge_nodes.shape[0], dtype=np.int32),
            "directed_edge": np.arange(
                directed_edge_index.shape[0], dtype=np.int32
            ),
            "edge_endpoint": np.arange(2, dtype=np.int32),
            "edge_feature": np.arange(edge_attr.shape[1], dtype=np.int32),
        },
        attrs={
            "title": "Processed WAVI graph trajectory for IceGraph",
            "summary": (
                "Temporally bundled WAVI fields on a trajectory-wide filtered "
                "and Delaunay-remeshed graph."
            ),
            "source": "WAVI.jl .jld2 timestep files",
            "Conventions": "CF-1.12 UGRID-1.0",
            "trajectory_id": trajectory_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dt": float(dt),
            "time_units": str(time_units),
            "coordinate_units": str(coordinate_units),
            "past_steps": int(past_steps),
            "future_steps": int(n_future_steps),
            "selected_feature_indices_json": json.dumps(selected_features_x),
            "selected_feature_names_json": json.dumps(selected_names),
            "target_feature_indices_json": json.dumps(TARGET_FEATURE_INDICES),
            "target_feature_names_json": json.dumps(TARGET_FEATURE_NAMES),
            "forcing_feature_indices_json": json.dumps(FORCING_FEATURE_INDICES),
            "forcing_feature_names_json": json.dumps(FORCING_FEATURE_NAMES),
            "filter_info_json": json.dumps(_json_safe(filter_info or {})),
        },
    )

    dataset["x"].attrs.update(
        long_name="temporally bundled model input features",
        feature_names="selected_feature_names_json global attribute",
    )
    dataset["y"].attrs.update(
        long_name="one-step temporal derivatives of u, v and h",
        feature_names="target_feature_names_json global attribute",
    )
    dataset["future_states"].attrs.update(
        long_name="future true u, v and h states",
        feature_names="target_feature_names_json global attribute",
    )
    dataset["future_forcings"].attrs.update(
        long_name="future prescribed accumulation and basal melt",
        feature_names="forcing_feature_names_json global attribute",
    )
    dataset["model_time"].attrs.update(
        long_name="model time since trajectory start", units=str(time_units)
    )
    dataset["mesh_x"].attrs.update(
        standard_name="projection_x_coordinate",
        long_name="retained mesh node x coordinate",
        units=str(coordinate_units),
    )
    dataset["mesh_y"].attrs.update(
        standard_name="projection_y_coordinate",
        long_name="retained mesh node y coordinate",
        units=str(coordinate_units),
    )
    dataset["mesh2d"].attrs.update(
        cf_role="mesh_topology",
        topology_dimension=np.int32(2),
        node_coordinates="mesh_x mesh_y",
        face_node_connectivity="face_nodes",
        edge_node_connectivity="edge_nodes",
    )
    for name in ("face_nodes", "edge_nodes", "directed_edge_index"):
        dataset[name].attrs["start_index"] = np.int32(0)
    dataset["edge_attr"].attrs.update(
        long_name="directed edge attributes dx, dy and Euclidean distance",
        edge_feature_names="dx, dy, distance",
    )

    engine = _choose_netcdf_engine()
    if engine == "scipy" and n_future_steps == 0:
        raise RuntimeError(
            "The scipy backend cannot reliably write a zero-length future_step; "
            "install netCDF4/h5netcdf or set future_size > 0"
        )
    encoding: dict[str, dict[str, Any]] = {}
    if engine in {"netcdf4", "h5netcdf"}:
        encoding = {
            name: {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
            }
            for name, variable in dataset.data_vars.items()
            if variable.ndim > 0 and variable.dtype.kind in {"f", "i", "u"}
        }
    dataset.to_netcdf(
        output_path, mode="w", engine=engine, encoding=encoding
    )
    return dataset


def load_netcdf_as_pyg(
    netcdf_path: str | Path,
    return_metadata: bool = False,
) -> list[Data] | tuple[list[Data], dict[str, Any]]:
    """Reconstruct PyTorch Geometric samples from a NetCDF trajectory.

    Static graph tensors are shared by the reconstructed samples, while each
    sample receives its own temporal tensors and time index.

    Args:
        netcdf_path: Processed trajectory file.
        return_metadata: Whether to return decoded feature and filtering
            metadata.

    Returns:
        Reconstructed samples, optionally paired with metadata.
    """
    netcdf_path = Path(netcdf_path)
    if not netcdf_path.exists():
        raise FileNotFoundError(f"NetCDF file does not exist:\n{netcdf_path}")

    with xr.open_dataset(netcdf_path) as opened:
        dataset = opened.load()

    required = {
        "x",
        "y",
        "future_states",
        "future_forcings",
        "time_index",
        "mesh_x",
        "mesh_y",
        "directed_edge_index",
        "edge_attr",
        "face_nodes",
    }
    missing = sorted(required - set(dataset.variables))
    if missing:
        raise KeyError(f"Missing required NetCDF variables: {missing}")

    input_x = torch.from_numpy(np.asarray(dataset["x"].values, dtype=np.float32))
    target_y = torch.from_numpy(np.asarray(dataset["y"].values, dtype=np.float32))
    future_states = torch.from_numpy(
        np.asarray(dataset["future_states"].values, dtype=np.float32)
    )
    future_forcings = torch.from_numpy(
        np.asarray(dataset["future_forcings"].values, dtype=np.float32)
    )
    mesh_pos = torch.from_numpy(
        np.column_stack(
            [dataset["mesh_x"].values, dataset["mesh_y"].values]
        ).astype(np.float32, copy=False)
    )
    edge_index = torch.from_numpy(
        np.asarray(dataset["directed_edge_index"].values, dtype=np.int64)
    ).T.contiguous()
    edge_attr = torch.from_numpy(
        np.asarray(dataset["edge_attr"].values, dtype=np.float32)
    )
    face = torch.from_numpy(
        np.asarray(dataset["face_nodes"].values, dtype=np.int64)
    ).T.contiguous()
    time_indices = np.asarray(dataset["time_index"].values, dtype=np.int64)
    trajectory_id = str(dataset.attrs.get("trajectory_id", netcdf_path.stem))

    bundled_dataset = [
        Data(
            x=input_x[index],
            y=target_y[index],
            future_states=future_states[index],
            future_forcings=future_forcings[index],
            edge_index=edge_index,
            edge_attr=edge_attr,
            mesh_pos=mesh_pos,
            face=face,
            time_index=int(time_indices[index]),
            trajectory_id=trajectory_id,
        )
        for index in range(input_x.shape[0])
    ]

    if return_metadata:
        metadata = dict(dataset.attrs)
        metadata["filter_info"] = json.loads(
            metadata.get("filter_info_json", "{}")
        )
        metadata["selected_feature_indices"] = json.loads(
            metadata.get("selected_feature_indices_json", "[]")
        )
        metadata["selected_feature_names"] = json.loads(
            metadata.get("selected_feature_names_json", "[]")
        )
        return bundled_dataset, metadata
    return bundled_dataset


def inspect_netcdf(netcdf_path: str | Path) -> None:
    """Print structure, selected features and filtering metadata for a file."""
    with xr.open_dataset(Path(netcdf_path)) as dataset:
        print(dataset)
        print("\nTrajectory ID:", dataset.attrs.get("trajectory_id"))
        print(
            "Selected features:",
            json.loads(dataset.attrs.get("selected_feature_names_json", "[]")),
        )
        print(
            "Filter information:",
            json.loads(dataset.attrs.get("filter_info_json", "{}")),
        )


def _output_filename(model_config: Any, simulation: str) -> str:
    """Build a traceable filename from trajectory and temporal settings."""
    prefix = str(_setting(model_config, "file_prefix", default="")).strip("_")
    stem = simulation
    if prefix and not simulation.startswith(f"{prefix}_"):
        stem = f"{prefix}_{simulation}"
    past_steps = int(
        _setting(model_config, "history_size", "past_steps", default=0)
    )
    future_steps = int(
        _setting(model_config, "future_size", "future_steps", default=0)
    )
    return f"{stem}_past{past_steps}_fut{future_steps}.nc"


def wavi_to_mesh(config: Config, wavi_simulation: str = "") -> list[Path]:
    """Convert one or all configured WAVI trajectories to NetCDF graphs.

    Passing ``wavi_simulation`` selects one trajectory directory. An empty
    value processes every trajectory directory under
    ``data.wavi_trajectories_subdir``. Existing trajectory files are skipped unless
    ``model.overwrite`` is true.

    Args:
        config: Validated data and model configuration.
        wavi_simulation: Optional trajectory directory name.

    Returns:
        Expected output path for each selected trajectory, including skipped
        existing files.
    """
    data_config = _config_section(config, "data", "directories")
    model_config = _config_section(config, "model", "settings")
    root_dir = Path(_setting(data_config, "root_dir"))
    input_subdir = _setting(data_config, "wavi_trajectories_subdir")
    output_subdir = _setting(
        data_config,
        "mesh_subdir",
        "mesh_trajectories_subdir",
    )
    if input_subdir is None or output_subdir is None:
        raise AttributeError(
            "Data configuration requires wavi_trajectories_subdir and mesh_subdir"
        )

    wavi_dir = root_dir / input_subdir
    if not wavi_dir.is_dir():
        raise NotADirectoryError(f"WAVI trajectory directory does not exist:\n{wavi_dir}")
    simulations = (
        sorted(path.name for path in wavi_dir.iterdir() if path.is_dir())
        if not wavi_simulation
        else [wavi_simulation]
    )
    if not simulations:
        raise ValueError(f"No simulation folders found in:\n{wavi_dir}")

    output_dir = root_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    overwrite = bool(_setting(model_config, "overwrite", default=False))
    selected_features = validate_selected_features(
        _setting(
            model_config,
            "selected_features_x",
            default=DEFAULT_SELECTED_FEATURES,
        ),
        bool(_setting(model_config, "use_node_types", default=False)),
    )
    dt = float(_setting(model_config, "delta_time", "dt", default=1.0))
    coordinate_units = str(
        _setting(model_config, "coordinate_units", default="m")
    )
    time_units = str(_setting(model_config, "time_units", default="years"))
    output_paths: list[Path] = []
    summaries: list[dict[str, Any]] = []
    skipped = 0
    for simulation in tqdm(simulations, desc="Processing datasets", unit="sim"):
        simulation_dir = wavi_dir / simulation
        if not simulation_dir.is_dir():
            raise NotADirectoryError(
                f"Simulation directory does not exist:\n{simulation_dir}"
            )
        output_path = output_dir / _output_filename(model_config, simulation)
        output_paths.append(output_path)
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        dataset, filter_info = _build_bundled_dataset(
            model_config=model_config,
            simulation_dir=simulation_dir,
            trajectory_id=output_path.stem,
            print_filter_summary=bool(wavi_simulation),
            show_timestep_progress=bool(wavi_simulation),
        )
        if not dataset:
            raise ValueError(f"Constructed dataset is empty for:\n{simulation_dir}")
        save_bundled_dataset_to_netcdf(
            bundled_dataset=dataset,
            output_path=output_path,
            selected_features_x=selected_features,
            dt=dt,
            filter_info=filter_info,
            coordinate_units=coordinate_units,
            time_units=time_units,
            overwrite=overwrite,
        )
        summaries.append(
            {
                "trajectory": simulation,
                "samples": len(dataset),
                "removed_nodes": filter_info["n_removed_nodes"],
                "kept_nodes": filter_info["n_kept_nodes"],
                "isolated_nodes": filter_info["n_isolated_nodes_after"],
                "output_file": output_path.name,
            }
        )
        del dataset
        gc.collect()

    print("\nFinished processing datasets")
    print(f"Converted trajectories: {len(summaries)}")
    print(f"Skipped existing trajectories: {skipped}")
    if summaries:
        print(f"Total graph samples saved: {sum(item['samples'] for item in summaries)}")
        print(
            "Total retained nodes across trajectories: "
            f"{sum(item['kept_nodes'] for item in summaries)}"
        )
        print(
            "Total nodes removed across trajectories: "
            f"{sum(item['removed_nodes'] for item in summaries)}"
        )
    return output_paths


def read_netcdf_file(config: Config, mesh_filename: str):
    """Load a processed NetCDF graph trajectory and print a compact summary.

    The NetCDF file is reconstructed as a list of PyTorch Geometric ``Data``
    objects through ``load_netcdf_as_pyg``.

    Args:
        config: Validated configuration used to locate the output directory.
        mesh_filename: NetCDF filename relative to ``data.mesh_subdir``.

    Returns:
        Reconstructed temporal graph samples.
    """
    data_config = _config_section(config, "data", "directories")
    root_dir = Path(_setting(data_config, "root_dir"))
    output_subdir = _setting(
        data_config, "mesh_subdir", "mesh_trajectories_subdir"
    )
    load_path = root_dir / output_subdir / mesh_filename

    if load_path.suffix.lower() == ".nc":
        loaded_dataset, metadata = load_netcdf_as_pyg(
            load_path, return_metadata=True
        )
        print(f"Loaded {len(loaded_dataset)} samples.")
        print("First sample:")
        print(loaded_dataset[0])
        print("Selected feature names:")
        print(metadata["selected_feature_names"])
        return loaded_dataset

    loaded_dataset = torch.load(load_path, weights_only=False)
    print(f"Loaded {len(loaded_dataset)} samples.")
    print("First sample:")
    print(loaded_dataset[0])
    return loaded_dataset