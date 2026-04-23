"""Methods for converting WAVI spinup (grid) datasets to torch mesh datasets."""


from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay
from torch_geometric.data import Data
from tqdm import tqdm
import xarray as xr

# from icemesh.config import Config
from icemesh.config import *


def _grid_to_mesh(
    model_config: ModelConfig,
    ds: xr.Dataset,
):
    """Converts regular 2D grid data into unstructured mesh data.

    Performs Delaunay triangulation on the grid points and flattens all 2D fields
    into 1D tensors suitable for GNNs. Optionally computes one-hot node-type
    encodings for boundary-condition information.

    Args:
        coords: 1D arrays defining the grid axes.
        values: 2D numpy arrays of shape [H, W] representing physical fields.
        param: 2D arrays representing per-node parameter fields.

    Returns:
        Tuple containing mesh geometry, flattened feature tensors, and optionally
        node_type_oh if model_config.use_node_types=True.
    """
    # Mesh generation
    # Create coordinate grid and flatten to list of points (N = H * W)
    x_grid, y_grid = np.meshgrid(ds["x"], ds["y"])
    points = torch.tensor(np.vstack([x_grid.flatten(), y_grid.flatten()]).T)  # Shape: [N, 2]

    # Perform Delaunay triangulation to define connectivity
    triangles = torch.tensor(Delaunay(points).simplices)  # Shape: [Num_Triangles, 3]

    # Vector field processing
    # Stack U/V components into 2D tensors -> Shape: [N, 2]
    velocity = torch.tensor(np.vstack((ds["u"].data.flatten(), ds["v"].data.flatten())).T).float()
    velocity_surface = torch.tensor(np.vstack((ds["us"].data.flatten(), ds["vs"].data.flatten())).T).float()
    velocity_bed = torch.tensor(np.vstack((ds["ub"].data.flatten(), ds["vb"].data.flatten())).T).float()

    # Optional node-type one-hot encoding
    node_type_oh = None
    if model_config.use_node_types:
        H, W = ds["u"].shape

        node_type_2d = np.full((H, W), model_config.node_type_interior, dtype=np.int64)
        node_type_2d[0, :] = model_config.node_type_free_slip
        node_type_2d[-1, :] = model_config.node_type_free_slip
        node_type_2d[:, 0] = model_config.node_type_left_no_slip

        node_type = torch.tensor(node_type_2d.flatten(), dtype=torch.long)
        node_type_oh = F.one_hot(node_type, num_classes=model_config.num_node_types).float()

    return (
        points,
        triangles,
        velocity,
        velocity_surface,
        velocity_bed,
        node_type_oh,
    )


def _triangles_to_edges(triangles: torch.Tensor) -> torch.Tensor:
    """Converts a list of triangles (Simplex) into a list of unique undirected edges
    (Graph Connectivity).

    Args:
        triangles: Tensor of shape [Num_Triangles, 3] containing node indices.

    Returns:
        edge_index: Tensor of shape [2, Num_Edges], sorted lexicographically.
                    Format matches PyTorch Geometric standards.
    """
    edge_set = set()

    # Iterate over every triangle (i, j, k) and extract unique edges
    for tri in triangles.numpy().tolist():
        i, j, k = tri
        # Sort node pairs (i, j) to ensure undirected uniqueness (0,1 is same as 1,0)
        edge_set.update(
            [
                tuple(sorted((i, j))),
                tuple(sorted((j, k))),
                tuple(sorted((k, i))),
            ],
        )

    # Convert set of tuples to tensor -> Shape: [2, E]
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).T

    # Lexicographical sort (sort by source node, then target node)
    # This ensures deterministic graph structure for the GNN.
    edge_np = edge_index.numpy()
    sorted_idx = np.lexsort((edge_np[1], edge_np[0]))
    edge_index = torch.tensor(edge_np[:, sorted_idx], dtype=torch.long)

    return edge_index


def _build_bundled_dataset(
    model_config: ModelConfig,
    simulation_dir: Path,
):
    """Parses a directory of time-series HDF5 files into a list of PyTorch Geometric Data objects.

    Can structure data as single-step transitions or multi-step time bundles.
    Optionally appends one-hot node-type encodings to node features. When enabled,
    node-type channels are included in x but excluded from derivative targets y.

    Args:
        model_config: model configuration parameters.
        simulation_dir: Path to directory containing HDF5 files.

    Returns:
        List of PyG Data objects ready for training.
    """
    # WAVI file discovery.
    wavi_filenames = sorted([f.name for f in simulation_dir.iterdir() if f.is_file()])
    timesteps = len(wavi_filenames)
    dataset = []

    # Sequential parsing loop
    # First, load every file individually and build the graph snapshot for that timestamp.
    for i in tqdm(range(timesteps), desc=f"  Parsing {simulation_dir.name[:15]}...", leave=False):
        file_path = simulation_dir / wavi_filenames[i]

        with xr.open_dataset(file_path, engine="h5netcdf", phony_dims="access") as ds:
            # Note that this dataset has unnamed dimensions and no coordinates.
            # These should be replaced by the two 2D arrays 'x' and 'y'
            ds = ds.rename_vars({"x": "cx", "y": "cy"}) # Coordinates cannot share names with variables.
            ds = ds.assign_coords({"x": ds["cx"][0, :], "y": ds["cy"][:, 0]})
            dims = [d for d in ds.dims]
            ds = ds.swap_dims({dims[0]: "y", dims[1]: "x"})
            ds = ds.drop_vars(["cx", "cy"])

            # Create dt field manually (constant across grid)
            ds = ds.assign(param_dt=(("y", "x"), np.full(ds["param_gt"].shape, model_config.delta_time)))

            # Convert grid to graph nodes
            (
                points, triangles,
                velocity, velocity_surface, velocity_bed,
                node_type_oh
            ) = _grid_to_mesh(model_config, ds)

            # Build graph connectivity
            cells = triangles.long()
            edge_index = _triangles_to_edges(triangles)

            # Feature assembly
            # Stack all node features into a single matrix X of shape [Num_Nodes, Num_Features]
            feature_tensors = [
                velocity,
                velocity_surface,
                velocity_bed
            ]

            # Order the variables to add to the feature tensors (for comparison with existing code).
            # TODO(tvl) replace by iterating over all da variables.
            vars = [
                "h", "b", "s",
                "dhdt", "accumulation",
                "basal_melt",
                "grounded_frac",
                "av_speed", "bed_speed",
                "weertman_c",
                "haf", "dsdh",
                "shelf_strain",
                "β", "βeff",
                "τbed",
                "ηav",
                "quad_f1", "quad_f2",
                "mask",
                "param_gt", "param_dt"
            ]
            for v in vars:
                feature_tensors.append(torch.tensor(ds[v].data.flatten()).float().unsqueeze(-1))
            # for _, variable in ds.data_vars.items():
            #     if variable.shape == ds["param_gt"].shape:
            #         feature_tensors.append(torch.tensor(variable.data.flatten()).float().unsqueeze(-1))

            # Add the one-hot node encoding.
            if model_config.use_node_types:
                feature_tensors.append(node_type_oh)

            x = torch.cat(feature_tensors, dim=-1)

            # Edge attributes
            # Calculate relative position (u_ij) and distance (norm) for edges
            u_i = points[edge_index[0]]  # source node pos
            u_j = points[edge_index[1]]  # target node pos
            u_ij = u_i - u_j  # relative vector
            u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)  # distance

            # Edge features: [dx, dy, distance]
            edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).float()

            # Construct PyG Data object for this timestep
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mesh_pos=points.float(),
                cells=cells,
                trajectory_id=simulation_dir.name,
            )
            dataset.append(data)

    # Dataset construction loop with targets and inputs depending on history size.
    bundled_dataset = []

    if model_config.history_size == 0:
        # Mode: Only present timestep used to predict derivative
        # Input: state at t
        # Target (y): Time derivative approx ((x_{t+1} - x_t) / dt)
        for i in range(timesteps - 1):
            x = dataset[i].x.unsqueeze(1)
            edge_index = dataset[i].edge_index
            edge_attr = dataset[i].edge_attr

            # Calculate finite difference target
            if model_config.use_node_types:
                C = model_config.num_node_types
                y = ((dataset[i + 1].x[:, :-C] - dataset[i].x[:, :-C]) / model_config.delta_time).float()
            else:
                y = ((dataset[i + 1].x - dataset[i].x) / model_config.delta_time).float()

            data = dataset[i]
            data.x = x
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            data.y = y
            data.cells = dataset[i].cells
            data.trajectory_id = simulation_dir.name
            bundled_dataset.append(data)

        # Last element has no target (cannot calculate future derivative)
        last = dataset[-1]
        last.x = last.x.unsqueeze(1)
        last.y = None
        last.cells = dataset[-1].cells
        last.trajectory_id = simulation_dir.name
        bundled_dataset.append(last)

    else:
        bundled_dataset = []
        # valid t: k .. timesteps-2  (need t+1 for derivative target)
        for t in range(model_config.history_size, timesteps - 1):
            # bundle past -> current: [t-k, ..., t]
            x_bundled = torch.stack(
                [dataset[t - model_config.history_size + j].x for j in range(model_config.history_size + 1)], dim=1
            )  # [N, k+1, F]

            # keep target at time t (here: derivative at t)
            if model_config.use_node_types:
                C = model_config.num_node_types
                y = ((dataset[t + 1].x[:, :-C] - dataset[t].x[:, :-C]) / model_config.delta_time).float()
            else:
                y = ((dataset[t + 1].x - dataset[t].x) / model_config.delta_time).float()

            bundled_dataset.append(
                Data(
                    x=x_bundled,
                    edge_index=dataset[t].edge_index,
                    edge_attr=dataset[t].edge_attr,
                    y=y,
                    mesh_pos=dataset[t].mesh_pos,
                    cells=dataset[t].cells,
                    trajectory_id=simulation_dir.name,
                ),
            )

    return bundled_dataset


def wavi_to_mesh(
    config: Config,
    wavi_simulation: str = "",
):
    # Get a list of valid simulation directories
    wavi_dir = config.data.root_dir / config.data.wavi_outputs_subdir
    simulations = (
        sorted([f.name for f in wavi_dir.iterdir() if f.is_dir()]) if wavi_simulation == "" else [wavi_simulation]
    )
    print("simulations:")
    print(simulations)

    output_dir = config.data.root_dir / config.data.mesh_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main Loop with Outer Progress Bar
    for simulation in tqdm(simulations, desc="Processing Datasets", unit="sim"):
        simulation_dir = wavi_dir / simulation
        output_filename = f"{simulation}_incl_node_enc.pt" if config.model.use_node_types else f"{simulation}.pt"
        output_path = output_dir / output_filename

        # Skip if already processed
        if output_path.exists():
            continue

        bundled_dataset = _build_bundled_dataset(
            model_config=config.model,
            simulation_dir=simulation_dir,
        )

        torch.save(bundled_dataset, output_path)


def read_pt_file(config: Config, mesh_filename: str):
    # Load path to the saved dataset
    output_dir = config.data.root_dir / config.data.mesh_subdir
    load_path = output_dir / mesh_filename

    # Load the datasetgit
    loaded_dataset = torch.load(load_path, weights_only=False)

    # Check what was loaded
    print(f"Loaded {len(loaded_dataset)} samples.")
    print("First sample:")
    print(loaded_dataset[0])
