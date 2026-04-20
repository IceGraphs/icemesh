"""Methods for converting WAVI spinup (grid) datasets to torch mesh datasets."""

 # TODO rem tmp imports
import copy
import subprocess


from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay
from torch_geometric.data import Data
from tqdm import tqdm

# from icemesh.config import Config
from icemesh.config import *


class GridAxes:
    x: np.ndarray
    y: np.ndarray

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y


class Model: # TODO: replace by xarray
    # Core variables.
    u: np.ndarray|torch.Tensor
    v: np.ndarray|torch.Tensor
    us: np.ndarray|torch.Tensor
    vs: np.ndarray|torch.Tensor
    ub: np.ndarray|torch.Tensor
    vb: np.ndarray|torch.Tensor
    h: np.ndarray|torch.Tensor
    b: np.ndarray|torch.Tensor  # bedrock elevation (?)
    s: np.ndarray|torch.Tensor  # surface elevation (?)
    accumulation: np.ndarray|torch.Tensor  # accumulation rate (?)
    basal_melt: np.ndarray|torch.Tensor
    haf: np.ndarray|torch.Tensor  # height above floatation (?)

    # Optional variables, for visualization purposes.    
    dhdt: np.ndarray|torch.Tensor = None
    grounded_frac: np.ndarray|torch.Tensor = None
    av_speed: np.ndarray|torch.Tensor = None
    bed_speed: np.ndarray|torch.Tensor = None
    weertman_c: np.ndarray|torch.Tensor = None
    dsdh: np.ndarray|torch.Tensor = None
    shelf_strain: np.ndarray|torch.Tensor = None
    beta: np.ndarray|torch.Tensor = None
    beta_eff: np.ndarray|torch.Tensor = None
    tau_bed: np.ndarray|torch.Tensor = None
    eta_av: np.ndarray|torch.Tensor = None
    quad_f1: np.ndarray|torch.Tensor = None
    quad_f2: np.ndarray|torch.Tensor = None
    mask: np.ndarray|torch.Tensor = None

    def __init__(self, dictionary: dict):
        for key, value in dictionary.items():
            setattr(self, key, value)
    

class ParameterFields:
    gt: np.ndarray|torch.Tensor
    dt: np.ndarray|torch.Tensor
    
    def __init__(self, gt: np.ndarray, dt: np.ndarray):
        self.gt = gt
        self.dt = dt



# TODO: define classes to hold, [physical fields]
def grid_to_mesh(
    model_config: ModelConfig,
    coords: GridAxes,
    values: Model,
    param: ParameterFields
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
    x_grid, y_grid = np.meshgrid(coords.x, coords.y)
    points = np.vstack([x_grid.flatten(), y_grid.flatten()]).T  # Shape: [N, 2]

    # Perform Delaunay triangulation to define connectivity
    triangles = Delaunay(points).simplices  # Shape: [Num_Triangles, 3]

    # Optional node-type one-hot encoding
    node_type_oh = None

    if model_config.use_node_types:
        H, W = values.u.shape

        node_type_2d = np.full((H, W), model_config.node_type_interior, dtype=np.int64)
        node_type_2d[0, :] = model_config.node_type_free_slip
        node_type_2d[-1, :] = model_config.node_type_free_slip
        node_type_2d[:, 0] = model_config.node_type_left_no_slip

        node_type = torch.tensor(node_type_2d.flatten(), dtype=torch.long)
        node_type_oh = F.one_hot(node_type, num_classes=model_config.num_node_types).float()

    # Vector field processing
    # Stack U/V components into 2D tensors -> Shape: [N, 2]
    vel = torch.tensor(np.vstack((values.u.flatten(), values.v.flatten())).T).float()
    surf_vel = torch.tensor(np.vstack((values.us.flatten(), values.vs.flatten())).T).float()
    bed_vel = torch.tensor(np.vstack((values.ub.flatten(), values.vb.flatten())).T).float()

    # Scalar field processing
    # Helper to convert 2D array [H, W] -> Column Tensor [N, 1]
    def to_col(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x.flatten()).float().unsqueeze(-1)

    # All variables
    values.h = to_col(values.h)
    values.b = to_col(values.b)
    values.s = to_col(values.s)
    values.accumulation = to_col(values.accumulation)
    values.basal_melt = to_col(values.basal_melt)
    values.haf = to_col(values.haf)

    values.dhdt = to_col(values.dhdt)
    values.grounded_frac = to_col(values.grounded_frac)
    values.av_speed = to_col(values.av_speed)
    values.bed_speed = to_col(values.bed_speed)
    values.weertman_c = to_col(values.weertman_c)
    values.dsdh = to_col(values.dsdh)
    values.shelf_strain = to_col(values.shelf_strain)
    values.beta = to_col(values.beta)
    values.beta_eff = to_col(values.beta_eff)
    values.tau_bed = to_col(values.tau_bed)
    values.eta_av = to_col(values.eta_av)
    values.quad_f1 = to_col(values.quad_f1)
    values.quad_f2 = to_col(values.quad_f2)
    values.mask = to_col(values.mask)
    
    param.gt = to_col(param.gt)
    param.dt = to_col(param.dt)

    return (
        points, triangles, vel, surf_vel, bed_vel,
        values,
        param,
        node_type_oh
    )


def triangles_to_edges(triangles: torch.Tensor) -> torch.Tensor:
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
            ]
        )

    # Convert set of tuples to tensor -> Shape: [2, E]
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).T

    # Lexicographical sort (sort by source node, then target node)
    # This ensures deterministic graph structure for the GNN.
    edge_np = edge_index.numpy()
    sorted_idx = np.lexsort((edge_np[1], edge_np[0]))
    edge_index = torch.tensor(edge_np[:, sorted_idx], dtype=torch.long)

    return edge_index


def build_bundled_dataset(
    model_config: ModelConfig,
    simulation_dir: Path
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

        with h5py.File(file_path, "r") as f:
            # Read Raw Arrays from HDF5.
            coords = GridAxes(
                x=f["x"][0, :],
                y=f["y"][:, 0]
            )
            
            values = Model({
                "u": f["u"][:, :],
                "v": f["v"][:, :],
                "us": f["us"][:, :],
                "vs": f["vs"][:, :],
                "ub": f["ub"][:, :],
                "vb": f["vb"][:, :],
                "h": f["h"][:, :],
                "b": f["b"][:, :],
                "s": f["s"][:, :],
                "accumulation": f["accumulation"][:, :],
                "basal_melt": f["basal_melt"][:, :],
                "haf": f["haf"][:, :],
                "dhdt": f["dhdt"][:, :],
                "grounded_frac": f["grounded_frac"][:, :],
                "av_speed": f["av_speed"][:, :],
                "bed_speed": f["bed_speed"][:, :],
                "weertman_c": f["weertman_c"][:, :],
                "dsdh": f["dsdh"][:, :],
                "shelf_strain": f["shelf_strain"][:, :],
                "beta": f["β"][:, :],
                "beta_eff": f["βeff"][:, :],
                "tau_bed": f["τbed"][:, :],
                "eta_av": f["ηav"][:, :],
                "quad_f1": f["quad_f1"][:, :],
                "quad_f2": f["quad_f2"][:, :],
                "mask": f["mask"][:, :],
                }
            )

            param = ParameterFields(
                gt=f["param_gt"][:, :],
                dt=np.full(values.h.shape, model_config.delta_time)  # Create dt field manually (constant across grid)
            )

            # Convert grid to graph nodes
            outputs = grid_to_mesh(
                model_config,
                coords,
                values,
                param
            )

            (mesh_pos, triangles, vel, surf_vel, bed_vel,
            values,
            param,
            node_type_oh) = outputs

            # Build graph connectivity
            cells = torch.tensor(triangles).long()
            edge_index = triangles_to_edges(torch.tensor(triangles))

            # Feature assembly
            # Stack all node features into a single matrix X of shape [Num_Nodes, Num_Features]
            feature_tensors = [
                vel, surf_vel, bed_vel,  # [N, 2] fields
                values.h, values.b, values.s,
                values.dhdt, values.accumulation, values.basal_melt, values.grounded_frac,  # [N, 1] fields
                values.av_speed, values.bed_speed,
                values.weertman_c, values.haf, values.dsdh, values.shelf_strain,
                values.beta, values.beta_eff, values.tau_bed, values.eta_av,
                values.quad_f1, values.quad_f2,
                values.mask,
                param.gt, param.dt  # append parameter fields to every node
            ]

            if model_config.use_node_types:
                feature_tensors.append(node_type_oh)

            x = torch.cat(feature_tensors, dim=-1)

            # Edge attributes
            # Calculate relative position (u_ij) and distance (norm) for edges
            u_i = torch.tensor(mesh_pos)[edge_index[0]]  # source node pos
            u_j = torch.tensor(mesh_pos)[edge_index[1]]  # target node pos
            u_ij = u_i - u_j  # relative vector
            u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)  # distance

            # Edge features: [dx, dy, distance]
            edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).float()

            # Construct PyG Data object for this timestep
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mesh_pos=torch.tensor(mesh_pos).float(),
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
            x_bundled = torch.stack([dataset[t - model_config.history_size + j].x for j in range(model_config.history_size + 1)], dim=1)  # [N, k+1, F]

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
                )
            )

    return bundled_dataset


def wavi_to_mesh(
    config: Config, 
    wavi_simulation: str = ""
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

        bundled_dataset = build_bundled_dataset(
            model_config = config.model,
            simulation_dir = simulation_dir
        )

        torch.save(bundled_dataset, output_path)


#TODO(tvl) rem: temporary placeholders.
def one_simulation(config: Config, wavi_simulation: str):
    wavi_to_mesh(config = config, wavi_simulation = wavi_simulation)


def multi_simulation(config: Config):
    wavi_to_mesh(config = config)


def read_pt_file(config: Config, mesh_filename: str):
    # Load path to the saved dataset
    output_dir = config.data.root_dir / config.data.mesh_subdir
    load_path = output_dir / mesh_filename

    # Load the datasetgit
    loaded_dataset = torch.load(load_path, weights_only = False)

    # Check what was loaded
    print(f"Loaded {len(loaded_dataset)} samples.")
    print("First sample:")
    print(loaded_dataset[0])


# Testing...
if __name__ == "__main__":
    print("exec main")
    config = Config.from_yaml("tests/wavi_to_mesh_config.yaml")


    config_single = copy.deepcopy(config)
    config_single.model.use_node_types=True
    os_filename = "rw_lhs_from_SMB0.29_gT3.12e-03__T100_traj001_incl_node_enc.pt" if config_single.model.use_node_types else "rw_lhs_from_SMB0.29_gT3.12e-03__T100_traj001.pt"
    os_out = config.data.root_dir / config.data.mesh_subdir / os_filename
    if os_out.exists(): os_out.unlink()


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
    
