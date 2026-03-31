"""
Methods for converting WAVI spinup (grid) datasets to torch mesh datasets.
"""

import h5py
# from icemesh.config import Config
from icemesh.config import *
import numpy as np
from pathlib import Path
from scipy.spatial import Delaunay
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm


# class Mesh:
#     points: np.ndarray
#     triangles: np.ndarray
    
#     vel: torch.Tensor
#     surf_vel: torch.Tensor
#     bed_vel: torch.Tensor
    
#     h: torch.Tensor
#     b: torch.Tensor
#     s: torch.Tensor
#     dhdt: torch.Tensor
#     accumulation: torch.Tensor
#     basal_melt: torch.Tensor
#     grounded_frac: torch.Tensor
#     av_speed: torch.Tensor
#     bed_speed: torch.Tensor
#     weertman_c: torch.Tensor
#     haf: torch.Tensor
#     dsdh: torch.Tensor
#     shelf_strain: torch.Tensor
#     beta: torch.Tensor
#     beta_eff: torch.Tensor
#     tau_bed: torch.Tensor
#     eta_av: torch.Tensor
#     quad_f1: torch.Tensor
#     quad_f2: torch.Tensor
#     mask: torch.Tensor
#     p_c: torch.Tensor
#     p_a: torch.Tensor
#     p_gt: torch.Tensor
#     p_dt: torch.Tensor

# Configuration
NT_INTERIOR = 0
NT_LEFT_NO_SLIP = 1
NT_FREE_SLIP = 2
NUM_NODE_TYPES = 3


# TODO: define classes to hold [x_coords, y_coords], [physical fields], [param_gt, param_dt], [NT_...]
def grid_to_mesh(
    x_coords: np.ndarray, y_coords: np.ndarray, 
    u: np.ndarray, v: np.ndarray, h: np.ndarray, b: np.ndarray, s: np.ndarray, 
    dhdt: np.ndarray, accumulation: np.ndarray, basal_melt: np.ndarray, 
    grounded_frac: np.ndarray, av_speed: np.ndarray, 
    us: np.ndarray, vs: np.ndarray, ub: np.ndarray, vb: np.ndarray, 
    bed_speed: np.ndarray, weertman_c: np.ndarray, haf: np.ndarray, 
    dsdh: np.ndarray, shelf_strain: np.ndarray, beta: np.ndarray, 
    beta_eff: np.ndarray, tau_bed: np.ndarray, eta_av: np.ndarray, 
    quad_f1: np.ndarray, quad_f2: np.ndarray, mask: np.ndarray, 
    param_gt: np.ndarray, param_dt: np.ndarray,
    use_node_types: bool = False  
):
    """
    Converts regular 2D grid data into unstructured mesh data.

    Performs Delaunay triangulation on the grid points and flattens all 2D fields
    into 1D tensors suitable for GNNs. Optionally computes one-hot node-type
    encodings for boundary-condition information.

    Args:
        x_coords, y_coords: 1D arrays defining the grid axes.
        u, v, h, etc.: 2D numpy arrays of shape [H, W] representing physical fields.
        param_gt, param_dt: 2D arrays representing per-node parameter fields.
        use_node_types: If True, compute and return one-hot node-type encodings.

    Returns:
        Tuple containing mesh geometry, flattened feature tensors, and optionally
        node_type_oh if use_node_types=True.
    """
    # Mesh generation
    # Create coordinate grid and flatten to list of points (N = H * W)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    points = np.vstack([x_grid.flatten(), y_grid.flatten()]).T  # Shape: [N, 2]
    
    # Perform Delaunay triangulation to define connectivity
    triangles = Delaunay(points).simplices  # Shape: [Num_Triangles, 3]

    # Optional node-type one-hot encoding
    node_type_oh = None

    if use_node_types:
        H, W = u.shape

        node_type_2d = np.full((H, W), NT_INTERIOR, dtype=np.int64)
        node_type_2d[0, :] = NT_FREE_SLIP
        node_type_2d[-1, :] = NT_FREE_SLIP
        node_type_2d[:, 0] = NT_LEFT_NO_SLIP

        node_type = torch.tensor(node_type_2d.flatten(), dtype=torch.long)
        node_type_oh = F.one_hot(node_type, num_classes=NUM_NODE_TYPES).float()

    # Vector field processing
    # Stack U/V components into 2D tensors -> Shape: [N, 2]
    vel = torch.tensor(np.vstack((u.flatten(), v.flatten())).T).float()
    surf_vel = torch.tensor(np.vstack((us.flatten(), vs.flatten())).T).float()
    bed_vel = torch.tensor(np.vstack((ub.flatten(), vb.flatten())).T).float()

    # Scalar field processing
    # Helper to convert 2D array [H, W] -> Column Tensor [N, 1]
    def to_col(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x.flatten()).float().unsqueeze(-1)

    # All variables
    h = to_col(h)
    b = to_col(b)
    s = to_col(s)
    dhdt = to_col(dhdt)
    accumulation = to_col(accumulation)
    basal_melt = to_col(basal_melt)
    grounded_frac = to_col(grounded_frac)
    av_speed = to_col(av_speed)
    bed_speed = to_col(bed_speed)
    weertman_c = to_col(weertman_c)
    haf = to_col(haf)
    dsdh = to_col(dsdh)
    shelf_strain = to_col(shelf_strain)
    beta = to_col(beta)
    beta_eff = to_col(beta_eff)
    tau_bed = to_col(tau_bed)
    eta_av = to_col(eta_av)
    quad_f1 = to_col(quad_f1)
    quad_f2 = to_col(quad_f2)
    mask = to_col(mask)
    p_gt = to_col(param_gt)
    p_dt = to_col(param_dt)

    if use_node_types:
        return (
            points, triangles, vel, surf_vel, bed_vel,
            node_type_oh,
            h, b, s, dhdt, accumulation, basal_melt, grounded_frac,
            av_speed, bed_speed, weertman_c, haf, dsdh, shelf_strain,
            beta, beta_eff, tau_bed, eta_av, quad_f1, quad_f2, mask,
            p_gt, p_dt
        )
    else:
        return (
            points, triangles, vel, surf_vel, bed_vel,
            h, b, s, dhdt, accumulation, basal_melt, grounded_frac,
            av_speed, bed_speed, weertman_c, haf, dsdh, shelf_strain,
            beta, beta_eff, tau_bed, eta_av, quad_f1, quad_f2, mask,
            p_gt, p_dt
        )
    
    

def triangles_to_edges(triangles: torch.Tensor) -> torch.Tensor:
    """
    Converts a list of triangles (Simplex) into a list of unique undirected edges 
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
        edge_set.update([
            tuple(sorted((i, j))),
            tuple(sorted((j, k))),
            tuple(sorted((k, i))),
        ])
    
    # Convert set of tuples to tensor -> Shape: [2, E]
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).T
    
    # Lexicographical sort (sort by source node, then target node)
    # This ensures deterministic graph structure for the GNN.
    edge_np = edge_index.numpy()
    sorted_idx = np.lexsort((edge_np[1], edge_np[0]))
    edge_index = torch.tensor(edge_np[:, sorted_idx], dtype=torch.long)
    
    return edge_index



def build_bundled_dataset(
    folder_path: Path, 
    k: int = 0, 
    dt: float = 1.0, 
    use_node_types: bool = False
):
    """
    Parses a directory of time-series HDF5 files into a list of PyTorch Geometric Data objects.

    Can structure data as single-step transitions or multi-step time bundles.
    Optionally appends one-hot node-type encodings to node features. When enabled,
    node-type channels are included in x but excluded from derivative targets y.

    Args:
        folder_path: Path to directory containing HDF5 files.
        k: History size; additional timesteps besides current to include in input.
        dt: Time delta between files used for finite-difference targets.
        use_node_types: If True, append one-hot node-type encodings to x.

    Returns:
        List of PyG Data objects ready for training.
    """
    # folder_name = os.path.basename(os.path.normpath(folder_path))
    folder_name = folder_path.name
    trajectory_id = folder_name
    
    # File discovery
    filenames = sorted([f.name for f in folder_path.iterdir() if f.is_file()])
    timesteps = len(filenames)
    dataset = []

    # Sequential parsing loop
    # First, load every file individually and build the graph snapshot for that timestamp.
    for i in tqdm(range(timesteps), desc=f"  Parsing {folder_name[:15]}...", leave=False):
        file_path = Path(folder_path) / filenames[i]
        
        with h5py.File(file_path, "r") as f:
            # Read Raw Arrays from HDF5
            x_coords = f["x"][0, :]; y_coords = f["y"][:, 0]
            
            u = f["u"][:, :]; v = f["v"][:, :]
            us = f["us"][:, :]; vs = f["vs"][:, :]
            ub = f["ub"][:, :]; vb = f["vb"][:, :]
            h = f["h"][:, :]; b = f["b"][:, :]; s = f["s"][:, :]
            dhdt = f["dhdt"][:, :]; accumulation = f["accumulation"][:, :]
            basal_melt = f["basal_melt"][:, :]; grounded_frac = f["grounded_frac"][:, :]
            av_speed = f["av_speed"][:, :]
            bed_speed = f["bed_speed"][:, :]
            weertman_c = f["weertman_c"][:, :]
            haf = f["haf"][:, :]; dsdh = f["dsdh"][:, :]
            shelf_strain = f["shelf_strain"][:, :]
            beta = f["β"][:, :]; beta_eff = f["βeff"][:, :]
            tau_bed = f["τbed"][:, :]; eta_av = f["ηav"][:, :]
            quad_f1 = f["quad_f1"][:, :]; quad_f2 = f["quad_f2"][:, :]
            mask = f["mask"][:, :]
            raw_gt = f["param_gt"][:, :]
            raw_dt = np.full(h.shape, dt)  # Create dt field manually (constant across grid)

            # Convert grid to graph nodes
            outputs = grid_to_mesh(
                x_coords, y_coords,
                u, v, h, b, s,
                dhdt, accumulation, basal_melt,
                grounded_frac, av_speed,
                us, vs, ub, vb,
                bed_speed, weertman_c, haf,
                dsdh, shelf_strain, beta,
                beta_eff, tau_bed, eta_av,
                quad_f1, quad_f2, mask,
                param_gt=raw_gt, param_dt=raw_dt,
                use_node_types=use_node_types
            )
            
            if use_node_types:
                (mesh_pos, triangles, vel, surf_vel, bed_vel,
                node_type_oh,
                h, b, s, dhdt, accumulation, basal_melt, grounded_frac,
                av_speed, bed_speed, weertman_c, haf, dsdh,
                shelf_strain, beta, beta_eff, tau_bed, eta_av,
                quad_f1, quad_f2, mask, t_gt, t_dt) = outputs
            else:
                (mesh_pos, triangles, vel, surf_vel, bed_vel,
                h, b, s, dhdt, accumulation, basal_melt, grounded_frac,
                av_speed, bed_speed, weertman_c, haf, dsdh,
                shelf_strain, beta, beta_eff, tau_bed, eta_av,
                quad_f1, quad_f2, mask, t_gt, t_dt) = outputs
                        
            # Build graph connectivity
            cells = torch.tensor(triangles).long()
            edge_index = triangles_to_edges(torch.tensor(triangles))

            # Feature assembly
            # Stack all node features into a single matrix X of shape [Num_Nodes, Num_Features]
            feature_tensors = [
                vel, surf_vel, bed_vel,  # [N, 2] fields
                h, b, s, dhdt, accumulation, basal_melt, grounded_frac,  # [N, 1] fields
                av_speed, bed_speed,
                weertman_c, haf, dsdh, shelf_strain,
                beta, beta_eff, tau_bed, eta_av,
                quad_f1, quad_f2,
                mask,
                t_gt, t_dt  # append parameter fields to every node
            ]
            
            if use_node_types:
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
                trajectory_id=trajectory_id, 
            )
            dataset.append(data)

    # Dataset construction loop with targets and inputs depending on k 
    bundled_dataset = []
    
    if k == 0:
        # Mode: Only present timestep used to predict derivative
        # Input: state at t
        # Target (y): Time derivative approx ((x_{t+1} - x_t) / dt)
        for i in range(timesteps - 1):
            x = dataset[i].x.unsqueeze(1)
            edge_index = dataset[i].edge_index
            edge_attr = dataset[i].edge_attr
            
            # Calculate finite difference target
            if use_node_types:
                C = NUM_NODE_TYPES
                y = ((dataset[i + 1].x[:, :-C] - dataset[i].x[:, :-C]) / dt).float()
            else:
                y = ((dataset[i + 1].x - dataset[i].x) / dt).float()
                
            data = dataset[i]
            data.x = x
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            data.y = y
            data.cells = dataset[i].cells
            data.trajectory_id = trajectory_id
            bundled_dataset.append(data)
        
        # Last element has no target (cannot calculate future derivative)
        last = dataset[-1]
        last.x = last.x.unsqueeze(1)
        last.y = None
        last.cells = dataset[-1].cells
        last.trajectory_id = trajectory_id
        bundled_dataset.append(last)

    else:
        bundled_dataset = []
        # valid t: k .. timesteps-2  (need t+1 for derivative target)
        for t in range(k, timesteps - 1):
            # bundle past -> current: [t-k, ..., t]
            x_bundled = torch.stack([dataset[t - k + j].x for j in range(k + 1)], dim=1)  # [N, k+1, F]

            # keep target at time t (here: derivative at t)
            if use_node_types:
                C = NUM_NODE_TYPES
                y = ((dataset[t + 1].x[:, :-C] - dataset[t].x[:, :-C]) / dt).float()
            else:
                y = ((dataset[t + 1].x - dataset[t].x) / dt).float()
                
            bundled_dataset.append(Data(
                x=x_bundled,
                edge_index=dataset[t].edge_index,
                edge_attr=dataset[t].edge_attr,
                y=y,
                mesh_pos=dataset[t].mesh_pos,
                cells=dataset[t].cells,
                trajectory_id=trajectory_id,
            ))

    return bundled_dataset


def wavi_to_mesh(
    wavi_dir: Path,
    output_dir: Path,
    wavi_simulation: str = "",
    history_size: int = 0, 
    delta_time: float = 1.0,
    use_node_types = False
):
    # Get a list of valid simulation directories
    simulations = sorted([f.name for f in wavi_dir.iterdir() if f.is_dir()]) if wavi_simulation == "" else [wavi_simulation]
    print("simulations:") #TODO(tvl) temp for debugging
    print(simulations) #TODO(tvl) temp for debugging

    output_dir.mkdir(parents=True, exist_ok=True)

    # Main Loop with Outer Progress Bar
    for simulation in tqdm(simulations, desc="Processing Datasets", unit="sim"):
        simulation_dir = wavi_dir / simulation
        output_filename = f"{simulation}_incl_node_enc.pt" if use_node_types else f"{simulation}.pt"
        output_path = output_dir / output_filename
        
        # Skip if already processed
        if output_path.exists():
            continue

        bundled_dataset = build_bundled_dataset(
            simulation_dir,
            k = history_size,
            dt = delta_time,
            use_node_types=use_node_types,
        )

        torch.save(bundled_dataset, output_path)


def one_simulation(config: Config):
    wavi_to_mesh(
        wavi_dir = config.data.root_dir / config.data.wavi_outputs_subdir,
        output_dir = config.data.root_dir / config.data.mesh_subdir,
        wavi_simulation = config.data.wavi_simulation,
        history_size = config.model.history_size, 
        delta_time = config.model.delta_time
    )


def multi_simulation(config: Config):
    wavi_to_mesh(
        wavi_dir = config.data.root_dir / config.data.wavi_outputs_subdir,
        output_dir = config.data.root_dir / config.data.mesh_subdir,
        history_size = config.model.history_size, 
        delta_time = config.model.delta_time
    )


def read_pt_file(config: Config):
    # Load path to the saved dataset
    output_dir = config.data.root_dir / config.data.mesh_subdir
    filename = config.data.mesh_filename  # example file
    load_path = output_dir / filename

    # Load the datasetgit 
    loaded_dataset = torch.load(load_path, weights_only=False)

    # Check what was loaded
    print(f"Loaded {len(loaded_dataset)} samples.")
    print("First sample:")
    print(loaded_dataset[0])


    






# Testing...
# if __name__ == "__main__":
#     print("exec main")
#     config = Config.from_yaml("tests/wavi_to_mesh_config.yaml")
#     one_simulation(config)
#     # transform_single_grid_to_mesh(config) <- TODO(tvl) rename function to something like this...
#     multi_simulation(config)
#     read_pt_file(config)
