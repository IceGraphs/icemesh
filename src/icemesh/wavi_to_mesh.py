"""
Methods for converting WAVI spinup (grid) datasets to torch mesh datasets.
"""

import h5py
# from icemesh.config import Config
import numpy as np
from pathlib import Path
from scipy.spatial import Delaunay
import torch
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
    param_c: np.ndarray, param_a: np.ndarray, param_gt: np.ndarray, param_dt: np.ndarray
):
    """
    Converts regular 2D grid data into unstructured mesh data (nodes and attributes).
    
    Performs Delaunay triangulation on the grid points and flattens all 2D fields 
    into 1D column vectors suitable for GNNs.

    Args:
        x_coords, y_coords: 1D arrays defining the grid axes.
        u, v, h, etc.: 2D numpy arrays of shape [H, W] representing physical fields.
        param_*: scalar or 2D arrays representing constant physical parameters.

    Returns:
        A tuple containing:
        - points: Node coordinates [N, 2]
        - triangles: Mesh connectivity [T, 3]
        - Tensors for every field flattened to shape [N, 1] or [N, 2]
    """
    
    # Mesh generation
    # Create coordinate grid and flatten to list of points (N = H * W)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    points = np.vstack([x_grid.flatten(), y_grid.flatten()]).T  # Shape: [N, 2]
    
    # Perform Delaunay triangulation to define connectivity
    triangles = Delaunay(points).simplices  # Shape: [Num_Triangles, 3]

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
    p_c  = to_col(param_c)
    p_a  = to_col(param_a)
    p_gt = to_col(param_gt)
    p_dt = to_col(param_dt)

    return (
        points, triangles, vel, surf_vel, bed_vel,
        h, b, s, dhdt, accumulation, basal_melt, grounded_frac,
        av_speed, bed_speed, weertman_c, haf, dsdh, shelf_strain,
        beta, beta_eff, tau_bed, eta_av, quad_f1, quad_f2, mask,
        p_c, p_a, p_gt, p_dt
    )
    




# def grid_to_mesh(
#     x_coords: np.ndarray, y_coords: np.ndarray, 
#     u: np.ndarray, v: np.ndarray, h: np.ndarray, b: np.ndarray, s: np.ndarray, 
#     dhdt: np.ndarray, accumulation: np.ndarray, basal_melt: np.ndarray, 
#     grounded_frac: np.ndarray, av_speed: np.ndarray, 
#     us: np.ndarray, vs: np.ndarray, ub: np.ndarray, vb: np.ndarray, 
#     bed_speed: np.ndarray, weertman_c: np.ndarray, haf: np.ndarray, 
#     dsdh: np.ndarray, shelf_strain: np.ndarray, beta: np.ndarray, 
#     beta_eff: np.ndarray, tau_bed: np.ndarray, eta_av: np.ndarray, 
#     quad_f1: np.ndarray, quad_f2: np.ndarray, mask: np.ndarray, 
#     param_c: np.ndarray, param_a: np.ndarray, param_gt: np.ndarray, param_dt: np.ndarray
# ) -> Mesh:
#     """
#     Converts regular 2D grid data into unstructured mesh data (nodes and attributes).
    
#     Performs Delaunay triangulation on the grid points and flattens all 2D fields 
#     into 1D column vectors suitable for GNNs.

#     Args:
#         x_coords, y_coords: 1D arrays defining the grid axes.
#         u, v, h, etc.: 2D numpy arrays of shape [H, W] representing physical fields.
#         param_*: scalar or 2D arrays representing constant physical parameters.

#     Returns:
#         A tuple containing:
#         - points: Node coordinates [N, 2]
#         - triangles: Mesh connectivity [T, 3]
#         - Tensors for every field flattened to shape [N, 1] or [N, 2]
#     """

#     # Mesh generation
#     # Create coordinate grid and flatten to list of points (N = H * W)
#     x_grid, y_grid = np.meshgrid(x_coords, y_coords)
#     points = np.vstack([x_grid.flatten(), y_grid.flatten()]).T  # Shape: [N, 2]
    
#     # Scalar field processing
#     # Helper to convert 2D array [H, W] -> Column Tensor [N, 1]
#     def to_col(x: np.ndarray) -> torch.Tensor:
#         return torch.tensor(x.flatten()).float().unsqueeze(-1)
    
#     return Mesh(
#         points = points,

#         # Perform Delaunay triangulation to define connectivity
#         triangles = Delaunay(points).simplices,  # Shape: [Num_Triangles, 3]

#         # Vector field processing
#         # Stack U/V components into 2D tensors -> Shape: [N, 2]
#         vel = torch.tensor(np.vstack((u.flatten(), v.flatten())).T).float(),
#         surf_vel = torch.tensor(np.vstack((us.flatten(), vs.flatten())).T).float(),
#         bed_vel = torch.tensor(np.vstack((ub.flatten(), vb.flatten())).T).float(),

#         # All variables
#         h = to_col(h),
#         b = to_col(b),
#         s = to_col(s),
#         dhdt = to_col(dhdt),
#         accumulation = to_col(accumulation),
#         basal_melt = to_col(basal_melt),
#         grounded_frac = to_col(grounded_frac),
#         av_speed = to_col(av_speed),
#         bed_speed = to_col(bed_speed),
#         weertman_c = to_col(weertman_c),
#         haf = to_col(haf),
#         dsdh = to_col(dsdh),
#         shelf_strain = to_col(shelf_strain),
#         beta = to_col(beta),
#         beta_eff = to_col(beta_eff),
#         tau_bed = to_col(tau_bed),
#         eta_av = to_col(eta_av),
#         quad_f1 = to_col(quad_f1),
#         quad_f2 = to_col(quad_f2),
#         mask = to_col(mask),
#         p_c  = to_col(param_c),
#         p_a  = to_col(param_a),
#         p_gt = to_col(param_gt),
#         p_dt = to_col(param_dt)
#     )

# def triangles_to_edges(triangles: torch.Tensor) -> torch.Tensor:
#     """
#     Converts a list of triangles (Simplex) into a list of unique undirected edges 
#     (Graph Connectivity).

#     Args:
#         triangles: Tensor of shape [Num_Triangles, 3] containing node indices.

#     Returns:
#         edge_index: Tensor of shape [2, Num_Edges], sorted lexicographically.
#                     Format matches PyTorch Geometric standards.
#     """
#     edge_set = set()
    
#     # Iterate over every triangle (i, j, k) and extract unique edges
#     for tri in triangles.numpy().tolist():
#         i, j, k = tri
#         # Sort node pairs (i, j) to ensure undirected uniqueness (0,1 is same as 1,0)
#         edge_set.update([
#             tuple(sorted((i, j))),
#             tuple(sorted((j, k))),
#             tuple(sorted((k, i))),
#         ])
    
#     # Convert set of tuples to tensor -> Shape: [2, E]
#     edge_index = torch.tensor(list(edge_set), dtype=torch.long).T
    
#     # Lexicographical sort (sort by source node, then target node)
#     # This ensures deterministic graph structure for the GNN.
#     edge_np = edge_index.numpy()
#     sorted_idx = np.lexsort((edge_np[1], edge_np[0]))
#     edge_index = torch.tensor(edge_np[:, sorted_idx], dtype=torch.long)
    
#     return edge_index


# def bundle_datasets(wavi_outputs_dir: Path, delta_time: float, history_size: int):
#     """
#     Parses a directory of time-series HDF5 files into a list of PyTorch Geometric Data objects.
#     Can structure data as single-step transitions or multi-step time bundles.

#     Args:
#         wavi_outputs_dir: Path to directory containing HDF5 files, sorted by name/time.
#         delta_time: Time delta (dt) between files, used to calculate finite diff derivatives.
#         history_size: Additional timesteps (k) besides current to include in input.  
#            If history_size=0, produces (Input: t, Target: dy/dt). 
#            If history_size>0, produces (Input: Stack[t...t+k], Target: dy/dt at t+k).

#     Returns:
#         List of PyG Data objects ready for training.
#     """

#     # Collect all files in the WAVI output directory.
#     wavi_filenames = sorted([f for f in wavi_outputs_dir.glob("*") if f.is_file()])
#     num_timesteps = len(wavi_filenames)

#     dataset = []

#     # Sequentially parse loop.
#     # First, load every file individually and build the graph snapshot for that timestamp.
#     folder_name = wavi_outputs_dir.name
#     for i in tqdm(range(num_timesteps), desc=f"  Parsing {folder_name[:15]}...", leave=False):    
#         wavi_filename = wavi_filenames[i]

#         with h5py.File(wavi_filename, "r") as wavi_file:
#             # Read Raw Arrays from HDF5 and convert grid to graph nodes
#             h = wavi_file["h"][:, :],
#             mesh = grid_to_mesh(
#                 x_coords = wavi_file["x"][0, :],
#                 y_coords = wavi_file["y"][:, 0],
#                 u = wavi_file["u"][:, :],
#                 v = wavi_file["v"][:, :],
#                 h = h,
#                 b = wavi_file["b"][:, :],
#                 s = wavi_file["s"][:, :],
#                 dhdt = wavi_file["dhdt"][:, :],
#                 accumulation = wavi_file["accumulation"][:, :],
#                 basal_melt = wavi_file["basal_melt"][:, :],
#                 grounded_frac = wavi_file["grounded_frac"][:, :],
#                 av_speed = wavi_file["av_speed"][:, :],
#                 us = wavi_file["us"][:, :],
#                 vs = wavi_file["vs"][:, :],
#                 ub = wavi_file["ub"][:, :],
#                 vb = wavi_file["vb"][:, :],
#                 bed_speed = wavi_file["bed_speed"][:, :],
#                 weertman_c = wavi_file["weertman_c"][:, :],
#                 haf = wavi_file["haf"][:, :],
#                 dsdh = wavi_file["dsdh"][:, :],
#                 shelf_strain = wavi_file["shelf_strain"][:, :],
#                 beta = wavi_file["β"][:, :],
#                 beta_eff = wavi_file["βeff"][:, :],
#                 tau_bed = wavi_file["τbed"][:, :],
#                 eta_av = wavi_file["ηav"][:, :],
#                 quad_f1 = wavi_file["quad_f1"][:, :],
#                 quad_f2 = wavi_file["quad_f2"][:, :],
#                 mask = wavi_file["mask"][:, :],
#                 param_c  = wavi_file["param_c"][:, :],
#                 param_a  = wavi_file["param_a"][:, :],
#                 param_gt = wavi_file["param_gt"][:, :],
#                 param_dt = np.full(h.shape, delta_time)  # Create dt field manually (constant across grid)
#             )

#             # Build graph connectivity
#             edge_index = triangles_to_edges(torch.tensor(mesh.triangles))

#             # Feature assembly
#             # Stack all node features into a single matrix X of shape [Num_Nodes, Num_Features]
#             x = torch.cat([
#                     mesh.vel, mesh.surf_vel, mesh.bed_vel,  # [N, 2] fields
#                     mesh.h, mesh.b, mesh.s, mesh.dhdt, mesh.accumulation, mesh.basal_melt, mesh.grounded_frac,  # [N, 1] fields
#                     mesh.av_speed, mesh.bed_speed,
#                     mesh.weertman_c, mesh.haf, mesh.dsdh, mesh.shelf_strain,
#                     mesh.beta, mesh.beta_eff, mesh.tau_bed, mesh.eta_av,
#                     mesh.quad_f1, mesh.quad_f2,
#                     mesh.mask,
#                     mesh.t_c, mesh.t_a, mesh.t_gt, mesh.t_dt  # append global parameters to every node
#                 ],
#                 dim=-1
#             )
            
#             # Edge attributes
#             # Calculate relative position (u_ij) and distance (norm) for edges
#             u_i = torch.tensor(mesh.points)[edge_index[0]]  # source node pos
#             u_j = torch.tensor(mesh.points)[edge_index[1]]  # target node pos
#             u_ij = u_i - u_j  # relative vector
#             u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)  # distance
            
#             # Edge features: [dx, dy, distance]
#             edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).float()

#             # Construct PyG Data object for this timestep
#             data = Data(
#                 x = x,
#                 edge_index = edge_index,
#                 edge_attr = edge_attr,
#                 mesh_pos = torch.tensor(mesh.points).float(),
#                 cells = torch.tensor(mesh.triangles).long()
#             )
#             dataset.append(data)

#     # Dataset construction loop with targets and inputs depending on k 
#     bundled_mesh = []
    
#     if history_size == 0:
#         # Mode: Only present timestep used to predict derivative
#         # Input: state at t
#         # Target (y): Time derivative approx ((x_{t+1} - x_t) / dt)
#         for i in range(num_timesteps - 1):
#             x = dataset[i].x
#             edge_index = dataset[i].edge_index
#             edge_attr = dataset[i].edge_attr
            
#             # Calculate finite difference target
#             y = ((dataset[i + 1].x - dataset[i].x) / delta_time).float()

#             data = dataset[i]
#             data.x = x
#             data.edge_index = edge_index
#             data.edge_attr = edge_attr
#             data.y = y
#             data.cells = dataset[i].cells
#             bundled_mesh.append(data)
        
#         # Last element has no target (cannot calculate future derivative)
#         last = dataset[-1]
#         last.y = None
#         last.cells = dataset[-1].cells
#         bundled_mesh.append(last)

#     else:
#         # valid t: k .. num_timesteps-2  (need t+1 for derivative target)
#         for t in range(history_size, num_timesteps - 1):
#             # bundle past -> current: [t-k, ..., t]
#             x_bundled = torch.stack([dataset[t - history_size + j].x for j in range(history_size + 1)], dim=1)  # [N, k+1, F]

#             # keep target at time t (here: derivative at t)
#             y = ((dataset[t + 1].x - dataset[t].x) / delta_time).float()

#             bundled_mesh.append(Data(
#                 x=x_bundled,
#                 edge_index=dataset[t].edge_index,
#                 edge_attr=dataset[t].edge_attr,
#                 y=y,
#                 mesh_pos=dataset[t].mesh_pos,
#                 cells=dataset[t].cells,
#             ))

#     return bundled_mesh


# def transform_single_grid_to_mesh(args: Config):
#     """
#     Transform a directory of time-series HDF5 files into a list of PyTorch Geometric Data objects and saves these to file.
    
#     Args:
#         Config: Collection of all processing parameters.
#     """

#     # Prepare directories.
#     wavi_outputs_dir = args.data.root_dir / args.data.wavi_outputs_subdir
#     mesh_dir = args.data.root_dir / args.data.mesh_subdir
#     mesh_filename = mesh_dir / args.data.mesh_filename
#     mesh_dir.mkdir(parents=True, exist_ok=True)

#     # Transform the dataset(s).
#     bundled_mesh = bundle_datasets(
#         wavi_outputs_dir=wavi_outputs_dir,
#         delta_time=args.model.delta_time,
#         history_size=args.model.history_size
#     )

#     # Save the mesh.
#     torch.save(bundled_mesh, mesh_filename)




root_dir = Path("/storage/IceGraphs/data/IceGraph")
# root_dir

    

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
    overlapping: bool = False
):
    """
    Parses a directory of time-series HDF5 files into a list of PyTorch Geometric Data objects.
    Can structure data as single-step transitions or multi-step time bundles.

    Args:
        folder_path: Path to directory containing HDF5 files (sorted by name/time).
        k: History size; additional timesteps besides current to include in input.  
           If k=0, produces (Input: t, Target: dy/dt). 
           If k>0, produces (Input: Stack[t...t+k], Target: dy/dt at t+k).
        dt: Time delta between files (used to calculate finite diff derivatives).
        overlapping: If k>0, determines if bundles slide by 1 (True) or by bundle_size (False).

    Returns:
        List of PyG Data objects ready for training.
    """
    
    folder_name = folder_path.resolve().name
    
    # File discovery
    filenames = sorted([f for f in folder_path.glob("*") if f.is_file()])
    timesteps = len(filenames)
    dataset = []

    # Sequential parsing loop
    # First, load every file individually and build the graph snapshot for that timestamp.
    for i in tqdm(range(timesteps), desc=f"  Parsing {folder_name[:15]}...", leave=False):
        file_path = folder_path / filenames[i]
       
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
            raw_c  = f["param_c"][:, :]
            raw_a  = f["param_a"][:, :]
            raw_gt = f["param_gt"][:, :]
            raw_dt = np.full(h.shape, dt)  # Create dt field manually (constant across grid)

            # Convert grid to graph nodes
            (mesh_pos, triangles, vel, surf_vel, bed_vel, h, b, s, dhdt, accumulation,
             basal_melt, grounded_frac, av_speed, bed_speed, weertman_c, haf, dsdh,
             shelf_strain, beta, beta_eff, tau_bed, eta_av, quad_f1, quad_f2, mask,
             t_c, t_a, t_gt, t_dt) = grid_to_mesh(
                x_coords, y_coords, u, v, h, b, s, dhdt, accumulation,
                basal_melt, grounded_frac, av_speed, us, vs, ub, vb,
                bed_speed, weertman_c, haf, dsdh, shelf_strain, beta,
                beta_eff, tau_bed, eta_av, quad_f1, quad_f2, mask,
                param_c=raw_c, param_a=raw_a, param_gt=raw_gt, param_dt=raw_dt
            )
            
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
                t_c, t_a, t_gt, t_dt  # append global parameters to every node
            ]
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
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                        mesh_pos=torch.tensor(mesh_pos).float(), cells=cells)
            dataset.append(data)

    # Dataset construction loop with targets and inputs depending on k 
    bundled_dataset = []
    
    if k == 0:
        # Mode: Only present timestep used to predict derivative
        # Input: state at t
        # Target (y): Time derivative approx ((x_{t+1} - x_t) / dt)
        for i in range(timesteps - 1):
            x = dataset[i].x
            edge_index = dataset[i].edge_index
            edge_attr = dataset[i].edge_attr
            
            # Calculate finite difference target
            y = ((dataset[i + 1].x - dataset[i].x) / dt).float()

            data = dataset[i]
            data.x = x
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            data.y = y
            data.cells = dataset[i].cells
            bundled_dataset.append(data)
        
        # Last element has no target (cannot calculate future derivative)
        last = dataset[-1]
        last.y = None
        last.cells = dataset[-1].cells
        bundled_dataset.append(last)

    else:
        bundled_dataset = []
        # valid t: k .. timesteps-2  (need t+1 for derivative target)
        for t in range(k, timesteps - 1):
            # bundle past -> current: [t-k, ..., t]
            x_bundled = torch.stack([dataset[t - k + j].x for j in range(k + 1)], dim=1)  # [N, k+1, F]

            # keep target at time t (here: derivative at t)
            y = ((dataset[t + 1].x - dataset[t].x) / dt).float()

            bundled_dataset.append(Data(
                x=x_bundled,
                edge_index=dataset[t].edge_index,
                edge_attr=dataset[t].edge_attr,
                y=y,
                mesh_pos=dataset[t].mesh_pos,
                cells=dataset[t].cells,
            ))

    return bundled_dataset


def one_simulation():
    # Execute the transformation
    folder_path = root_dir / "WAVI_simulations/8km_1param_perturb/1 simulation/outputs"  # map with the output files
    output_dir = root_dir / 'preprocessed_datasets_py'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define parameters
    k = 9
    dt = 1

    # Perform the transformation
    bundled_dataset = build_bundled_dataset(
        folder_path, 
        k=k, 
        dt=dt, 
        overlapping=False
    )

    filename = f'8km_1param_perturb_SMB.pt'
    torch.save(bundled_dataset, output_dir / filename)


def multi_simulation():
    # Configuration
    folder_path = root_dir / "WAVI_simulations/8km_1param_perturb/training_datasets/outputs"

    output_dir = root_dir / 'preprocessed_datasets_py'
    print(f"output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    k = 0
    dt = 1

    # Get list of valid simulation directories
    sim_folders = sorted([f for f in folder_path.glob("*") if f.is_dir()])
    sim_folders = sim_folders[:2]  # TODO(tvl) NOTE: this is only temporarily for faster testing. Should be removed at the end and fully tested...
    print(sim_folders)

    # Main Loop with Outer Progress Bar
    for sim_folder in tqdm(sim_folders, desc="Processing Datasets", unit="sim"):
        
        # new_folder_path = folder_path / folder_name
        new_folder_path = sim_folder

        # Build the dataset (this now has an inner progress bar!)
        bundled_dataset = build_bundled_dataset(
            new_folder_path, 
            k = k, 
            dt = dt, 
            overlapping = False,  # risk of data leakage
        )

        # Save with matching filename
        filename = f'{sim_folder.name}.pt'
        print(f"save to: {output_dir / filename}")
        torch.save(bundled_dataset, output_dir / filename)

    # 34 seconds for 3 datasets of 800 nodes (8km resolution) with 301 timesteps each
    # 667 seconds 11min7sec) for 20 datasets of 800 nodes (8km resolution) with 1000 timesteps each
    # 173 seconds (2min53sec) for 3 datasets of 12800 nodes (2km resolution) with 301 timesteps each
    # 3863 seconds (64min23sec) for 20 datasets of 12800 nodes (2km resolution) with 1000 timesteps each


def read_pt_file():
    # Load path to the saved dataset
    output_dir = root_dir / 'preprocessed_datasets_py'
    filename = f'8km_1param_perturb_SMB.pt'
    load_path = output_dir / filename

    # Load the dataset
    loaded_dataset = torch.load(load_path, weights_only=False)

    # Check what was loaded
    print(f"Loaded {len(loaded_dataset)} samples.")
    print("First sample:")
    print(loaded_dataset[0])






# Testing...
# if __name__ == "__main__":
#     print("exec main")
#     config = Config.from_yaml("../../tests/single_simulation_config.yaml")
#     transform_single_grid_to_mesh(config)
