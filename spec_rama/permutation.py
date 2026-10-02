from typing import Tuple

import torch


def compute_greedy_tsp_1d(matrix: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    Computes a 1D Greedy TSP permutation of rows (axis=0) or columns (axis=1)
    to minimize distance between consecutive rows/columns, maximizing spectral energy concentration.
    Fast PyTorch Tensor C++/CUDA implementation.
    """
    data = matrix.t().detach() if axis == 1 else matrix.detach()
    num_vecs = data.shape[0]

    if num_vecs <= 2:
        return torch.arange(num_vecs, dtype=torch.long, device=matrix.device)

    # Fast pairwise L2 distance using PyTorch cdist
    dist_matrix = torch.cdist(data.float(), data.float())

    visited = torch.zeros(num_vecs, dtype=torch.bool, device=matrix.device)
    path = torch.zeros(num_vecs, dtype=torch.long, device=matrix.device)

    curr = 0
    path[0] = curr
    visited[curr] = True

    inf_val = torch.finfo(dist_matrix.dtype).max
    dist_matrix.fill_diagonal_(inf_val)

    for i in range(1, num_vecs):
        dists = dist_matrix[curr]
        dists_masked = dists.masked_fill(visited, inf_val)
        next_node = torch.argmin(dists_masked)
        path[i] = next_node
        visited[next_node] = True
        curr = next_node

    return path


def compute_pca_permutation(matrix: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    Orders rows or columns according to their first principal component score.
    """
    if axis == 1:
        data = matrix.t().detach()
    else:
        data = matrix.detach()

    if data.shape[0] <= 2:
        return torch.arange(data.shape[0], dtype=torch.long)

    # Center data
    mean = data.mean(dim=0, keepdim=True)
    centered = data - mean

    # SVD
    try:
        _, _, V = torch.pca_lowrank(centered, q=1)
        scores = torch.matmul(centered, V[:, 0])
    except Exception:
        # Fallback if SVD doesn't converge
        scores = torch.norm(data, dim=1)

    perm = torch.argsort(scores)
    return perm


def compute_total_variation_2d(matrix: torch.Tensor) -> float:
    """Computes total variation (TV norm) of a 2D matrix measuring spatial noise."""
    tv_rows = torch.abs(matrix[1:, :] - matrix[:-1, :]).sum()
    tv_cols = torch.abs(matrix[:, 1:] - matrix[:, :-1]).sum()
    return (tv_rows + tv_cols).item()


def compute_joint_bipartite_2d_permutation(
    weight: torch.Tensor, max_iters: int = 5, tol: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a joint 2D bipartite permutation by alternating TSP sorting on rows and columns
    until Total Variation (TV) converges within tolerance 'tol' or reaches 'max_iters'.
    This creates a smooth 2D manifold preserving 2D spatial topology.
    """
    out_dim, in_dim = weight.shape
    row_perm = torch.arange(out_dim, dtype=torch.long)
    col_perm = torch.arange(in_dim, dtype=torch.long)

    w_curr = weight.detach().clone()
    prev_tv = compute_total_variation_2d(w_curr)

    for iter_idx in range(max_iters):
        r_p = compute_greedy_tsp_1d(w_curr, axis=0)
        row_perm = row_perm[r_p]
        w_curr = w_curr[r_p, :]

        c_p = compute_greedy_tsp_1d(w_curr, axis=1)
        col_perm = col_perm[c_p]
        w_curr = w_curr[:, c_p]

        curr_tv = compute_total_variation_2d(w_curr)
        rel_improvement = (prev_tv - curr_tv) / (prev_tv + 1e-8)

        if rel_improvement < tol and iter_idx >= 1:
            # Reached Total Variation convergence threshold
            break

        prev_tv = curr_tv

    return row_perm, col_perm


_PERMUTATION_REGISTRY = {}


def register_permutation_method(name: str, fn):
    """
    Registers a custom 2D permutation function.
    Function signature: fn(weight: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]
    """
    _PERMUTATION_REGISTRY[name.lower()] = fn


def list_permutation_methods():
    """Returns all registered and built-in permutation methods."""
    builtins = ["bipartite_tsp", "tsp", "pca", "random", "none"]
    return sorted(list(set(builtins + list(_PERMUTATION_REGISTRY.keys()))))


def compute_2d_permutations(
    weight: torch.Tensor,
    method: str = "bipartite_tsp",
    max_iters: int = 5,
    tol: float = 1e-3,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes both row and column permutations for a 2D weight matrix (out_features, in_features).

    Args:
        weight: PyTorch Tensor (out_features, in_features)
        method: "bipartite_tsp", "tsp", "pca", "random", "none", or any custom registered method name
        max_iters: Maximum iterations for bipartite TSP (default: 5)
        tol: Convergence tolerance for relative total variation improvement
    Returns:
        row_perm: indices for rows (length out_features)
        col_perm: indices for columns (length in_features)
    """
    method_key = method.lower()
    if method_key in _PERMUTATION_REGISTRY:
        return _PERMUTATION_REGISTRY[method_key](weight, max_iters=max_iters, tol=tol, **kwargs)
    elif method_key == "bipartite_tsp":
        return compute_joint_bipartite_2d_permutation(weight, max_iters=max_iters, tol=tol)
    elif method_key == "tsp":
        row_perm = compute_greedy_tsp_1d(weight, axis=0)
        col_perm = compute_greedy_tsp_1d(weight, axis=1)
        return row_perm, col_perm
    elif method_key == "pca":
        row_perm = compute_pca_permutation(weight, axis=0)
        col_perm = compute_pca_permutation(weight, axis=1)
        return row_perm, col_perm
    elif method_key == "random":
        row_perm = torch.randperm(weight.shape[0])
        col_perm = torch.randperm(weight.shape[1])
        return row_perm, col_perm
    else:
        row_perm = torch.arange(weight.shape[0], dtype=torch.long)
        col_perm = torch.arange(weight.shape[1], dtype=torch.long)
        return row_perm, col_perm



def compute_3d_tensor_permutations(
    tensor_3d: torch.Tensor, method: str = "tsp"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes permutations across all 3 axes of a 3D Tensor (Depth/Layers L, Out_features Y, In_features X).

    Returns:
        perm_z: permutation along depth axis Z (layers/heads)
        perm_y: permutation along row axis Y
        perm_x: permutation along column axis X
    """
    L, Y, X = tensor_3d.shape

    # 1. Depth axis Z permutation (correlate 2D weight slices across layers)
    flat_layers = tensor_3d.view(L, -1)
    perm_z = (
        compute_greedy_tsp_1d(flat_layers, axis=0) if method == "tsp" else compute_pca_permutation(flat_layers, axis=0)
    )

    # 2. Average weight across depth to find global Y and X permutations
    mean_2d = tensor_3d[perm_z].mean(dim=0)
    perm_y, perm_x = compute_2d_permutations(mean_2d, method=method)

    return perm_z, perm_y, perm_x
