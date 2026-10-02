"""
Ripple Insertion TSP Solver for Spec-RAMA Matrix Permutation.
Original algorithm and concept by Mario Raúl Carbonell Martínez.

This module implements a dynamic recursive cheapest-insertion TSP algorithm with
cascading wavefront relaxation ('ripple' effect) and 2-opt post-processing to minimize
Total Variation and optimize 2D spectral energy concentration for Spec-RAMA adapters.
"""

from typing import Tuple

import numpy as np
import torch


def compute_ripple_tsp_1d(
    matrix: torch.Tensor,
    axis: int = 0,
    max_k_neighbors: int = 25,
    max_ripple_depth: int = 50,
    enable_2opt: bool = True,
    max_2opt_passes: int = 5,
) -> torch.Tensor:
    """
    Computes a 1D Ripple Insertion TSP permutation along rows (axis=0) or columns (axis=1).
    Minimizes Euclidean path distance between consecutive vectors.

    Args:
        matrix: 2D Tensor (d_out, d_in)
        axis: 0 for rows, 1 for columns
        max_k_neighbors: Neighborhood size for candidate insertion and ripple moves
        max_ripple_depth: Maximum cascading ripple iterations per insertion
        enable_2opt: Whether to run 2-opt local search refinement after insertion
        max_2opt_passes: Number of 2-opt passes

    Returns:
        1D LongTensor of permutation indices of shape (N,)
    """
    data = matrix.t().detach() if axis == 1 else matrix.detach()
    n = data.shape[0]

    if n <= 2:
        return torch.arange(n, dtype=torch.long, device=matrix.device)

    # Fast pairwise Euclidean distance matrix
    dist_t = torch.cdist(data.float(), data.float())
    dist = dist_t.cpu().numpy()

    # Run Ripple Insertion
    tour = _solve_ripple_tsp(
        dist,
        max_k_neighbors=max_k_neighbors,
        max_ripple_depth=max_ripple_depth,
        enable_2opt=enable_2opt,
        max_2opt_passes=max_2opt_passes,
    )

    return torch.tensor(tour, dtype=torch.long, device=matrix.device)


def _solve_ripple_tsp(
    dist: np.ndarray,
    max_k_neighbors: int = 25,
    max_ripple_depth: int = 50,
    enable_2opt: bool = True,
    max_2opt_passes: int = 5,
) -> np.ndarray:
    """
    Core Ripple Insertion solver using circular doubly-linked tour.
    """
    n = dist.shape[0]
    if n <= 3:
        return np.arange(n)

    # Precompute top nearest neighbors for each node for fast spatial candidate search
    k_cand = min(max_k_neighbors, n - 1)
    # Sort neighbors by distance ascending, excluding self
    sorted_neighbors = np.argsort(dist, axis=1)[:, 1 : k_cand + 1]

    # Doubly-linked tour arrays: next_node[i], prev_node[i]
    next_node = np.zeros(n, dtype=np.int64)
    prev_node = np.zeros(n, dtype=np.int64)
    in_tour = np.zeros(n, dtype=bool)

    # Initialize triangle with node 0 and its closest neighbors
    c0 = 0
    c1 = int(sorted_neighbors[c0][0])
    c2 = int(sorted_neighbors[c0][1])

    # Form initial circle: c0 -> c1 -> c2 -> c0
    next_node[c0] = c1
    prev_node[c1] = c0
    next_node[c1] = c2
    prev_node[c2] = c1
    next_node[c2] = c0
    prev_node[c0] = c2

    in_tour[c0] = True
    in_tour[c1] = True
    in_tour[c2] = True
    tour_size = 3

    # Remaining nodes to insert
    remaining = [i for i in range(n) if not in_tour[i]]

    # Sort remaining nodes: inserting outer/distant nodes first (heuristic onion order)
    center_dist = np.mean(dist, axis=1)
    remaining.sort(key=lambda idx: center_dist[idx], reverse=True)

    # Set up reusable wavefront queue for ripple
    for city in remaining:
        # Step 1: Find best insertion position for 'city'
        # Evaluate candidate edges adjacent to nearest neighbors already in the tour
        candidates = []
        for nb in sorted_neighbors[city]:
            if in_tour[nb]:
                candidates.append((prev_node[nb], nb))
                candidates.append((nb, next_node[nb]))

        if not candidates:
            # Fallback: check an arbitrary edge in tour
            curr = c0
            candidates = [(curr, next_node[curr])]

        best_cost = float("inf")
        best_u, best_v = candidates[0]

        for u, v in candidates:
            cost = dist[u, city] + dist[city, v] - dist[u, v]
            if cost < best_cost:
                best_cost = cost
                best_u, best_v = u, v

        # Insert city between best_u and best_v
        next_node[best_u] = city
        prev_node[city] = best_u
        next_node[city] = best_v
        prev_node[best_v] = city
        in_tour[city] = True
        tour_size += 1

        # Step 2: Ripple Effect (Wavefront Propagation)
        # Check if re-routing nearby nodes relieves geometric tension
        wavefront = [city, best_u, best_v]
        ripple_iters = 0

        while wavefront and ripple_iters < max_ripple_depth:
            ripple_iters += 1
            curr_node = wavefront.pop(0)
            if tour_size < 4:
                break

            p = prev_node[curr_node]
            q = next_node[curr_node]
            saving = dist[p, curr_node] + dist[curr_node, q] - dist[p, q]

            best_gain = 0.0
            best_target_u = None

            # Test inserting curr_node around its spatial neighbors in the tour
            for nb in sorted_neighbors[curr_node]:
                if not in_tour[nb] or nb == curr_node or nb == p or nb == q:
                    continue

                for u, v in [(prev_node[nb], nb), (nb, next_node[nb])]:
                    if u == curr_node or v == curr_node or u == p:
                        continue
                    cost_to_insert = dist[u, curr_node] + dist[curr_node, v] - dist[u, v]
                    gain = saving - cost_to_insert
                    if gain > best_gain:
                        best_gain = gain
                        best_target_u = u

            if best_gain > 1e-6 and best_target_u is not None:
                # Relocate curr_node to after best_target_u
                target_v = next_node[best_target_u]

                # Splice out curr_node
                next_node[p] = q
                prev_node[q] = p

                # Insert between target_u and target_v
                next_node[best_target_u] = curr_node
                prev_node[curr_node] = best_target_u
                next_node[curr_node] = target_v
                prev_node[target_v] = curr_node

                # Ripple propagates to neighbors
                wavefront.extend([p, q, best_target_u, target_v])

    # Convert linked list to tour array
    tour = np.empty(n, dtype=np.int64)
    curr = c0
    for idx in range(n):
        tour[idx] = curr
        curr = next_node[curr]

    # Step 3: Optional 2-opt post-processing
    if enable_2opt and n >= 4:
        tour = _apply_2opt(tour, dist, max_passes=max_2opt_passes)

    # Step 4: Cut the cyclic tour at its longest edge to produce the optimal open path
    # In matrix row/column ordination, we want an open 1D path that minimizes consecutive distance
    max_edge_dist = -1.0
    cut_idx = 0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        if dist[u, v] > max_edge_dist:
            max_edge_dist = dist[u, v]
            cut_idx = (i + 1) % n

    # Unroll path starting after the longest edge
    open_tour = np.roll(tour, -cut_idx)
    return open_tour


def _apply_2opt(tour: np.ndarray, dist: np.ndarray, max_passes: int = 5) -> np.ndarray:
    """Fast 2-opt local search to eliminate crossing edges."""
    n = len(tour)
    for _ in range(max_passes):
        improved = False
        for a in range(n - 1):
            u_a = tour[a]
            u_b = tour[a + 1]
            d_ab = dist[u_a, u_b]

            for b in range(a + 2, n if a > 0 else n - 1):
                u_c = tour[b]
                u_d = tour[(b + 1) % n]

                curr_len = d_ab + dist[u_c, u_d]
                new_len = dist[u_a, u_c] + dist[u_b, u_d]

                if new_len < curr_len - 1e-6:
                    # Reverse segment from a+1 to b
                    tour[a + 1 : b + 1] = tour[a + 1 : b + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return tour


def compute_joint_bipartite_ripple_permutation(
    weight: torch.Tensor,
    max_iters: int = 5,
    tol: float = 1e-3,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a joint 2D bipartite permutation by alternating Ripple Insertion TSP on rows and columns.
    Author: Mario Raúl Carbonell Martínez.
    """
    from spec_rama.permutation import compute_total_variation_2d

    out_dim, in_dim = weight.shape
    row_perm = torch.arange(out_dim, dtype=torch.long, device=weight.device)
    col_perm = torch.arange(in_dim, dtype=torch.long, device=weight.device)

    w_curr = weight.detach().clone()
    prev_tv = compute_total_variation_2d(w_curr)

    for iter_idx in range(max_iters):
        r_p = compute_ripple_tsp_1d(w_curr, axis=0, **kwargs)
        row_perm = row_perm[r_p]
        w_curr = w_curr[r_p, :]

        c_p = compute_ripple_tsp_1d(w_curr, axis=1, **kwargs)
        col_perm = col_perm[c_p]
        w_curr = w_curr[:, c_p]

        curr_tv = compute_total_variation_2d(w_curr)
        rel_improvement = (prev_tv - curr_tv) / (prev_tv + 1e-8)

        if rel_improvement < tol and iter_idx >= 1:
            break

        prev_tv = curr_tv

    return row_perm, col_perm
