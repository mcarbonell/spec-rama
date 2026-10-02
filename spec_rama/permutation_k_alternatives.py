"""
k-Alternatives Meta-Heuristic TSP Solver for Spec-RAMA Matrix Permutation.
Original algorithm and concept by Mario Raúl Carbonell Martínez.

This module implements Limited Discrepancy Search (LDS) with k-deviations,
multi-start exploration, and reinforcement-learning-inspired adaptive heuristic
policy updates without neural networks.
"""

from typing import Optional, Tuple

import numpy as np
import torch


def compute_k_alternatives_tsp_1d(
    matrix: torch.Tensor,
    axis: int = 0,
    max_k: int = 2,
    num_starts: int = 8,
    candidate_size: int = 20,
    seed: int = 42,
) -> torch.Tensor:
    """
    Computes a 1D k-Alternatives TSP permutation along rows (axis=0) or columns (axis=1).
    Minimizes Euclidean path distance between consecutive vectors.

    Args:
        matrix: 2D Tensor (d_out, d_in)
        axis: 0 for rows, 1 for columns
        max_k: Maximum discrepancy budget (number of alternative heuristic choices)
        num_starts: Number of different starting points for multi-start exploration
        candidate_size: Number of nearest neighbors tracked in local heuristic lists
        seed: Random seed for starting node selection

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

    tour = _solve_k_alternatives_tsp(
        dist,
        max_k=max_k,
        num_starts=num_starts,
        candidate_size=candidate_size,
        seed=seed,
    )

    return torch.tensor(tour, dtype=torch.long, device=matrix.device)


def _solve_k_alternatives_tsp(
    dist: np.ndarray,
    max_k: int = 2,
    num_starts: int = 8,
    candidate_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Core k-Alternatives solver with adaptive policy learning.
    """
    n = dist.shape[0]
    if n <= 3:
        return np.arange(n)

    rng = np.random.RandomState(seed)

    # 1. Initialize candidate lists: local_heuristics[i] = top sorted neighbors
    cand_k = min(candidate_size, n - 1)
    # Exclude self (column 0 in argsort is self)
    local_heuristics = [
        list(np.argsort(dist[i])[1 : cand_k + 1]) for i in range(n)
    ]

    best_tour: Optional[np.ndarray] = None
    best_cost = float("inf")

    # Helper: calculate 1D open path distance
    def eval_path_cost(path: np.ndarray) -> float:
        return float(np.sum(dist[path[:-1], path[1:]]))

    # Helper: Adaptive policy update (promote successful edges to front of candidate lists)
    def update_heuristics(improved_tour: np.ndarray):
        for i in range(len(improved_tour) - 1):
            u = improved_tour[i]
            v = improved_tour[i + 1]

            # Promote v in u's candidate list
            h_u = local_heuristics[u]
            if v in h_u:
                h_u.remove(v)
                h_u.insert(0, v)
            else:
                h_u.insert(0, v)
                if len(h_u) > cand_k:
                    h_u.pop()

            # Promote u in v's candidate list (symmetric edge reinforcement)
            h_v = local_heuristics[v]
            if u in h_v:
                h_v.remove(u)
                h_v.insert(0, u)
            else:
                h_v.insert(0, u)
                if len(h_v) > cand_k:
                    h_v.pop()

    # Determine starting cities for multi-start
    # Always include 0, most central, most eccentric, plus random samples
    row_sums = np.sum(dist, axis=1)
    starts = [0, int(np.argmin(row_sums)), int(np.argmax(row_sums))]
    if num_starts > len(starts):
        extra_starts = list(rng.choice(n, size=min(n, num_starts * 2), replace=False))
        for s in extra_starts:
            if s not in starts:
                starts.append(int(s))
            if len(starts) >= num_starts:
                break

    # Search loop progressing across discrepancy levels k = 0, 1, ..., max_k
    for k_budget in range(max_k + 1):
        for start_node in starts:
            # Construct solution with discrepancy budget k_budget
            tour = np.empty(n, dtype=np.int64)
            visited = np.zeros(n, dtype=bool)

            tour[0] = start_node
            visited[start_node] = True
            curr = start_node
            k_left = k_budget
            curr_cost = 0.0
            pruned = False

            for step in range(1, n):
                # Fetch candidate choices from current learned policy
                cands = local_heuristics[curr]
                valid_choices = [c for c in cands if not visited[c]]

                if not valid_choices:
                    # Fallback to nearest unvisited among all remaining nodes
                    unvisited_indices = np.where(~visited)[0]
                    dists_to_unvisited = dist[curr, unvisited_indices]
                    next_node = unvisited_indices[np.argmin(dists_to_unvisited)]
                elif k_left == 0 or len(valid_choices) == 1:
                    # Pure exploitation: take top learned choice (rank 0)
                    next_node = valid_choices[0]
                else:
                    # Exploration: allow deviation if within budget k
                    # Pick between best (rank 0) and alternatives (rank 1..k)
                    max_alt = min(len(valid_choices) - 1, k_left)
                    # Stochastic or deterministic discrepancy choice
                    alt_rank = rng.randint(0, max_alt + 1)
                    next_node = valid_choices[alt_rank]
                    if alt_rank > 0:
                        k_left -= alt_rank

                inc_cost = dist[curr, next_node]
                curr_cost += inc_cost

                # Branch & bound pruning
                if curr_cost >= best_cost:
                    pruned = True
                    break

                tour[step] = next_node
                visited[next_node] = True
                curr = next_node

            if not pruned:
                final_cost = curr_cost
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_tour = tour.copy()
                    # Reinforce the policy with the winning edges!
                    update_heuristics(best_tour)

    return best_tour if best_tour is not None else np.arange(n)


def compute_joint_bipartite_k_alternatives_permutation(
    weight: torch.Tensor,
    max_iters: int = 5,
    tol: float = 1e-3,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a joint 2D bipartite permutation by alternating k-Alternatives TSP on rows and columns.
    Author: Mario Raúl Carbonell Martínez.
    """
    from spec_rama.permutation import compute_total_variation_2d

    out_dim, in_dim = weight.shape
    row_perm = torch.arange(out_dim, dtype=torch.long, device=weight.device)
    col_perm = torch.arange(in_dim, dtype=torch.long, device=weight.device)

    w_curr = weight.detach().clone()
    prev_tv = compute_total_variation_2d(w_curr)

    for iter_idx in range(max_iters):
        r_p = compute_k_alternatives_tsp_1d(w_curr, axis=0, **kwargs)
        row_perm = row_perm[r_p]
        w_curr = w_curr[r_p, :]

        c_p = compute_k_alternatives_tsp_1d(w_curr, axis=1, **kwargs)
        col_perm = col_perm[c_p]
        w_curr = w_curr[:, c_p]

        curr_tv = compute_total_variation_2d(w_curr)
        rel_improvement = (prev_tv - curr_tv) / (prev_tv + 1e-8)

        if rel_improvement < tol and iter_idx >= 1:
            break

        prev_tv = curr_tv

    return row_perm, col_perm
