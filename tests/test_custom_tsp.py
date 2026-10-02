"""
Unit tests for Ripple Insertion and k-Alternatives TSP permutation solvers.
"""

import torch

from spec_rama.permutation import (
    compute_2d_permutations,
    compute_total_variation_2d,
    list_permutation_methods,
)
from spec_rama.permutation_k_alternatives import compute_k_alternatives_tsp_1d
from spec_rama.permutation_ripple import compute_ripple_tsp_1d


def test_registry_contains_custom_solvers():
    methods = list_permutation_methods()
    assert "ripple_tsp" in methods
    assert "k_alternatives_tsp" in methods


def test_ripple_tsp_1d_validity_and_smoothness():
    torch.manual_seed(42)
    # Generate 1D sequence with spatial structure + noise
    t = torch.linspace(0, 4 * 3.14159, 64)
    smooth_features = torch.stack([torch.sin(t), torch.cos(t), t], dim=1)

    # Shuffle rows randomly
    perm = torch.randperm(64)
    shuffled = smooth_features[perm]

    # Run Ripple TSP
    solved_perm = compute_ripple_tsp_1d(shuffled, axis=0, enable_2opt=True)

    # Check validity
    assert len(solved_perm) == 64
    assert set(solved_perm.tolist()) == set(range(64))

    # Check that solved tour reduces path distance compared to shuffled
    ordered = shuffled[solved_perm]
    shuffled_dist = torch.norm(shuffled[1:] - shuffled[:-1], dim=1).sum().item()
    ordered_dist = torch.norm(ordered[1:] - ordered[:-1], dim=1).sum().item()

    assert ordered_dist < shuffled_dist


def test_k_alternatives_tsp_1d_validity_and_smoothness():
    torch.manual_seed(42)
    t = torch.linspace(0, 4 * 3.14159, 64)
    smooth_features = torch.stack([torch.sin(t), torch.cos(t), t], dim=1)

    perm = torch.randperm(64)
    shuffled = smooth_features[perm]

    # Run k-Alternatives TSP
    solved_perm = compute_k_alternatives_tsp_1d(
        shuffled, axis=0, max_k=2, num_starts=4
    )

    # Check validity
    assert len(solved_perm) == 64
    assert set(solved_perm.tolist()) == set(range(64))

    # Check that solved tour reduces path distance compared to shuffled
    ordered = shuffled[solved_perm]
    shuffled_dist = torch.norm(shuffled[1:] - shuffled[:-1], dim=1).sum().item()
    ordered_dist = torch.norm(ordered[1:] - ordered[:-1], dim=1).sum().item()

    assert ordered_dist < shuffled_dist


def test_bipartite_ripple_permutation_2d():
    torch.manual_seed(42)
    # 2D smooth matrix with random shuffle
    x = torch.linspace(-2, 2, 48)
    y = torch.linspace(-2, 2, 48)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    smooth_grid = torch.sin(xx) * torch.cos(yy)

    # Randomly shuffle rows and columns
    r_perm = torch.randperm(48)
    c_perm = torch.randperm(48)
    shuffled = smooth_grid[r_perm, :][:, c_perm]

    tv_shuffled = compute_total_variation_2d(shuffled)

    row_p, col_p = compute_2d_permutations(shuffled, method="ripple_tsp", max_iters=3)

    assert len(row_p) == 48 and set(row_p.tolist()) == set(range(48))
    assert len(col_p) == 48 and set(col_p.tolist()) == set(range(48))

    reconstructed = shuffled[row_p, :][:, col_p]
    tv_reconstructed = compute_total_variation_2d(reconstructed)

    assert tv_reconstructed < tv_shuffled


def test_bipartite_k_alternatives_permutation_2d():
    torch.manual_seed(42)
    x = torch.linspace(-2, 2, 48)
    y = torch.linspace(-2, 2, 48)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    smooth_grid = torch.sin(xx) * torch.cos(yy)

    r_perm = torch.randperm(48)
    c_perm = torch.randperm(48)
    shuffled = smooth_grid[r_perm, :][:, c_perm]

    tv_shuffled = compute_total_variation_2d(shuffled)

    row_p, col_p = compute_2d_permutations(
        shuffled, method="k_alternatives_tsp", max_iters=2, max_k=1, num_starts=3
    )

    assert len(row_p) == 48 and set(row_p.tolist()) == set(range(48))
    assert len(col_p) == 48 and set(col_p.tolist()) == set(range(48))

    reconstructed = shuffled[row_p, :][:, col_p]
    tv_reconstructed = compute_total_variation_2d(reconstructed)

    assert tv_reconstructed < tv_shuffled
