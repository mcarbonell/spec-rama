import torch

from spec_rama.permutation import (
    compute_3d_tensor_permutations,
    compute_greedy_tsp_1d,
    compute_joint_bipartite_2d_permutation,
    compute_pca_permutation,
    compute_total_variation_2d,
)


def test_greedy_tsp_valid_permutation():
    """Verify that greedy TSP returns a valid permutation of indices."""
    torch.manual_seed(42)
    matrix = torch.randn(64, 32)

    row_perm = compute_greedy_tsp_1d(matrix, axis=0)
    assert len(row_perm) == 64
    assert set(row_perm.tolist()) == set(range(64))

    col_perm = compute_greedy_tsp_1d(matrix, axis=1)
    assert len(col_perm) == 32
    assert set(col_perm.tolist()) == set(range(32))


def test_greedy_tsp_reduces_total_variation():
    """Verify that TSP ordering reduces 2D Total Variation compared to random ordering."""
    torch.manual_seed(42)
    # Create smooth 1D signals randomly shuffled
    t = torch.linspace(0, 4 * 3.14159, 100)
    smooth_matrix = torch.stack([torch.sin(t + i * 0.1) for i in range(50)])

    # Randomly shuffle rows
    rand_perm = torch.randperm(50)
    shuffled = smooth_matrix[rand_perm, :]

    tv_shuffled = compute_total_variation_2d(shuffled)

    # Sort via TSP
    tsp_perm = compute_greedy_tsp_1d(shuffled, axis=0)
    sorted_matrix = shuffled[tsp_perm, :]
    tv_sorted = compute_total_variation_2d(sorted_matrix)

    assert tv_sorted < tv_shuffled, f"TSP failed to reduce TV: {tv_sorted} >= {tv_shuffled}"


def test_bipartite_tsp_convergence():
    """Verify bipartite TSP converges and outputs valid permutations."""
    torch.manual_seed(42)
    weight = torch.randn(32, 24)

    row_perm, col_perm = compute_joint_bipartite_2d_permutation(weight, max_iters=5, tol=1e-3)

    assert len(row_perm) == 32
    assert set(row_perm.tolist()) == set(range(32))
    assert len(col_perm) == 24
    assert set(col_perm.tolist()) == set(range(24))


def test_pca_permutation():
    """Verify PCA permutation produces valid indices."""
    torch.manual_seed(42)
    weight = torch.randn(40, 20)

    row_perm = compute_pca_permutation(weight, axis=0)
    col_perm = compute_pca_permutation(weight, axis=1)

    assert len(row_perm) == 40
    assert set(row_perm.tolist()) == set(range(40))
    assert len(col_perm) == 20
    assert set(col_perm.tolist()) == set(range(20))


def test_3d_tensor_permutations():
    """Verify 3D tensor permutations across (L, Y, X) axes."""
    torch.manual_seed(42)
    tensor_3d = torch.randn(6, 16, 12)

    pz, py, px = compute_3d_tensor_permutations(tensor_3d, method="tsp")
    assert len(pz) == 6
    assert len(py) == 16
    assert len(px) == 12
    assert set(pz.tolist()) == set(range(6))
    assert set(py.tolist()) == set(range(16))
    assert set(px.tolist()) == set(range(12))


def test_custom_permutation_registry():
    """Verify custom permutation functions can be registered and dispatched."""
    from spec_rama.permutation import (
        compute_2d_permutations,
        list_permutation_methods,
        register_permutation_method,
    )

    def custom_reverse_permutation(matrix: torch.Tensor, **kwargs):
        r = torch.arange(matrix.shape[0] - 1, -1, -1, dtype=torch.long)
        c = torch.arange(matrix.shape[1] - 1, -1, -1, dtype=torch.long)
        return r, c

    register_permutation_method("custom_reverse", custom_reverse_permutation)
    methods = list_permutation_methods()
    assert "custom_reverse" in methods

    w = torch.randn(10, 8)
    r_p, c_p = compute_2d_permutations(w, method="custom_reverse")
    assert r_p.tolist() == list(range(9, -1, -1))
    assert c_p.tolist() == list(range(7, -1, -1))

