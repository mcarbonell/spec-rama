import torch

from spec_rama.transforms import (
    dct_3d_project,
    get_dct_matrix_1d,
    get_walsh_matrix_1d,
    haar_dwt_2d,
    haar_idwt_2d,
)


def test_dct_orthogonality():
    """Verify that DCT-II matrix is strictly orthogonal: D @ D.T = I."""
    for N in [8, 16, 32, 64, 128]:
        D = get_dct_matrix_1d(N)
        eye_mat = torch.eye(N, dtype=torch.float32)
        diff = torch.matmul(D, D.t()) - eye_mat
        assert torch.max(torch.abs(diff)).item() < 1e-5, f"DCT matrix of size {N} is not orthogonal"


def test_dct_roundtrip():
    """Verify 1D DCT roundtrip reconstruction."""
    torch.manual_seed(42)
    N = 32
    x = torch.randn(N, dtype=torch.float32)
    D = get_dct_matrix_1d(N)

    # Analysis and synthesis
    X = torch.matmul(D, x)
    x_rec = torch.matmul(D.t(), X)

    assert torch.max(torch.abs(x - x_rec)).item() < 1e-5


def test_walsh_orthogonality_powers_of_two():
    """Verify Walsh-Hadamard matrix orthogonality for powers of 2."""
    for N in [2, 4, 8, 16, 64]:
        H = get_walsh_matrix_1d(N)
        eye_mat = torch.eye(N, dtype=torch.float32)
        diff = torch.matmul(H, H.t()) - eye_mat
        assert torch.max(torch.abs(diff)).item() < 1e-5, f"Walsh matrix of size {N} is not orthogonal"


def test_walsh_orthogonality_non_power_of_two():
    """Verify QR fallback guarantees orthogonality for arbitrary dimensions."""
    for N in [3, 7, 10, 23, 50]:
        H = get_walsh_matrix_1d(N)
        eye_mat = torch.eye(N, dtype=torch.float32)
        diff = torch.matmul(H, H.t()) - eye_mat
        assert torch.max(torch.abs(diff)).item() < 1e-5, f"Non-power-of-2 Walsh matrix of size {N} is not orthogonal"


def test_haar_dwt_idwt_roundtrip_even():
    """Verify exact lossless invertibility of 2D Haar Wavelet on even dimensions."""
    torch.manual_seed(42)
    x = torch.randn(32, 48, dtype=torch.float32)
    LL, LH, HL, HH = haar_dwt_2d(x)
    x_rec = haar_idwt_2d(LL, LH, HL, HH, target_shape=x.shape)

    assert torch.max(torch.abs(x - x_rec)).item() < 1e-5


def test_haar_dwt_idwt_roundtrip_odd():
    """Verify invertibility of 2D Haar Wavelet on odd dimensions (boundary padding)."""
    torch.manual_seed(42)
    x = torch.randn(33, 47, dtype=torch.float32)
    LL, LH, HL, HH = haar_dwt_2d(x)
    x_rec = haar_idwt_2d(LL, LH, HL, HH, target_shape=x.shape)

    assert torch.max(torch.abs(x - x_rec)).item() < 1e-5


def test_dct_3d_project():
    """Verify 3D DCT projection shape and forward mapping."""
    L, Y, X = 4, 16, 16
    kz, ky, kx = 2, 4, 4
    core_3d = torch.randn(kz, ky, kx)

    Dz = get_dct_matrix_1d(L)
    Dy = get_dct_matrix_1d(Y)
    Dx = get_dct_matrix_1d(X)

    spatial_3d = dct_3d_project(core_3d, Dz, Dy, Dx, target_shape=(L, Y, X))
    assert spatial_3d.shape == (L, Y, X)
