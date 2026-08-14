import torch
import math
from typing import Tuple

def get_dct_matrix_1d(N: int) -> torch.Tensor:
    """
    Computes the 1D Orthogonal DCT-II matrix of dimension (N, N).
    D @ D.T = I
    """
    n = torch.arange(N, dtype=torch.float32)
    k = torch.arange(N, dtype=torch.float32).unsqueeze(1)
    
    # DCT-II formula
    D = torch.cos((math.pi / N) * (n + 0.5) * k)
    D[0, :] *= math.sqrt(1.0 / N)
    D[1:, :] *= math.sqrt(2.0 / N)
    return D


def get_walsh_matrix_1d(N: int) -> torch.Tensor:
    """
    Computes the 1D Normalized Walsh-Hadamard Matrix of size (N, N).
    Guarantees strict orthogonality (H @ H.T = I) for any dimension N.
    """
    if (N & (N - 1)) != 0 or N <= 0:
        p = 2 ** math.ceil(math.log2(N))
        H_full = get_walsh_matrix_1d(p)
        # Re-orthogonalize truncated Hadamard submatrix via QR decomposition
        Q, _ = torch.linalg.qr(H_full[:N, :N])
        return Q

    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < N:
        top = torch.cat([H, H], dim=1)
        bottom = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bottom], dim=0)
        
    return H / math.sqrt(N)


def haar_dwt_2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes 1-level 2D Haar Discrete Wavelet Transform (DWT) for a 2D tensor x (H, W).
    """
    H, W = x.shape
    pad_h = 1 if H % 2 != 0 else 0
    pad_w = 1 if W % 2 != 0 else 0
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(
            x.unsqueeze(0).unsqueeze(0),
            (0, pad_w, 0, pad_h),
            mode='replicate'
        ).squeeze(0).squeeze(0)
        
    x00 = x[0::2, 0::2]
    x01 = x[0::2, 1::2]
    x10 = x[1::2, 0::2]
    x11 = x[1::2, 1::2]
    
    LL = (x00 + x01 + x10 + x11) / 2.0
    LH = (x00 - x01 + x10 - x11) / 2.0
    HL = (x00 + x01 - x10 - x11) / 2.0
    HH = (x00 - x01 - x10 + x11) / 2.0
    
    return LL, LH, HL, HH


def haar_idwt_2d(
    LL: torch.Tensor,
    LH: torch.Tensor,
    HL: torch.Tensor,
    HH: torch.Tensor,
    target_shape: Tuple[int, int] = None
) -> torch.Tensor:
    """
    Inverse 2D Haar Wavelet Transform. Reconstructs original 2D tensor from (LL, LH, HL, HH).
    """
    H_sub, W_sub = LL.shape
    H_out = H_sub * 2
    W_out = W_sub * 2
    
    x00 = (LL + LH + HL + HH) / 2.0
    x01 = (LL - LH + HL - HH) / 2.0
    x10 = (LL + LH - HL - HH) / 2.0
    x11 = (LL - LH - HL + HH) / 2.0
    
    out = torch.zeros((H_out, W_out), dtype=LL.dtype, device=LL.device)
    out[0::2, 0::2] = x00
    out[0::2, 1::2] = x01
    out[1::2, 0::2] = x10
    out[1::2, 1::2] = x11
    
    if target_shape is not None:
        out = out[:target_shape[0], :target_shape[1]]
        
    return out


def dct_3d_project(
    core_3d: torch.Tensor,
    D_z: torch.Tensor,
    D_y: torch.Tensor,
    D_x: torch.Tensor,
    target_shape: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Synthesizes a 3D spatial tensor (L, Y, X) from a small 3D spectral core (k_z, k_y, k_x)
    using 3D DCT projections.
    
    Args:
        core_3d: PyTorch Tensor of shape (k_z, k_y, k_x)
        D_z: DCT matrix for Z axis (L, L)
        D_y: DCT matrix for Y axis (Y, Y)
        D_x: DCT matrix for X axis (X, X)
        target_shape: (L, Y, X)
    Returns:
        spatial_tensor_3d: PyTorch Tensor of shape (L, Y, X)
    """
    L, Y, X = target_shape
    k_z, k_y, k_x = core_3d.shape
    
    # 1. Expand X and Y dimensions: (k_z, Y, X)
    Dy_sub = D_y[:k_y, :] # (k_y, Y)
    Dx_sub = D_x[:k_x, :] # (k_x, X)
    Dz_sub = D_z[:k_z, :] # (k_z, L)
    
    # temp_2d shape: (k_z, Y, X)
    temp_2d = torch.einsum('ym, zyx, xn -> zmn', Dy_sub, core_3d, Dx_sub)
    
    # 2. Projection along Z depth axis: (L, Y, X)
    spatial_3d = torch.einsum('zl, zmn -> lmn', Dz_sub, temp_2d)
    
    return spatial_3d
