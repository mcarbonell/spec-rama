import math
from typing import Any, Dict

import torch

from .permutation import compute_2d_permutations
from .transforms import get_dct_matrix_1d, get_walsh_matrix_1d, haar_dwt_2d, haar_idwt_2d


class HierarchicalSpectralQuantizer:
    """
    Hierarchical Spectral Quantizer:

    1. Permutes linear weight matrix W_0 (out_dim, in_dim) via TSP/PCA to maximize spatial smoothness.
    2. Transforms permuted weight matrix into DCT, FWHT, or 2D Haar Wavelet spectral space.
    3. Hierarchical Bit Allocation:
       - 8-bit Quantization for Low-Frequency Core (top-left k_out x k_in quadrant, ~6.25% of coefficients).
       - 4-bit (or 2-bit / 0-bit cutoff) Quantization for High-Frequency Rest.
    """

    def __init__(
        self,
        transform_type: str = "dct",
        core_ratio: float = 0.0625,  # 6.25% low-frequency core
        core_bits: int = 8,
        rest_bits: int = 4,
        permutation_method: str = "bipartite_tsp",
    ):
        self.transform_type = transform_type.lower()
        self.core_ratio = core_ratio
        self.core_bits = core_bits
        self.rest_bits = rest_bits
        self.permutation_method = permutation_method

    def quantize_matrix(self, weight: torch.Tensor) -> Dict[str, Any]:
        out_dim, in_dim = weight.shape

        # 1. Compute 2D Permutations
        row_perm, col_perm = compute_2d_permutations(weight, method=self.permutation_method)
        w_perm = weight[row_perm, :][:, col_perm]

        # 2. Compute Spectral Representation
        if self.transform_type == "dct":
            B_out = get_dct_matrix_1d(out_dim)
            B_in = get_dct_matrix_1d(in_dim)
            coeff = torch.matmul(B_out, torch.matmul(w_perm, B_in.t()))
            coeff_shape = (out_dim, in_dim)
        elif self.transform_type == "walsh":
            B_out = get_walsh_matrix_1d(out_dim)
            B_in = get_walsh_matrix_1d(in_dim)
            coeff = torch.matmul(B_out, torch.matmul(w_perm, B_in.t()))
            coeff_shape = (out_dim, in_dim)
        elif self.transform_type == "wavelet":
            LL, LH, HL, HH = haar_dwt_2d(w_perm)
            top = torch.cat([LL, LH], dim=1)
            bottom = torch.cat([HL, HH], dim=1)
            coeff = torch.cat([top, bottom], dim=0)
            coeff_shape = coeff.shape
        else:
            raise ValueError(f"Unsupported transform: {self.transform_type}")

        # 3. Hierarchical Partitioning
        c_h, c_w = coeff_shape
        k_out = max(1, int(math.ceil(c_h * math.sqrt(self.core_ratio))))
        k_in = max(1, int(math.ceil(c_w * math.sqrt(self.core_ratio))))

        core = coeff[:k_out, :k_in]

        # Quantize Core to core_bits
        core_min, core_max = core.min(), core.max()
        core_scale = (core_max - core_min) / (2**self.core_bits - 1) + 1e-8
        core_q = torch.round((core - core_min) / core_scale).to(torch.uint8)

        # Quantize Rest to rest_bits
        rest_mask = torch.ones_like(coeff, dtype=torch.bool)
        rest_mask[:k_out, :k_in] = False

        rest_vals = coeff[rest_mask]
        rest_min, rest_max = rest_vals.min(), rest_vals.max()
        rest_scale = (rest_max - rest_min) / (2**self.rest_bits - 1) + 1e-8
        rest_q = torch.round((rest_vals - rest_min) / rest_scale).to(torch.uint8)

        # Calculate average bits per weight
        total_weights = out_dim * in_dim
        core_weights = k_out * k_in
        rest_weights = coeff.numel() - core_weights
        avg_bits = (core_weights * self.core_bits + rest_weights * self.rest_bits) / total_weights

        return {
            "shape": (out_dim, in_dim),
            "coeff_shape": coeff_shape,
            "core_shape": (k_out, k_in),
            "core_q": core_q,
            "core_min": core_min.item(),
            "core_scale": core_scale.item(),
            "rest_q": rest_q,
            "rest_min": rest_min.item(),
            "rest_scale": rest_scale.item(),
            "row_perm": row_perm,
            "col_perm": col_perm,
            "transform_type": self.transform_type,
            "avg_bits": avg_bits,
        }

    def dequantize_matrix(self, payload: Dict[str, Any]) -> torch.Tensor:
        out_dim, in_dim = payload["shape"]
        coeff_shape = payload.get("coeff_shape", (out_dim, in_dim))
        k_out, k_in = payload["core_shape"]

        # Dequantize Core
        core = payload["core_q"].float() * payload["core_scale"] + payload["core_min"]

        # Dequantize Rest
        rest = payload["rest_q"].float() * payload["rest_scale"] + payload["rest_min"]

        # Reconstruct Spectral Matrix C
        coeff = torch.zeros(coeff_shape, dtype=torch.float32)
        coeff[:k_out, :k_in] = core

        rest_mask = torch.ones(coeff_shape, dtype=torch.bool)
        rest_mask[:k_out, :k_in] = False
        coeff[rest_mask] = rest

        # Compute Inverse Spectral Transform
        if payload["transform_type"] == "dct":
            B_out = get_dct_matrix_1d(out_dim)
            B_in = get_dct_matrix_1d(in_dim)
            w_perm = torch.matmul(B_out.t(), torch.matmul(coeff, B_in))
        elif payload["transform_type"] == "walsh":
            B_out = get_walsh_matrix_1d(out_dim)
            B_in = get_walsh_matrix_1d(in_dim)
            w_perm = torch.matmul(B_out.t(), torch.matmul(coeff, B_in))
        elif payload["transform_type"] == "wavelet":
            h_sub = (out_dim + 1) // 2
            w_sub = (in_dim + 1) // 2
            LL = coeff[:h_sub, :w_sub]
            LH = coeff[:h_sub, w_sub : 2 * w_sub]
            HL = coeff[h_sub : 2 * h_sub, :w_sub]
            HH = coeff[h_sub : 2 * h_sub, w_sub : 2 * w_sub]
            w_perm = haar_idwt_2d(LL, LH, HL, HH, target_shape=(out_dim, in_dim))
        else:
            raise ValueError(f"Unknown transform: {payload['transform_type']}")

        # Un-permutating rows and columns
        row_inv_perm = torch.argsort(payload["row_perm"])
        col_inv_perm = torch.argsort(payload["col_perm"])

        weight_reconstructed = w_perm[row_inv_perm, :][:, col_inv_perm]
        return weight_reconstructed
