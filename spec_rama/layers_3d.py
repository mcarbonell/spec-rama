import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .permutation import compute_3d_tensor_permutations
from .transforms import dct_3d_project, get_dct_matrix_1d


class SpecRAMA3DLinearGroup(nn.Module):
    """
    SpecRAMA3DLinearGroup: Adapts a group of L linear layers (e.g., across model depth or attention heads)
    as a single 3D Tensor using a 3D Spectral Core (k_z x k_y x k_x).

    Mathematical Formulation:
        W_eff^(l) = W_0^(l) * (1 + scaling_m * M^(l)) + scaling_a * A^(l)

    where M and A are 3D tensors (L, Y, X) synthesized from 3D Spectral Cores via 3D-DCT,
    scaled consistently with Parseval RMS energy normalization:
        scaling_m = alpha_m / (sqrt(k_z * k_y * k_x) * w_std)
        scaling_a = alpha_a / sqrt(k_z * k_y * k_x)
    """

    def __init__(
        self,
        base_layers: List[nn.Linear],
        core_size_3d: Tuple[int, int, int] = (4, 8, 8),
        alpha_m: float = 1.0,
        alpha_a: float = 8.0,
        permutation_method: str = "tsp",
    ):
        super().__init__()
        self.base_layers = nn.ModuleList(base_layers)
        self.num_layers = len(base_layers)
        self.out_features = base_layers[0].out_features
        self.in_features = base_layers[0].in_features

        self.core_size_3d = (
            min(core_size_3d[0], self.num_layers),
            min(core_size_3d[1], self.out_features),
            min(core_size_3d[2], self.in_features),
        )
        self.alpha_m = alpha_m
        self.alpha_a = alpha_a

        # Freeze base parameters
        for layer in self.base_layers:
            layer.weight.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False

        # Stack base weights into a 3D Tensor (L, out_features, in_features)
        with torch.no_grad():
            stacked_w = torch.stack([layer.weight.data for layer in self.base_layers], dim=0)
            self.w_std = max(stacked_w.std().item(), 1e-5)

        # 1. Compute 3D Permutations (P_Z, P_Y, P_X)
        perm_z, perm_y, perm_x = compute_3d_tensor_permutations(stacked_w, method=permutation_method)

        self.register_buffer("perm_z", perm_z)
        self.register_buffer("perm_y", perm_y)
        self.register_buffer("perm_x", perm_x)
        self.register_buffer("inv_perm_z", torch.argsort(perm_z))
        self.register_buffer("inv_perm_y", torch.argsort(perm_y))
        self.register_buffer("inv_perm_x", torch.argsort(perm_x))

        # 2. Register 1D DCT Bases for Z, Y, X
        self.register_buffer("D_z", get_dct_matrix_1d(self.num_layers))
        self.register_buffer("D_y", get_dct_matrix_1d(self.out_features))
        self.register_buffer("D_x", get_dct_matrix_1d(self.in_features))

        # 3. Trainable 3D Cores
        k_z, k_y, k_x = self.core_size_3d
        self.core_3d_m = nn.Parameter(torch.zeros(k_z, k_y, k_x))
        self.core_3d_a = nn.Parameter(torch.zeros(k_z, k_y, k_x))

        self.merged = False
        self.original_weights: Optional[List[torch.Tensor]] = None

    def _synthesize_3d_tensor_from_core(self, core_3d: torch.Tensor) -> torch.Tensor:
        """
        Synthesizes a full 3D update tensor (L, Y, X) from a small 3D core (k_z, k_y, k_x).
        """
        spatial_perm_3d = dct_3d_project(
            core_3d, self.D_z, self.D_y, self.D_x, target_shape=(self.num_layers, self.out_features, self.in_features)
        )

        # Restore 3D inverse permutations across all 3 axes
        spatial_3d = spatial_perm_3d[self.inv_perm_z, :, :][:, self.inv_perm_y, :][:, :, self.inv_perm_x]
        return spatial_3d

    def get_effective_weights(self) -> List[torch.Tensor]:
        """
        Returns list of effective weights W_eff for all L layers using Parseval RMS scaling.
        """
        k_z, k_y, k_x = self.core_size_3d
        core_dim_scale = math.sqrt(k_z * k_y * k_x)

        mod_tensor_3d = self._synthesize_3d_tensor_from_core(self.core_3d_m)
        add_tensor_3d = self._synthesize_3d_tensor_from_core(self.core_3d_a)

        scaling_m = self.alpha_m / (core_dim_scale * self.w_std)
        scaling_a = self.alpha_a / core_dim_scale

        effective_weights = []
        for l_idx, layer in enumerate(self.base_layers):
            w0 = layer.weight
            w_eff = w0 * (1.0 + scaling_m * mod_tensor_3d[l_idx]) + scaling_a * add_tensor_3d[l_idx]
            effective_weights.append(w_eff)

        return effective_weights

    def get_effective_weight(self, layer_idx: int) -> torch.Tensor:
        """Returns the effective weight for a specific layer index."""
        weights = self.get_effective_weights()
        return weights[layer_idx]

    def forward(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Forward pass for a specific layer in the 3D group."""
        if self.merged:
            return self.base_layers[layer_idx](x)
        w_eff = self.get_effective_weight(layer_idx)
        return F.linear(x, w_eff, self.base_layers[layer_idx].bias)

    def merge(self):
        """Merges 3D spectral updates into base layer weights."""
        if self.merged:
            return
        self.original_weights = [layer.weight.data.clone() for layer in self.base_layers]
        with torch.no_grad():
            w_effs = self.get_effective_weights()
            for l_idx, layer in enumerate(self.base_layers):
                layer.weight.copy_(w_effs[l_idx])
        self.merged = True

    def unmerge(self):
        """Reverts merged weights to original base weights."""
        if not self.merged:
            return
        if self.original_weights is not None:
            for l_idx, layer in enumerate(self.base_layers):
                layer.weight.copy_(self.original_weights[l_idx])
            self.original_weights = None
        self.merged = False
