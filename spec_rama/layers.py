import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from .permutation import compute_2d_permutations
from .transforms import get_dct_matrix_1d, get_walsh_matrix_1d, haar_idwt_2d

class SpecRAMALinear(nn.Module):
    """
    SpecRAMALinear: Permutated Spectral Core Adaptation (RAMA-DCT / RAMA-FWHT / RAMA-Wavelet)
    Supports both standard PyTorch nn.Linear and HuggingFace Conv1D (used in GPT-2).
    """
    def __init__(
        self,
        base_layer: nn.Module,
        transform_type: str = "dct",
        core_size: Tuple[int, int] = (8, 8),
        alpha_m: float = 1.0,
        alpha_a: float = 8.0,
        permutation_method: str = "tsp",
        use_multiplicative: bool = True,
        use_additive: bool = True,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.is_conv1d = (base_layer.__class__.__name__ == "Conv1D")
        
        if self.is_conv1d:
            # Conv1D weight shape: (in_features, out_features)
            self.in_features = base_layer.weight.shape[0]
            self.out_features = base_layer.weight.shape[1]
            base_weight_2d = base_layer.weight.data.t()
        else:
            self.in_features = getattr(base_layer, 'in_features', base_layer.weight.shape[1])
            self.out_features = getattr(base_layer, 'out_features', base_layer.weight.shape[0])
            base_weight_2d = base_layer.weight.data

        self.transform_type = transform_type.lower()
        self.core_size = (
            min(core_size[0], self.out_features),
            min(core_size[1], self.in_features),
        )
        self.alpha_m = alpha_m
        self.alpha_a = alpha_a
        self.use_multiplicative = use_multiplicative
        self.use_additive = use_additive

        # Freeze base parameters
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # 1. Compute Permutations (P_L, P_R) on 2D weight (out_features, in_features)
        row_perm, col_perm = compute_2d_permutations(base_weight_2d, method=permutation_method)
        
        self.register_buffer("row_perm", row_perm)
        self.register_buffer("col_perm", col_perm)
        self.register_buffer("row_inv_perm", torch.argsort(row_perm))
        self.register_buffer("col_inv_perm", torch.argsort(col_perm))

        # 2. Register Spectral Transform Bases
        if self.transform_type == "dct":
            self.register_buffer("B_out", get_dct_matrix_1d(self.out_features))
            self.register_buffer("B_in", get_dct_matrix_1d(self.in_features))
        elif self.transform_type == "walsh":
            self.register_buffer("B_out", get_walsh_matrix_1d(self.out_features))
            self.register_buffer("B_in", get_walsh_matrix_1d(self.in_features))
        elif self.transform_type == "wavelet":
            self.ll_h = (self.out_features + 1) // 2
            self.ll_w = (self.in_features + 1) // 2
            self.core_size = (
                min(self.core_size[0], self.ll_h),
                min(self.core_size[1], self.ll_w),
            )

        # 3. Trainable Spectral Cores
        k_out, k_in = self.core_size
        
        if self.use_multiplicative:
            self.core_m = nn.Parameter(torch.zeros(k_out, k_in))
        else:
            self.register_parameter("core_m", None)

        if self.use_additive:
            self.core_a = nn.Parameter(torch.zeros(k_out, k_in))
        else:
            self.register_parameter("core_a", None)

        # Scale factor & Base weight std
        with torch.no_grad():
            w_std = base_weight_2d.std().item()
            self.w_std = max(w_std, 1e-5)

        self.merged = False
        self.original_weight: Optional[torch.Tensor] = None

    def _synthesize_matrix_from_core(self, core: torch.Tensor) -> torch.Tensor:
        k_out, k_in = core.shape
        dev = core.device

        if self.transform_type in ["dct", "walsh"]:
            B_out_sub = self.B_out[:k_out, :].to(dev)
            B_in_sub = self.B_in[:k_in, :].to(dev)
            M_perm = torch.matmul(
                B_out_sub.t(),
                torch.matmul(core, B_in_sub)
            )
        elif self.transform_type == "wavelet":
            LL = torch.zeros((self.ll_h, self.ll_w), dtype=core.dtype, device=dev)
            LL[:k_out, :k_in] = core
            
            LH = torch.zeros_like(LL)
            HL = torch.zeros_like(LL)
            HH = torch.zeros_like(LL)
            
            M_perm = haar_idwt_2d(LL, LH, HL, HH, target_shape=(self.out_features, self.in_features))
        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")

        row_inv = self.row_inv_perm.to(dev)
        col_inv = self.col_inv_perm.to(dev)
        M = M_perm[row_inv, :][:, col_inv]
        return M

    def get_effective_weight(
        self,
        custom_core_m: Optional[torch.Tensor] = None,
        custom_core_a: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        w_base_2d = self.base_layer.weight.data.t().clone() if self.is_conv1d else self.base_layer.weight.clone()
        w_eff = w_base_2d
        
        core_m_val = custom_core_m if custom_core_m is not None else self.core_m
        core_a_val = custom_core_a if custom_core_a is not None else self.core_a

        core_dim_scale = math.sqrt(self.core_size[0] * self.core_size[1])
        if self.use_multiplicative and core_m_val is not None:
            mod_matrix = self._synthesize_matrix_from_core(core_m_val)
            scaling = self.alpha_m / (core_dim_scale * self.w_std)
            w_eff = w_eff * (1.0 + scaling * mod_matrix)
            
        if self.use_additive and core_a_val is not None:
            add_matrix = self._synthesize_matrix_from_core(core_a_val)
            scaling = self.alpha_a / core_dim_scale
            w_eff = w_eff + scaling * add_matrix
            
        return w_eff.t() if self.is_conv1d else w_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)
            
        w_eff = self.get_effective_weight()
        if self.is_conv1d:
            # Conv1D in HF: x @ weight + bias
            out = torch.matmul(x, w_eff)
            if self.base_layer.bias is not None:
                out = out + self.base_layer.bias
            return out
        else:
            return F.linear(x, w_eff, self.base_layer.bias)

    def merge(self):
        if self.merged:
            return
        self.original_weight = self.base_layer.weight.data.clone()
        with torch.no_grad():
            w_eff = self.get_effective_weight()
            self.base_layer.weight.copy_(w_eff)
        self.merged = True
