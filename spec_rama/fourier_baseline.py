import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import get_dct_matrix_1d


class FourierFTLinear(nn.Module):
    """FourierFT (Gao et al., ICML 2024): Parameter-Efficient Fine-Tuning via Frequency Domain Adaptation.

    Adapts weights directly in the orthogonal spectral domain without spatial reordering/permutations:
        W_eff = W_0 + (alpha / sqrt(k_out * k_in)) * (B_out.T @ S @ B_in)
    where B_out and B_in are standard orthogonal DCT/Fourier projection bases.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        core_size: Tuple[int, int] = (16, 16),
        alpha: float = 16.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.is_conv1d = "Conv1D" in base_layer.__class__.__name__

        if self.is_conv1d:
            self.in_features = base_layer.weight.shape[0]
            self.out_features = base_layer.weight.shape[1]
        else:
            self.in_features = getattr(base_layer, "in_features", base_layer.weight.shape[1])
            self.out_features = getattr(base_layer, "out_features", base_layer.weight.shape[0])

        self.core_size = (
            min(core_size[0], self.out_features),
            min(core_size[1], self.in_features),
        )
        self.alpha = alpha
        self.scaling = alpha / math.sqrt(self.core_size[0] * self.core_size[1])

        # Freeze base parameters
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Register spectral projection bases (pure frequency domain, NO permutation)
        self.register_buffer("B_out", get_dct_matrix_1d(self.out_features))
        self.register_buffer("B_in", get_dct_matrix_1d(self.in_features))

        # Trainable spectral core initialized to zero (identity at init)
        self.fourier_core = nn.Parameter(torch.zeros(*self.core_size))

        self.merged = False
        self.original_weight: Optional[torch.Tensor] = None

    def _synthesize_delta(self) -> torch.Tensor:
        k_out, k_in = self.core_size
        dev = self.fourier_core.device
        B_out_sub = self.B_out[:k_out, :].to(dev)
        B_in_sub = self.B_in[:k_in, :].to(dev)
        delta = torch.matmul(B_out_sub.t(), torch.matmul(self.fourier_core, B_in_sub))
        return delta * self.scaling

    def get_effective_weight(self) -> torch.Tensor:
        w_base = self.base_layer.weight.data.t().clone() if self.is_conv1d else self.base_layer.weight.clone()
        delta = self._synthesize_delta()
        w_eff = w_base + delta
        return w_eff.t() if self.is_conv1d else w_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        delta = self._synthesize_delta()
        if self.is_conv1d:
            # x @ (W_base + delta.T) + bias
            out = self.base_layer(x) + torch.matmul(x, delta.t())
            return out
        else:
            return F.linear(x, self.base_layer.weight, self.base_layer.bias) + F.linear(x, delta)

    def merge(self):
        """Fuses FourierFT frequency updates into the base weight."""
        if self.merged:
            return
        self.original_weight = self.base_layer.weight.data.clone()
        with torch.no_grad():
            w_eff = self.get_effective_weight()
            self.base_layer.weight.copy_(w_eff)
        self.merged = True

    def unmerge(self):
        """Restores original frozen base weight."""
        if not self.merged:
            return
        if self.original_weight is not None:
            self.base_layer.weight.data.copy_(self.original_weight)
            self.original_weight = None
        self.merged = False


def inject_fourier_ft_in_model(
    model: nn.Module,
    target_modules: List[str],
    core_size: Tuple[int, int] = (16, 16),
    alpha: float = 16.0,
    freeze_base: bool = True,
) -> List[FourierFTLinear]:
    """Recursively wraps target linear/Conv1D layers with FourierFTLinear."""
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    injected = []

    def _inject(module: nn.Module):
        for name, child in list(module.named_children()):
            is_target = (
                isinstance(child, nn.Linear)
                or "Conv1D" in child.__class__.__name__
                or "Linear" in child.__class__.__name__
            )
            if is_target and any(t in name for t in target_modules):
                wrapped = FourierFTLinear(child, core_size=core_size, alpha=alpha)
                wrapped.fourier_core.requires_grad = True
                setattr(module, name, wrapped)
                injected.append(wrapped)
            else:
                _inject(child)

    _inject(model)
    return injected
