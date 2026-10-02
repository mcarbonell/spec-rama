import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VeRALinear(nn.Module):
    """VeRA (Kopiczko et al., ICLR 2024): Vector-based Random Matrix Adaptation.

    Shares a single pair of frozen random projection matrices (A, B) across the entire network,
    and learns only layer-specific diagonal vector scalings:
        W_eff = W_0 + (alpha / rank) * (diag(d) @ B @ diag(b) @ A)
    where:
        A: (rank, in_features) - FROZEN random matrix
        B: (out_features, rank) - FROZEN random matrix
        d: (out_features,) - TRAINABLE vector initialized to zero
        b: (rank,) - TRAINABLE vector initialized to one
    """

    def __init__(
        self,
        base_layer: nn.Module,
        shared_A: nn.Parameter,
        shared_B: nn.Parameter,
        rank: int = 256,
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

        self.rank = rank
        self.scaling = alpha / rank

        # Freeze base parameters
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Shared frozen projection matrices
        self.shared_A = shared_A
        self.shared_B = shared_B

        # Trainable scaling vectors (d initialized to 0 for identity at init, b initialized to 1)
        self.d = nn.Parameter(torch.zeros(self.out_features))
        self.b = nn.Parameter(torch.ones(self.rank))

        self.merged = False
        self.original_weight: Optional[torch.Tensor] = None

    def _synthesize_delta(self) -> torch.Tensor:
        # B_scaled = diag(d) @ B: shape (out_features, rank)
        B_scaled = self.shared_B * self.d.unsqueeze(1)
        # A_scaled = diag(b) @ A: shape (rank, in_features)
        A_scaled = self.shared_A * self.b.unsqueeze(1)
        delta = torch.matmul(B_scaled, A_scaled) * self.scaling
        return delta

    def get_effective_weight(self) -> torch.Tensor:
        w_base = self.base_layer.weight.data.t().clone() if self.is_conv1d else self.base_layer.weight.clone()
        delta = self._synthesize_delta()
        w_eff = w_base + delta
        return w_eff.t() if self.is_conv1d else w_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        if self.is_conv1d:
            delta = self._synthesize_delta()
            return self.base_layer(x) + torch.matmul(x, delta.t())
        else:
            # Efficient decomposed computation: x @ A.T -> scaled by b -> @ B.T -> scaled by d
            # x: (..., in_features)
            # A_scaled: (rank, in_features)
            Ax = F.linear(x, self.shared_A * self.b.unsqueeze(1))  # (..., rank)
            BAx = F.linear(Ax, self.shared_B)  # (..., out_features)
            delta_out = BAx * self.d * self.scaling
            return F.linear(x, self.base_layer.weight, self.base_layer.bias) + delta_out

    def merge(self):
        """Fuses VeRA adaptation in-place into the base layer weight."""
        if self.merged:
            return
        self.original_weight = self.base_layer.weight.data.clone()
        with torch.no_grad():
            w_eff = self.get_effective_weight()
            self.base_layer.weight.copy_(w_eff)
        self.merged = True

    def unmerge(self):
        """Restores original base weight."""
        if not self.merged:
            return
        if self.original_weight is not None:
            self.base_layer.weight.data.copy_(self.original_weight)
            self.original_weight = None
        self.merged = False


class VeRAModel(nn.Module):
    """Wraps a model to inject VeRA with shared frozen random projection matrices."""

    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        rank: int = 256,
        alpha: float = 16.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha

        if freeze_base:
            for p in model.parameters():
                p.requires_grad = False

        # Find max dimensions for shared random projections
        max_in = 0
        max_out = 0
        for name, child in model.named_modules():
            if ("Linear" in child.__class__.__name__ or "Conv1D" in child.__class__.__name__) and any(
                t in name for t in target_modules
            ):
                if "Conv1D" in child.__class__.__name__:
                    in_f, out_f = child.weight.shape[0], child.weight.shape[1]
                else:
                    in_f = getattr(child, "in_features", child.weight.shape[1])
                    out_f = getattr(child, "out_features", child.weight.shape[0])
                max_in = max(max_in, in_f)
                max_out = max(max_out, out_f)

        # Allocate shared frozen random projections
        shared_A = torch.randn(rank, max_in) / math.sqrt(rank)
        shared_B = torch.randn(max_out, rank) / math.sqrt(rank)
        self.register_buffer("shared_A", shared_A)
        self.register_buffer("shared_B", shared_B)

        self.vera_layers = nn.ModuleList()
        self._inject(self.model)

    def _inject(self, module: nn.Module):
        for name, child in list(module.named_children()):
            is_target = ("Linear" in child.__class__.__name__ or "Conv1D" in child.__class__.__name__) and any(
                t in name for t in self.target_modules
            )
            if is_target:
                if "Conv1D" in child.__class__.__name__:
                    in_f, out_f = child.weight.shape[0], child.weight.shape[1]
                else:
                    in_f = getattr(child, "in_features", child.weight.shape[1])
                    out_f = getattr(child, "out_features", child.weight.shape[0])

                sub_A = self.shared_A[:, :in_f]
                sub_B = self.shared_B[:out_f, :]

                wrapped = VeRALinear(
                    base_layer=child,
                    shared_A=sub_A,
                    shared_B=sub_B,
                    rank=self.rank,
                    alpha=self.alpha,
                )
                setattr(module, name, wrapped)
                self.vera_layers.append(wrapped)
            else:
                self._inject(child)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def merge(self):
        for layer in self.vera_layers:
            layer.merge()

    def unmerge(self):
        for layer in self.vera_layers:
            layer.unmerge()
