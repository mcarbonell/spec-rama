import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Standard LoRA (Low-Rank Adaptation) wrapper for linear / Conv1D layers.
    W_eff = W_0 + (alpha / rank) * (B @ A)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.is_conv1d = "Conv1D" in base_layer.__class__.__name__

        if self.is_conv1d:
            self.in_features = base_layer.weight.shape[0]
            self.out_features = base_layer.weight.shape[1]
        else:
            self.in_features = getattr(base_layer, "in_features", base_layer.weight.shape[1])
            self.out_features = getattr(base_layer, "out_features", base_layer.weight.shape[0])

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable low-rank matrices: A (rank x in_features), B (out_features x rank)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.merged = False
        self.original_weight = None

    def get_effective_weight(self) -> torch.Tensor:
        w_base_2d = self.base_layer.weight.data.t().clone() if self.is_conv1d else self.base_layer.weight.clone()
        delta_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        w_eff = w_base_2d + delta_w
        return w_eff.t() if self.is_conv1d else w_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        delta = torch.matmul(self.lora_B, self.lora_A) * self.scaling  # (out_features, in_features)

        if self.is_conv1d:
            # Conv1D in HF: x @ weight + bias + x @ delta.T
            out = self.base_layer(x) + torch.matmul(x, delta.t())
            return out
        else:
            out = F.linear(x, self.base_layer.weight, self.base_layer.bias) + F.linear(x, delta)
            return out

    def merge(self):
        if self.merged:
            return
        self.original_weight = self.base_layer.weight.data.clone()
        with torch.no_grad():
            w_base_2d = self.base_layer.weight.data.t() if self.is_conv1d else self.base_layer.weight.data
            delta = torch.matmul(self.lora_B, self.lora_A) * self.scaling
            w_eff_2d = w_base_2d + delta
            w_eff = w_eff_2d.t() if self.is_conv1d else w_eff_2d
            self.base_layer.weight.copy_(w_eff)
        self.merged = True

    def unmerge(self):
        if not self.merged:
            return
        if self.original_weight is not None:
            self.base_layer.weight.data.copy_(self.original_weight)
            self.original_weight = None
        self.merged = False


def inject_lora_in_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    freeze_base: bool = True,
) -> List[LoRALinear]:
    """
    Recursively finds target Linear layers in a PyTorch model and wraps them with LoRALinear.
    When freeze_base=True, freezes all base model parameters so ONLY LoRA parameters train.
    Returns a list of injected LoRALinear layers.
    """
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    injected = []

    def _inject(module: nn.Module):
        for name, child in list(module.named_children()):
            is_target = isinstance(child, nn.Linear) or child.__class__.__name__ in ["Conv1D", "Linear"]
            if is_target and any(target in name for target in target_modules):
                wrapped = LoRALinear(child, rank=rank, alpha=alpha)
                # Ensure LoRA parameters are trainable
                wrapped.lora_A.requires_grad = True
                wrapped.lora_B.requires_grad = True
                setattr(module, name, wrapped)
                injected.append(wrapped)
            else:
                _inject(child)

    _inject(model)
    return injected
