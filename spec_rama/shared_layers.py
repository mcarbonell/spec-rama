import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from .layers import SpecRAMALinear

class SharedSpecRAMALinear(nn.Module):
    """
    Sub-layer for Cross-Layer Shared Spectral Core adaptation.
    References a single global master spectral core shared across multiple layers,
    and learns layer-specific scalar gains (gamma_m, gamma_a).
    """
    def __init__(
        self,
        base_layer: nn.Module,
        master_core_m: Optional[nn.Parameter],
        master_core_a: Optional[nn.Parameter],
        transform_type: str = "dct",
        core_size: Tuple[int, int] = (8, 8),
        alpha_m: float = 1.0,
        alpha_a: float = 8.0,
        permutation_method: str = "tsp",
    ):
        super().__init__()
        # Instantiate SpecRAMALinear without allocating local cores (0 dead parameters)
        self.spec_layer = SpecRAMALinear(
            base_layer=base_layer,
            transform_type=transform_type,
            core_size=core_size,
            alpha_m=alpha_m,
            alpha_a=alpha_a,
            permutation_method=permutation_method,
            use_multiplicative=False,
            use_additive=False,
        )
        
        # References to global master cores
        self.master_core_m = master_core_m
        self.master_core_a = master_core_a
        
        # Layer-specific scalar gains (initialized to 1.0)
        if master_core_m is not None:
            self.gamma_m = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("gamma_m", None)
            
        if master_core_a is not None:
            self.gamma_a = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("gamma_a", None)

    def get_effective_weight(self) -> torch.Tensor:
        core_m_scaled = (self.master_core_m * self.gamma_m) if self.master_core_m is not None else None
        core_a_scaled = (self.master_core_a * self.gamma_a) if self.master_core_a is not None else None
        return self.spec_layer.get_effective_weight(
            custom_core_m=core_m_scaled,
            custom_core_a=core_a_scaled
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec_layer.merged:
            return self.spec_layer.base_layer(x)
            
        w_eff = self.get_effective_weight()
        if self.spec_layer.is_conv1d:
            out = torch.matmul(x, w_eff)
            if self.spec_layer.base_layer.bias is not None:
                out = out + self.spec_layer.base_layer.bias
            return out
        else:
            return F.linear(x, w_eff, self.spec_layer.base_layer.bias)

    def merge(self):
        if self.spec_layer.merged:
            return
        w_eff = self.get_effective_weight()
        self.spec_layer.base_layer.weight.data.copy_(w_eff)
        self.spec_layer.merged = True


class SharedSpecRAMAModel(nn.Module):
    """
    Wraps a model to inject a single shared Master Spectral Core across all targeted layers.
    Total trainable parameters = 1 Master Core (e.g. 8x8 = 64 params) + 2 * (Number of Layers) gains.
    True Sub-Kilobyte PEFT!
    """
    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        transform_type: str = "dct",
        core_size: Tuple[int, int] = (8, 8),
        alpha_m: float = 1.0,
        alpha_a: float = 8.0,
        permutation_method: str = "tsp",
        freeze_base: bool = True,
    ):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        
        if freeze_base:
            for p in model.parameters():
                p.requires_grad = False

        # Master Cores (shared globally)
        self.master_core_m = nn.Parameter(torch.zeros(*core_size))
        self.master_core_a = nn.Parameter(torch.zeros(*core_size))
        
        self.shared_layers: List[SharedSpecRAMALinear] = []
        self._inject_shared_cores(self.model, transform_type, core_size, alpha_m, alpha_a, permutation_method)

    def _inject_shared_cores(self, module, transform_type, core_size, alpha_m, alpha_a, permutation_method):
        for name, child in list(module.named_children()):
            is_target_layer = isinstance(child, nn.Linear) or child.__class__.__name__ in ["Conv1D", "Linear"]
            if is_target_layer and any(t in name for t in self.target_modules):
                wrapped = SharedSpecRAMALinear(
                    base_layer=child,
                    master_core_m=self.master_core_m,
                    master_core_a=self.master_core_a,
                    transform_type=transform_type,
                    core_size=core_size,
                    alpha_m=alpha_m,
                    alpha_a=alpha_a,
                    permutation_method=permutation_method,
                )
                setattr(module, name, wrapped)
                self.shared_layers.append(wrapped)
            else:
                self._inject_shared_cores(child, transform_type, core_size, alpha_m, alpha_a, permutation_method)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def merge(self):
        for layer in self.shared_layers:
            layer.merge()
