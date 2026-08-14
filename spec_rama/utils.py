import torch
import torch.nn as nn
from typing import List, Tuple
from .layers import SpecRAMALinear
from .shared_layers import SharedSpecRAMALinear

def inject_spec_rama_in_model(
    model: nn.Module,
    target_modules: List[str],
    transform_type: str = "dct",
    core_size: Tuple[int, int] = (8, 8),
    alpha_m: float = 1.0,
    alpha_a: float = 8.0,
    permutation_method: str = "tsp",
    use_multiplicative: bool = True,
    use_additive: bool = True,
    freeze_base: bool = True,
) -> List[SpecRAMALinear]:
    """
    Recursively finds target Linear layers in a PyTorch model and wraps them with SpecRAMALinear.
    When freeze_base=True, freezes all base model parameters so ONLY SpecRAMA cores train.
    Returns a list of injected layers.
    """
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    injected_layers = []

    def _inject(module: nn.Module):
        for name, child in list(module.named_children()):
            is_target_layer = isinstance(child, nn.Linear) or child.__class__.__name__ in ["Conv1D", "Linear"]
            if is_target_layer and any(target in name for target in target_modules):
                wrapped = SpecRAMALinear(
                    base_layer=child,
                    transform_type=transform_type,
                    core_size=core_size,
                    alpha_m=alpha_m,
                    alpha_a=alpha_a,
                    permutation_method=permutation_method,
                    use_multiplicative=use_multiplicative,
                    use_additive=use_additive,
                )
                if wrapped.core_m is not None:
                    wrapped.core_m.requires_grad = True
                if wrapped.core_a is not None:
                    wrapped.core_a.requires_grad = True
                setattr(module, name, wrapped)
                injected_layers.append(wrapped)
            else:
                _inject(child)

    _inject(model)
    return injected_layers


def merge_spec_rama_modules(model: nn.Module):
    """
    Recursively calls merge() on all SpecRAMALinear and SharedSpecRAMALinear modules in a model.
    """
    for module in model.modules():
        if isinstance(module, (SpecRAMALinear, SharedSpecRAMALinear)):
            module.merge()


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Returns (trainable_params, total_params, percentage_trainable).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = (trainable / total * 100.0) if total > 0 else 0.0
    return trainable, total, ratio
