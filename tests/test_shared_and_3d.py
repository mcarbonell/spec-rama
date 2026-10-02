import pytest
import torch
import torch.nn as nn

from spec_rama.layers_3d import SpecRAMA3DLinearGroup
from spec_rama.lora_baseline import LoRALinear
from spec_rama.shared_layers import SharedSpecRAMAModel
from spec_rama.utils import (
    assert_strictly_frozen_base,
    merge_spec_rama_modules,
    unmerge_spec_rama_modules,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)
        self.head = nn.Linear(16, 8)

    def forward(self, x):
        return self.head(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def test_shared_spec_rama_model():
    """Verify cross-layer shared spectral adaptation with scalar gains."""
    torch.manual_seed(42)
    model = DummyModel()

    shared_model = SharedSpecRAMAModel(
        model=model,
        target_modules=["fc1", "fc2"],
        transform_type="dct",
        core_size=(8, 8),
        freeze_base=True,
    )

    # Check that shared_layers is an nn.ModuleList
    assert isinstance(shared_model.shared_layers, nn.ModuleList)
    assert len(shared_model.shared_layers) == 2

    # Check trainable parameters: Master cores + layer gains
    trainable_names = [name for name, p in shared_model.named_parameters() if p.requires_grad]
    assert "master_core_m" in trainable_names
    assert "master_core_a" in trainable_names

    # Forward pass
    x = torch.randn(2, 32)
    out = shared_model(x)
    assert out.shape == (2, 8)

    # Merge and unmerge
    shared_model.merge()
    assert all(layer.spec_layer.merged for layer in shared_model.shared_layers)

    shared_model.unmerge()
    assert all(not layer.spec_layer.merged for layer in shared_model.shared_layers)


def test_3d_linear_group():
    """Verify SpecRAMA3DLinearGroup forward pass, RMS Parseval scaling, and merge/unmerge."""
    torch.manual_seed(42)
    layers = [nn.Linear(24, 16) for _ in range(3)]
    orig_w0 = layers[0].weight.data.clone()

    group = SpecRAMA3DLinearGroup(
        base_layers=layers,
        core_size_3d=(2, 4, 4),
        alpha_m=1.0,
        alpha_a=8.0,
    )

    x = torch.randn(4, 24)
    out0 = group(x, layer_idx=0)
    assert out0.shape == (4, 16)

    # Merge
    group.merge()
    assert group.merged

    # Unmerge
    group.unmerge()
    assert not group.merged
    assert torch.allclose(layers[0].weight.data, orig_w0, atol=1e-7)


def test_assert_strictly_frozen_base():
    """Verify assert_strictly_frozen_base prevents gradient leakage."""
    model = DummyModel()
    for p in model.parameters():
        p.requires_grad = False
    model.head.weight.requires_grad = True

    # Should pass
    assert_strictly_frozen_base(model, allowed_substrings=["head.weight"])

    # Should fail if gradient leaks to fc1
    model.fc1.weight.requires_grad = True
    with pytest.raises(AssertionError, match="Gradient leakage"):
        assert_strictly_frozen_base(model, allowed_substrings=["head.weight"])


def test_lora_merge_and_unmerge():
    """Verify LoRALinear merge() and unmerge() cycle."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 16)
    orig_weight = base_linear.weight.data.clone()

    lora = LoRALinear(base_linear, rank=4, alpha=8.0)
    # Set non-zero weights
    with torch.no_grad():
        lora.lora_A.fill_(0.1)
        lora.lora_B.fill_(0.1)

    lora.merge()
    assert lora.merged
    assert not torch.allclose(base_linear.weight.data, orig_weight, atol=1e-4)

    lora.unmerge()
    assert not lora.merged
    assert torch.allclose(base_linear.weight.data, orig_weight, atol=1e-7)


def test_inject_spec_rama_and_utils():
    """Verify recursive injection, parameter counting, and batch merge/unmerge."""
    from spec_rama.lora_baseline import inject_lora_in_model
    from spec_rama.utils import count_trainable_parameters, inject_spec_rama_in_model

    # 1. SpecRAMA injection
    model = DummyModel()
    injected = inject_spec_rama_in_model(
        model,
        target_modules=["fc1", "fc2"],
        transform_type="wavelet",
        freeze_base=True,
    )
    assert len(injected) == 2

    trainable, total, ratio = count_trainable_parameters(model)
    assert trainable > 0
    assert 0 < ratio < 50.0

    # Batch merge & unmerge
    merge_spec_rama_modules(model)
    assert all(layer.merged for layer in injected)
    unmerge_spec_rama_modules(model)
    assert all(not layer.merged for layer in injected)

    # 2. LoRA injection
    model2 = DummyModel()
    lora_injected = inject_lora_in_model(model2, target_modules=["fc1"], rank=4)
    assert len(lora_injected) == 1
    assert lora_injected[0].lora_A.requires_grad
