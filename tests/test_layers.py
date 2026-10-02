import pytest
import torch
import torch.nn as nn

from spec_rama.layers import SpecRAMALinear


class DummyConv1D(nn.Module):
    """Simulates HuggingFace Conv1D layer used in GPT-2."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # HF Conv1D stores weight with shape (in_features, out_features)
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF Conv1D compute: x @ weight + bias
        return torch.matmul(x, self.weight) + self.bias


@pytest.mark.parametrize("transform_type", ["dct", "walsh", "wavelet"])
def test_zero_init_identity(transform_type):
    """
    Core theoretical requirement:
    At initialization, trainable cores S_m and S_a are zero, so W_eff MUST strictly equal W_0.
    """
    torch.manual_seed(42)
    base_linear = nn.Linear(64, 32)
    spec_layer = SpecRAMALinear(
        base_layer=base_linear,
        transform_type=transform_type,
        core_size=(8, 8),
    )

    w_eff = spec_layer.get_effective_weight()
    diff = torch.max(torch.abs(w_eff - base_linear.weight))
    assert diff.item() < 1e-6, f"W_eff diverged from W_0 at init for {transform_type}: max diff = {diff.item()}"


@pytest.mark.parametrize("transform_type", ["dct", "walsh", "wavelet"])
def test_forward_equivalence_at_init(transform_type):
    """Forward pass of SpecRAMALinear must match base layer output at init."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 16)
    x = torch.randn(4, 32)

    base_out = base_linear(x)
    spec_layer = SpecRAMALinear(base_layer=base_linear, transform_type=transform_type)
    spec_out = spec_layer(x)

    assert torch.allclose(base_out, spec_out, atol=1e-5)


def test_hf_conv1d_support():
    """Verify full compatibility with HuggingFace Conv1D weight layout (transpose handling)."""
    torch.manual_seed(42)
    in_features, out_features = 32, 64
    conv1d = DummyConv1D(in_features, out_features)
    x = torch.randn(2, 8, in_features)

    base_out = conv1d(x)
    spec_layer = SpecRAMALinear(base_layer=conv1d, transform_type="wavelet", core_size=(8, 8))
    spec_out = spec_layer(x)

    assert spec_layer.is_conv1d
    assert torch.allclose(base_out, spec_out, atol=1e-5)


def test_merge_and_unmerge_linear():
    """Verify that merge() fuses weights in-place, and unmerge() exactly restores original weights."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 16)
    orig_weight_copy = base_linear.weight.data.clone()

    spec_layer = SpecRAMALinear(base_layer=base_linear, transform_type="wavelet")

    # Simulate adaptation by assigning non-zero values to cores
    with torch.no_grad():
        spec_layer.core_m.fill_(0.5)
        spec_layer.core_a.fill_(0.2)

    w_adapted = spec_layer.get_effective_weight().clone()
    assert not torch.allclose(w_adapted, orig_weight_copy, atol=1e-4)

    # 1. Merge
    spec_layer.merge()
    assert spec_layer.merged
    assert torch.allclose(base_linear.weight.data, w_adapted, atol=1e-6)

    x = torch.randn(2, 32)
    merged_out = spec_layer(x)
    base_out_merged = base_linear(x)
    assert torch.allclose(merged_out, base_out_merged, atol=1e-6)

    # 2. Unmerge
    spec_layer.unmerge()
    assert not spec_layer.merged
    assert torch.allclose(base_linear.weight.data, orig_weight_copy, atol=1e-7), (
        "unmerge failed to restore exact original weight"
    )


def test_merge_and_unmerge_conv1d():
    """Verify merge() and unmerge() cycle on HF Conv1D."""
    torch.manual_seed(42)
    conv1d = DummyConv1D(24, 48)
    orig_weight_copy = conv1d.weight.data.clone()

    spec_layer = SpecRAMALinear(base_layer=conv1d, transform_type="dct")
    with torch.no_grad():
        spec_layer.core_m.fill_(0.3)

    spec_layer.merge()
    assert spec_layer.merged
    assert not torch.allclose(conv1d.weight.data, orig_weight_copy, atol=1e-4)

    spec_layer.unmerge()
    assert not spec_layer.merged
    assert torch.allclose(conv1d.weight.data, orig_weight_copy, atol=1e-7)


def test_gradient_flow_and_freezing():
    """Verify base parameters remain strictly frozen while cores receive gradients."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 16)
    spec_layer = SpecRAMALinear(base_layer=base_linear, transform_type="wavelet")

    # Base weight must be frozen
    assert not base_linear.weight.requires_grad
    assert spec_layer.core_m.requires_grad
    assert spec_layer.core_a.requires_grad

    x = torch.randn(4, 32)
    out = spec_layer(x)
    loss = out.sum()
    loss.backward()

    assert base_linear.weight.grad is None
    assert spec_layer.core_m.grad is not None and spec_layer.core_m.grad.abs().sum() > 0
    assert spec_layer.core_a.grad is not None and spec_layer.core_a.grad.abs().sum() > 0
