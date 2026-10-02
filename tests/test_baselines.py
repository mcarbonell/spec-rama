import torch
import torch.nn as nn

from spec_rama.fourier_baseline import FourierFTLinear
from spec_rama.vera_baseline import VeRALinear, VeRAModel


class DummyConv1D(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 24)
        self.fc2 = nn.Linear(24, 16)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ==============================================================================
# FourierFT (ICML 2024) Tests
# ==============================================================================


def test_fourier_ft_zero_init_identity():
    """Verify FourierFT produces exact identity at initialization (S = 0)."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 16)
    fourier_layer = FourierFTLinear(base_linear, core_size=(8, 8))

    w_eff = fourier_layer.get_effective_weight()
    assert torch.allclose(w_eff, base_linear.weight, atol=1e-6)

    x = torch.randn(2, 32)
    assert torch.allclose(fourier_layer(x), base_linear(x), atol=1e-5)


def test_fourier_ft_merge_unmerge():
    """Verify FourierFT merge() and unmerge() cycle."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 16)
    orig_weight = base_linear.weight.data.clone()

    fourier_layer = FourierFTLinear(base_linear, core_size=(8, 8))
    with torch.no_grad():
        fourier_layer.fourier_core.fill_(0.5)

    w_adapted = fourier_layer.get_effective_weight()
    assert not torch.allclose(w_adapted, orig_weight, atol=1e-4)

    fourier_layer.merge()
    assert fourier_layer.merged
    assert torch.allclose(base_linear.weight.data, w_adapted, atol=1e-6)

    fourier_layer.unmerge()
    assert not fourier_layer.merged
    assert torch.allclose(base_linear.weight.data, orig_weight, atol=1e-7)


def test_fourier_ft_conv1d():
    """Verify FourierFT compatibility with HuggingFace Conv1D."""
    torch.manual_seed(42)
    conv1d = DummyConv1D(24, 36)
    fourier_layer = FourierFTLinear(conv1d, core_size=(8, 8))

    x = torch.randn(2, 4, 24)
    assert torch.allclose(fourier_layer(x), conv1d(x), atol=1e-5)


def test_fourier_ft_gradient_flow():
    """Verify base weights are frozen and only frequency core trains."""
    base_linear = nn.Linear(32, 16)
    fourier_layer = FourierFTLinear(base_linear, core_size=(8, 8))

    assert not base_linear.weight.requires_grad
    assert fourier_layer.fourier_core.requires_grad

    x = torch.randn(2, 32)
    out = fourier_layer(x)
    loss = out.sum()
    loss.backward()

    assert base_linear.weight.grad is None
    assert fourier_layer.fourier_core.grad is not None


# ==============================================================================
# VeRA (ICLR 2024) Tests
# ==============================================================================


def test_vera_zero_init_identity():
    """Verify VeRA produces exact identity at initialization (d = 0)."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 16)
    shared_A = torch.randn(16, 32)
    shared_B = torch.randn(16, 16)

    vera_layer = VeRALinear(base_linear, shared_A=shared_A, shared_B=shared_B, rank=16)

    w_eff = vera_layer.get_effective_weight()
    assert torch.allclose(w_eff, base_linear.weight, atol=1e-6)

    x = torch.randn(2, 32)
    assert torch.allclose(vera_layer(x), base_linear(x), atol=1e-5)


def test_vera_decomposed_forward_consistency():
    """Verify decomposed forward pass matches explicit weight reconstruction."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 16)
    shared_A = torch.randn(16, 32)
    shared_B = torch.randn(16, 16)

    vera_layer = VeRALinear(base_linear, shared_A=shared_A, shared_B=shared_B, rank=16)
    with torch.no_grad():
        vera_layer.d.fill_(0.2)
        vera_layer.b.fill_(0.8)

    x = torch.randn(4, 32)
    out_decomposed = vera_layer(x)

    # Explicit W_eff forward
    w_eff = vera_layer.get_effective_weight()
    out_explicit = torch.nn.functional.linear(x, w_eff, base_linear.bias)

    assert torch.allclose(out_decomposed, out_explicit, atol=1e-5)


def test_vera_merge_unmerge():
    """Verify VeRA merge() and unmerge() cycle."""
    torch.manual_seed(42)
    base_linear = nn.Linear(32, 16)
    orig_weight = base_linear.weight.data.clone()

    shared_A = torch.randn(16, 32)
    shared_B = torch.randn(16, 16)
    vera_layer = VeRALinear(base_linear, shared_A=shared_A, shared_B=shared_B, rank=16)

    with torch.no_grad():
        vera_layer.d.fill_(0.3)

    vera_layer.merge()
    assert vera_layer.merged
    assert not torch.allclose(base_linear.weight.data, orig_weight, atol=1e-4)

    vera_layer.unmerge()
    assert not vera_layer.merged
    assert torch.allclose(base_linear.weight.data, orig_weight, atol=1e-7)


def test_vera_model_wrapper():
    """Verify VeRAModel wraps multiple layers with shared frozen projections."""
    torch.manual_seed(42)
    net = DummyNet()

    vera_model = VeRAModel(net, target_modules=["fc1", "fc2"], rank=16)
    assert len(vera_model.vera_layers) == 2

    # Trainable parameters should ONLY be d and b vectors (24+16 + 16+16 = 72 params)
    trainable = sum(p.numel() for p in vera_model.parameters() if p.requires_grad)
    assert trainable == (24 + 16) + (16 + 16)

    x = torch.randn(2, 32)
    out = vera_model(x)
    assert out.shape == (2, 16)

    vera_model.merge()
    assert all(layer.merged for layer in vera_model.vera_layers)

    vera_model.unmerge()
    assert all(not layer.merged for layer in vera_model.vera_layers)

