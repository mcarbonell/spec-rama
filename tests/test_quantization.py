import pytest
import torch

from spec_rama.quantization import HierarchicalSpectralQuantizer


@pytest.mark.parametrize("transform_type", ["dct", "walsh", "wavelet"])
def test_quantize_dequantize_roundtrip(transform_type):
    """
    Verify hierarchical spectral quantization and dequantization roundtrip.
    Reconstruction SNR should be high and shape preserved.
    """
    torch.manual_seed(42)
    weight = torch.randn(64, 48)

    quantizer = HierarchicalSpectralQuantizer(
        transform_type=transform_type,
        core_ratio=0.0625,
        core_bits=8,
        rest_bits=4,
        permutation_method="bipartite_tsp",
    )

    payload = quantizer.quantize_matrix(weight)

    assert payload["shape"] == (64, 48)
    assert 3.5 <= payload["avg_bits"] <= 5.0
    assert payload["core_q"].dtype == torch.uint8
    assert payload["rest_q"].dtype == torch.uint8

    w_rec = quantizer.dequantize_matrix(payload)
    assert w_rec.shape == weight.shape

    # Calculate Mean Squared Error
    mse = torch.mean((weight - w_rec) ** 2).item()
    # Normalized MSE should be low (< 0.15 for 4-bit rest / 8-bit core)
    nmse = mse / torch.var(weight).item()
    assert nmse < 0.20, f"Excessive quantization error for {transform_type}: NMSE = {nmse}"


def test_quantization_odd_dimensions():
    """Verify quantizer handles non-even / odd matrix shapes gracefully."""
    torch.manual_seed(42)
    weight = torch.randn(33, 47)

    quantizer = HierarchicalSpectralQuantizer(
        transform_type="wavelet",
        core_ratio=0.1,
        core_bits=8,
        rest_bits=4,
    )

    payload = quantizer.quantize_matrix(weight)
    w_rec = quantizer.dequantize_matrix(payload)

    assert w_rec.shape == (33, 47)
