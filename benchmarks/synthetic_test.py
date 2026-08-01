import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama import (
    SpecRAMALinear,
    SharedSpecRAMAModel,
    SpecRAMA3DLinearGroup,
    inject_spec_rama_in_model,
    merge_spec_rama_modules,
    count_trainable_parameters,
    compute_greedy_tsp_1d,
    compute_joint_bipartite_2d_permutation,
    compute_3d_tensor_permutations,
    get_dct_matrix_1d,
    get_walsh_matrix_1d,
    haar_dwt_2d,
    haar_idwt_2d,
    dct_3d_project,
)

def test_spectral_transforms():
    print("[1/6] Testing Spectral Transforms (DCT, FWHT, DWT)...")
    # DCT Round-trip
    D = get_dct_matrix_1d(64)
    I_dct = torch.matmul(D, D.t())
    assert torch.allclose(I_dct, torch.eye(64), atol=1e-5), "DCT matrix is not orthogonal!"
    
    # FWHT Round-trip
    W = get_walsh_matrix_1d(64)
    I_walsh = torch.matmul(W, W.t())
    assert torch.allclose(I_walsh, torch.eye(64), atol=1e-5), "Walsh-Hadamard matrix is not orthogonal!"
    
    # Haar Wavelet Round-trip
    x = torch.randn(32, 64)
    LL, LH, HL, HH = haar_dwt_2d(x)
    x_rec = haar_idwt_2d(LL, LH, HL, HH, target_shape=x.shape)
    assert torch.allclose(x, x_rec, atol=1e-5), "2D Haar Wavelet round-trip failed!"
    print("  -> All transform round-trips passed perfectly!")


def test_spec_rama_layer(transform_type="dct"):
    print(f"[2/6] Testing SpecRAMALinear ({transform_type.upper()})...")
    base = nn.Linear(128, 64)
    x = torch.randn(16, 128)
    
    # Step 0 output
    orig_out = base(x).detach().clone()
    
    spec_layer = SpecRAMALinear(
        base,
        transform_type=transform_type,
        core_size=(8, 8),
        permutation_method="bipartite_tsp",
    )
    
    # Verify Step 0 exact equality (zero initialization)
    step0_out = spec_layer(x)
    assert torch.allclose(orig_out, step0_out, atol=1e-5), f"Step 0 output mismatch for {transform_type}!"
    
    # Backward pass & Gradient flow check
    loss = step0_out.sum()
    loss.backward()
    assert spec_layer.core_m.grad is not None, "Gradient didn't reach core_m!"
    assert spec_layer.core_a.grad is not None, "Gradient didn't reach core_a!"
    
    # Merge test
    spec_layer.core_m.data.add_(torch.randn_like(spec_layer.core_m) * 0.1)
    spec_layer.core_a.data.add_(torch.randn_like(spec_layer.core_a) * 0.1)
    
    unmerged_out = spec_layer(x).detach()
    spec_layer.merge()
    merged_out = spec_layer(x).detach()
    
    assert torch.allclose(unmerged_out, merged_out, atol=1e-4), f"Merge output mismatch for {transform_type}!"
    print(f"  -> SpecRAMALinear ({transform_type.upper()}) passed step 0, gradient flow, and merge tests!")


def test_model_injection():
    print("[3/6] Testing Auto-Injection in Complex Model...")
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        def forward(self, x):
            return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    net = SimpleNet()
    trainable_before, total, _ = count_trainable_parameters(net)
    
    injected = inject_spec_rama_in_model(
        net,
        target_modules=["fc1", "fc2"],
        transform_type="dct",
        core_size=(8, 8),
        permutation_method="bipartite_tsp",
    )
    
    trainable_after, total_after, ratio = count_trainable_parameters(net)
    assert len(injected) == 2, "Failed to inject into 2 target layers!"
    print(f"  -> Injected {len(injected)} layers. Trainable params reduced from {trainable_before} to {trainable_after} ({ratio:.2f}% of total).")


def test_shared_core_model():
    print("[4/6] Testing Cross-Layer Shared Core Model (Sub-Kilobyte PEFT)...")
    class MultiLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(256, 256)
            self.layer2 = nn.Linear(256, 256)
            self.layer3 = nn.Linear(256, 256)
            self.layer4 = nn.Linear(256, 256)
        def forward(self, x):
            return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    net = MultiLayerNet()
    shared_net = SharedSpecRAMAModel(
        net,
        target_modules=["layer"],
        transform_type="dct",
        core_size=(8, 8),
    )
    
    trainable, total, ratio = count_trainable_parameters(shared_net)
    print(f"  -> Shared Core Model: Only {trainable} trainable parameters across 4 layers! ({trainable * 4} bytes = {trainable*4/1024:.2f} KB)")
    
    x = torch.randn(8, 256)
    out = shared_net(x)
    loss = out.sum()
    loss.backward()
    
    assert shared_net.master_core_m.grad is not None, "Gradient didn't reach master_core_m!"
    print("  -> Shared Core Model passed forward & backward tests!")


def test_3d_tensor_core_group():
    print("[5/6] Testing 3D Tensor Core Group across Layers/Depth...")
    layers = [nn.Linear(256, 256) for _ in range(8)]
    
    group = SpecRAMA3DLinearGroup(
        base_layers=layers,
        core_size_3d=(4, 8, 8), # 4 x 8 x 8 = 256 params for 8 layers combined!
        permutation_method="tsp",
    )
    
    w_effs = group.get_effective_weights()
    assert len(w_effs) == 8, "Failed to produce 8 effective weight matrices!"
    
    loss = sum(w.sum() for w in w_effs)
    loss.backward()
    
    assert group.core_3d_m.grad is not None, "Gradient didn't reach 3D core_3d_m!"
    assert group.core_3d_a.grad is not None, "Gradient didn't reach 3D core_3d_a!"
    print("  -> 3D Tensor Core Group passed 3D-DCT projection & gradient flow tests!")


def test_hierarchical_spectral_quantizer():
    print("[6/7] Testing Hierarchical Spectral Quantizer (v289 + v290)...")
    from spec_rama import HierarchicalSpectralQuantizer
    
    W = torch.randn(128, 128)
    quantizer = HierarchicalSpectralQuantizer(
        transform_type="dct",
        core_ratio=0.0625,
        core_bits=8,
        rest_bits=4,
        permutation_method="bipartite_tsp",
    )
    
    payload = quantizer.quantize_matrix(W)
    avg_bits = payload["avg_bits"]
    print(f"  -> Quantized 128x128 matrix down to {avg_bits:.2f} bits/weight!")
    
    W_rec = quantizer.dequantize_matrix(payload)
    diff = torch.norm(W - W_rec) / torch.norm(W)
    print(f"  -> Quantization relative error: {diff.item():.4f}")
    assert payload["core_q"].dtype == torch.uint8, "Core bits are not uint8!"
    print("  -> Hierarchical Spectral Quantizer passed successfully!")


def test_parameter_budget_comparison():
    print("[7/7] Parameter Budget Comparison (SpecRAMA vs Standard LoRA)...")
    d_out, d_in = 4096, 4096
    
    # Standard LoRA r=8
    lora_params = 2 * (d_out * 8)
    
    # SpecRAMA Core 8x8
    spec_rama_params = 8 * 8 * 2
    
    reduction = lora_params / spec_rama_params
    print(f"  -> Single 4096x4096 Layer:")
    print(f"     * LoRA (rank=8): {lora_params:,} params")
    print(f"     * SpecRAMA (core=8x8): {spec_rama_params:,} params")
    print(f"     * Parameter Reduction: {reduction:.1f}x FEWER parameters than LoRA!")


if __name__ == "__main__":
    print("==================================================")
    print("     RUNNING SPEC-RAMA SYNTHETIC BENCHMARK        ")
    print("==================================================")
    test_spectral_transforms()
    test_spec_rama_layer("dct")
    test_spec_rama_layer("walsh")
    test_spec_rama_layer("wavelet")
    test_model_injection()
    test_shared_core_model()
    test_3d_tensor_core_group()
    test_hierarchical_spectral_quantizer()
    test_parameter_budget_comparison()
    print("==================================================")
    print("  ALL SPEC-RAMA TESTS PASSED WITH 100% SUCCESS!   ")
    print("==================================================")
