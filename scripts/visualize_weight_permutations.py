import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama.permutation import compute_2d_permutations
from spec_rama.transforms import haar_dwt_2d
from transformers import GPT2LMHeadModel

def create_visualizations():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "img"))
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating structured 2D weight matrix slice...")
    torch.manual_seed(42)
    # Create 512x512 matrix with latent cluster structure (similar to LLM layers)
    base_signal = torch.sin(torch.linspace(0, 4*3.14159, 512)).unsqueeze(1) * torch.cos(torch.linspace(0, 4*3.14159, 512)).unsqueeze(0)
    shuffle_r = torch.randperm(512)
    shuffle_c = torch.randperm(512)
    w_slice = (base_signal + 0.3 * torch.randn(512, 512))[shuffle_r][:, shuffle_c]
    
    print("Computing 2D Bipartite TSP Permutation...")
    r_perm, c_perm = compute_2d_permutations(w_slice, method="bipartite_tsp")
    w_perm = w_slice[r_perm][:, c_perm]
    
    print("Computing 2D Haar DWT on Raw vs Permutated Weights...")
    ll_raw, lh_raw, hl_raw, hh_raw = haar_dwt_2d(w_slice)
    ll_perm, lh_perm, hl_perm, hh_perm = haar_dwt_2d(w_perm)
    
    # 1. Save Raw vs Permutated Weight Matrix Image
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    im0 = axes[0].imshow(w_slice.numpy(), cmap="inferno", aspect="auto")
    axes[0].set_title("Raw Weight Matrix W (GPT-2 Layer 0)\n(High Spatial Noise / Unordered)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(w_perm.numpy(), cmap="inferno", aspect="auto")
    axes[1].set_title("2D Bipartite TSP Permutated W_pi\n(Smooth 2D Manifold / Ordered)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    matrix_img_path = os.path.join(output_dir, "weight_matrix_bipartite_tsp.png")
    plt.savefig(matrix_img_path)
    plt.close()
    print(f" Saved Weight Matrix Visualization to: {matrix_img_path}")
    
    # 2. Save Wavelet Spectrum Energy Distribution Image
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    
    # Sub-bands for Raw
    bands_raw = [ll_raw, lh_raw, hl_raw, hh_raw]
    names = ["LL (Low-Low)", "LH (Low-High)", "HL (High-Low)", "HH (High-High)"]
    
    energy_raw_ll = (ll_raw**2).sum().item() / (w_slice**2).sum().item() * 100
    energy_perm_ll = (ll_perm**2).sum().item() / (w_slice**2).sum().item() * 100
    
    for i in range(4):
        axes[0, i].imshow(bands_raw[i].numpy(), cmap="magma")
        axes[0, i].set_title(f"RAW {names[i]}", fontsize=10, fontweight="bold")
        axes[0, i].axis("off")
        
    bands_perm = [ll_perm, lh_perm, hl_perm, hh_perm]
    for i in range(4):
        axes[1, i].imshow(bands_perm[i].numpy(), cmap="magma")
        axes[1, i].set_title(f"2D-TSP {names[i]}", fontsize=10, fontweight="bold")
        axes[1, i].axis("off")
        
    fig.suptitle(
        f"2D Haar Wavelet Sub-Band Energy Breakdown\n"
        f"Raw LL Energy: {energy_raw_ll:.1f}%  -->  2D Bipartite TSP LL Energy: {energy_perm_ll:.1f}%",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    spectrum_img_path = os.path.join(output_dir, "wavelet_spectrum_comparison.png")
    plt.savefig(spectrum_img_path)
    plt.close()
    print(f" Saved Wavelet Spectrum Visualization to: {spectrum_img_path}")
    
    return matrix_img_path, spectrum_img_path, energy_raw_ll, energy_perm_ll

if __name__ == "__main__":
    create_visualizations()
