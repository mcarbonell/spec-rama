"""
Benchmark EXP-17: Comparative Evaluation of TSP Algorithms for Spectral Permutation
Author of custom TSP algorithms: Mario Raúl Carbonell Martínez

Compares:
1. None (Natural Weight Matrix Order)
2. Random Permutation
3. PCA / SVD Ordination
4. Standard 1D Greedy TSP (Nearest Neighbor, k=0)
5. Alternating 2D Bipartite Greedy TSP
6. k-Alternatives TSP (LDS + Multi-Start + Adaptive RL-style Heuristics)
7. Ripple Insertion TSP (Cheapest Insertion + Wavefront Ripple + 2-opt)

Metrics:
- 2D Total Variation (TV)
- Relative TV Reduction (%)
- 2D Wavelet LL Energy Concentration Ratio (32x32 Core) (%)
- Execution Time (ms)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Model

from spec_rama.permutation import (
    compute_2d_permutations,
    compute_greedy_tsp_1d,
    compute_pca_permutation,
    compute_total_variation_2d,
)
from spec_rama.permutation_k_alternatives import (
    compute_joint_bipartite_k_alternatives_permutation,
)
from spec_rama.permutation_ripple import compute_joint_bipartite_ripple_permutation
from spec_rama.transforms import haar_dwt_2d


def compute_wavelet_energy_ratio(matrix: torch.Tensor, core_h: int = 32, core_w: int = 32) -> float:
    """Computes percentage of Frobenius energy captured by the top-left low-frequency LL core."""
    LL, LH, HL, HH = haar_dwt_2d(matrix)
    core = LL[:core_h, :core_w]
    core_energy = torch.sum(core ** 2).item()
    total_energy = torch.sum(LL ** 2 + LH ** 2 + HL ** 2 + HH ** 2).item()
    return (core_energy / (total_energy + 1e-12)) * 100.0


def benchmark_matrix(name: str, weight: torch.Tensor):
    print("\n=======================================================")
    print(f" Matrix: {name} | Shape: {tuple(weight.shape)}")
    print("=======================================================")

    methods = [
        ("None (Original)", "none"),
        ("Random Permutation", "random"),
        ("PCA / SVD Ordination", "pca"),
        ("1D Greedy TSP (NN, k=0)", "greedy_1d"),
        ("2D Bipartite Greedy TSP", "bipartite_greedy"),
        ("k-Alternatives TSP (LDS+RL)", "k_alternatives"),
        ("Ripple Insertion TSP (+2opt)", "ripple"),
    ]

    base_tv = compute_total_variation_2d(weight)

    results = []

    for label, method_key in methods:
        print(f"  Testing {label}...", end="", flush=True)
        start_t = time.perf_counter()

        if method_key == "none":
            r_perm = torch.arange(weight.shape[0])
            c_perm = torch.arange(weight.shape[1])
        elif method_key == "random":
            torch.manual_seed(42)
            r_perm = torch.randperm(weight.shape[0])
            c_perm = torch.randperm(weight.shape[1])
        elif method_key == "pca":
            r_perm = compute_pca_permutation(weight, axis=0)
            c_perm = compute_pca_permutation(weight, axis=1)
        elif method_key == "greedy_1d":
            r_perm = compute_greedy_tsp_1d(weight, axis=0)
            c_perm = compute_greedy_tsp_1d(weight, axis=1)
        elif method_key == "bipartite_greedy":
            r_perm, c_perm = compute_2d_permutations(weight, method="bipartite_tsp", max_iters=3)
        elif method_key == "k_alternatives":
            r_perm, c_perm = compute_joint_bipartite_k_alternatives_permutation(
                weight, max_iters=2, max_k=2, num_starts=4
            )
        elif method_key == "ripple":
            r_perm, c_perm = compute_joint_bipartite_ripple_permutation(
                weight, max_iters=2, max_k_neighbors=25, enable_2opt=True, max_2opt_passes=3
            )
        else:
            raise ValueError(f"Unknown method {method_key}")

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        print(f" done ({elapsed_ms:.1f} ms)", flush=True)

        permuted = weight[r_perm, :][:, c_perm]
        tv = compute_total_variation_2d(permuted)
        tv_reduction = ((base_tv - tv) / base_tv) * 100.0
        energy_ll = compute_wavelet_energy_ratio(permuted)

        results.append({
            "Method": label,
            "TV": tv,
            "TV Reduction (%)": tv_reduction,
            "LL Energy (%)": energy_ll,
            "Time (ms)": elapsed_ms,
        })

    # Print summary table
    print(f"{'Method':<32} | {'TV Norm':<10} | {'TV Red. %':<10} | {'LL Energy %':<12} | {'Time (ms)':<10}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['Method']:<32} | {r['TV']:<10.2f} | {r['TV Reduction (%)']:>9.2f}% | {r['LL Energy (%)']:>11.2f}% | {r['Time (ms)']:>9.1f}"
        )


def main():
    print("Loading GPT-2 base weights...")
    gpt2 = GPT2Model.from_pretrained("gpt2")
    gpt2.eval()

    # Layer 0 Attention QKV projection (768 x 768 slice)
    # GPT-2 c_attn weight is (768, 2304) containing Q, K, V
    w_attn_q = gpt2.h[0].attn.c_attn.weight.detach()[:, :768].t()  # Shape: (768, 768)
    benchmark_matrix("GPT-2 Layer 0 Self-Attention (W_q)", w_attn_q)

    # Layer 0 MLP c_fc weight (768, 3072 in GPT-2)
    w_mlp = gpt2.h[0].mlp.c_fc.weight.detach().t()  # Shape: (3072, 768) -> let's take (768, 768) slice or full
    benchmark_matrix("GPT-2 Layer 0 MLP (c_fc slice 768x768)", w_mlp[:768, :768])


if __name__ == "__main__":
    main()
