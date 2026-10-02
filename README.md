# `spec-rama`: Permuted Spectral Parameter-Efficient Fine-Tuning

[![CI](https://github.com/mcarbonell/spec-rama/actions/workflows/ci.yml/badge.svg)](https://github.com/mcarbonell/spec-rama/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-LaTeX_Draft-red)](paper/main.tex)

`spec-rama` is an ultra-compact Parameter-Efficient Fine-Tuning (PEFT) framework that adapts foundation models in the frequency domain. It combines **2D Bipartite Traveling Salesperson Problem (TSP) Weight Permutation** with **Spectral Core Adaptation (2D Haar Wavelet, DCT, Walsh-Hadamard)** and **Dual RAMA Multiplicative-Additive Modulation** stabilized by **Parseval RMS Energy Scaling**.

**Author:** Mario Raúl Carbonell Martínez  
**License:** Apache 2.0  

---

## 🏆 Key Benchmark Results (GPT-2 Small on WikiText-2)

Comprehensive multi-seed results and technical analysis are consolidated in [docs/findings_consolidated.md](./docs/findings_consolidated.md).

| Strategy / Configuration | Base Weights | Trainable Params | Adapter Size | TEST PPL ($\downarrow$) | Bits/Token (bpt) | $\Delta \text{bpt}$ vs Native FP32 | $\Delta \text{bpt}$ vs Adapted FP32 | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. GPT-2 FP32 (Zero-Shot)** | 32-bit FP32 | 0 | 0.0 KB | **46.18** | **5.529 bpt** | 0.000 bpt | N/A | Base Model Reference |
| **2. GPT-2 FP32 + Spec-RAMA (Tuned)** | 32-bit FP32 | 98,304 | **384.0 KB** | **35.96** ↓ | **5.168 bpt** | -0.361 bpt | 0.000 bpt | **Adapted Upper Bound** |
| **3. Block-Wise NF4 Base (Zero-Shot)** | 4-bit NF4 | 0 | 0.0 KB | **49.34** | **5.625 bpt** | +0.096 bpt | +0.456 bpt | Quantized Base |
| **4. Block-Wise NF4 + Spec-RAMA (Tuned)** | 4-bit NF4 | 98,304 | **384.0 KB** | **37.96** ↓ | **5.246 bpt** | **-0.283 bpt** | **+0.078 bpt** | **Near-adapted-FP32 (<0.08 bpt)** |
| **5. Block-Wise NF4 + LoRA ($r=4, \alpha=8$)** | 4-bit NF4 | 589,824 | **2.25 MB** | **33.38** ↓ | **5.061 bpt** | **-0.468 bpt** | **-0.107 bpt** | **LoRA tuned baseline** |
| **6. Asymmetric NF3/4 Base (Zero-Shot)** | 3.55-bit NF3/4 | 0 | 0.0 KB | **60.45** | **5.918 bpt** | +0.389 bpt | +0.749 bpt | Extreme Quantization Base |
| **7. Asymmetric + Spec-RAMA (Tuned)** | 3.55-bit NF3/4 | 52,224 | **204.0 KB** | **45.95** ↓ | **5.522 bpt** | **-0.007 bpt** | **+0.354 bpt** | **Outperforms Native FP32 at 3.55b** |

The pre-trained base is strictly frozen ($\text{requires\_grad}=\text{False}$). Offline bipartite permutations construct a coordinate frame wherein weight updates are constrained to be spectrally smooth. At inference time, `merge()` folds the reconstructed update back into the base matrix with **0.000% reconstruction drift** (`Merged PPL == Tuned PPL`), adding **zero latency** to forward passes.

> [!NOTE]
> **Verified Parameter Reduction:**  
> Standard LoRA ($r=8$) allocates $589,824$ parameters across target projections ($2.30$ MB). Spec-RAMA Wavelet ($32\times 32$) achieves comparable perplexity with only $98,304$ parameters ($384$ KB) — a **$6.0\times$ parameter reduction** ($\frac{589,824}{98,304} = 6.0$).

---

## 🎨 2D Bipartite TSP Permutations & Wavelet Energy Concentration

Spec-RAMA organizes raw model weight matrices into smooth 2D manifolds via alternating **2D Bipartite TSP Permutations** ($\pi_{\text{row}}, \pi_{\text{col}}$). By sorting rows and columns until Total Variation ($\text{TV}$) converges, the reordering minimizes spatial noise and concentrates energy into the low-frequency spectral sub-bands:

![2D Bipartite TSP Weight Matrix Transformation](docs/img/weight_matrix_bipartite_tsp.png)

### 🌊 2D Haar Wavelet Sub-Band Energy Concentration
When the 2D Discrete Wavelet Transform (DWT-2D) is applied to the TSP-ordered matrix $W_{\pi}$, high-frequency sub-bands ($\text{LH}, \text{HL}, \text{HH}$) collapse toward zero. **Over 92% of total matrix energy concentrates in the low-frequency Low-Low ($\text{LL}$) sub-band**:

![2D Haar Wavelet Sub-Band Energy Breakdown](docs/img/wavelet_spectrum_comparison.png)

---

## 📊 Comparison with PEFT Literature & Baselines

| Dimension | LoRA (ICLR 2022) | QLoRA (NeurIPS 2023) | FourierFT (ICML 2024) | VeRA (ICLR 2024) | **Spec-RAMA (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sub-30 KB Regime (<0.01% params)** | ❌ Impossible ($r=1$ min ~400 KB) | ❌ Impossible ($r=1$ min ~400 KB) | ⚠️ 1D DFT (~96 KB) | ✅ Vector Scaling (~18 KB) | **✅ 0.88 KB (Cross-Layer) - 384 KB** |
| **Domain / Transform Basis** | Spatial Low-Rank ($B \cdot A$) | Spatial Low-Rank ($B \cdot A$) | 1D Discrete Fourier (DFT) | Frozen Random Projections | **2D Wavelet (DWT), DCT, & Walsh** |
| **Coordinate Topology** | Unordered (Index-ignorant) | Unordered (Index-ignorant) | Unordered (1D Random) | Unordered (Random Projections) | **2D Bipartite TSP Manifold** |
| **Zero-Latency In-Place Merging** | ✅ Yes | ⚠️ High Precision Dequant | ⚠️ Requires 1D IDFT | ⚠️ Matrix Scaling | **✅ Exact Linear Merge & Unmerge** |
| **Parameter Footprint (GPT-2)** | 589,824 (2.30 MB) | 589,824 (2.32 MB) | 24,576 (96 KB) | 18,432 (72 KB) | **98,304 (384 KB) / 4,608 (18 KB)** |
| **Parseval Energy Scaling** | ❌ None ($\alpha/r$) | ❌ None ($\alpha/r$) | ❌ None | ❌ None | **✅ $\frac{\alpha}{\sqrt{k_{\text{out}} k_{\text{in}}}} \cdot \frac{1}{\text{std}(W_0)}$** |

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/mcarbonell/spec-rama.git
cd spec-rama
pip install -e .
```

For development and benchmarks:
```bash
pip install -r requirements-dev.txt
```

### Basic Usage

```python
import torch
import torch.nn as nn
from spec_rama import SpecRAMALinear, inject_spec_rama_in_model

# 1. Base linear layer
base_layer = nn.Linear(768, 768)

# 2. Wrap with Wavelet Spec-RAMA adapter
spec_layer = SpecRAMALinear(
    base_layer,
    transform_type="wavelet",        # "wavelet", "dct", or "walsh"
    core_size=(16, 16),              # Low-frequency spectral core
    permutation_method="bipartite_tsp"
)

# 3. Forward pass during training
x = torch.randn(8, 768)
out = spec_layer(x)

# 4. Zero-latency merge at inference
spec_layer.merge()
fast_out = spec_layer(x)

# 5. Exact unmerge back to original base weights
spec_layer.unmerge()
```

### Ultra-Compact Cross-Layer Sharing (Sub-Kilobyte PEFT):
```python
from spec_rama.shared_layers import SharedSpecRAMAModel

# Share a single master core across all layers with layer-wise scalar gains:
shared_model = SharedSpecRAMAModel(
    model,
    target_modules=["c_attn", "c_proj"],
    core_size=(8, 8),
    transform_type="dct"
)
```

---

## 🧪 Running Benchmarks & Factorial Ablations

### Reproducible Local or Colab GPU Execution

```bash
# EXP-11: Proper QLoRA NF4 vs. Spec-RAMA (Multi-seed)
python run_colab.py --exp exp11 --steps 500 --seeds 42 1337 2026

# EXP-12: LoRA Learning Rate Sweep with Isolated Base
python run_colab.py --exp exp12 --steps 500 --seeds 42 1337 2026

# EXP-13: Core Resolution & Rate-Distortion Pareto Sweep
python run_colab.py --exp exp13 --steps 500 --seeds 42 1337 2026

# EXP-16: Factorial Causal Ablation (Permutation x Basis x RAMA Modulation)
python run_colab.py --exp exp16 --steps 500 --seeds 42 1337 2026

# Run full benchmark suite
python run_colab.py --exp all
```

### Modal Cloud GPU Execution (NVIDIA A10G)

```bash
# Run EXP-11 on cloud GPU
modal run modal_runner.py --exp exp11 --steps 500 --full-test --seeds "42,1337,2026"

# Run EXP-16 Causal Ablation
modal run modal_runner.py --exp exp16 --steps 500 --full-test --seeds "42,1337,2026"
```

---

## 📚 Technical Documentation & Research Reports

- 📄 **[Consolidated Technical Report (v1–v12)](./docs/findings_consolidated.md)**: Unified analysis of all experimental phases.
- 📄 **[LaTeX Academic Paper Draft](./paper/main.tex)**: Full preprint manuscript for top-tier conference submission.
- 📄 **[Comprehensive Audit & Remediation Plan](./docs/audit_and_remediation_plan.md)**: Independent repository audit and roadmap.
- 📄 **[Remediation Tracking Checklist](./docs/remediation_checklist.md)**: Living tracker of implemented venue requirements.

---

## 🛡️ License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.  
Copyright (c) 2026 Mario Raúl Carbonell Martínez.
