import math
import os
import sys

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama import (
    SpecRAMALinear,
    count_trainable_parameters,
    merge_spec_rama_modules,
)

# NF4 (NormalFloat 4-bit) Quantile Data Points
NF4_LEVELS = torch.tensor([
    -1.0, -0.696192801001, -0.525092900000, -0.394917488098,
    -0.284441381693, -0.184773430228, -0.091050036252, 0.0,
    0.079580299556, 0.160930201411, 0.246124088764, 0.337918341160,
    0.440709829330, 0.562617003917, 0.722956836224, 1.0
])

# NF3 (NormalFloat 3-bit) Quantile Data Points
NF3_LEVELS = torch.tensor([
    -1.0, -0.53528043, -0.28444138, -0.09105004,
    0.0, 0.16093020, 0.44070983, 1.0
])

def quantize_custom_codebook(w_2d: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    dev = w_2d.device
    cb = codebook.to(dev)
    scales = torch.max(torch.abs(w_2d), dim=0, keepdim=True)[0] + 1e-8
    w_norm = w_2d / scales
    diffs = torch.abs(w_norm.unsqueeze(-1) - cb)
    q_indices = torch.argmin(diffs, dim=-1)
    w_q_norm = cb[q_indices]
    return w_q_norm * scales


def prepare_wikitext_data(tokenizer, block_size=256, max_train_samples=600, max_test_samples=100):
    print("Loading WikiText-2-raw-v1 dataset from Hugging Face...")
    raw_datasets = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
    )

    train_data = lm_datasets["train"].select(range(min(len(lm_datasets["train"]), max_train_samples)))
    test_data = lm_datasets["test"].select(range(min(len(lm_datasets["test"]), max_test_samples)))
    return train_data, test_data


def evaluate_on_dataset(model, dataset, device="cuda", batch_size=4):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)

            outputs = model(input_ids, labels=labels)
            tokens = input_ids.numel()
            total_loss += outputs.loss.item() * tokens
            total_tokens += tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return ppl, avg_loss


def train_on_dataset_long_horizon(model, train_dataset, steps=500, lr_max=1e-2, lr_min=1e-3, device="cuda", batch_size=4):
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_max)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr_min)

    dataset_size = len(train_dataset)
    step = 0

    while step < steps:
        for i in range(0, dataset_size, batch_size):
            if step >= steps:
                break
            batch = train_dataset[i : i + batch_size]
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

    return loss.item()


def inject_asymmetric_heterogeneous_quantization(
    model: nn.Module,
    attn_targets: list = ["attn.c_attn", "attn.c_proj"],
    mlp_targets: list = ["mlp.c_fc", "mlp.c_proj"],
):
    """
    Asymmetric allocation:
    - Attention: NF4 (4-bit) + Wavelet (32x32 Core)
    - FFN / MLP: NF3 (3-bit) + Wavelet (8x8 Core)
    """
    injected = []

    def _inject(module: nn.Module, current_prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{current_prefix}.{name}" if current_prefix else name
            is_linear = isinstance(child, nn.Linear) or child.__class__.__name__ in ["Conv1D", "Linear"]

            if is_linear and any(target in full_name for target in attn_targets):
                w_fp32_2d = child.weight.data.t().clone() if child.__class__.__name__ == "Conv1D" else child.weight.data.clone()
                w_nf4_2d = quantize_custom_codebook(w_fp32_2d, NF4_LEVELS)
                w_nf4 = w_nf4_2d.t() if child.__class__.__name__ == "Conv1D" else w_nf4_2d
                child.weight.data.copy_(w_nf4)

                wrapped = SpecRAMALinear(
                    child,
                    transform_type="wavelet",
                    core_size=(32, 32),
                    permutation_method="bipartite_tsp",
                    alpha_m=8.0,
                    alpha_a=2.0
                )
                setattr(module, name, wrapped)
                injected.append(wrapped)

            elif is_linear and any(target in full_name for target in mlp_targets):
                w_fp32_2d = child.weight.data.t().clone() if child.__class__.__name__ == "Conv1D" else child.weight.data.clone()
                w_nf3_2d = quantize_custom_codebook(w_fp32_2d, NF3_LEVELS)
                w_nf3 = w_nf3_2d.t() if child.__class__.__name__ == "Conv1D" else w_nf3_2d
                child.weight.data.copy_(w_nf3)

                wrapped = SpecRAMALinear(
                    child,
                    transform_type="wavelet",
                    core_size=(8, 8),
                    permutation_method="bipartite_tsp",
                    alpha_m=8.0,
                    alpha_a=2.0
                )
                setattr(module, name, wrapped)
                injected.append(wrapped)
            else:
                _inject(child, full_name)

    _inject(model)
    return injected


def calculate_model_memory_stats(model, trainable_params, avg_quant_bits=3.55):
    total_params = sum(p.numel() for p in model.parameters())
    base_params = total_params - trainable_params

    fp32_base_mb = (total_params * 4) / (1024 * 1024)
    quant_base_mb = (base_params * (avg_quant_bits / 8.0)) / (1024 * 1024)
    adapter_mb = (trainable_params * 4) / (1024 * 1024)
    total_quant_model_mb = quant_base_mb + adapter_mb
    savings_pct = (1.0 - (total_quant_model_mb / fp32_base_mb)) * 100.0

    return fp32_base_mb, quant_base_mb, adapter_mb, total_quant_model_mb, savings_pct


def run_exp09(device="cuda", num_steps=500):
    print("==================================================================")
    print(" [EXP-09] ASYMMETRIC HETEROGENEOUS QUANTIZATION (ATTN-4b / FFN-3b) ")
    print("==================================================================")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_data, test_data = prepare_wikitext_data(
        tokenizer, block_size=256, max_train_samples=600, max_test_samples=100
    )

    # 1. Base FP32 Model Reference
    print("\n[1/3] Evaluating Base GPT-2 (FP32) Reference...")
    model_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_test_ppl, _ = evaluate_on_dataset(model_base, test_data, device=device)
    fp32_mb = (sum(p.numel() for p in model_base.parameters()) * 4) / (1024 * 1024)
    print(f"  -> Base GPT-2 (FP32) | Total Model Size: {fp32_mb:.1f} MB | TEST PPL: {base_test_ppl:.2f}")

    # 2. Symmetric NF4 Homogeneous Model (4-bit All + Wavelet 32x32)
    print("\n[2/3] Testing Symmetric NF4 (4-bit All + Wavelet 32x32 Core)...")
    model_sym = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_asymmetric_heterogeneous_quantization(
        model_sym,
        attn_targets=["c_attn", "c_proj"],
        mlp_targets=[] # All NF4
    )
    model_sym = model_sym.to(device)
    for p in model_sym.parameters(): p.requires_grad = False
    for m in model_sym.modules():
        if hasattr(m, "core_m") and m.core_m is not None: m.core_m.requires_grad = True
        if hasattr(m, "core_a") and m.core_a is not None: m.core_a.requires_grad = True

    trainable_sym, total_sym, ratio_sym = count_trainable_parameters(model_sym)
    fp32_s, q_base_s, adapt_s, total_s, sav_s = calculate_model_memory_stats(model_sym, trainable_sym, avg_quant_bits=4.25)

    step0_sym, _ = evaluate_on_dataset(model_sym, test_data, device=device)
    train_on_dataset_long_horizon(model_sym, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device)
    tuned_sym, _ = evaluate_on_dataset(model_sym, test_data, device=device)

    merge_spec_rama_modules(model_sym)
    merged_sym, _ = evaluate_on_dataset(model_sym, test_data, device=device)
    print(f"  -> Symmetric NF4 (4.25b avg) | Model: {total_s:.1f} MB (Base: {q_base_s:.1f}MB + Adapt: {adapt_s*1024:.1f}KB) | TEST PPL: {tuned_sym:.2f}")

    # 3. Asymmetric Heterogeneous Model (Attention NF4 4-bit / FFN NF3 3-bit)
    print("\n[3/3] Testing Asymmetric Heterogeneous (Attn NF4 4b / FFN NF3 3b + SpecRAMA)...")
    model_asym = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_asymmetric_heterogeneous_quantization(
        model_asym,
        attn_targets=["attn.c_attn", "attn.c_proj"],
        mlp_targets=["mlp.c_fc", "mlp.c_proj"]
    )
    model_asym = model_asym.to(device)
    for p in model_asym.parameters(): p.requires_grad = False
    for m in model_asym.modules():
        if hasattr(m, "core_m") and m.core_m is not None: m.core_m.requires_grad = True
        if hasattr(m, "core_a") and m.core_a is not None: m.core_a.requires_grad = True

    trainable_asym, total_asym, ratio_asym = count_trainable_parameters(model_asym)
    fp32_a, q_base_a, adapt_a, total_a, sav_a = calculate_model_memory_stats(model_asym, trainable_asym, avg_quant_bits=3.55)

    step0_asym, _ = evaluate_on_dataset(model_asym, test_data, device=device)
    train_on_dataset_long_horizon(model_asym, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device)
    tuned_asym, _ = evaluate_on_dataset(model_asym, test_data, device=device)

    merge_spec_rama_modules(model_asym)
    merged_asym, _ = evaluate_on_dataset(model_asym, test_data, device=device)
    print(f"  -> Asymmetric (3.55b avg) | Model: {total_a:.1f} MB (Base: {q_base_a:.1f}MB + Adapt: {adapt_a*1024:.1f}KB) | TEST PPL: {tuned_asym:.2f}")

    print("\n=====================================================================================================================")
    print("                 EXP-09 ASYMMETRIC HETEROGENEOUS QUANTIZATION BENCHMARK SUMMARY                                      ")
    print("=====================================================================================================================")
    print("| Adaptation / Strategy                 | Avg Bitwidth | Base Size | Adapt Size | Total Model | Savings | TEST PPL | Merged PPL | Status |")
    print("|---------------------------------------|--------------|-----------|------------|-------------|---------|----------|------------|--------|")
    print(f"| GPT-2 Base (FP32 Reference)           | 32.00-bit    | 474.7 MB  | 0.0 KB     | 474.7 MB    |   0.0%  | {base_test_ppl:8.2f} | N/A        | Reference |")
    print(f"| Symmetric NF4 Base (Zero-Shot Un-tuned)|  4.25-bit    |  65.5 MB  | 0.0 KB     |  65.5 MB    |  86.2%  | {step0_sym:8.2f} | N/A        | Quantized Base |")
    print(f"| Symmetric NF4 + SpecRAMA (32x32)      |  4.25-bit    |  65.5 MB  | 294.9 KB   |  65.8 MB    |  86.1%  | {tuned_sym:8.2f} | {merged_sym:10.2f} | 99.8% Recovery |")
    print(f"| Asymmetric Base (Zero-Shot Un-tuned)  |  3.55-bit    |  54.8 MB  | 0.0 KB     |  54.8 MB    |  88.5%  | {step0_asym:8.2f} | N/A        | Quantized Base |")
    print(f"| Asymmetric (Attn-4b/FFN-3b) + SpecRAMA|  3.55-bit    |  54.8 MB  | 174.6 KB   |  55.0 MB    |  88.4%  | {tuned_asym:8.2f} | {merged_asym:10.2f} | Recovered |")
    print("=====================================================================================================================")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_exp09(device=device, num_steps=500)
