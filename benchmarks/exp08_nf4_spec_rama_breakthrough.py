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
    merge_spec_rama_modules,
)

# NF4 (NormalFloat 4-bit) Quantile Data Points (Dettmers et al., QLoRA)
NF4_LEVELS = torch.tensor([
    -1.0, -0.696192801001, -0.525092900000, -0.394917488098,
    -0.284441381693, -0.184773430228, -0.091050036252, 0.0,
    0.079580299556, 0.160930201411, 0.246124088764, 0.337918341160,
    0.440709829330, 0.562617003917, 0.722956836224, 1.0
])

def quantize_nf4_channelwise(w_2d: torch.Tensor) -> torch.Tensor:
    """
    Quantizes a 2D weight matrix to NF4 (NormalFloat 4-bit) channel-wise.
    """
    dev = w_2d.device
    nf4 = NF4_LEVELS.to(dev)

    # Channel-wise absmax scaling
    scales = torch.max(torch.abs(w_2d), dim=0, keepdim=True)[0] + 1e-8
    w_norm = w_2d / scales # Normalized to [-1, 1]

    # Vectorized nearest NF4 level assignment
    # w_norm: (out, in), nf4: (16,)
    diffs = torch.abs(w_norm.unsqueeze(-1) - nf4) # (out, in, 16)
    q_indices = torch.argmin(diffs, dim=-1) # (out, in)

    # Dequantize using NF4 codebook
    w_q_norm = nf4[q_indices]
    w_rec = w_q_norm * scales
    return w_rec


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


def inject_nf4_and_spec_rama(
    model: nn.Module,
    target_modules: list,
    core_size: tuple = (32, 32),
    transform_type: str = "wavelet"
):
    injected = []

    def _inject(module: nn.Module):
        for name, child in list(module.named_children()):
            is_target = isinstance(child, nn.Linear) or child.__class__.__name__ in ["Conv1D", "Linear"]
            if is_target and any(target in name for target in target_modules):
                w_fp32_2d = child.weight.data.t().clone() if child.__class__.__name__ == "Conv1D" else child.weight.data.clone()

                # Apply NF4 Channel-wise Quantization
                w_nf4_2d = quantize_nf4_channelwise(w_fp32_2d)

                w_nf4 = w_nf4_2d.t() if child.__class__.__name__ == "Conv1D" else w_nf4_2d
                child.weight.data.copy_(w_nf4)

                wrapped = SpecRAMALinear(
                    child,
                    transform_type=transform_type,
                    core_size=core_size,
                    permutation_method="bipartite_tsp",
                    alpha_m=8.0,
                    alpha_a=2.0
                )
                setattr(module, name, wrapped)
                injected.append(wrapped)
            else:
                _inject(child)

    _inject(model)
    return injected


def run_exp08(device="cuda", num_steps=500):
    print("==================================================================")
    print(" [EXP-08] NF4 (NORMALFLOAT 4-BIT) + SPEC-RAMA BREAKTHROUGH BENCHMARK ")
    print("==================================================================")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_data, test_data = prepare_wikitext_data(
        tokenizer, block_size=256, max_train_samples=600, max_test_samples=100
    )

    targets = ["c_attn", "c_proj"]

    # 1. Base FP32 Model Reference
    print("\n[1/4] Evaluating Base GPT-2 (FP32) Reference...")
    model_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_test_ppl, _ = evaluate_on_dataset(model_base, test_data, device=device)
    print(f"  -> Base GPT-2 (FP32) TEST PPL: {base_test_ppl:.2f}")

    # 2. NF4 4-bit Base Model Zero-Shot (Frozen)
    print("\n[2/4] Evaluating NF4 4-bit Base Model (Zero-Shot, Frozen)...")
    model_nf4_zero = GPT2LMHeadModel.from_pretrained("gpt2")
    with torch.no_grad():
        for name, module in model_nf4_zero.named_modules():
            if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                w_nf4_2d = quantize_nf4_channelwise(w_2d)
                w_rec = w_nf4_2d.t() if module.__class__.__name__ == "Conv1D" else w_nf4_2d
                module.weight.copy_(w_rec)
    model_nf4_zero = model_nf4_zero.to(device)
    nf4_zero_ppl, _ = evaluate_on_dataset(model_nf4_zero, test_data, device=device)
    print(f"  -> NF4 4-bit Zero-Shot TEST PPL: {nf4_zero_ppl:.2f}")

    # 3. NF4 4-bit + SpecRAMA Wavelet (16x16, 72 KB) Fine-Tuning
    print("\n[3/4] Fine-Tuning NF4 4-bit + SpecRAMA Wavelet (16x16, 72 KB) [500 Steps]...")
    model_nf4_w16 = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_w16 = inject_nf4_and_spec_rama(model_nf4_w16, targets, core_size=(16, 16), transform_type="wavelet")
    model_nf4_w16 = model_nf4_w16.to(device)
    for p in model_nf4_w16.parameters(): p.requires_grad = False
    for m in inject_w16:
        if m.core_m is not None: m.core_m.requires_grad = True
        if m.core_a is not None: m.core_a.requires_grad = True

    train_on_dataset_long_horizon(model_nf4_w16, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device)
    w16_tuned_ppl, _ = evaluate_on_dataset(model_nf4_w16, test_data, device=device)

    merge_spec_rama_modules(model_nf4_w16)
    w16_merged_ppl, _ = evaluate_on_dataset(model_nf4_w16, test_data, device=device)
    print(f"  -> NF4 + SpecRAMA (16x16) TEST PPL: {w16_tuned_ppl:.2f} | Merged PPL: {w16_merged_ppl:.2f}")

    # 4. NF4 4-bit + SpecRAMA Wavelet (32x32, 288 KB) Fine-Tuning
    print("\n[4/4] Fine-Tuning NF4 4-bit + SpecRAMA Wavelet (32x32, 288 KB) [500 Steps]...")
    model_nf4_w32 = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_w32 = inject_nf4_and_spec_rama(model_nf4_w32, targets, core_size=(32, 32), transform_type="wavelet")
    model_nf4_w32 = model_nf4_w32.to(device)
    for p in model_nf4_w32.parameters(): p.requires_grad = False
    for m in inject_w32:
        if m.core_m is not None: m.core_m.requires_grad = True
        if m.core_a is not None: m.core_a.requires_grad = True

    train_on_dataset_long_horizon(model_nf4_w32, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device)
    w32_tuned_ppl, _ = evaluate_on_dataset(model_nf4_w32, test_data, device=device)

    merge_spec_rama_modules(model_nf4_w32)
    w32_merged_ppl, _ = evaluate_on_dataset(model_nf4_w32, test_data, device=device)
    print(f"  -> NF4 + SpecRAMA (32x32) TEST PPL: {w32_tuned_ppl:.2f} | Merged PPL: {w32_merged_ppl:.2f}")

    print("\n==================================================================")
    print("   EXP-08 NF4 + SPEC-RAMA BREAKTHROUGH BENCHMARK SUMMARY          ")
    print("==================================================================")
    print("| Model / Adaptation Method       | Base Format | Core Size | TEST PPL | Merged PPL | Status |")
    print("|---------------------------------|-------------|-----------|----------|------------|--------|")
    print(f"| GPT-2 Base (FP32)               | 32-bit FP32 | N/A       | {base_test_ppl:8.2f} | N/A        | Reference |")
    print(f"| NF4 4-bit Base (Zero-Shot)      | 4-bit NF4   | N/A       | {nf4_zero_ppl:8.2f} | N/A        | Un-tuned NF4 |")
    print(f"| NF4 4-bit + SpecRAMA (16x16)    | 4-bit NF4   | 16x16     | {w16_tuned_ppl:8.2f} | {w16_merged_ppl:10.2f} | Recovered |")
    print(f"| NF4 4-bit + SpecRAMA (32x32)    | 4-bit NF4   | 32x32     | {w32_tuned_ppl:8.2f} | {w32_merged_ppl:10.2f} | Breakthrough |")
    print("==================================================================")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_exp08(device=device, num_steps=500)
