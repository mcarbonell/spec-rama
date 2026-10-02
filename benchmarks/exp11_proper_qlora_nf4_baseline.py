import argparse
import json
import math
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama import (
    SpecRAMALinear,
    assert_strictly_frozen_base,
    count_trainable_parameters,
)
from spec_rama.lora_baseline import inject_lora_in_model

# NF4 (NormalFloat 4-bit) Quantile Data Points (Dettmers et al., QLoRA)
NF4_LEVELS = torch.tensor(
    [
        -1.0,
        -0.696192801001,
        -0.525092900000,
        -0.394917488098,
        -0.284441381693,
        -0.184773430228,
        -0.091050036252,
        0.0,
        0.079580299556,
        0.160930201411,
        0.246124088764,
        0.337918341160,
        0.440709829330,
        0.562617003917,
        0.722956836224,
        1.0,
    ]
)

# NF3 (NormalFloat 3-bit) Quantile Data Points
NF3_LEVELS = torch.tensor(
    [
        -1.0,
        -0.53528043,
        -0.28444138,
        -0.09105004,
        0.0,
        0.16093020,
        0.44070983,
        1.0,
    ]
)


def quantize_blockwise_nf(w_2d: torch.Tensor, codebook: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Proper QLoRA-style block-wise NF4 quantization (block_size=64).

    Calculates absmax scaling per 64-element block instead of entire column/tensor.
    """
    dev = w_2d.device
    cb = codebook.to(dev)

    orig_shape = w_2d.shape
    flat_w = w_2d.flatten()

    # Pad to multiple of block_size
    pad_len = (block_size - (flat_w.numel() % block_size)) % block_size
    if pad_len > 0:
        flat_w = torch.cat([flat_w, torch.zeros(pad_len, device=dev)])

    num_blocks = flat_w.numel() // block_size
    blocks = flat_w.view(num_blocks, block_size)

    # Per-block absmax scale
    scales = torch.max(torch.abs(blocks), dim=1, keepdim=True)[0] + 1e-8
    blocks_norm = blocks / scales

    # Quantize to nearest codebook level
    diffs = torch.abs(blocks_norm.unsqueeze(-1) - cb)
    q_indices = torch.argmin(diffs, dim=-1)

    # Dequantize
    blocks_q = cb[q_indices] * scales
    flat_q = blocks_q.view(-1)

    if pad_len > 0:
        flat_q = flat_q[:-pad_len]

    return flat_q.view(orig_shape)


def prepare_wikitext_data(tokenizer, block_size=256, max_train_samples=600, max_test_samples=100, full_test=False):
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

    train_len = len(lm_datasets["train"])
    if max_train_samples is not None:
        train_len = min(train_len, max_train_samples)
    train_data = lm_datasets["train"].select(range(train_len))

    test_len = len(lm_datasets["test"])
    if not full_test and max_test_samples is not None:
        test_len = min(test_len, max_test_samples)
    test_data = lm_datasets["test"].select(range(test_len))

    train_tokens = len(train_data) * block_size
    test_tokens = len(test_data) * block_size
    mode_str = (
        "Full Test Set (~287k tokens)"
        if full_test or max_test_samples is None
        else f"Subset ({test_len} blocks = {test_tokens:,} tokens)"
    )
    print(
        f"Data Prepared: Train = {len(train_data)} blocks ({train_tokens:,} tokens) | Test = {len(test_data)} blocks ({test_tokens:,} tokens) [{mode_str}]"
    )
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
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    bits_per_token = avg_loss / math.log(2.0)
    return ppl, avg_loss, bits_per_token, total_tokens


def train_on_dataset_long_horizon(
    model, train_dataset, steps=500, lr_max=1e-2, lr_min=1e-3, device="cuda", batch_size=4, seed=42
):
    torch.manual_seed(seed)
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


def format_adapter_size(trainable_params: int) -> str:
    bytes_fp32 = trainable_params * 4
    if bytes_fp32 < 1024 * 1024:
        return f"{bytes_fp32 / 1024:.1f} KB"
    else:
        return f"{bytes_fp32 / (1024 * 1024):.2f} MB"


def build_and_evaluate_arm(
    arm: int,
    seed: int,
    train_data,
    test_data,
    device: str,
    num_steps: int,
    targets: List[str],
):
    """Builds, trains (if applicable), and evaluates a specific experimental arm with a given seed."""
    torch.manual_seed(seed)

    if arm == 1:
        # GPT-2 FP32 Native (Zero-Shot)
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        for p in model.parameters():
            p.requires_grad = False
        train_params, _, _ = count_trainable_parameters(model)
        ppl, _, bpt, _ = evaluate_on_dataset(model, test_data, device=device)
        return ppl, bpt, train_params

    elif arm == 2:
        # GPT-2 FP32 + SpecRAMA Wavelet (32x32) [Tuned]
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        inject_layers = []
        for name, module in list(model.named_modules()):
            if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                wrapped = SpecRAMALinear(
                    module,
                    transform_type="wavelet",
                    core_size=(32, 32),
                    permutation_method="bipartite_tsp",
                    alpha_m=8.0,
                    alpha_a=2.0,
                )
                setattr(parent, child_name, wrapped)
                inject_layers.append(wrapped)

        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        for m in inject_layers:
            if m.core_m is not None:
                m.core_m.requires_grad = True
            if m.core_a is not None:
                m.core_a.requires_grad = True

        assert_strictly_frozen_base(model, allowed_substrings=["core_m", "core_a"])
        train_params, _, _ = count_trainable_parameters(model)
        train_on_dataset_long_horizon(
            model, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device, seed=seed
        )
        ppl, _, bpt, _ = evaluate_on_dataset(model, test_data, device=device)
        return ppl, bpt, train_params

    elif arm == 3:
        # Block-Wise NF4 Base (Zero-Shot)
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        with torch.no_grad():
            for name, module in model.named_modules():
                if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                    w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                    w_q_2d = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                    w_rec = w_q_2d.t() if module.__class__.__name__ == "Conv1D" else w_q_2d
                    module.weight.copy_(w_rec)
        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        train_params, _, _ = count_trainable_parameters(model)
        ppl, _, bpt, _ = evaluate_on_dataset(model, test_data, device=device)
        return ppl, bpt, train_params

    elif arm == 4:
        # Block-Wise NF4 + SpecRAMA Wavelet (32x32) [Tuned]
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        inject_layers = []
        for name, module in list(model.named_modules()):
            if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                w_2d = (
                    module.weight.data.t().clone()
                    if module.__class__.__name__ == "Conv1D"
                    else module.weight.data.clone()
                )
                w_q_2d = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                w_rec = w_q_2d.t() if module.__class__.__name__ == "Conv1D" else w_q_2d
                module.weight.data.copy_(w_rec)

                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                wrapped = SpecRAMALinear(
                    module,
                    transform_type="wavelet",
                    core_size=(32, 32),
                    permutation_method="bipartite_tsp",
                    alpha_m=8.0,
                    alpha_a=2.0,
                )
                setattr(parent, child_name, wrapped)
                inject_layers.append(wrapped)

        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        for m in inject_layers:
            if m.core_m is not None:
                m.core_m.requires_grad = True
            if m.core_a is not None:
                m.core_a.requires_grad = True

        assert_strictly_frozen_base(model, allowed_substrings=["core_m", "core_a"])
        train_params, _, _ = count_trainable_parameters(model)
        train_on_dataset_long_horizon(
            model, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device, seed=seed
        )
        ppl, _, bpt, _ = evaluate_on_dataset(model, test_data, device=device)
        return ppl, bpt, train_params

    elif arm == 5:
        # Block-Wise NF4 + Standard LoRA (r=4) [Tuned]
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        with torch.no_grad():
            for name, module in model.named_modules():
                if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                    w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                    w_q_2d = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                    w_rec = w_q_2d.t() if module.__class__.__name__ == "Conv1D" else w_q_2d
                    module.weight.copy_(w_rec)

        inject_lora_in_model(
            model,
            target_modules=["c_attn", "c_proj", "c_fc"],
            rank=4,
            alpha=8.0,
            freeze_base=True,
        )
        model = model.to(device)
        assert_strictly_frozen_base(model, allowed_substrings=["lora_A", "lora_B"])
        train_params, _, _ = count_trainable_parameters(model)
        train_on_dataset_long_horizon(
            model, train_data, steps=num_steps, lr_max=1e-4, lr_min=1e-5, device=device, seed=seed
        )
        ppl, _, bpt, _ = evaluate_on_dataset(model, test_data, device=device)
        return ppl, bpt, train_params

    elif arm == 6:
        # Proper Asymmetric NF3/NF4 Block-Wise Base (Zero-Shot)
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        with torch.no_grad():
            for name, module in model.named_modules():
                if module.__class__.__name__ in ["Conv1D", "Linear"]:
                    if any(t in name for t in ["attn.c_attn", "attn.c_proj"]):
                        w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                        w_rec = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                        module.weight.copy_(w_rec.t() if module.__class__.__name__ == "Conv1D" else w_rec)
                    elif any(t in name for t in ["mlp.c_fc", "mlp.c_proj"]):
                        w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                        w_rec = quantize_blockwise_nf(w_2d, NF3_LEVELS, block_size=64)
                        module.weight.copy_(w_rec.t() if module.__class__.__name__ == "Conv1D" else w_rec)

        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        train_params, _, _ = count_trainable_parameters(model)
        ppl, _, bpt, _ = evaluate_on_dataset(model, test_data, device=device)
        return ppl, bpt, train_params

    elif arm == 7:
        # Proper Asymmetric NF3/NF4 + SpecRAMA [Tuned]
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        inject_layers = []
        for name, module in list(model.named_modules()):
            if module.__class__.__name__ in ["Conv1D", "Linear"]:
                if any(t in name for t in ["attn.c_attn", "attn.c_proj"]):
                    w_2d = (
                        module.weight.data.t().clone()
                        if module.__class__.__name__ == "Conv1D"
                        else module.weight.data.clone()
                    )
                    w_rec = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                    module.weight.data.copy_(w_rec.t() if module.__class__.__name__ == "Conv1D" else w_rec)

                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    wrapped = SpecRAMALinear(
                        module,
                        transform_type="wavelet",
                        core_size=(32, 32),
                        permutation_method="bipartite_tsp",
                        alpha_m=8.0,
                        alpha_a=2.0,
                    )
                    setattr(parent, child_name, wrapped)
                    inject_layers.append(wrapped)
                elif any(t in name for t in ["mlp.c_fc", "mlp.c_proj"]):
                    w_2d = (
                        module.weight.data.t().clone()
                        if module.__class__.__name__ == "Conv1D"
                        else module.weight.data.clone()
                    )
                    w_rec = quantize_blockwise_nf(w_2d, NF3_LEVELS, block_size=64)
                    module.weight.data.copy_(w_rec.t() if module.__class__.__name__ == "Conv1D" else w_rec)

                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    wrapped = SpecRAMALinear(
                        module,
                        transform_type="wavelet",
                        core_size=(8, 8),
                        permutation_method="bipartite_tsp",
                        alpha_m=8.0,
                        alpha_a=2.0,
                    )
                    setattr(parent, child_name, wrapped)
                    inject_layers.append(wrapped)

        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        for m in inject_layers:
            if m.core_m is not None:
                m.core_m.requires_grad = True
            if m.core_a is not None:
                m.core_a.requires_grad = True

        assert_strictly_frozen_base(model, allowed_substrings=["core_m", "core_a"])
        train_params, _, _ = count_trainable_parameters(model)
        train_on_dataset_long_horizon(
            model, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device, seed=seed
        )
        ppl, _, bpt, _ = evaluate_on_dataset(model, test_data, device=device)
        return ppl, bpt, train_params

    else:
        raise ValueError(f"Unknown arm index: {arm}")


def run_exp11(
    device: str = "cuda",
    num_steps: int = 500,
    max_test_samples: int = 100,
    full_test: bool = False,
    seeds: Optional[List[int]] = None,
    save_json: bool = True,
):
    if seeds is None:
        seeds = [42]

    print("=" * 105)
    print(" [EXP-11] PROPER QLORA BLOCK-WISE NF4 & HEAD-TO-HEAD VS SPEC-RAMA BENCHMARK")
    print(f" Seeds: {seeds} | Full Test: {full_test} | Steps: {num_steps} | Device: {device.upper()}")
    print("=" * 105)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_data, test_data = prepare_wikitext_data(
        tokenizer,
        block_size=256,
        max_train_samples=600,
        max_test_samples=max_test_samples,
        full_test=full_test,
    )
    test_token_count = len(test_data) * 256
    targets = ["c_attn", "c_proj", "c_fc"]

    arm_meta = {
        1: {"name": "GPT-2 FP32 Native (Zero-Shot)", "format": "FP32", "is_trained": False, "status": "Reference"},
        2: {
            "name": "GPT-2 FP32 + SpecRAMA (Tuned)",
            "format": "FP32",
            "is_trained": True,
            "status": "Adapted Upper Bound",
        },
        3: {
            "name": "Block-Wise NF4 Base (Zero-Shot)",
            "format": "4-bit NF4",
            "is_trained": False,
            "status": "Proper Quantized Base",
        },
        4: {
            "name": "Block-Wise NF4 + SpecRAMA (Tuned)",
            "format": "4-bit NF4",
            "is_trained": True,
            "status": "SpecRAMA Quantized",
        },
        5: {
            "name": "Block-Wise NF4 + LoRA (r=4 Tuned)",
            "format": "4-bit NF4",
            "is_trained": True,
            "status": "Head-to-Head QLoRA",
        },
        6: {
            "name": "Asymmetric NF3/4 Base (Zero-Shot)",
            "format": "3.55-bit",
            "is_trained": False,
            "status": "Quantized Base",
        },
        7: {
            "name": "Asymmetric + SpecRAMA (Tuned)",
            "format": "3.55-bit",
            "is_trained": True,
            "status": "Extreme Savings",
        },
    }

    results_records = []
    ref_bpt_native = None
    ref_bpt_adapted = None

    for arm_id in range(1, 8):
        meta = arm_meta[arm_id]
        print(f"\n[{arm_id}/7] Evaluating {meta['name']}...")

        arm_seeds = seeds if meta["is_trained"] else [seeds[0]]
        seed_runs = []
        train_params = 0

        for s_idx, s in enumerate(arm_seeds):
            if meta["is_trained"] and len(arm_seeds) > 1:
                print(f"  --> Seed {s} ({s_idx + 1}/{len(arm_seeds)})...")
            ppl_val, bpt_val, t_params = build_and_evaluate_arm(
                arm=arm_id,
                seed=s,
                train_data=train_data,
                test_data=test_data,
                device=device,
                num_steps=num_steps,
                targets=targets,
            )
            train_params = t_params
            seed_runs.append({"seed": s, "ppl": ppl_val, "bpt": bpt_val})

        ppls = [r["ppl"] for r in seed_runs]
        bpts = [r["bpt"] for r in seed_runs]

        mean_ppl = float(np.mean(ppls))
        std_ppl = float(np.std(ppls)) if len(ppls) > 1 else 0.0
        mean_bpt = float(np.mean(bpts))
        std_bpt = float(np.std(bpts)) if len(bpts) > 1 else 0.0

        if arm_id == 1:
            ref_bpt_native = mean_bpt
        elif arm_id == 2:
            ref_bpt_adapted = mean_bpt

        delta_nat = round(mean_bpt - ref_bpt_native, 3) if ref_bpt_native is not None else 0.0
        delta_adp = round(mean_bpt - ref_bpt_adapted, 3) if ref_bpt_adapted is not None else None

        if len(arm_seeds) > 1:
            print(f"  -> {meta['name']}: {mean_ppl:.2f} +/- {std_ppl:.2f} PPL ({mean_bpt:.3f} +/- {std_bpt:.3f} bpt)")
        else:
            print(f"  -> {meta['name']}: {mean_ppl:.2f} PPL ({mean_bpt:.3f} bpt)")

        record = {
            "arm": arm_id,
            "name": meta["name"],
            "format": meta["format"],
            "trainable_params": train_params,
            "adapter_size": format_adapter_size(train_params),
            "test_ppl": round(mean_ppl, 2),
            "test_ppl_std": round(std_ppl, 2),
            "bits_per_token": round(mean_bpt, 3),
            "bits_per_token_std": round(std_bpt, 3),
            "delta_bpt_vs_native": delta_nat,
            "delta_bpt_vs_adapted": delta_adp,
            "status": meta["status"],
            "runs": seed_runs,
        }
        results_records.append(record)

    # Print markdown table
    print("\n" + "=" * 135)
    print(f"      EXP-11 MEASURED BENCHMARK SUMMARY (Evaluated on {test_token_count:,} tokens, seeds={seeds})")
    print("=" * 135)
    if len(seeds) > 1:
        print(
            "| Arm | Strategy / Experimental Arm                 | Format   | Adapt Size | Params   | TEST PPL (mean+/-std) | Bits/Token (mean+/-std) | Delta Native | Status |"
        )
        print(
            "|-----|---------------------------------------------|----------|------------|----------|-----------------------|-------------------------|--------------|--------|"
        )
        for r in results_records:
            ppl_str = f"{r['test_ppl']:.2f} +/- {r['test_ppl_std']:.2f}"
            bpt_str = f"{r['bits_per_token']:.3f} +/- {r['bits_per_token_std']:.3f}"
            delta_nat_str = (
                f"{r['delta_bpt_vs_native']:+6.3f} bpt" if r["delta_bpt_vs_native"] is not None else "   N/A   "
            )
            print(
                f"|  {r['arm']}  | {r['name']:<43} | {r['format']:<8} | {r['adapter_size']:<10} | {r['trainable_params']:>8,} | {ppl_str:<21} | {bpt_str:<23} | {delta_nat_str:<12} | {r['status']} |"
            )
    else:
        print(
            "| Arm | Strategy / Experimental Arm                 | Format   | Adapt Size | Params   | TEST PPL | Bits/Token | Delta Native | Delta Adapted | Status |"
        )
        print(
            "|-----|---------------------------------------------|----------|------------|----------|----------|------------|--------------|---------------|--------|"
        )
        for r in results_records:
            delta_nat_str = (
                f"{r['delta_bpt_vs_native']:+6.3f} bpt" if r["delta_bpt_vs_native"] is not None else "   N/A   "
            )
            delta_adp_str = (
                f"{r['delta_bpt_vs_adapted']:+6.3f} bpt" if r["delta_bpt_vs_adapted"] is not None else "   N/A   "
            )
            print(
                f"|  {r['arm']}  | {r['name']:<43} | {r['format']:<8} | {r['adapter_size']:<10} | {r['trainable_params']:>8,} | {r['test_ppl']:8.2f} |   {r['bits_per_token']:6.3f}   | {delta_nat_str:<12} | {delta_adp_str:<13} | {r['status']} |"
            )
    print("=" * 135)

    if save_json:
        filename = "exp11_multiseed_results.json" if len(seeds) > 1 else "exp11_results.json"
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", filename))
        output_payload = {
            "experiment": "EXP-11",
            "eval_tokens": test_token_count,
            "eval_blocks": len(test_data),
            "full_test_set": full_test,
            "num_steps": num_steps,
            "seeds": seeds,
            "results": results_records,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\nSaved experimental results to: {output_path}")

    return results_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EXP-11 NF4 + SpecRAMA vs LoRA Benchmark")
    parser.add_argument("--steps", type=int, default=500, help="Number of fine-tuning steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--test-samples", type=int, default=100, help="Number of test blocks (default: 100 blocks = 25.6k tokens)"
    )
    parser.add_argument(
        "--full-test", action="store_true", help="Evaluate on the full WikiText-2 test set (~287k tokens)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="List of random seeds (default: 42, recommended: 42 1337 2026)",
    )
    args = parser.parse_args()

    run_exp11(
        device=args.device,
        num_steps=args.steps,
        max_test_samples=args.test_samples,
        full_test=args.full_test,
        seeds=args.seeds,
    )
