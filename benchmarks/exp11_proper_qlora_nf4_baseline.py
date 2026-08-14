import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama import (
    SpecRAMALinear,
    count_trainable_parameters,
    merge_spec_rama_modules,
)
from spec_rama.lora_baseline import inject_lora_in_model

# NF4 (NormalFloat 4-bit) Quantile Data Points (Dettmers et al., QLoRA)
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

def quantize_blockwise_nf(w_2d: torch.Tensor, codebook: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """
    Proper QLoRA-style block-wise NF4 quantization (block_size=64).
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


import json
import argparse

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
    mode_str = "Full Test Set" if full_test or max_test_samples is None else f"Subset ({test_len} blocks)"
    print(f"Data Prepared: Train = {len(train_data)} blocks ({train_tokens:,} tokens) | Test = {len(test_data)} blocks ({test_tokens:,} tokens) [{mode_str}]")
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
    bits_per_token = avg_loss / math.log(2.0)
    return ppl, avg_loss, bits_per_token, total_tokens


def train_on_dataset_long_horizon(model, train_dataset, steps=500, lr_max=1e-2, lr_min=1e-3, device="cuda", batch_size=4, seed=42):
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


def run_exp11(device="cuda", num_steps=500, max_test_samples=100, full_test=False, save_json=True):
    print("====================================================================================================")
    print(" [EXP-11] PROPER QLORA BLOCK-WISE NF4 (block_size=64) & HEAD-TO-HEAD VS STANDARD LORA CONTROL BENCHMARK ")
    print("====================================================================================================")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_data, test_data = prepare_wikitext_data(
        tokenizer, block_size=256, max_train_samples=600, max_test_samples=max_test_samples, full_test=full_test
    )
    test_token_count = len(test_data) * 256
    
    targets = ["c_attn", "c_proj", "c_fc"]
    results_records = []
    
    # 1. Native FP32 Reference (Zero-Shot)
    print("\n[1/7] Evaluating Native GPT-2 (FP32) Reference...")
    model_fp32_native = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in model_fp32_native.parameters(): p.requires_grad = False
    train_p1, _, _ = count_trainable_parameters(model_fp32_native)
    ppl_fp32_native, _, bpt_fp32_native, _ = evaluate_on_dataset(model_fp32_native, test_data, device=device)
    print(f"  -> Native FP32 (Zero-Shot): {ppl_fp32_native:.2f} PPL ({bpt_fp32_native:.3f} bits/token) [0 params]")
    results_records.append({
        "arm": 1,
        "name": "GPT-2 FP32 Native (Zero-Shot)",
        "format": "FP32",
        "trainable_params": train_p1,
        "adapter_size": format_adapter_size(train_p1),
        "test_ppl": round(ppl_fp32_native, 2),
        "bits_per_token": round(bpt_fp32_native, 3),
        "delta_bpt_vs_native": 0.0,
        "delta_bpt_vs_adapted": None,
        "status": "Reference"
    })

    # 2. Tuned FP32 Reference + SpecRAMA (Domain Adapted Upper Bound)
    print("\n[2/7] Evaluating Tuned GPT-2 (FP32) + SpecRAMA Wavelet (32x32) [Domain Adapted Upper Bound]...")
    model_fp32_tuned = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_fp32_tuned = []
    for name, module in list(model_fp32_tuned.named_modules()):
        if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model_fp32_tuned.get_submodule(parent_name) if parent_name else model_fp32_tuned
            wrapped = SpecRAMALinear(
                module,
                transform_type="wavelet",
                core_size=(32, 32),
                permutation_method="bipartite_tsp",
                alpha_m=8.0,
                alpha_a=2.0
            )
            setattr(parent, child_name, wrapped)
            inject_fp32_tuned.append(wrapped)
            
    model_fp32_tuned = model_fp32_tuned.to(device)
    for p in model_fp32_tuned.parameters(): p.requires_grad = False
    for m in inject_fp32_tuned:
        if m.core_m is not None: m.core_m.requires_grad = True
        if m.core_a is not None: m.core_a.requires_grad = True
    train_p2, _, _ = count_trainable_parameters(model_fp32_tuned)
    print(f"  -> Trainable Parameters: {train_p2:,} ({format_adapter_size(train_p2)})")
    train_on_dataset_long_horizon(model_fp32_tuned, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device, seed=42)
    ppl_fp32_tuned, _, bpt_fp32_tuned, _ = evaluate_on_dataset(model_fp32_tuned, test_data, device=device)
    print(f"  -> FP32 + SpecRAMA (Tuned Upper Bound): {ppl_fp32_tuned:.2f} PPL ({bpt_fp32_tuned:.3f} bits/token)")
    results_records.append({
        "arm": 2,
        "name": "GPT-2 FP32 + SpecRAMA (Tuned)",
        "format": "FP32",
        "trainable_params": train_p2,
        "adapter_size": format_adapter_size(train_p2),
        "test_ppl": round(ppl_fp32_tuned, 2),
        "bits_per_token": round(bpt_fp32_tuned, 3),
        "delta_bpt_vs_native": round(bpt_fp32_tuned - bpt_fp32_native, 3),
        "delta_bpt_vs_adapted": 0.0,
        "status": "Adapted Upper Bound"
    })

    # 3. Proper QLoRA Block-Wise NF4 Base (block_size=64, Zero-Shot)
    print("\n[3/7] Evaluating Proper Block-Wise NF4 Base (block_size=64, Frozen)...")
    model_nf4_block_zero = GPT2LMHeadModel.from_pretrained("gpt2")
    with torch.no_grad():
        for name, module in model_nf4_block_zero.named_modules():
            if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                w_q_2d = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                w_rec = w_q_2d.t() if module.__class__.__name__ == "Conv1D" else w_q_2d
                module.weight.copy_(w_rec)
    model_nf4_block_zero = model_nf4_block_zero.to(device)
    for p in model_nf4_block_zero.parameters(): p.requires_grad = False
    train_p3, _, _ = count_trainable_parameters(model_nf4_block_zero)
    ppl_nf4_block_zero, _, bpt_nf4_block_zero, _ = evaluate_on_dataset(model_nf4_block_zero, test_data, device=device)
    print(f"  -> Proper Block-Wise NF4 Base (Zero-Shot): {ppl_nf4_block_zero:.2f} PPL ({bpt_nf4_block_zero:.3f} bits/token)")
    results_records.append({
        "arm": 3,
        "name": "Block-Wise NF4 Base (Zero-Shot)",
        "format": "4-bit NF4",
        "trainable_params": train_p3,
        "adapter_size": format_adapter_size(train_p3),
        "test_ppl": round(ppl_nf4_block_zero, 2),
        "bits_per_token": round(bpt_nf4_block_zero, 3),
        "delta_bpt_vs_native": round(bpt_nf4_block_zero - bpt_fp32_native, 3),
        "delta_bpt_vs_adapted": round(bpt_nf4_block_zero - bpt_fp32_tuned, 3),
        "status": "Proper Quantized Base"
    })

    # 4. Proper Block-Wise NF4 + SpecRAMA Wavelet (32x32) [Tuned]
    print("\n[4/7] Evaluating Proper Block-Wise NF4 + SpecRAMA Wavelet (32x32) [Tuned]...")
    model_nf4_spec = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_nf4_spec = []
    for name, module in list(model_nf4_spec.named_modules()):
        if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
            w_2d = module.weight.data.t().clone() if module.__class__.__name__ == "Conv1D" else module.weight.data.clone()
            w_q_2d = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
            w_rec = w_q_2d.t() if module.__class__.__name__ == "Conv1D" else w_q_2d
            module.weight.data.copy_(w_rec)
            
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model_nf4_spec.get_submodule(parent_name) if parent_name else model_nf4_spec
            wrapped = SpecRAMALinear(
                module,
                transform_type="wavelet",
                core_size=(32, 32),
                permutation_method="bipartite_tsp",
                alpha_m=8.0,
                alpha_a=2.0
            )
            setattr(parent, child_name, wrapped)
            inject_nf4_spec.append(wrapped)
            
    model_nf4_spec = model_nf4_spec.to(device)
    for p in model_nf4_spec.parameters(): p.requires_grad = False
    for m in inject_nf4_spec:
        if m.core_m is not None: m.core_m.requires_grad = True
        if m.core_a is not None: m.core_a.requires_grad = True
    train_p4, _, _ = count_trainable_parameters(model_nf4_spec)
    print(f"  -> Trainable Parameters: {train_p4:,} ({format_adapter_size(train_p4)})")
    train_on_dataset_long_horizon(model_nf4_spec, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device, seed=42)
    ppl_nf4_spec, _, bpt_nf4_spec, _ = evaluate_on_dataset(model_nf4_spec, test_data, device=device)
    print(f"  -> Proper Block-Wise NF4 + SpecRAMA: {ppl_nf4_spec:.2f} PPL ({bpt_nf4_spec:.3f} bits/token)")
    results_records.append({
        "arm": 4,
        "name": "Block-Wise NF4 + SpecRAMA (Tuned)",
        "format": "4-bit NF4",
        "trainable_params": train_p4,
        "adapter_size": format_adapter_size(train_p4),
        "test_ppl": round(ppl_nf4_spec, 2),
        "bits_per_token": round(bpt_nf4_spec, 3),
        "delta_bpt_vs_native": round(bpt_nf4_spec - bpt_fp32_native, 3),
        "delta_bpt_vs_adapted": round(bpt_nf4_spec - bpt_fp32_tuned, 3),
        "status": "SpecRAMA Quantized"
    })

    # 5. Proper Block-Wise NF4 + Standard LoRA (rank=4) [Tuned]
    print("\n[5/7] Evaluating Proper Block-Wise NF4 + Standard LoRA (r=4) [Tuned]...")
    model_nf4_lora = GPT2LMHeadModel.from_pretrained("gpt2")
    with torch.no_grad():
        for name, module in model_nf4_lora.named_modules():
            if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                w_q_2d = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                w_rec = w_q_2d.t() if module.__class__.__name__ == "Conv1D" else w_q_2d
                module.weight.copy_(w_rec)
                
    inject_lora_in_model(model_nf4_lora, target_modules=["c_attn", "c_proj", "c_fc"], rank=4, alpha=8.0, freeze_base=True)
    model_nf4_lora = model_nf4_lora.to(device)
    train_p5, total_p5, ratio_p5 = count_trainable_parameters(model_nf4_lora)
    print(f"  -> LoRA Trainable Parameters: {train_p5:,} / {total_p5:,} ({format_adapter_size(train_p5)})")
    train_on_dataset_long_horizon(model_nf4_lora, train_data, steps=num_steps, lr_max=1e-4, lr_min=1e-5, device=device, seed=42)
    ppl_nf4_lora, _, bpt_nf4_lora, _ = evaluate_on_dataset(model_nf4_lora, test_data, device=device)
    print(f"  -> Proper Block-Wise NF4 + Standard LoRA (r=4): {ppl_nf4_lora:.2f} PPL ({bpt_nf4_lora:.3f} bits/token)")
    results_records.append({
        "arm": 5,
        "name": "Block-Wise NF4 + LoRA (r=4 Tuned)",
        "format": "4-bit NF4",
        "trainable_params": train_p5,
        "adapter_size": format_adapter_size(train_p5),
        "test_ppl": round(ppl_nf4_lora, 2),
        "bits_per_token": round(bpt_nf4_lora, 3),
        "delta_bpt_vs_native": round(bpt_nf4_lora - bpt_fp32_native, 3),
        "delta_bpt_vs_adapted": round(bpt_nf4_lora - bpt_fp32_tuned, 3),
        "status": "Head-to-Head QLoRA"
    })

    # 6. Proper Asymmetric NF3/NF4 Block-Wise Base (Zero-Shot, Frozen)
    print("\n[6/7] Evaluating Proper Asymmetric NF3/NF4 Block-Wise Base (Zero-Shot, Frozen)...")
    model_asym_block_zero = GPT2LMHeadModel.from_pretrained("gpt2")
    with torch.no_grad():
        for name, module in model_asym_block_zero.named_modules():
            if module.__class__.__name__ in ["Conv1D", "Linear"]:
                if any(t in name for t in ["attn.c_attn", "attn.c_proj"]):
                    w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                    w_rec = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64).t() if module.__class__.__name__ == "Conv1D" else quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                    module.weight.copy_(w_rec)
                elif any(t in name for t in ["mlp.c_fc", "mlp.c_proj"]):
                    w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                    w_rec = quantize_blockwise_nf(w_2d, NF3_LEVELS, block_size=64).t() if module.__class__.__name__ == "Conv1D" else quantize_blockwise_nf(w_2d, NF3_LEVELS, block_size=64)
                    module.weight.copy_(w_rec)
                    
    model_asym_block_zero = model_asym_block_zero.to(device)
    for p in model_asym_block_zero.parameters(): p.requires_grad = False
    train_p6, _, _ = count_trainable_parameters(model_asym_block_zero)
    ppl_asym_zero, _, bpt_asym_zero, _ = evaluate_on_dataset(model_asym_block_zero, test_data, device=device)
    print(f"  -> Proper Asymmetric Block-Wise Base Zero-Shot: {ppl_asym_zero:.2f} PPL ({bpt_asym_zero:.3f} bits/token)")
    results_records.append({
        "arm": 6,
        "name": "Asymmetric NF3/4 Base (Zero-Shot)",
        "format": "3.55-bit",
        "trainable_params": train_p6,
        "adapter_size": format_adapter_size(train_p6),
        "test_ppl": round(ppl_asym_zero, 2),
        "bits_per_token": round(bpt_asym_zero, 3),
        "delta_bpt_vs_native": round(bpt_asym_zero - bpt_fp32_native, 3),
        "delta_bpt_vs_adapted": round(bpt_asym_zero - bpt_fp32_tuned, 3),
        "status": "Quantized Base"
    })

    # 7. Proper Asymmetric NF3/NF4 + SpecRAMA (Tuned)
    print("\n[7/7] Evaluating Proper Asymmetric NF3/NF4 + SpecRAMA [Tuned]...")
    model_asym_spec = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_asym_spec = []
    for name, module in list(model_asym_spec.named_modules()):
        if module.__class__.__name__ in ["Conv1D", "Linear"]:
            if any(t in name for t in ["attn.c_attn", "attn.c_proj"]):
                w_2d = module.weight.data.t().clone() if module.__class__.__name__ == "Conv1D" else module.weight.data.clone()
                w_rec = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64).t() if module.__class__.__name__ == "Conv1D" else quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                module.weight.data.copy_(w_rec)
                
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model_asym_spec.get_submodule(parent_name) if parent_name else model_asym_spec
                wrapped = SpecRAMALinear(module, transform_type="wavelet", core_size=(32, 32), permutation_method="bipartite_tsp", alpha_m=8.0, alpha_a=2.0)
                setattr(parent, child_name, wrapped)
                inject_asym_spec.append(wrapped)
            elif any(t in name for t in ["mlp.c_fc", "mlp.c_proj"]):
                w_2d = module.weight.data.t().clone() if module.__class__.__name__ == "Conv1D" else module.weight.data.clone()
                w_rec = quantize_blockwise_nf(w_2d, NF3_LEVELS, block_size=64).t() if module.__class__.__name__ == "Conv1D" else quantize_blockwise_nf(w_2d, NF3_LEVELS, block_size=64)
                module.weight.data.copy_(w_rec)
                
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model_asym_spec.get_submodule(parent_name) if parent_name else model_asym_spec
                wrapped = SpecRAMALinear(module, transform_type="wavelet", core_size=(8, 8), permutation_method="bipartite_tsp", alpha_m=8.0, alpha_a=2.0)
                setattr(parent, child_name, wrapped)
                inject_asym_spec.append(wrapped)
                
    model_asym_spec = model_asym_spec.to(device)
    for p in model_asym_spec.parameters(): p.requires_grad = False
    for m in inject_asym_spec:
        if m.core_m is not None: m.core_m.requires_grad = True
        if m.core_a is not None: m.core_a.requires_grad = True
    train_p7, _, _ = count_trainable_parameters(model_asym_spec)
    print(f"  -> Trainable Parameters: {train_p7:,} ({format_adapter_size(train_p7)})")
    train_on_dataset_long_horizon(model_asym_spec, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device, seed=42)
    ppl_asym_spec, _, bpt_asym_spec, _ = evaluate_on_dataset(model_asym_spec, test_data, device=device)
    print(f"  -> Proper Asymmetric + SpecRAMA: {ppl_asym_spec:.2f} PPL ({bpt_asym_spec:.3f} bits/token)")
    results_records.append({
        "arm": 7,
        "name": "Asymmetric + SpecRAMA (Tuned)",
        "format": "3.55-bit",
        "trainable_params": train_p7,
        "adapter_size": format_adapter_size(train_p7),
        "test_ppl": round(ppl_asym_spec, 2),
        "bits_per_token": round(bpt_asym_spec, 3),
        "delta_bpt_vs_native": round(bpt_asym_spec - bpt_fp32_native, 3),
        "delta_bpt_vs_adapted": round(bpt_asym_spec - bpt_fp32_tuned, 3),
        "status": "Extreme Savings"
    })

    print(f"\n=========================================================================================================================================")
    print(f"      EXP-11 MEASURED BENCHMARK SUMMARY (Evaluated on {test_token_count:,} tokens)                                                     ")
    print(f"=========================================================================================================================================")
    print(f"| Strategy / Experimental Arm                 | Format   | Adapt Size | Params   | TEST PPL | Bits/Token | Delta vs Native | Delta vs Adapted | Status |")
    print(f"|---------------------------------------------|----------|------------|----------|----------|------------|-----------------|------------------|--------|")
    for r in results_records:
        delta_nat_str = f"{r['delta_bpt_vs_native']:+6.3f} bpt" if r['delta_bpt_vs_native'] is not None else "   N/A   "
        delta_adp_str = f"{r['delta_bpt_vs_adapted']:+6.3f} bpt" if r['delta_bpt_vs_adapted'] is not None else "   N/A   "
        print(f"| {r['arm']}. {r['name']:<42} | {r['format']:<8} | {r['adapter_size']:<10} | {r['trainable_params']:>8,} | {r['test_ppl']:8.2f} |   {r['bits_per_token']:6.3f}   |   {delta_nat_str:<11}   |   {delta_adp_str:<12}   | {r['status']} |")
    print(f"=========================================================================================================================================")

    if save_json:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "exp11_results.json"))
        output_payload = {
            "experiment": "EXP-11",
            "eval_tokens": test_token_count,
            "eval_blocks": len(test_data),
            "full_test_set": full_test,
            "num_steps": num_steps,
            "results": results_records
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\n Saved verified experimental results to: {output_path}")

    return results_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EXP-11 NF4 + SpecRAMA vs LoRA Benchmark")
    parser.add_argument("--steps", type=int, default=500, help="Number of fine-tuning steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-samples", type=int, default=100, help="Number of test blocks (default: 100 blocks = 25.6k tokens)")
    parser.add_argument("--full-test", action="store_true", help="Evaluate on the full WikiText-2 test set (~287k tokens)")
    args = parser.parse_args()

    run_exp11(
        device=args.device,
        num_steps=args.steps,
        max_test_samples=args.test_samples,
        full_test=args.full_test
    )
