import argparse
import json
import math
import os
import random
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama import (
    assert_strictly_frozen_base,
    count_trainable_parameters,
    inject_spec_rama_in_model,
    merge_spec_rama_modules,
)
from spec_rama.lora_baseline import inject_lora_in_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_wikitext_data(
    tokenizer,
    block_size: int = 256,
    max_train_samples: int = 600,
    max_test_samples: int = 100,
    full_test: bool = False,
):
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
    if full_test:
        test_data = lm_datasets["test"]
    else:
        test_data = lm_datasets["test"].select(range(min(len(lm_datasets["test"]), max_test_samples)))

    eval_tokens = len(test_data) * block_size
    test_desc = f"Full Test Set ({len(test_data)} blocks = {eval_tokens:,} tokens)" if full_test else f"Subset ({len(test_data)} blocks = {eval_tokens:,} tokens)"
    print(f"Data Prepared: Train = {len(train_data)} blocks ({len(train_data)*block_size:,} tokens) | Test = {test_desc}")
    return train_data, test_data


def evaluate_on_dataset(model: nn.Module, dataset, device: str = "cuda", batch_size: int = 4):
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
    bpt = avg_loss / math.log(2)
    return ppl, avg_loss, bpt, total_tokens


def train_on_dataset(
    model: nn.Module,
    train_dataset,
    steps: int = 500,
    lr: float = 1e-2,
    device: str = "cuda",
    batch_size: int = 4,
    seed: int = 42,
):
    set_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    dataset_size = len(train_dataset)
    step = 0
    epoch = 0

    while step < steps:
        epoch += 1
        indices = list(range(0, dataset_size, batch_size))
        random.shuffle(indices)

        for i in indices:
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
            step += 1

    return loss.item()


def run_exp13(
    device: str = "cuda",
    num_steps: int = 500,
    max_test_samples: int = 100,
    full_test: bool = False,
    seeds: Optional[List[int]] = None,
    save_json: bool = True,
):
    if seeds is None:
        seeds = [42]

    print("=" * 115)
    print(" [EXP-13] SPECTRAL CORE RATE-DISTORTION & RESOLUTION SWEEP (PARETO TRADEOFF)")
    print(f" Seeds: {seeds} | Full Test: {full_test} | Steps: {num_steps} | Device: {device.upper()}")
    print("=" * 115)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_data, test_data = prepare_wikitext_data(
        tokenizer,
        block_size=256,
        max_train_samples=600,
        max_test_samples=max_test_samples,
        full_test=full_test,
    )
    eval_tokens = len(test_data) * 256

    targets = ["c_attn", "c_proj"]

    # Base FP32 Model Eval
    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_ppl, _, base_bpt, _ = evaluate_on_dataset(base_model, test_data, device=device)
    print(f"Base GPT-2 (FP32 Zero-Shot) -> Test PPL: {base_ppl:.2f} | Bits/Token: {base_bpt:.3f}")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sweep_configs = [
        # Wavelet resolution ladder
        {"name": "SpecRAMA Wavelet (4x4)", "type": "wavelet", "core": (4, 4), "lr": 1e-2, "is_lora": False},
        {"name": "SpecRAMA Wavelet (8x8)", "type": "wavelet", "core": (8, 8), "lr": 1e-2, "is_lora": False},
        {"name": "SpecRAMA Wavelet (16x16)", "type": "wavelet", "core": (16, 16), "lr": 1e-2, "is_lora": False},
        {"name": "SpecRAMA Wavelet (32x32)", "type": "wavelet", "core": (32, 32), "lr": 1e-2, "is_lora": False},
        {"name": "SpecRAMA Wavelet (64x64)", "type": "wavelet", "core": (64, 64), "lr": 1e-2, "is_lora": False},
        # DCT resolution ladder
        {"name": "SpecRAMA DCT (8x8)", "type": "dct", "core": (8, 8), "lr": 1e-2, "is_lora": False},
        {"name": "SpecRAMA DCT (16x16)", "type": "dct", "core": (16, 16), "lr": 1e-2, "is_lora": False},
        {"name": "SpecRAMA DCT (32x32)", "type": "dct", "core": (32, 32), "lr": 1e-2, "is_lora": False},
        # Direct LoRA baseline reference
        {"name": "LoRA (r=8, alpha=16)", "type": "lora", "rank": 8, "lr": 1e-3, "is_lora": True},
    ]

    all_results = []

    for cfg in sweep_configs:
        name = cfg["name"]
        print(f"\nEvaluating Configuration: {name} across {len(seeds)} seed(s)...")
        seed_runs = []
        trainable_params = 0
        total_params = 0
        trainable_pct = 0.0

        for s in seeds:
            set_seed(s)
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            for p in model.parameters():
                p.requires_grad = False

            if cfg["is_lora"]:
                inject_lora_in_model(model, target_modules=targets, rank=cfg["rank"], alpha=float(cfg["rank"] * 2))
                model = model.to(device)
                assert_strictly_frozen_base(model, allowed_substrings=["lora_A", "lora_B"])
            else:
                inject_spec_rama_in_model(
                    model,
                    target_modules=targets,
                    transform_type=cfg["type"],
                    core_size=cfg["core"],
                    alpha_m=8.0,
                    alpha_a=2.0,
                    permutation_method="bipartite_tsp",
                )
                model = model.to(device)
                assert_strictly_frozen_base(model, allowed_substrings=["core_m", "core_a"])

            trainable_params, total_params, trainable_pct = count_trainable_parameters(model)

            train_on_dataset(
                model,
                train_data,
                steps=num_steps,
                lr=cfg["lr"],
                device=device,
                seed=s,
            )

            test_ppl, _, test_bpt, _ = evaluate_on_dataset(model, test_data, device=device)

            # Test merge / unmerge
            merge_spec_rama_modules(model)
            merged_ppl, _, merged_bpt, _ = evaluate_on_dataset(model, test_data, device=device)

            seed_runs.append({
                "seed": s,
                "test_ppl": test_ppl,
                "test_bpt": test_bpt,
                "merged_ppl": merged_ppl,
                "merged_bpt": merged_bpt,
            })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ppls = [r["test_ppl"] for r in seed_runs]
        bpts = [r["test_bpt"] for r in seed_runs]
        merged_ppls = [r["merged_ppl"] for r in seed_runs]

        mean_ppl = float(np.mean(ppls))
        std_ppl = float(np.std(ppls)) if len(ppls) > 1 else 0.0
        mean_bpt = float(np.mean(bpts))
        std_bpt = float(np.std(bpts)) if len(bpts) > 1 else 0.0
        mean_merged_ppl = float(np.mean(merged_ppls))
        std_merged_ppl = float(np.std(merged_ppls)) if len(merged_ppls) > 1 else 0.0

        size_bytes = trainable_params * 4
        size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.2f} MB"

        if len(seeds) > 1:
            print(f"  -> {name} | Params: {trainable_params:,} ({size_str}) | PPL: {mean_ppl:.2f} +/- {std_ppl:.2f} | Merged PPL: {mean_merged_ppl:.2f}")
        else:
            print(f"  -> {name} | Params: {trainable_params:,} ({size_str}) | PPL: {mean_ppl:.2f} | Merged PPL: {mean_merged_ppl:.2f}")

        all_results.append({
            "name": name,
            "type": cfg["type"],
            "core_size": list(cfg["core"]) if "core" in cfg else None,
            "trainable_params": trainable_params,
            "trainable_pct": round(trainable_pct, 4),
            "adapter_size": size_str,
            "adapter_bytes": size_bytes,
            "mean_test_ppl": round(mean_ppl, 2),
            "std_test_ppl": round(std_ppl, 2),
            "mean_bpt": round(mean_bpt, 3),
            "std_bpt": round(std_bpt, 3),
            "mean_merged_ppl": round(mean_merged_ppl, 2),
            "std_merged_ppl": round(std_merged_ppl, 2),
            "runs": seed_runs,
        })

    print("\n" + "=" * 115)
    print(f"  EXP-13 RATE-DISTORTION & CORE RESOLUTION SUMMARY (Evaluated on {eval_tokens:,} tokens, seeds={seeds})")
    print("=" * 115)
    print("| Configuration                 | Params     | Adapter Size | Test PPL (mean+/-std)   | Merged PPL (mean+/-std) | Bits/Token       |")
    print("|-------------------------------|------------|--------------|-------------------------|-------------------------|------------------|")
    print(f"| Base GPT-2 (FP32 zero-shot)   | 0          | 0 KB         | {base_ppl:7.2f}                 | {base_ppl:7.2f}                 | {base_bpt:6.3f}          |")
    for r in all_results:
        ppl_str = f"{r['mean_test_ppl']:.2f} +/- {r['std_test_ppl']:.2f}" if len(seeds) > 1 else f"{r['mean_test_ppl']:.2f}"
        mppl_str = f"{r['mean_merged_ppl']:.2f} +/- {r['std_merged_ppl']:.2f}" if len(seeds) > 1 else f"{r['mean_merged_ppl']:.2f}"
        bpt_str = f"{r['mean_bpt']:.3f} +/- {r['std_bpt']:.3f}" if len(seeds) > 1 else f"{r['mean_bpt']:.3f}"
        print(f"| {r['name']:29s} | {r['trainable_params']:10,d} | {r['adapter_size']:12s} | {ppl_str:23s} | {mppl_str:23s} | {bpt_str:16s} |")
    print("=" * 115)

    if save_json:
        out_path = os.path.join(os.path.dirname(__file__), "exp13_rate_distortion_results.json")
        payload = {
            "experiment": "EXP-13",
            "eval_tokens": eval_tokens,
            "full_test_set": full_test,
            "num_steps": num_steps,
            "seeds": seeds,
            "base_ppl": round(base_ppl, 2),
            "base_bpt": round(base_bpt, 3),
            "results": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved rate-distortion sweep results to: {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="EXP-13: Core Resolution Rate-Distortion Sweep")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    run_exp13(
        device=args.device,
        num_steps=args.steps,
        max_test_samples=args.test_samples,
        full_test=args.full_test,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
