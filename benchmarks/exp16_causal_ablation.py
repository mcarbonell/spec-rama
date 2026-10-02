import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.exp11_proper_qlora_nf4_baseline import (
    NF4_LEVELS,
    evaluate_on_dataset,
    format_adapter_size,
    prepare_wikitext_data,
    quantize_blockwise_nf,
    train_on_dataset_long_horizon,
)
from spec_rama import SpecRAMALinear, assert_strictly_frozen_base, count_trainable_parameters


def build_and_evaluate_ablation_arm(
    perm_method: str,
    transform_type: str,
    use_multiplicative: bool,
    use_additive: bool,
    core_size: tuple,
    seed: int,
    train_data,
    test_data,
    device: str,
    num_steps: int,
    targets: List[str],
):
    """Builds, trains, and evaluates a single factorial ablation arm on NF4 quantized GPT-2."""
    torch.manual_seed(seed)
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Quantize base layers to NF4 block-wise
    with torch.no_grad():
        for name, module in model.named_modules():
            if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                w_q_2d = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                w_rec = w_q_2d.t() if module.__class__.__name__ == "Conv1D" else w_q_2d
                module.weight.copy_(w_rec)

    # Inject SpecRAMA with specific permutation and transform configuration
    inject_layers = []
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            wrapped = SpecRAMALinear(
                module,
                transform_type=transform_type,
                core_size=core_size,
                permutation_method=perm_method,
                alpha_m=8.0,
                alpha_a=2.0,
                use_multiplicative=use_multiplicative,
                use_additive=use_additive,
            )
            setattr(parent, child_name, wrapped)
            inject_layers.append(wrapped)

    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    allowed_subs = []
    if use_multiplicative:
        allowed_subs.append("core_m")
        for m in inject_layers:
            if m.core_m is not None:
                m.core_m.requires_grad = True
    if use_additive:
        allowed_subs.append("core_a")
        for m in inject_layers:
            if m.core_a is not None:
                m.core_a.requires_grad = True

    assert_strictly_frozen_base(model, allowed_substrings=allowed_subs)
    train_params, _, _ = count_trainable_parameters(model)

    train_on_dataset_long_horizon(
        model, train_data, steps=num_steps, lr_max=1e-2, lr_min=1e-3, device=device, seed=seed
    )
    ppl, _, bpt, _ = evaluate_on_dataset(model, test_data, device=device)
    return ppl, bpt, train_params


def run_exp16(
    device: str = "cuda",
    num_steps: int = 500,
    max_test_samples: int = 100,
    full_test: bool = False,
    seeds: Optional[List[int]] = None,
    arm_ids: Optional[List[int]] = None,
    save_json: bool = True,
):
    if seeds is None:
        seeds = [42]

    print("=" * 115)
    print(" [EXP-16] CAUSAL FACTORIAL ABLATION BENCHMARK (PERMUTATION x SPECTRAL BASE x RAMA MODULATION)")
    print(f" Seeds: {seeds} | Full Test: {full_test} | Steps: {num_steps} | Arms: {arm_ids or 'All'} | Device: {device.upper()}")
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
    test_token_count = len(test_data) * 256
    targets = ["c_attn", "c_proj", "c_fc"]

    # Factorial Arms Definition
    ablation_arms = [
        {
            "id": 1,
            "name": "Identity (None) x Wavelet (M+A)",
            "perm": "none",
            "transform": "wavelet",
            "mult": True,
            "add": True,
            "core": (32, 32),
            "hypothesis": "Evaluates Wavelet without TSP ordering (baseline spectral)",
        },
        {
            "id": 2,
            "name": "Random Permutation x Wavelet (M+A)",
            "perm": "random",
            "transform": "wavelet",
            "mult": True,
            "add": True,
            "core": (32, 32),
            "hypothesis": "Shuffled control to prove TSP topology is non-trivial",
        },
        {
            "id": 3,
            "name": "Bipartite TSP x Wavelet (M+A) [Full Spec-RAMA]",
            "perm": "bipartite_tsp",
            "transform": "wavelet",
            "mult": True,
            "add": True,
            "core": (32, 32),
            "hypothesis": "Flagship Spec-RAMA Wavelet configuration",
        },
        {
            "id": 4,
            "name": "Identity (None) x DCT (M+A) [FourierFT-like]",
            "perm": "none",
            "transform": "dct",
            "mult": True,
            "add": True,
            "core": (16, 16),
            "hypothesis": "DCT without TSP permutation",
        },
        {
            "id": 5,
            "name": "Bipartite TSP x DCT (M+A)",
            "perm": "bipartite_tsp",
            "transform": "dct",
            "mult": True,
            "add": True,
            "core": (16, 16),
            "hypothesis": "Proves TSP benefit specifically on DCT",
        },
        {
            "id": 6,
            "name": "Bipartite TSP x Walsh (M+A)",
            "perm": "bipartite_tsp",
            "transform": "walsh",
            "mult": True,
            "add": True,
            "core": (16, 16),
            "hypothesis": "Hadamard binary basis under TSP ordering",
        },
        {
            "id": 7,
            "name": "Bipartite TSP x Wavelet (Multiplicative Only)",
            "perm": "bipartite_tsp",
            "transform": "wavelet",
            "mult": True,
            "add": False,
            "core": (32, 32),
            "hypothesis": "Isolates multiplicative RAMA component (core_a = 0)",
        },
        {
            "id": 8,
            "name": "Bipartite TSP x Wavelet (Additive Only)",
            "perm": "bipartite_tsp",
            "transform": "wavelet",
            "mult": False,
            "add": True,
            "core": (32, 32),
            "hypothesis": "Isolates additive RAMA component (core_m = 0)",
        },
        {
            "id": 9,
            "name": "Single-Pass 1D TSP x Wavelet (M+A)",
            "perm": "tsp",
            "transform": "wavelet",
            "mult": True,
            "add": True,
            "core": (32, 32),
            "hypothesis": "Compares 1D TSP vs alternating 2D bipartite TSP",
        },
        {
            "id": 10,
            "name": "PCA SVD Permutation x Wavelet (M+A)",
            "perm": "pca",
            "transform": "wavelet",
            "mult": True,
            "add": True,
            "core": (32, 32),
            "hypothesis": "Evaluates spectral SVD ordination vs metric TSP",
        },
        {
            "id": 11,
            "name": "k-Alternatives TSP x Wavelet (M+A)",
            "perm": "k_alternatives_tsp",
            "transform": "wavelet",
            "mult": True,
            "add": True,
            "core": (32, 32),
            "hypothesis": "Evaluates k-Alternatives LDS search with RL-style heuristic promotion vs Greedy TSP",
        },
        {
            "id": 12,
            "name": "Ripple Insertion TSP x Wavelet (M+A)",
            "perm": "ripple_tsp",
            "transform": "wavelet",
            "mult": True,
            "add": True,
            "core": (32, 32),
            "hypothesis": "Evaluates Ripple Insertion with wavefront relaxation and 2-opt vs Greedy TSP",
        },
    ]

    if arm_ids is not None:
        ablation_arms = [a for a in ablation_arms if a["id"] in arm_ids]

    results = []

    for arm in ablation_arms:
        print(f"\n--- Arm {arm['id']}/{len(ablation_arms)}: {arm['name']} ---")
        print(f"    Hypothesis: {arm['hypothesis']}")
        seed_runs = []
        train_params = 0

        for s in seeds:
            if len(seeds) > 1:
                print(f"  --> Running seed {s}...")
            ppl, bpt, t_params = build_and_evaluate_ablation_arm(
                perm_method=arm["perm"],
                transform_type=arm["transform"],
                use_multiplicative=arm["mult"],
                use_additive=arm["add"],
                core_size=arm["core"],
                seed=s,
                train_data=train_data,
                test_data=test_data,
                device=device,
                num_steps=num_steps,
                targets=targets,
            )
            train_params = t_params
            seed_runs.append({"seed": s, "test_ppl": ppl, "bits_per_token": bpt})

        ppls = [r["test_ppl"] for r in seed_runs]
        bpts = [r["bits_per_token"] for r in seed_runs]

        mean_ppl = float(np.mean(ppls))
        std_ppl = float(np.std(ppls)) if len(ppls) > 1 else 0.0
        mean_bpt = float(np.mean(bpts))
        std_bpt = float(np.std(bpts)) if len(bpts) > 1 else 0.0

        if len(seeds) > 1:
            print(f"  Result: PPL = {mean_ppl:.2f} +/- {std_ppl:.2f} | bpt = {mean_bpt:.3f} +/- {std_bpt:.3f}")
        else:
            print(f"  Result: PPL = {mean_ppl:.2f} | bpt = {mean_bpt:.3f}")

        results.append({
            "arm_id": arm["id"],
            "name": arm["name"],
            "perm_method": arm["perm"],
            "transform_type": arm["transform"],
            "use_multiplicative": arm["mult"],
            "use_additive": arm["add"],
            "core_size": list(arm["core"]),
            "trainable_params": train_params,
            "adapter_size": format_adapter_size(train_params),
            "test_ppl": round(mean_ppl, 2),
            "test_ppl_std": round(std_ppl, 2),
            "bits_per_token": round(mean_bpt, 3),
            "bits_per_token_std": round(std_bpt, 3),
            "hypothesis": arm["hypothesis"],
            "runs": seed_runs,
        })

    # Summary table
    print("\n" + "=" * 130)
    print(f"      EXP-16 CAUSAL ABLATION SUMMARY (Evaluated on {test_token_count:,} tokens, seeds={seeds})")
    print("=" * 130)
    print("| Arm | Permutation   | Transform | Modes | Params   | TEST PPL (mean+/-std) | Bits/Token (mean+/-std) |")
    print("|-----|---------------|-----------|-------|----------|-----------------------|-------------------------|")
    for r in results:
        modes = f"{'M' if r['use_multiplicative'] else ''}{'+' if r['use_multiplicative'] and r['use_additive'] else ''}{'A' if r['use_additive'] else ''}"
        ppl_str = f"{r['test_ppl']:.2f} +/- {r['test_ppl_std']:.2f}" if len(seeds) > 1 else f"{r['test_ppl']:.2f}"
        bpt_str = f"{r['bits_per_token']:.3f} +/- {r['bits_per_token_std']:.3f}" if len(seeds) > 1 else f"{r['bits_per_token']:.3f}"
        print(f"|  {r['arm_id']}  | {r['perm_method']:<13} | {r['transform_type']:<9} | {modes:<5} | {r['trainable_params']:>8,} | {ppl_str:<21} | {bpt_str:<23} |")
    print("=" * 130)

    if save_json:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "exp16_results.json"))
        final_results = results
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                old_map = {r["arm_id"]: r for r in old_data.get("results", [])}
                for r in results:
                    old_map[r["arm_id"]] = r
                final_results = sorted(list(old_map.values()), key=lambda x: x["arm_id"])
            except Exception:
                final_results = results

        output_payload = {
            "experiment": "EXP-16",
            "eval_tokens": test_token_count,
            "eval_blocks": len(test_data),
            "full_test_set": full_test,
            "num_steps": num_steps,
            "seeds": seeds,
            "results": final_results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\nSaved ablation results to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EXP-16 Causal Factorial Ablation")
    parser.add_argument("--steps", type=int, default=500, help="Number of fine-tuning steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-samples", type=int, default=100, help="Number of test blocks")
    parser.add_argument("--full-test", action="store_true", help="Evaluate on full test set")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of random seeds (default: 42, recommended: 42 1337 2026)")
    parser.add_argument("--arm-ids", type=int, nargs="+", default=None, help="List of arm IDs to run (default: run all 1-10)")
    args = parser.parse_args()

    run_exp16(
        device=args.device,
        num_steps=args.steps,
        max_test_samples=args.test_samples,
        full_test=args.full_test,
        seeds=args.seeds,
        arm_ids=args.arm_ids,
    )
