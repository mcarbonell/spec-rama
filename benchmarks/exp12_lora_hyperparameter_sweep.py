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
    prepare_wikitext_data,
    quantize_blockwise_nf,
    train_on_dataset_long_horizon,
)
from spec_rama import assert_strictly_frozen_base, count_trainable_parameters
from spec_rama.lora_baseline import inject_lora_in_model


def run_exp12(
    device: str = "cuda",
    num_steps: int = 500,
    max_test_samples: int = 100,
    full_test: bool = False,
    seeds: Optional[List[int]] = None,
    save_json: bool = True,
):
    if seeds is None:
        seeds = [42]

    print("=" * 95)
    print(" [EXP-12] LORA HYPERPARAMETER SWEEP & BASELINE SANITY AUDIT (NF4 BASE)")
    print(f" Seeds: {seeds} | Full Test: {full_test} | Steps: {num_steps} | Device: {device.upper()}")
    print("=" * 95)

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

    # Test different LRs for LoRA (r=4)
    lrs = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    alpha = 8.0

    results = []

    for lr in lrs:
        print(f"\n--- Testing Standard LoRA (r=4, alpha={alpha}) with lr_max={lr} across {len(seeds)} seeds ---")
        seed_runs = []
        train_params = 0

        for s_idx, s in enumerate(seeds):
            torch.manual_seed(s)
            model_nf4_lora = GPT2LMHeadModel.from_pretrained("gpt2")
            with torch.no_grad():
                for name, module in model_nf4_lora.named_modules():
                    if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                        w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                        w_q_2d = quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
                        w_rec = w_q_2d.t() if module.__class__.__name__ == "Conv1D" else w_q_2d
                        module.weight.copy_(w_rec)

            inject_lora_in_model(model_nf4_lora, target_modules=targets, rank=4, alpha=alpha, freeze_base=True)
            model_nf4_lora = model_nf4_lora.to(device)

            # Strictly verify freeze
            assert_strictly_frozen_base(model_nf4_lora, allowed_substrings=["lora_A", "lora_B"])

            train_params, total_params, ratio = count_trainable_parameters(model_nf4_lora)
            if s_idx == 0:
                print(f"  Trainable Parameters: {train_params:,} / {total_params:,} ({ratio:.3f}%)")

            lr_min = lr / 10.0
            train_on_dataset_long_horizon(
                model_nf4_lora,
                train_data,
                steps=num_steps,
                lr_max=lr,
                lr_min=lr_min,
                device=device,
                seed=s,
            )
            ppl, _, bpt, _ = evaluate_on_dataset(model_nf4_lora, test_data, device=device)
            seed_runs.append({"seed": s, "test_ppl": ppl, "bits_per_token": bpt})
            if len(seeds) > 1:
                print(f"    Seed {s} -> PPL: {ppl:.2f} ({bpt:.3f} bpt)")

        ppls = [r["test_ppl"] for r in seed_runs]
        bpts = [r["bits_per_token"] for r in seed_runs]

        mean_ppl = float(np.mean(ppls))
        std_ppl = float(np.std(ppls)) if len(ppls) > 1 else 0.0
        mean_bpt = float(np.mean(bpts))
        std_bpt = float(np.std(bpts)) if len(bpts) > 1 else 0.0

        if len(seeds) > 1:
            print(
                f"  Result lr_max={lr:.1e}: PPL = {mean_ppl:.2f} +/- {std_ppl:.2f} ({mean_bpt:.3f} +/- {std_bpt:.3f} bpt)"
            )
        else:
            print(f"  Result lr_max={lr:.1e}: PPL = {mean_ppl:.2f} ({mean_bpt:.3f} bpt)")

        results.append(
            {
                "lr_max": lr,
                "alpha": alpha,
                "trainable_params": train_params,
                "test_ppl": round(mean_ppl, 2),
                "test_ppl_std": round(std_ppl, 2),
                "bits_per_token": round(mean_bpt, 3),
                "bits_per_token_std": round(std_bpt, 3),
                "runs": seed_runs,
            }
        )

    print("\n" + "=" * 95)
    print(f" LORA HYPERPARAMETER SWEEP SUMMARY (Evaluated on {test_token_count:,} tokens, seeds={seeds})")
    print("=" * 95)
    if len(seeds) > 1:
        for r in results:
            print(
                f"  lr_max: {r['lr_max']:6.1e} | alpha: {r['alpha']:4.1f} | Params: {r['trainable_params']:,} | PPL: {r['test_ppl']:6.2f} +/- {r['test_ppl_std']:4.2f} | bpt: {r['bits_per_token']:5.3f} +/- {r['bits_per_token_std']:5.3f}"
            )
    else:
        for r in results:
            print(
                f"  lr_max: {r['lr_max']:6.1e} | alpha: {r['alpha']:4.1f} | Params: {r['trainable_params']:,} | PPL: {r['test_ppl']:8.2f} | bpt: {r['bits_per_token']:6.3f}"
            )
    print("=" * 95)

    if save_json:
        filename = "exp12_multiseed_results.json" if len(seeds) > 1 else "exp12_results.json"
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", filename))
        output_payload = {
            "experiment": "EXP-12",
            "eval_tokens": test_token_count,
            "eval_blocks": len(test_data),
            "full_test_set": full_test,
            "num_steps": num_steps,
            "seeds": seeds,
            "sweep_results": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\nSaved sweep results to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EXP-12 LoRA Hyperparameter Sweep")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps per run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-samples", type=int, default=100, help="Number of test blocks")
    parser.add_argument("--full-test", action="store_true", help="Evaluate on full test set")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="List of random seeds (default: 42, recommended: 42 1337 2026)",
    )
    args = parser.parse_args()

    run_exp12(
        device=args.device,
        num_steps=args.steps,
        max_test_samples=args.test_samples,
        full_test=args.full_test,
        seeds=args.seeds,
    )
