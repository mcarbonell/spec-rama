import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import math
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama.lora_baseline import inject_lora_in_model
from spec_rama import count_trainable_parameters
from benchmarks.exp11_proper_qlora_nf4_baseline import (
    quantize_blockwise_nf,
    NF4_LEVELS,
    prepare_wikitext_data,
    evaluate_on_dataset,
    train_on_dataset_long_horizon
)

def run_exp12(device="cuda", num_steps=500, max_test_samples=100, full_test=False, save_json=True):
    print("=====================================================================================")
    print(" [EXP-12] LORA HYPERPARAMETER SWEEP & BASELINE SANITY AUDIT (NF4 BASE) ")
    print("=====================================================================================")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_data, test_data = prepare_wikitext_data(
        tokenizer, block_size=256, max_train_samples=600, max_test_samples=max_test_samples, full_test=full_test
    )
    test_token_count = len(test_data) * 256
    
    targets = ["c_attn", "c_proj", "c_fc"]
    
    # Test different LRs and Alphas for LoRA (r=4)
    lrs = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    alphas = [8.0]
    
    results = []
    
    for alpha in alphas:
        for lr in lrs:
            print(f"\n--- Testing Standard LoRA (r=4, alpha={alpha}) with lr_max={lr} ---")
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
            train_params, total_params, ratio = count_trainable_parameters(model_nf4_lora)
            print(f"  Trainable Parameters: {train_params:,} / {total_params:,} ({ratio:.3f}%)")
            
            lr_min = lr / 10.0
            train_on_dataset_long_horizon(
                model_nf4_lora, train_data, steps=num_steps, lr_max=lr, lr_min=lr_min, device=device, seed=42
            )
            ppl, _, bpt, _ = evaluate_on_dataset(model_nf4_lora, test_data, device=device)
            print(f"  Result: lr_max={lr:.1e} -> PPL: {ppl:.2f} ({bpt:.3f} bits/token)")
            results.append({
                "lr_max": lr,
                "alpha": alpha,
                "trainable_params": train_params,
                "test_ppl": round(ppl, 2),
                "bits_per_token": round(bpt, 3)
            })
            
    print("\n=====================================================================================")
    print(f" LORA HYPERPARAMETER SWEEP SUMMARY (Evaluated on {test_token_count:,} tokens) ")
    print("=====================================================================================")
    for r in results:
        print(f"  lr_max: {r['lr_max']:6.1e} | alpha: {r['alpha']:4.1f} | Params: {r['trainable_params']:,} | PPL: {r['test_ppl']:8.2f} | bpt: {r['bits_per_token']:6.3f}")

    if save_json:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "exp12_results.json"))
        output_payload = {
            "experiment": "EXP-12",
            "eval_tokens": test_token_count,
            "eval_blocks": len(test_data),
            "full_test_set": full_test,
            "num_steps": num_steps,
            "sweep_results": results
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\n Saved sweep results to: {output_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EXP-12 LoRA Hyperparameter Sweep")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps per run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-samples", type=int, default=100, help="Number of test blocks")
    parser.add_argument("--full-test", action="store_true", help="Evaluate on full test set")
    args = parser.parse_args()

    run_exp12(
        device=args.device,
        num_steps=args.steps,
        max_test_samples=args.test_samples,
        full_test=args.full_test
    )
