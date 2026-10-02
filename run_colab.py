"""
Spec-RAMA Colab & Local GPU Runner
Enables running all Spec-RAMA benchmarks (EXP-11, EXP-12, etc.) on Google Colab (Tesla T4) or local GPU.
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import torch


def check_environment():
    print("=" * 80)
    print(" SPEC-RAMA BENCHMARK RUNNER (CUDA / GOOGLE COLAB)")
    print("=" * 80)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Device:     {device_name} ({vram_gb:.2f} GB VRAM)")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        print("WARNING: Running on CPU. Benchmarks will be significantly slower.")
    print("=" * 80 + "\n")
    return "cuda" if cuda_available else "cpu"


def run_benchmark(
    exp_name: str,
    steps: int = 500,
    full_test: bool = False,
    test_samples: int = 100,
    seeds: Optional[List[int]] = None,
):
    device = check_environment()
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    if seeds is None:
        seeds = [42]

    t0 = time.time()

    if exp_name == "exp11":
        from benchmarks.exp11_proper_qlora_nf4_baseline import run_exp11

        print(f"--- Running EXP-11 (Proper QLoRA NF4 Baseline vs SpecRAMA) on {device.upper()} ---")
        run_exp11(
            device=device,
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seeds,
            save_json=True,
        )

    elif exp_name == "exp12":
        from benchmarks.exp12_lora_hyperparameter_sweep import run_exp12

        print(f"--- Running EXP-12 (LoRA Learning Rate Sweep with Isolated Base) on {device.upper()} ---")
        run_exp12(
            device=device,
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seeds,
            save_json=True,
        )

    elif exp_name == "exp13":
        from benchmarks.exp13_rate_distortion_sweep import run_exp13

        print(f"--- Running EXP-13 (Rate-Distortion Resolution Sweep) on {device.upper()} ---")
        run_exp13(
            device=device,
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seeds,
            save_json=True,
        )

    elif exp_name == "exp16":
        from benchmarks.exp16_causal_ablation import run_exp16

        print(f"--- Running EXP-16 (Causal Factorial Ablation) on {device.upper()} ---")
        run_exp16(
            device=device,
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seeds,
            save_json=True,
        )

    elif exp_name == "all":
        from benchmarks.exp11_proper_qlora_nf4_baseline import run_exp11
        from benchmarks.exp12_lora_hyperparameter_sweep import run_exp12
        from benchmarks.exp13_rate_distortion_sweep import run_exp13
        from benchmarks.exp16_causal_ablation import run_exp16

        print(f"--- Running FULL SUITE (EXP-11 + EXP-12 + EXP-13 + EXP-16) on {device.upper()} ---")
        print("\n>>> [1/4] Executing EXP-11...")
        run_exp11(
            device=device,
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seeds,
            save_json=True,
        )
        print("\n>>> [2/4] Executing EXP-12...")
        run_exp12(
            device=device,
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seeds,
            save_json=True,
        )
        print("\n>>> [3/4] Executing EXP-13...")
        run_exp13(
            device=device,
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seeds,
            save_json=True,
        )
        print("\n>>> [4/4] Executing EXP-16...")
        run_exp16(
            device=device,
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seeds,
            save_json=True,
        )
    else:
        raise ValueError(f"Unknown experiment '{exp_name}'. Supported: 'exp11', 'exp12', 'exp13', 'exp16', 'all'.")

    elapsed = time.time() - t0
    print(f"\n Benchmark execution completed in {elapsed:.1f} seconds ({elapsed / 60:.2f} minutes).")
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f" Peak GPU VRAM allocated: {peak_vram:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spec-RAMA benchmarks on Colab / Local GPU")
    parser.add_argument(
        "--exp", type=str, default="exp11", choices=["exp11", "exp12", "exp13", "exp16", "all"], help="Experiment to run"
    )

    parser.add_argument("--steps", type=int, default=500, help="Fine-tuning steps (default: 500)")
    parser.add_argument(
        "--test-samples", type=int, default=100, help="Number of test blocks (default: 100 = 25.6k tokens)"
    )
    parser.add_argument("--full-test", action="store_true", help="Evaluate on full WikiText-2 test set (~287k tokens)")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42], help="List of seeds (default: 42, recommended: 42 1337 2026)"
    )
    args = parser.parse_args()

    run_benchmark(
        exp_name=args.exp,
        steps=args.steps,
        full_test=args.full_test,
        test_samples=args.test_samples,
        seeds=args.seeds,
    )
