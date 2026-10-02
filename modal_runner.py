import modal

app = modal.App("spec-rama-benchmark")

# Define container image with PyTorch & Transformers
spec_rama_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "huggingface_hub>=0.20.0",
        "numpy",
        "datasets",
        "accelerate",
        "scipy",
    )
    .add_local_dir("spec_rama", remote_path="/root/spec_rama")
    .add_local_dir("benchmarks", remote_path="/root/benchmarks")
)


@app.function(
    image=spec_rama_image,
    gpu="A10G",
    timeout=7200,
)
def run_remote_spec_rama_experiment(
    exp_name: str = "exp11",
    steps: int = 500,
    full_test: bool = False,
    test_samples: int = 100,
    seeds: str = "42",
    arm_ids: str = "",
):
    import sys

    import torch

    sys.path.insert(0, "/root")

    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    if not seed_list:
        seed_list = [42]

    print(f"=== STARTING REMOTE GPU SPEC-RAMA EXPERIMENT ({exp_name.upper()}) ON MODAL ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Seeds: {seed_list} | Steps: {steps} | Full Test: {full_test}")

    res = None
    if exp_name == "exp01":
        from benchmarks.exp01_synthetic_dummy_eval import run_exp01

        res = run_exp01(device="cuda", max_steps=50)
    elif exp_name == "exp02":
        from benchmarks.exp02_wikitext2_head2head import run_exp02

        res = run_exp02(device="cuda", num_steps=200)
    elif exp_name == "exp03":
        from benchmarks.exp03_spectral_core_scaling_sweep import run_exp03

        res = run_exp03(device="cuda", num_steps=200)
    elif exp_name == "exp04":
        from benchmarks.exp04_equal_parameter_head2head import run_exp04

        res = run_exp04(device="cuda", num_steps=200)
    elif exp_name == "exp05":
        from benchmarks.exp05_spectral_quantization_recovery import run_exp05

        res = run_exp05(device="cuda", num_steps=200)
    elif exp_name == "exp06":
        from benchmarks.exp06_qspec_rama_bias_recovery import run_exp06

        res = run_exp06(device="cuda", num_steps=200)
    elif exp_name == "exp07":
        from benchmarks.exp07_long_horizon_qspec import run_exp07

        res = run_exp07(device="cuda", num_steps=steps)
    elif exp_name == "exp08":
        from benchmarks.exp08_nf4_spec_rama_breakthrough import run_exp08

        res = run_exp08(device="cuda", num_steps=steps)
    elif exp_name == "exp09":
        from benchmarks.exp09_heterogeneous_quantization import run_exp09

        res = run_exp09(device="cuda", num_steps=steps)
    elif exp_name == "exp10":
        from benchmarks.exp10_quantization_vs_adaptation_control import run_exp10

        res = run_exp10(device="cuda", num_steps=steps)
    elif exp_name == "exp11":
        from benchmarks.exp11_proper_qlora_nf4_baseline import run_exp11

        res = run_exp11(
            device="cuda",
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seed_list,
            save_json=True,
        )
    elif exp_name == "exp12":
        from benchmarks.exp12_lora_hyperparameter_sweep import run_exp12

        res = run_exp12(
            device="cuda",
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seed_list,
            save_json=True,
        )
    elif exp_name == "exp13":
        from benchmarks.exp13_rate_distortion_sweep import run_exp13

        res = run_exp13(
            device="cuda",
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seed_list,
            save_json=True,
        )
    elif exp_name == "exp16":
        from benchmarks.exp16_causal_ablation import run_exp16

        arm_id_list = [int(x.strip()) for x in arm_ids.split(",") if x.strip()] if arm_ids else None
        res = run_exp16(
            device="cuda",
            num_steps=steps,
            max_test_samples=test_samples,
            full_test=full_test,
            seeds=seed_list,
            arm_ids=arm_id_list,
            save_json=True,
        )
    else:
        raise ValueError(f"Unknown experiment name: {exp_name}. Choose 'exp01' through 'exp13', or 'exp16'.")

    print(f"=== REMOTE SPEC-RAMA EXPERIMENT ({exp_name.upper()}) COMPLETED ===")
    return res


@app.local_entrypoint()
def main(
    exp: str = "exp11",
    steps: int = 500,
    full_test: bool = False,
    test_samples: int = 100,
    seeds: str = "42,1337,2026",
    arm_ids: str = "",
):
    import json
    import os

    print(
        f"Launching SpecRAMA Experiment [{exp}] (steps={steps}, full_test={full_test}, seeds={seeds}, arms={arm_ids or 'all'}) on Modal cloud GPU..."
    )
    result = run_remote_spec_rama_experiment.remote(
        exp_name=exp,
        steps=steps,
        full_test=full_test,
        test_samples=test_samples,
        seeds=seeds,
        arm_ids=arm_ids,
    )
    if result is not None:
        filename = f"{exp}_multiseed_results.json" if "," in seeds else f"{exp}_results.json"
        local_path = os.path.join(os.path.dirname(__file__), "benchmarks", filename)
        final_results = result
        if os.path.exists(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                old_map = {r["arm_id"]: r for r in old_data.get("results", []) if "arm_id" in r}
                for r in result:
                    if "arm_id" in r:
                        old_map[r["arm_id"]] = r
                if old_map:
                    final_results = sorted(list(old_map.values()), key=lambda x: x["arm_id"])
            except Exception:
                final_results = result

        with open(local_path, "w", encoding="utf-8") as f:
            json.dump({"experiment": exp.upper(), "seeds": seeds, "results": final_results}, f, indent=2)
        print(f"\n[Local] Saved remote results to: {local_path}")
