import modal
import os

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
    )
    .add_local_dir("spec_rama", remote_path="/root/spec_rama")
    .add_local_dir("benchmarks", remote_path="/root/benchmarks")
)

@app.function(
    image=spec_rama_image,
    gpu="A10G",
    timeout=1800,
)
def run_remote_spec_rama_experiment(exp_name: str = "exp11"):
    import sys
    import torch
    
    sys.path.insert(0, "/root")
    
    print(f"=== STARTING REMOTE GPU SPEC-RAMA EXPERIMENT ({exp_name.upper()}) ON MODAL ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
    if exp_name == "exp01":
        from benchmarks.exp01_synthetic_dummy_eval import run_exp01
        run_exp01(device="cuda", max_steps=50)
    elif exp_name == "exp02":
        from benchmarks.exp02_wikitext2_head2head import run_exp02
        run_exp02(device="cuda", num_steps=200)
    elif exp_name == "exp03":
        from benchmarks.exp03_spectral_core_scaling_sweep import run_exp03
        run_exp03(device="cuda", num_steps=200)
    elif exp_name == "exp04":
        from benchmarks.exp04_equal_parameter_head2head import run_exp04
        run_exp04(device="cuda", num_steps=200)
    elif exp_name == "exp05":
        from benchmarks.exp05_spectral_quantization_recovery import run_exp05
        run_exp05(device="cuda", num_steps=200)
    elif exp_name == "exp06":
        from benchmarks.exp06_qspec_rama_bias_recovery import run_exp06
        run_exp06(device="cuda", num_steps=200)
    elif exp_name == "exp07":
        from benchmarks.exp07_long_horizon_qspec import run_exp07
        run_exp07(device="cuda", num_steps=500)
    elif exp_name == "exp08":
        from benchmarks.exp08_nf4_spec_rama_breakthrough import run_exp08
        run_exp08(device="cuda", num_steps=500)
    elif exp_name == "exp09":
        from benchmarks.exp09_heterogeneous_quantization import run_exp09
        run_exp09(device="cuda", num_steps=500)
    elif exp_name == "exp10":
        from benchmarks.exp10_quantization_vs_adaptation_control import run_exp10
        run_exp10(device="cuda", num_steps=500)
    elif exp_name == "exp11":
        from benchmarks.exp11_proper_qlora_nf4_baseline import run_exp11
        run_exp11(device="cuda", num_steps=500)
    else:
        raise ValueError(f"Unknown experiment name: {exp_name}. Choose 'exp01' through 'exp11'.")
        
    print(f"=== REMOTE SPEC-RAMA EXPERIMENT ({exp_name.upper()}) COMPLETED ===")

@app.local_entrypoint()
def main(exp: str = "exp11"):
    print(f"Launching SpecRAMA Experiment [{exp}] on Modal cloud GPU...")
    run_remote_spec_rama_experiment.remote(exp_name=exp)
