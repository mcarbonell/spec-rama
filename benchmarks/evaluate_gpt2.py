import math
import os
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama import (
    SharedSpecRAMAModel,
    count_trainable_parameters,
    inject_spec_rama_in_model,
    merge_spec_rama_modules,
)


def get_dummy_tokens(tokenizer, batch_size=8, seq_len=128):
    text = "The principles of compact spectral representation and parameter-efficient fine-tuning allow deep neural networks to adapt rapidly without overfitting."
    tokens = tokenizer.encode(text, return_tensors="pt")
    # Repeat to fill batch
    input_ids = tokens.repeat(batch_size, seq_len // tokens.shape[1] + 1)[:, :seq_len]
    return input_ids

def evaluate_ppl(model, input_ids, device="cuda"):
    model.eval()
    with torch.no_grad():
        inputs = input_ids.to(device)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss.item()
        ppl = math.exp(loss) if loss < 20 else float('inf')
    return ppl, loss

def train_steps(model, input_ids, steps=50, lr=1e-2, device="cuda"):
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    inputs = input_ids.to(device)

    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    return loss.item()

def run_benchmark(device="cuda", max_steps=50):
    print("==========================================================")
    print(f"      RUNNING REMOTE SPEC-RAMA BENCHMARK ON {device.upper()}")
    print("==========================================================")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = get_dummy_tokens(tokenizer, batch_size=4, seq_len=128)

    targets = ["c_attn", "c_proj"]

    # 1. Base FP32 Model
    model_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_ppl, base_loss = evaluate_ppl(model_base, input_ids, device=device)

    print(f"Original GPT-2 Baseline PPL: {base_ppl:.4f} (Loss: {base_loss:.4f})")
    print("\n--- SpecRAMA Parameter Efficiency & Fine-Tuning ---")

    configs = [
        ("SpecRAMA (DCT 8x8)", "dct", (8, 8)),
        ("SpecRAMA (FWHT 8x8)", "walsh", (8, 8)),
        ("SpecRAMA (Wavelet 8x8)", "wavelet", (8, 8)),
    ]

    results = []

    for name, transform, core_size in configs:
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        # Freeze ALL base parameters first!
        for p in model.parameters():
            p.requires_grad = False

        inject_spec_rama_in_model(model, target_modules=targets, transform_type=transform, core_size=core_size)
        model = model.to(device)
        trainable, total, ratio = count_trainable_parameters(model)

        # Step 0 PPL
        ppl_step0, _ = evaluate_ppl(model, input_ids, device=device)

        # 50-step fine-tuning of small spectral cores
        final_loss = train_steps(model, input_ids, steps=max_steps, lr=1e-2, device=device)
        ppl_tuned, _ = evaluate_ppl(model, input_ids, device=device)

        # Merge zero latency
        merge_spec_rama_modules(model)
        ppl_merged, _ = evaluate_ppl(model, input_ids, device=device)

        results.append((name, trainable, ratio, ppl_step0, ppl_tuned, ppl_merged))
        print(f"  * {name:22s} | Trainable Params: {trainable:7,d} ({ratio:.4f}%) | Step0 PPL: {ppl_step0:.2f} | Tuned PPL: {ppl_tuned:.2f} | Merged PPL: {ppl_merged:.2f}")

    # Sub-KB Shared Core Model
    model_shared_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in model_shared_base.parameters():
        p.requires_grad = False

    model_shared = SharedSpecRAMAModel(model_shared_base, target_modules=targets, transform_type="dct", core_size=(8, 8))
    model_shared = model_shared.to(device)
    trainable_s, total_s, ratio_s = count_trainable_parameters(model_shared)
    ppl_s_step0, _ = evaluate_ppl(model_shared, input_ids, device=device)
    train_steps(model_shared, input_ids, steps=max_steps, lr=1e-2, device=device)
    ppl_s_tuned, _ = evaluate_ppl(model_shared, input_ids, device=device)

    print(f"  * {'Shared Core (Sub-KB)':22s} | Trainable Params: {trainable_s:7,d} ({ratio_s:.5f}%) | Step0 PPL: {ppl_s_step0:.2f} | Tuned PPL: {ppl_s_tuned:.2f}")

    print("\n==========================================================")
    print("                BENCHMARK SUMMARY RESULTS                 ")
    print("==========================================================")
    print("| Model Configuration     | Trainable Params | % of Total | Step 0 PPL | Tuned PPL | Merged PPL |")
    print("|-------------------------|------------------|------------|------------|-----------|------------|")
    print(f"| GPT-2 Base (FP32)       | 124,439,808      | 100.00%    | {base_ppl:.2f}      | N/A       | N/A        |")
    for name, trainable, ratio, step0, tuned, merged in results:
        print(f"| {name:23s} | {trainable:16,d} | {ratio:9.4f}% | {step0:10.2f} | {tuned:9.2f} | {merged:10.2f} |")
    print(f"| Shared Core (Sub-KB)    | {trainable_s:16,d} | {ratio_s:9.5f}% | {ppl_s_step0:10.2f} | {ppl_s_tuned:9.2f} | N/A        |")
    print("==========================================================")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_benchmark(device=device, max_steps=50)
