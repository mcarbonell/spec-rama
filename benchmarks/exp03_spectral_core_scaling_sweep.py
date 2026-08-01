import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama import (
    inject_spec_rama_in_model,
    count_trainable_parameters,
    merge_spec_rama_modules,
)
from spec_rama.lora_baseline import inject_lora_in_model

def prepare_wikitext_data(tokenizer, block_size=256, max_train_samples=400, max_test_samples=100):
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
    test_data = lm_datasets["test"].select(range(min(len(lm_datasets["test"]), max_test_samples)))
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
    return ppl, avg_loss


def train_on_dataset(model, train_dataset, steps=200, lr=1e-3, device="cuda", batch_size=4):
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
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
            step += 1
            
    return loss.item()


def run_exp03(device="cuda", num_steps=200):
    print("==================================================================")
    print("   [EXP-03] SPECTRAL CORE RESOLUTION SWEEP (8x8 to 64x64) ON GPU   ")
    print("==================================================================")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_data, test_data = prepare_wikitext_data(
        tokenizer, block_size=256, max_train_samples=400, max_test_samples=100
    )
    
    targets = ["c_attn", "c_proj"]
    
    # 1. Base FP32 GPT-2
    model_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_test_ppl, _ = evaluate_on_dataset(model_base, test_data, device=device)
    print(f"Base GPT-2 (FP32) TEST PPL: {base_test_ppl:.2f}")

    # 2. LoRA r=8 Baseline
    model_lora = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in model_lora.parameters():
        p.requires_grad = False
    inject_lora_in_model(model_lora, target_modules=targets, rank=8, alpha=16.0)
    model_lora = model_lora.to(device)
    lora_trainable, _, lora_ratio = count_trainable_parameters(model_lora)
    train_on_dataset(model_lora, train_data, steps=num_steps, lr=1e-3, device=device)
    lora_test_ppl, _ = evaluate_on_dataset(model_lora, test_data, device=device)

    # 3. Spectral Core Resolution Sweep
    sweep_configs = [
        ("SpecRAMA Wavelet (8x8)", "wavelet", (8, 8), 1e-2),
        ("SpecRAMA Wavelet (16x16)", "wavelet", (16, 16), 1e-2),
        ("SpecRAMA Wavelet (32x32)", "wavelet", (32, 32), 1e-2),
        ("SpecRAMA Wavelet (64x64)", "wavelet", (64, 64), 1e-2),
        ("SpecRAMA DCT (32x32)", "dct", (32, 32), 1e-2),
    ]

    results = []

    for name, transform, core_size, lr in sweep_configs:
        print(f"\nTraining {name} (Core Size={core_size})...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        for p in model.parameters():
            p.requires_grad = False
            
        inject_spec_rama_in_model(model, target_modules=targets, transform_type=transform, core_size=core_size)
        model = model.to(device)
        trainable, total, ratio = count_trainable_parameters(model)
        
        train_on_dataset(model, train_data, steps=num_steps, lr=lr, device=device)
        test_ppl, _ = evaluate_on_dataset(model, test_data, device=device)
        
        merge_spec_rama_modules(model)
        merged_ppl, _ = evaluate_on_dataset(model, test_data, device=device)
        
        # Calculate adapter size in KB/MB
        size_bytes = trainable * 4
        size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024*1024 else f"{size_bytes / (1024*1024):.2f} MB"
        
        results.append((name, trainable, ratio, size_str, test_ppl, merged_ppl))
        print(f"  -> {name} | Trainable: {trainable:,} ({ratio:.4f}%) [{size_str}] | TEST PPL: {test_ppl:.2f}")

    print("\n==================================================================")
    print("       EXP-03 CORE RESOLUTION SWEEP BENCHMARK SUMMARY             ")
    print("==================================================================")
    print(f"| Model Configuration     | Trainable Params | % Total | Adapter Size | TEST PPL | Merged PPL |")
    print(f"|-------------------------|------------------|---------|--------------|----------|------------|")
    print(f"| GPT-2 Base (FP32)       | 124,439,808      | 100.00% | N/A          | {base_test_ppl:8.2f} | N/A        |")
    print(f"| LoRA (rank=8, alpha=16) | {lora_trainable:16,d} | {lora_ratio:6.3f}% | 3.10 MB      | {lora_test_ppl:8.2f} | N/A        |")
    for name, trainable, ratio, size_str, test_ppl, merged_ppl in results:
        print(f"| {name:23s} | {trainable:16,d} | {ratio:6.4f}% | {size_str:12s} | {test_ppl:8.2f} | {merged_ppl:10.2f} |")
    print("==================================================================")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_exp03(device=device, num_steps=200)
