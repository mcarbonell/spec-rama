import math
import os
import sys

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama import (
    SharedSpecRAMAModel,
    count_trainable_parameters,
    inject_spec_rama_in_model,
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


def run_exp04(device="cuda", num_steps=200):
    print("==================================================================")
    print("  [EXP-04] EXACT ISO-PARAMETER HEAD-TO-HEAD: SPEC-RAMA VS LORA    ")
    print("==================================================================")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_data, test_data = prepare_wikitext_data(
        tokenizer, block_size=256, max_train_samples=400, max_test_samples=100
    )

    targets = ["c_attn", "c_proj"]

    # Baseline FP32
    model_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_test_ppl, _ = evaluate_on_dataset(model_base, test_data, device=device)
    print(f"Base GPT-2 (FP32) TEST PPL: {base_test_ppl:.2f}")

    # Exact Equal Parameter Budget Pairs
    configs = [
        # Tier 0: Sub-50K Budget (Where LoRA cannot exist!)
        ("SpecRAMA Shared (Sub-KB)", "shared", None, 0, 1e-2),
        ("SpecRAMA Wavelet (8x8)", "wavelet", (8, 8), 0, 1e-2),

        # Tier 1: EXACT ~101K Parameter Budget Pair (LoRA r=1 vs SpecRAMA 37x38)
        ("LoRA (rank=1, alpha=2)", "lora", None, 1, 1e-3),
        ("SpecRAMA Wavelet (37x38)", "wavelet", (37, 38), 0, 1e-2),
        ("SpecRAMA DCT (37x38)", "dct", (37, 38), 0, 1e-2),

        # Tier 2: EXACT ~405K Parameter Budget Pair (LoRA r=4 vs SpecRAMA 75x75)
        ("LoRA (rank=4, alpha=8)", "lora", None, 4, 1e-3),
        ("SpecRAMA Wavelet (75x75)", "wavelet", (75, 75), 0, 1e-2),
    ]

    results = []

    for name, method, core_size, rank, lr in configs:
        print(f"\nTraining {name}...")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        for p in model.parameters():
            p.requires_grad = False

        if method == "shared":
            model = SharedSpecRAMAModel(model, target_modules=targets, transform_type="wavelet", core_size=(8, 8))
            model = model.to(device)
        elif method == "lora":
            inject_lora_in_model(model, target_modules=targets, rank=rank, alpha=rank * 2.0)
            model = model.to(device)
        else:
            inject_spec_rama_in_model(model, target_modules=targets, transform_type=method, core_size=core_size)
            model = model.to(device)

        trainable, total, ratio = count_trainable_parameters(model)

        train_on_dataset(model, train_data, steps=num_steps, lr=lr, device=device)
        test_ppl, _ = evaluate_on_dataset(model, test_data, device=device)

        if method in ["wavelet", "dct"]:
            merge_spec_rama_modules(model)
            merged_ppl, _ = evaluate_on_dataset(model, test_data, device=device)
        elif method == "lora":
            for m in model.modules():
                if hasattr(m, "merge"):
                    m.merge()
            merged_ppl, _ = evaluate_on_dataset(model, test_data, device=device)
        else:
            merged_ppl = test_ppl

        size_bytes = trainable * 4
        size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024*1024 else f"{size_bytes / (1024*1024):.2f} MB"

        results.append((name, trainable, ratio, size_str, test_ppl, merged_ppl))
        print(f"  -> {name} | Trainable: {trainable:,} ({ratio:.4f}%) [{size_str}] | TEST PPL: {test_ppl:.2f}")

    print("\n==================================================================")
    print("      EXP-04 EXACT ISO-PARAMETER BENCHMARK SUMMARY                ")
    print("==================================================================")
    print("| Model / Adaptation Method  | Trainable Params | % Total | Adapter Size | TEST PPL | Merged PPL |")
    print("|----------------------------|------------------|---------|--------------|----------|------------|")
    print(f"| GPT-2 Base (FP32)          | 124,439,808      | 100.00% | N/A          | {base_test_ppl:8.2f} | N/A        |")
    for name, trainable, ratio, size_str, test_ppl, merged_ppl in results:
        print(f"| {name:26s} | {trainable:16,d} | {ratio:6.4f}% | {size_str:12s} | {test_ppl:8.2f} | {merged_ppl:10.2f} |")
    print("==================================================================")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_exp04(device=device, num_steps=200)
