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
    SharedSpecRAMAModel,
    HierarchicalSpectralQuantizer,
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
    epoch = 0
    
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
            
        epoch += 1
        
    return loss.item()


def run_exp02(device="cuda", num_steps=200):
    print("==================================================================")
    print("   [EXP-02] HEAD-TO-HEAD BENCHMARK: SPEC-RAMA VS. LORA ON WIKITEXT-2 ")
    print("==================================================================")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_data, test_data = prepare_wikitext_data(
        tokenizer, block_size=256, max_train_samples=400, max_test_samples=100
    )
    
    targets = ["c_attn", "c_proj"]
    
    # 1. GPT-2 Base FP32
    print("\n[1/6] Evaluating Base GPT-2 (FP32) on WikiText-2 TEST set...")
    model_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_test_ppl, _ = evaluate_on_dataset(model_base, test_data, device=device)
    print(f"  -> Base GPT-2 (FP32) TEST PPL: {base_test_ppl:.2f}")

    # 2. Standard LoRA (rank=8)
    print("\n[2/6] Training Standard LoRA (rank=8, alpha=16)...")
    model_lora = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in model_lora.parameters():
        p.requires_grad = False
    inject_lora_in_model(model_lora, target_modules=targets, rank=8, alpha=16.0)
    model_lora = model_lora.to(device)
    lora_trainable, _, lora_ratio = count_trainable_parameters(model_lora)
    
    train_on_dataset(model_lora, train_data, steps=num_steps, lr=1e-3, device=device)
    lora_test_ppl, _ = evaluate_on_dataset(model_lora, test_data, device=device)
    print(f"  -> LoRA (rank=8) | Trainable Params: {lora_trainable:,} ({lora_ratio:.4f}%) | TEST PPL: {lora_test_ppl:.2f}")

    # 3. SpecRAMA (DCT 8x8)
    print("\n[3/6] Training SpecRAMA (DCT 8x8)...")
    model_dct = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in model_dct.parameters():
        p.requires_grad = False
    inject_spec_rama_in_model(model_dct, target_modules=targets, transform_type="dct", core_size=(8, 8))
    model_dct = model_dct.to(device)
    dct_trainable, _, dct_ratio = count_trainable_parameters(model_dct)
    
    train_on_dataset(model_dct, train_data, steps=num_steps, lr=1e-2, device=device)
    dct_test_ppl, _ = evaluate_on_dataset(model_dct, test_data, device=device)
    
    merge_spec_rama_modules(model_dct)
    dct_merged_ppl, _ = evaluate_on_dataset(model_dct, test_data, device=device)
    print(f"  -> SpecRAMA (DCT) | Trainable Params: {dct_trainable:,} ({dct_ratio:.4f}%) | TEST PPL: {dct_test_ppl:.2f} | Merged PPL: {dct_merged_ppl:.2f}")

    # 4. SpecRAMA (Wavelet 8x8)
    print("\n[4/6] Training SpecRAMA (Wavelet 8x8)...")
    model_wav = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in model_wav.parameters():
        p.requires_grad = False
    inject_spec_rama_in_model(model_wav, target_modules=targets, transform_type="wavelet", core_size=(8, 8))
    model_wav = model_wav.to(device)
    wav_trainable, _, wav_ratio = count_trainable_parameters(model_wav)
    
    train_on_dataset(model_wav, train_data, steps=num_steps, lr=1e-2, device=device)
    wav_test_ppl, _ = evaluate_on_dataset(model_wav, test_data, device=device)
    
    merge_spec_rama_modules(model_wav)
    wav_merged_ppl, _ = evaluate_on_dataset(model_wav, test_data, device=device)
    print(f"  -> SpecRAMA (Wavelet) | Trainable Params: {wav_trainable:,} ({wav_ratio:.4f}%) | TEST PPL: {wav_test_ppl:.2f} | Merged PPL: {wav_merged_ppl:.2f}")

    # 5. Shared Core (Sub-KB)
    print("\n[5/6] Training Shared Core Model (Sub-KB)...")
    model_shared_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in model_shared_base.parameters():
        p.requires_grad = False
    model_shared = SharedSpecRAMAModel(model_shared_base, target_modules=targets, transform_type="dct", core_size=(8, 8))
    model_shared = model_shared.to(device)
    shared_trainable, _, shared_ratio = count_trainable_parameters(model_shared)
    
    train_on_dataset(model_shared, train_data, steps=num_steps, lr=1e-2, device=device)
    shared_test_ppl, _ = evaluate_on_dataset(model_shared, test_data, device=device)
    print(f"  -> Shared Core (Sub-KB) | Trainable Params: {shared_trainable:,} ({shared_ratio:.5f}%) | TEST PPL: {shared_test_ppl:.2f}")

    # 6. SpecQuant (3.5-bit Hierarchical Zero-Shot Quantization)
    print("\n[6/6] Evaluating SpecQuant (3.5-bit Hierarchical Zero-Shot Quantization)...")
    model_quant_base = GPT2LMHeadModel.from_pretrained("gpt2")
    quantizer = HierarchicalSpectralQuantizer(transform_type="dct", core_ratio=0.0625, core_bits=8, rest_bits=4)
    
    with torch.no_grad():
        for name, module in model_quant_base.named_modules():
            if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                payload = quantizer.quantize_matrix(w_2d)
                w_rec_2d = quantizer.dequantize_matrix(payload)
                w_rec = w_rec_2d.t() if module.__class__.__name__ == "Conv1D" else w_rec_2d
                module.weight.copy_(w_rec)
                
    model_quant = model_quant_base.to(device)
    quant_test_ppl, _ = evaluate_on_dataset(model_quant, test_data, device=device)
    print(f"  -> SpecQuant (3.5 bits/weight) TEST PPL: {quant_test_ppl:.2f}")

    print("\n==================================================================")
    print("           EXP-02 WIKITEXT-2 TEST SET BENCHMARK RESULTS           ")
    print("==================================================================")
    print(f"| Model / Adaptation Method  | Trainable Params | % Total | TEST PPL | Merged PPL |")
    print(f"|----------------------------|------------------|---------|----------|------------|")
    print(f"| GPT-2 Base (FP32)          | 124,439,808      | 100.00% | {base_test_ppl:8.2f} | N/A        |")
    print(f"| LoRA (rank=8, alpha=16)    | {lora_trainable:16,d} | {lora_ratio:6.3f}% | {lora_test_ppl:8.2f} | N/A        |")
    print(f"| SpecRAMA (DCT 8x8)         | {dct_trainable:16,d} | {dct_ratio:6.4f}% | {dct_test_ppl:8.2f} | {dct_merged_ppl:10.2f} |")
    print(f"| SpecRAMA (Wavelet 8x8)     | {wav_trainable:16,d} | {wav_ratio:6.4f}% | {wav_test_ppl:8.2f} | {wav_merged_ppl:10.2f} |")
    print(f"| Shared Core (Sub-KB)       | {shared_trainable:16,d} | {shared_ratio:6.5f}% | {shared_test_ppl:8.2f} | N/A        |")
    print(f"| SpecQuant (3.5 bit Zero)   | 0 (Quantized)    |   0.00% | {quant_test_ppl:8.2f} | N/A        |")
    print("==================================================================")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_exp02(device=device, num_steps=200)
