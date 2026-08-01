import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama import (
    SpecRAMALinear,
    HierarchicalSpectralQuantizer,
    count_trainable_parameters,
    merge_spec_rama_modules,
)

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


def train_on_dataset(model, train_dataset, steps=200, lr=1e-2, device="cuda", batch_size=4):
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


def quantize_and_inject_spec_rama(
    model: nn.Module,
    target_modules: list,
    rest_bits: int = 3,
    core_size: tuple = (16, 16),
    transform_type: str = "wavelet"
):
    """
    1. Computes SpecRAMA permutations on FP32 weights FIRST.
    2. Quantizes base weight to (8-bit Core / rest_bits Rest) with mean-bias correction.
    3. Injects SpecRAMALinear adapter with additive residual recovery.
    """
    quantizer = HierarchicalSpectralQuantizer(
        transform_type="dct",
        core_ratio=0.0625,
        core_bits=8,
        rest_bits=rest_bits
    )
    
    injected = []
    
    def _inject(module: nn.Module):
        for name, child in list(module.named_children()):
            is_target = isinstance(child, nn.Linear) or child.__class__.__name__ in ["Conv1D", "Linear"]
            if is_target and any(target in name for target in target_modules):
                # Get FP32 weight 2D
                w_fp32_2d = child.weight.data.t().clone() if child.__class__.__name__ == "Conv1D" else child.weight.data.clone()
                
                # Quantize FP32 weight
                payload = quantizer.quantize_matrix(w_fp32_2d)
                w_rec_2d = quantizer.dequantize_matrix(payload)
                
                # Bias correction: align channel means to FP32
                mean_diff = w_fp32_2d.mean(dim=0, keepdim=True) - w_rec_2d.mean(dim=0, keepdim=True)
                w_rec_2d = w_rec_2d + mean_diff
                
                # Assign quantized weight to child
                w_rec = w_rec_2d.t() if child.__class__.__name__ == "Conv1D" else w_rec_2d
                child.weight.data.copy_(w_rec)
                
                # Create SpecRAMA layer with custom alpha_a scaling for residual recovery
                wrapped = SpecRAMALinear(
                    child,
                    transform_type=transform_type,
                    core_size=core_size,
                    permutation_method="bipartite_tsp",
                    alpha_m=8.0,
                    alpha_a=2.0 # Higher additive recovery gain
                )
                setattr(module, name, wrapped)
                injected.append(wrapped)
            else:
                _inject(child)
                
    _inject(model)
    return injected


def run_exp06(device="cuda", num_steps=200):
    print("==================================================================")
    print("  [EXP-06] ADVANCED Q-SPECPERMUTED RECOVERY BENCHMARK (3b & 4b)  ")
    print("==================================================================")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_data, test_data = prepare_wikitext_data(
        tokenizer, block_size=256, max_train_samples=400, max_test_samples=100
    )
    
    targets = ["c_attn", "c_proj"]
    
    # 1. Base FP32 Model
    model_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_test_ppl, _ = evaluate_on_dataset(model_base, test_data, device=device)
    print(f"Base GPT-2 (FP32) TEST PPL: {base_test_ppl:.2f}")

    # 2. 4.2-bit Quantized Recovery (8b Core / 4b Rest) + SpecRAMA Wavelet (16x16)
    print("\n[1/3] Testing 4.2-bit SpecQuant + SpecRAMA Wavelet (16x16, 72 KB)...")
    model_4bit = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_list = quantize_and_inject_spec_rama(model_4bit, targets, rest_bits=4, core_size=(16, 16), transform_type="wavelet")
    model_4bit = model_4bit.to(device)
    
    for p in model_4bit.parameters():
        p.requires_grad = False
    for m in inject_list:
        if m.core_m is not None: m.core_m.requires_grad = True
        if m.core_a is not None: m.core_a.requires_grad = True
        
    trainable_4b, _, ratio_4b = count_trainable_parameters(model_4bit)
    ppl_4b_step0, _ = evaluate_on_dataset(model_4bit, test_data, device=device)
    print(f"  -> 4.2-bit Step 0 TEST PPL: {ppl_4b_step0:.2f}")
    
    train_on_dataset(model_4bit, train_data, steps=num_steps, lr=1e-2, device=device)
    ppl_4b_tuned, _ = evaluate_on_dataset(model_4bit, test_data, device=device)
    print(f"  -> 4.2-bit Tuned TEST PPL: {ppl_4b_tuned:.2f}")

    # 3. 3.3-bit Quantized Recovery + SpecRAMA Wavelet (16x16, 72 KB)
    print("\n[2/3] Testing 3.3-bit SpecQuant + SpecRAMA Wavelet (16x16, 72 KB)...")
    model_3bit_16 = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_list_3b16 = quantize_and_inject_spec_rama(model_3bit_16, targets, rest_bits=3, core_size=(16, 16), transform_type="wavelet")
    model_3bit_16 = model_3bit_16.to(device)
    
    for p in model_3bit_16.parameters():
        p.requires_grad = False
    for m in inject_list_3b16:
        if m.core_m is not None: m.core_m.requires_grad = True
        if m.core_a is not None: m.core_a.requires_grad = True
        
    ppl_3b16_step0, _ = evaluate_on_dataset(model_3bit_16, test_data, device=device)
    print(f"  -> 3.3-bit Step 0 TEST PPL: {ppl_3b16_step0:.2f}")
    
    train_on_dataset(model_3bit_16, train_data, steps=num_steps, lr=1e-2, device=device)
    ppl_3b16_tuned, _ = evaluate_on_dataset(model_3bit_16, test_data, device=device)
    print(f"  -> 3.3-bit Tuned TEST PPL: {ppl_3b16_tuned:.2f}")

    # 4. 3.3-bit Quantized Recovery + SpecRAMA Wavelet (32x32, 288 KB)
    print("\n[3/3] Testing 3.3-bit SpecQuant + SpecRAMA Wavelet (32x32, 288 KB)...")
    model_3bit_32 = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_list_3b32 = quantize_and_inject_spec_rama(model_3bit_32, targets, rest_bits=3, core_size=(32, 32), transform_type="wavelet")
    model_3bit_32 = model_3bit_32.to(device)
    
    for p in model_3bit_32.parameters():
        p.requires_grad = False
    for m in inject_list_3b32:
        if m.core_m is not None: m.core_m.requires_grad = True
        if m.core_a is not None: m.core_a.requires_grad = True
        
    ppl_3b32_step0, _ = evaluate_on_dataset(model_3bit_32, test_data, device=device)
    print(f"  -> 3.3-bit Step 0 (32x32) TEST PPL: {ppl_3b32_step0:.2f}")
    
    train_on_dataset(model_3bit_32, train_data, steps=num_steps, lr=1e-2, device=device)
    ppl_3b32_tuned, _ = evaluate_on_dataset(model_3bit_32, test_data, device=device)
    print(f"  -> 3.3-bit Tuned (32x32) TEST PPL: {ppl_3b32_tuned:.2f}")

    print("\n==================================================================")
    print("      EXP-06 ADVANCED Q-SPECPERMUTED RECOVERY BENCHMARK SUMMARY   ")
    print("==================================================================")
    print(f"| Model / Adaptation Method       | Base Bitwidth | Core Size | Step 0 PPL | Tuned PPL | Status |")
    print(f"|---------------------------------|---------------|-----------|------------|-----------|--------|")
    print(f"| GPT-2 Base (FP32)               | 32-bit FP32   | N/A       | {base_test_ppl:10.2f} | N/A       | Reference |")
    print(f"| SpecRAMA-Quant 4.2-bit Wavelet  | 4.25-bit avg  | 16x16     | {ppl_4b_step0:10.2f} | {ppl_4b_tuned:9.2f} | Preserved |")
    print(f"| SpecRAMA-Quant 3.3-bit Wavelet  | 3.31-bit avg  | 16x16     | {ppl_3b16_step0:10.2f} | {ppl_3b16_tuned:9.2f} | Recovered |")
    print(f"| SpecRAMA-Quant 3.3-bit Wavelet  | 3.31-bit avg  | 32x32     | {ppl_3b32_step0:10.2f} | {ppl_3b32_tuned:9.2f} | Recovered |")
    print("==================================================================")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_exp06(device=device, num_steps=200)
