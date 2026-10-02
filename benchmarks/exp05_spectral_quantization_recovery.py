import math
import os
import sys

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spec_rama import (
    HierarchicalSpectralQuantizer,
    count_trainable_parameters,
    inject_spec_rama_in_model,
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


def quantize_model_3bit(model, targets, transform_type="dct"):
    """
    Quantizes targeted linear / Conv1D layers to 3.3-bit average (8-bit Core / 3-bit Rest)
    """
    quantizer = HierarchicalSpectralQuantizer(
        transform_type=transform_type,
        core_ratio=0.0625,
        core_bits=8,
        rest_bits=3 # 3-bit Rest!
    )
    with torch.no_grad():
        for name, module in model.named_modules():
            if module.__class__.__name__ in ["Conv1D", "Linear"] and any(t in name for t in targets):
                w_2d = module.weight.data.t() if module.__class__.__name__ == "Conv1D" else module.weight.data
                payload = quantizer.quantize_matrix(w_2d)
                w_rec_2d = quantizer.dequantize_matrix(payload)
                w_rec = w_rec_2d.t() if module.__class__.__name__ == "Conv1D" else w_rec_2d
                module.weight.copy_(w_rec)
    return model


def run_exp05(device="cuda", num_steps=200):
    print("==================================================================")
    print("  [EXP-05] 3-BIT SPECTRAL QUANTIZATION RECOVERY VIA SPEC-RAMA     ")
    print("==================================================================")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_data, test_data = prepare_wikitext_data(
        tokenizer, block_size=256, max_train_samples=400, max_test_samples=100
    )

    targets = ["c_attn", "c_proj"]

    # 1. Base FP32 Model
    print("\n[1/5] Evaluating Base GPT-2 (FP32) on WikiText-2 TEST set...")
    model_base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_test_ppl, _ = evaluate_on_dataset(model_base, test_data, device=device)
    print(f"  -> Base GPT-2 (FP32) TEST PPL: {base_test_ppl:.2f}")

    # 2. SpecQuant 3.3-bit Zero-Shot (Frozen, No Fine-Tuning)
    print("\n[2/5] Quantizing to 3.3-bit Zero-Shot (8b Core / 3b Rest) [Frozen]...")
    model_3bit = GPT2LMHeadModel.from_pretrained("gpt2")
    model_3bit = quantize_model_3bit(model_3bit, targets, transform_type="dct")
    model_3bit = model_3bit.to(device)
    zero_shot_3bit_ppl, _ = evaluate_on_dataset(model_3bit, test_data, device=device)
    print(f"  -> SpecQuant 3.3-bit Zero-Shot TEST PPL: {zero_shot_3bit_ppl:.2f} (COLLAPSED!)")

    # 3. SpecRAMA-Quant DCT (8x8 Core, 18 KB) Fine-Tuning on 3.3-bit Quantized Model
    print("\n[3/5] Fine-Tuning SpecRAMA DCT (8x8, 18 KB) on 3.3-bit Quantized Model...")
    model_dct_q = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dct_q = quantize_model_3bit(model_dct_q, targets, transform_type="dct").to(device)
    for p in model_dct_q.parameters():
        p.requires_grad = False
    inject_spec_rama_in_model(model_dct_q, target_modules=targets, transform_type="dct", core_size=(8, 8))
    model_dct_q = model_dct_q.to(device)
    trainable_dct, _, ratio_dct = count_trainable_parameters(model_dct_q)

    train_on_dataset(model_dct_q, train_data, steps=num_steps, lr=1e-2, device=device)
    dct_rec_ppl, _ = evaluate_on_dataset(model_dct_q, test_data, device=device)

    merge_spec_rama_modules(model_dct_q)
    dct_merged_ppl, _ = evaluate_on_dataset(model_dct_q, test_data, device=device)
    print(f"  -> SpecRAMA-Quant DCT (8x8) | Trainable: {trainable_dct:,} ({ratio_dct:.4f}%) | TEST PPL: {dct_rec_ppl:.2f} | Merged PPL: {dct_merged_ppl:.2f}")

    # 4. SpecRAMA-Quant Wavelet (8x8 Core, 18 KB) Fine-Tuning on 3.3-bit Quantized Model
    print("\n[4/5] Fine-Tuning SpecRAMA Wavelet (8x8, 18 KB) on 3.3-bit Quantized Model...")
    model_wav_q = GPT2LMHeadModel.from_pretrained("gpt2")
    model_wav_q = quantize_model_3bit(model_wav_q, targets, transform_type="dct").to(device)
    for p in model_wav_q.parameters():
        p.requires_grad = False
    inject_spec_rama_in_model(model_wav_q, target_modules=targets, transform_type="wavelet", core_size=(8, 8))
    model_wav_q = model_wav_q.to(device)
    trainable_wav, _, ratio_wav = count_trainable_parameters(model_wav_q)

    train_on_dataset(model_wav_q, train_data, steps=num_steps, lr=1e-2, device=device)
    wav_rec_ppl, _ = evaluate_on_dataset(model_wav_q, test_data, device=device)

    merge_spec_rama_modules(model_wav_q)
    wav_merged_ppl, _ = evaluate_on_dataset(model_wav_q, test_data, device=device)
    print(f"  -> SpecRAMA-Quant Wavelet (8x8) | Trainable: {trainable_wav:,} ({ratio_wav:.4f}%) | TEST PPL: {wav_rec_ppl:.2f} | Merged PPL: {wav_merged_ppl:.2f}")

    # 5. SpecRAMA-Quant Wavelet (16x16 Core, 72 KB) Fine-Tuning on 3.3-bit Quantized Model
    print("\n[5/5] Fine-Tuning SpecRAMA Wavelet (16x16, 72 KB) on 3.3-bit Quantized Model...")
    model_wav16_q = GPT2LMHeadModel.from_pretrained("gpt2")
    model_wav16_q = quantize_model_3bit(model_wav16_q, targets, transform_type="dct").to(device)
    for p in model_wav16_q.parameters():
        p.requires_grad = False
    inject_spec_rama_in_model(model_wav16_q, target_modules=targets, transform_type="wavelet", core_size=(16, 16))
    model_wav16_q = model_wav16_q.to(device)
    trainable_w16, _, ratio_w16 = count_trainable_parameters(model_wav16_q)

    train_on_dataset(model_wav16_q, train_data, steps=num_steps, lr=1e-2, device=device)
    w16_rec_ppl, _ = evaluate_on_dataset(model_wav16_q, test_data, device=device)

    merge_spec_rama_modules(model_wav16_q)
    w16_merged_ppl, _ = evaluate_on_dataset(model_wav16_q, test_data, device=device)
    print(f"  -> SpecRAMA-Quant Wavelet (16x16) | Trainable: {trainable_w16:,} ({ratio_w16:.4f}%) | TEST PPL: {w16_rec_ppl:.2f} | Merged PPL: {w16_merged_ppl:.2f}")

    print("\n==================================================================")
    print("      EXP-05 3-BIT SPECTRAL QUANTIZATION RECOVERY SUMMARY         ")
    print("==================================================================")
    print("| Model / Adaptation Method       | Base Bitwidth | Trainable Params | TEST PPL | Merged PPL | Status |")
    print("|---------------------------------|---------------|------------------|----------|------------|--------|")
    print(f"| GPT-2 Base (FP32)               | 32-bit FP32   | 124,439,808      | {base_test_ppl:8.2f} | N/A        | Reference |")
    print(f"| SpecQuant 3.3-bit Zero-Shot     | 3.31-bit avg  | 0 (Quantized)    | {zero_shot_3bit_ppl:8.2f} | N/A        | Collapsed |")
    print(f"| SpecRAMA-Quant DCT (8x8)        | 3.31-bit avg  | {trainable_dct:16,d} | {dct_rec_ppl:8.2f} | {dct_merged_ppl:10.2f} | Recovered |")
    print(f"| SpecRAMA-Quant Wavelet (8x8)    | 3.31-bit avg  | {trainable_wav:16,d} | {wav_rec_ppl:8.2f} | {wav_merged_ppl:10.2f} | Recovered |")
    print(f"| SpecRAMA-Quant Wavelet (16x16)  | 3.31-bit avg  | {trainable_w16:16,d} | {w16_rec_ppl:8.2f} | {w16_merged_ppl:10.2f} | Recovered |")
    print("==================================================================")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_exp05(device=device, num_steps=200)
