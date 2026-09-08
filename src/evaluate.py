"""
GSM8K evaluation for a trained OctoLoRA checkpoint.
Shared architecture/checkpoint-loading logic lives in octolora_core.py.
"""

import argparse
import json
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from octolora_core import (
    ALPHA,
    MODEL_ID,
    RANK,
    inject_octo_lora,
    load_checkpoint_weights,
    load_run_config,
    resolve_checkpoint,
)

CHECKPOINT_PATH = "./results/octo_lora_plus/checkpoint-702"


def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\d,.-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\d,.-]+", text)
    return numbers[-1].replace(",", "") if numbers else None


def evaluate_gsm8k(model, tokenizer, n_examples=None, device="cuda", data_path="data"):
    """n_examples=None evaluates the entire test file."""
    print("Loading local evaluation dataset...", flush=True)
    with open(os.path.join(data_path, "gsm8k_test_alpaca.json")) as f:
        dataset = json.load(f)
    if n_examples is not None:
        dataset = dataset[:min(n_examples, len(dataset))]
    print(f"Loaded {len(dataset)} examples. Starting live generation loops...", flush=True)

    model.eval()
    correct = 0
    for i, example in enumerate(dataset):
        # This test file's real question lives in "instruction"; "input" is
        # always empty here (unlike the alpaca-style train file, where the
        # roles of those two fields are reversed - see train.py).
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Solve this step by step:\n{example['instruction']}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(generated)
        target = extract_answer(example["output"])

        if pred and target and pred == target:
            correct += 1

    acc = correct / len(dataset)
    print("\n=========================================", flush=True)
    print(f"OCTOLORA GSM8K FINAL ACCURACY: {correct}/{len(dataset)} = {acc:.2%}", flush=True)
    print("=========================================", flush=True)
    return acc


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an OctoLoRA checkpoint on GSM8K")
    parser.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="Number of test examples to evaluate (default: all 1319)",
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_PATH,
        help=f"Path to the checkpoint directory to evaluate (default: {CHECKPOINT_PATH})",
    )
    args = parser.parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    # Runs produced by the ablation-aware train.py write octolora_run_config.json
    # alongside the checkpoint. Older checkpoints (e.g. checkpoint-702, trained
    # before this existed) don't have it - fall back to the historical
    # defaults (gate on, rank 16, alpha 32) in that case.
    run_config = load_run_config(checkpoint_path, defaults={"use_gate": True, "rank": RANK, "alpha": ALPHA})

    print("Initializing components...", flush=True)
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Reconstructing OctoLoRA routing framework...", flush=True)
    model = inject_octo_lora(
        base_model, rank=run_config["rank"], alpha=run_config["alpha"], use_gate=run_config["use_gate"]
    )

    print(f"Loading custom weights from checkpoint: {checkpoint_path}", flush=True)
    model = load_checkpoint_weights(model, checkpoint_path)

    print("Framework generation completed. Starting evaluation pipeline...", flush=True)
    evaluate_gsm8k(model, tokenizer, n_examples=args.n_examples)
