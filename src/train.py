"""
OctoLoRA Training Pipeline with LoRA+ Optimizer on GSM8K
--------------------------------------------------------
GSM8K-specific data loading, evaluation, and CLI. The shared adapter
architecture, optimizer, and Trainer live in octolora_core.py - see that
module's docstring for why they're factored out.

Supports ablating the two ideas independently via --gate/--lora-plus,
so the same script can produce the vanilla-LoRA, LoRA+-only,
gate-only, and full-OctoLoRA runs needed to isolate what each part
contributes.
"""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from octolora_core import (
    ALPHA,
    BASE_LR,
    B_LR_RATIO,
    MODEL_ID,
    RANK,
    SEED,
    GateStatsCallback,
    OctoLoraPlusTrainer,
    inject_octo_lora,
    logger,
    make_forward_safe,
    variant_name,
)

MAX_LEN    = 512
BATCH_SIZE = 4
GRAD_ACCUM = 8  # effective batch = 32
EPOCHS     = 3


# ─────────────────────────────────────────────────────────────
# GSM8K DATA & EVALUATION PIPELINES
# ─────────────────────────────────────────────────────────────

def load_gsm8k(tokenizer, data_files, split="train", max_len=MAX_LEN):
    dataset = load_dataset("json", data_files=data_files, split=split)

    def format_example(example):
        # In this dataset's alpaca schema, "input" holds the actual math
        # question and "instruction" is a fixed boilerplate string
        # ("Solve the following math problem step-by-step.") repeated on
        # every row. Reading "instruction" here would train the model on a
        # constant, uninformative prompt instead of the real question.
        question = example.get("input") or example.get("instruction")
        output = example.get("output")

        if not question or not output:
            # Fallback to avoid crashing the data collator down the line
            question = "Empty question fallback"
            output = "Empty answer fallback"

        messages = [
            {"role": "user", "content": f"Solve this step by step:\n{question}"},
            {"role": "assistant", "content": output}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

        tokenized = tokenizer(
            full_prompt, truncation=True, max_length=max_len, padding=False
        )

        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()

        # Sanitize target boundaries
        VOCAB_LIMIT = 128256
        for idx, token_id in enumerate(labels):
            if token_id >= VOCAB_LIMIT or token_id < 0:
                labels[idx] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

    # Map dataset and completely drop old raw columns
    mapped_dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # Strict validation filter: drop any row that somehow ended up empty
    mapped_dataset = mapped_dataset.filter(lambda x: len(x.get("input_ids", [])) > 0)

    logger.info("[Dataset Check] Processed dataset features contain keys: %s", list(mapped_dataset[0].keys()))
    return mapped_dataset

def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\d,.-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\d,.-]+", text)
    return numbers[-1].replace(",", "") if numbers else None


def evaluate_gsm8k(model, tokenizer, n_examples=200, device="cuda"):
    # data_files must be a dict here: passing a bare string/list always
    # names the resulting split "train", regardless of the file's content,
    # so split="test" would fail to find anything.
    dataset = load_dataset(
        "json", data_files={"test": "data/gsm8k_test_alpaca.json"}, split="test"
    )

    dataset = dataset.select(range(min(n_examples, len(dataset))))

    model.eval()
    correct = 0

    for example in dataset:

        question = example["instruction"]


        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Solve this step by step:\n{question}"
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

        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        pred   = extract_answer(generated)
        target = extract_answer(example["output"])

        if pred and target and pred == target:
            correct += 1

    acc = correct / len(dataset)
    logger.info("GSM8K Evaluation completeness: %d/%d Correct", correct, len(dataset))
    return acc


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train an OctoLoRA / ablation variant on GSM8K")
    parser.add_argument(
        "--gate", type=lambda s: s.lower() not in ("false", "0", "no"), default=True,
        help="Enable gradient-adaptive gating between A and B (default: true). "
             "--gate false trains plain LoRA up/down projections.",
    )
    parser.add_argument(
        "--lora-plus", type=lambda s: s.lower() not in ("false", "0", "no"), default=True,
        help="Enable LoRA+ split A/B learning rates (default: true). "
             "--lora-plus false uses a single flat learning rate for all trainable params.",
    )
    parser.add_argument(
        "--b-lr-ratio", type=float, default=B_LR_RATIO,
        help=f"LoRA+ B-matrix learning-rate multiplier (default: {B_LR_RATIO}). "
             "Only matters when --lora-plus is on. Lower this to test whether the "
             "default ratio is too aggressive and overfitting the training set.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed for init/data order/Trainer (default: {SEED}). "
             "Use a different value per run when checking whether a result "
             "holds up across seeds rather than being a lucky single run.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override the checkpoint output directory (default: derived from --gate/--lora-plus/--b-lr-ratio/--seed).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = args.output_dir or f"./results/{variant_name(args.gate, args.lora_plus, args.b_lr_ratio, args.seed)}"
    logger.info(
        "Run variant: gate=%s lora_plus=%s b_lr_ratio=%s seed=%s -> output_dir=%s",
        args.gate, args.lora_plus, args.b_lr_ratio, args.seed, output_dir,
    )

    # Persist the run config alongside the checkpoints so evaluate.py can
    # reconstruct the exact same architecture later without guessing.
    os.makedirs(output_dir, exist_ok=True)
    run_config = {
        "use_gate": args.gate, "use_lora_plus": args.lora_plus,
        "b_lr_ratio": args.b_lr_ratio, "seed": args.seed, "rank": RANK, "alpha": ALPHA,
    }
    with open(os.path.join(output_dir, "octolora_run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info("Loading tokenizers...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading GSM8K datasets...")
    train_files = "data/gsm8k_train_alpaca.json"
    train_data = load_gsm8k(tokenizer, train_files)
    logger.info("Found %d training samples", len(train_data))

    logger.info("Initializing Llama Foundation Architecture...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    model = make_forward_safe(model)

    model = inject_octo_lora(model, rank=RANK, alpha=ALPHA, use_gate=args.gate)

    # Track structural params
    total   = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable Density: %s / %s (%.4f%%)", f"{trained:,}", f"{total:,}", 100 * trained / total)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=BASE_LR,  # Passed down as initial reference for Group A
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        seed=args.seed,
        remove_unused_columns=False,
        report_to="none",
    )

    # Bootstrapping Custom Trainer executing Option 1
    trainer = OctoLoraPlusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100),
        use_lora_plus=args.lora_plus,
        b_lr_ratio=args.b_lr_ratio,
    )
    if args.gate:
        trainer.add_callback(GateStatsCallback(model))


    logger.info("Beginning fine-tuning engine pass...")
    trainer.train()

    logger.info("Initiating model verification matrix...")
    final_accuracy = evaluate_gsm8k(model, tokenizer)

    logger.info("Final Experiment Result Suite Complete.")
    logger.info("OctoLoRA + LoRA+ Balanced Accuracy Output: %.2f%%", final_accuracy * 100)
