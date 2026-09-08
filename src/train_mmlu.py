"""
OctoLoRA Training Pipeline on MMLU
-----------------------------------
Second dataset for the ablation, to check whether the GSM8K result
generalizes beyond math reasoning to multiple-choice knowledge recall.
Shares the adapter architecture/optimizer/Trainer with the GSM8K pipeline
via octolora_core.py; only the data format and evaluation protocol differ.

MMLU has no large natural "training set" analogous to GSM8K's - we use
the `auxiliary_train` split (~99.8k multiple-choice examples pooled from
other exam datasets: ARC, MC_TEST, OBQA, RACE), which is ~10.7x larger
than GSM8K's training data. To keep per-run compute in a reasonable range
while still covering the full ablation/ratio-sweep breadth, this script
defaults to 1 epoch (not GSM8K's 3) - at effective batch 32 that's still
~3120 steps, about 4.4x GSM8K's entire 3-epoch run (702 steps), so this
is already a substantial compute increase per run, not a like-for-like
match. Override with --epochs if a different budget is preferred.

Evaluation (see evaluate_mmlu.py and the shared evaluate_mmlu() below) uses
the standard MMLU protocol - 5-shot, scored via next-token log-likelihood
over the four answer letters - rather than free-form generation, both
because it's the standard/comparable way to score MMLU and because it's
far cheaper per example (one forward pass, not up to 256 generated
tokens).
"""

import argparse
import json
import os

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
from evaluate_mmlu import evaluate_mmlu

MAX_LEN    = 512
BATCH_SIZE = 4
GRAD_ACCUM = 8   # effective batch = 32
EPOCHS     = 1   # see module docstring - MMLU's train split is ~10.7x GSM8K's
VOCAB_LIMIT = 128256


def load_mmlu_train(tokenizer, max_len=MAX_LEN):
    dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")

    def format_example(example):
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]
        letter = "ABCD"[answer_idx]

        prompt = (
            "Answer the following multiple-choice question by giving only "
            f"the letter of the correct option (A, B, C, or D).\n\n{question}\n"
            f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": letter},
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

        tokenized = tokenizer(full_prompt, truncation=True, max_length=max_len, padding=False)
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()
        for idx, token_id in enumerate(labels):
            if token_id >= VOCAB_LIMIT or token_id < 0:
                labels[idx] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    mapped = dataset.map(format_example, remove_columns=dataset.column_names)
    mapped = mapped.filter(lambda x: len(x.get("input_ids", [])) > 0)
    logger.info("[Dataset Check] MMLU auxiliary_train processed, %d examples", len(mapped))
    return mapped


def parse_args():
    parser = argparse.ArgumentParser(description="Train an OctoLoRA / ablation variant on MMLU")
    parser.add_argument(
        "--gate", type=lambda s: s.lower() not in ("false", "0", "no"), default=True,
        help="Enable gradient-adaptive gating between A and B (default: true).",
    )
    parser.add_argument(
        "--lora-plus", type=lambda s: s.lower() not in ("false", "0", "no"), default=True,
        help="Enable LoRA+ split A/B learning rates (default: true).",
    )
    parser.add_argument(
        "--b-lr-ratio", type=float, default=B_LR_RATIO,
        help=f"LoRA+ B-matrix learning-rate multiplier (default: {B_LR_RATIO}).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED}).",
    )
    parser.add_argument(
        "--epochs", type=float, default=EPOCHS,
        help=f"Training epochs over auxiliary_train (default: {EPOCHS}; see module docstring for why "
             "this differs from GSM8K's default of 3).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override the checkpoint output directory (default: derived from --gate/--lora-plus/--b-lr-ratio/--seed).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = args.output_dir or f"./results/{variant_name(args.gate, args.lora_plus, args.b_lr_ratio, args.seed, prefix='mmlu_')}"
    logger.info(
        "Run variant: dataset=mmlu gate=%s lora_plus=%s b_lr_ratio=%s seed=%s epochs=%s -> output_dir=%s",
        args.gate, args.lora_plus, args.b_lr_ratio, args.seed, args.epochs, output_dir,
    )

    os.makedirs(output_dir, exist_ok=True)
    run_config = {
        "use_gate": args.gate, "use_lora_plus": args.lora_plus,
        "b_lr_ratio": args.b_lr_ratio, "seed": args.seed, "rank": RANK, "alpha": ALPHA,
        "dataset": "mmlu", "epochs": args.epochs,
    }
    with open(os.path.join(output_dir, "octolora_run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading MMLU auxiliary_train...")
    train_data = load_mmlu_train(tokenizer)
    logger.info("Found %d training samples", len(train_data))

    logger.info("Initializing Llama Foundation Architecture...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = make_forward_safe(model)
    model = inject_octo_lora(model, rank=RANK, alpha=ALPHA, use_gate=args.gate)

    total   = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable Density: %s / %s (%.4f%%)", f"{trained:,}", f"{total:,}", 100 * trained / total)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=BASE_LR,
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
    final_accuracy = evaluate_mmlu(model, tokenizer, n_examples=200)

    logger.info("Final Experiment Result Suite Complete.")
    logger.info("OctoLoRA + LoRA+ MMLU Accuracy Output: %.2f%%", final_accuracy * 100)
