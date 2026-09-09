"""
MMLU evaluation for a trained OctoLoRA checkpoint.

Uses the standard MMLU protocol: 5-shot, scored via next-token
log-likelihood over the four answer letters (A/B/C/D), rather than
free-form generation. This is both the conventional way to score MMLU
(comparable to published numbers) and far cheaper per example than GSM8K's
generation-based scoring - one forward pass per question, not up to 256
generated tokens - which matters given MMLU's ~14k-example test set.

evaluate_mmlu() is imported directly by train_mmlu.py for its quick
post-training sanity check; this file's __main__ block is the standalone
full-test-set evaluation submitted as its own SLURM job, mirroring
evaluate.py/GSM8K.
"""

import argparse

import torch
from datasets import load_dataset
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

CHECKPOINT_PATH = "./results/mmlu_octo_lora_plus"
LETTERS = "ABCD"


def format_question(question, choices):
    return (
        f"{question}\n"
        f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n"
        "Answer:"
    )


def build_fewshot_prefix(dev_by_subject, subject, k=5):
    examples = dev_by_subject.get(subject, [])[:k]
    prefix = ""
    for ex in examples:
        letter = LETTERS[ex["answer"]]
        prefix += format_question(ex["question"], ex["choices"]) + f" {letter}\n\n"
    return prefix


def evaluate_mmlu(model, tokenizer, n_examples=None, device="cuda", k_shot=5):
    """n_examples=None evaluates the entire ~14k-example test split."""
    print("Loading MMLU test/dev splits...", flush=True)
    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")

    dev_by_subject = {}
    for ex in dev_ds:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    if n_examples is not None:
        # MMLU's test split is grouped contiguously by subject (confirmed:
        # the first 200 rows are 100% abstract_algebra + anatomy, two
        # harder-than-average subjects out of 57) - taking a subset without
        # shuffling first would silently score only whichever subjects
        # happen to sort first, not a representative cross-section.
        test_ds = test_ds.shuffle(seed=42).select(range(min(n_examples, len(test_ds))))
    print(f"Loaded {len(test_ds)} test examples. Starting log-likelihood scoring...", flush=True)

    # Letter token ids, resolved once against this tokenizer. Taking the
    # last sub-token is a standard, robust-enough fallback if " A" happens
    # to tokenize to more than one piece.
    letter_ids = [tokenizer.encode(f" {L}", add_special_tokens=False)[-1] for L in LETTERS]

    model.eval()
    correct = 0
    for i, example in enumerate(test_ds):
        subject_readable = example["subject"].replace("_", " ")
        prefix = build_fewshot_prefix(dev_by_subject, example["subject"], k=k_shot)
        # Deliberately NOT wrapped in the chat template. The standard MMLU
        # protocol - and the only thing consistent with how the few-shot
        # examples above are formatted (flowing "...Answer: X" text, not
        # separated chat turns) - is plain continuation scoring: the model
        # predicts the very next token after "Answer:" as if completing
        # the same block of text the few-shot examples are written in. An
        # earlier version of this function wrapped the prompt in Llama-3's
        # chat template and scored the first token of a *fresh assistant
        # turn* instead, which isn't primed as a continuation of "Answer:"
        # the way plain text is - that scored 38% on the raw base model
        # (should be ~65-70%), a bug caught via scripts/diagnose_mmlu_baseline.py.
        prompt = (
            f"The following are multiple choice questions (with answers) about {subject_readable}.\n\n"
            f"{prefix}{format_question(example['question'], example['choices'])}"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        next_token_logits = outputs.logits[0, -1]
        choice_logits = next_token_logits[letter_ids]
        pred = choice_logits.argmax().item()
        target = example["answer"]

        if pred == target:
            correct += 1

        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(test_ds)} scored, running accuracy {correct / (i + 1):.2%}", flush=True)

    acc = correct / len(test_ds)
    print("\n=========================================", flush=True)
    print(f"OCTOLORA MMLU FINAL ACCURACY: {correct}/{len(test_ds)} = {acc:.2%}", flush=True)
    print("=========================================", flush=True)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an OctoLoRA checkpoint on MMLU")
    parser.add_argument(
        "--n-examples", type=int, default=None,
        help="Number of test examples to evaluate (default: all ~14,042)",
    )
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH,
        help=f"Path to the checkpoint directory to evaluate (default: {CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--k-shot", type=int, default=5,
        help="Number of few-shot examples per subject (default: 5, the standard MMLU protocol).",
    )
    args = parser.parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    run_config = load_run_config(checkpoint_path, defaults={"use_gate": True, "rank": RANK, "alpha": ALPHA})

    print("Initializing components...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

    print("Reconstructing OctoLoRA routing framework...", flush=True)
    model = inject_octo_lora(
        base_model, rank=run_config["rank"], alpha=run_config["alpha"], use_gate=run_config["use_gate"]
    )

    print(f"Loading custom weights from checkpoint: {checkpoint_path}", flush=True)
    model = load_checkpoint_weights(model, checkpoint_path)

    print("Framework generation completed. Starting evaluation pipeline...", flush=True)
    evaluate_mmlu(model, tokenizer, n_examples=args.n_examples, k_shot=args.k_shot)
