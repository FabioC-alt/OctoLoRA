"""One-off diagnostic: score the RAW base model (no LoRA at all) on the
same MMLU eval protocol, to check whether a low score after fine-tuning
reflects a real effect or a bug in the evaluation methodology itself. If
the raw base model also scores far below its expected ~65-70% published
range, the bug is in evaluate_mmlu.py, not in what was trained.
"""
import sys
sys.path.insert(0, "src")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from octolora_core import MODEL_ID
from evaluate_mmlu import evaluate_mmlu

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")

acc = evaluate_mmlu(model, tokenizer, n_examples=200)
print(f"RAW BASE MODEL (no LoRA) MMLU accuracy on 200 examples: {acc:.2%}")
