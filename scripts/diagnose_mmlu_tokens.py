"""One-off diagnostic: for a handful of real MMLU questions, print the
model's actual top-10 predicted next tokens (with their probabilities) at
the scoring position, so we can see with our own eyes whether letter
tokens are even meaningfully represented - rather than continuing to
guess at the scoring mechanism from accuracy numbers alone.
"""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from octolora_core import MODEL_ID
from evaluate_mmlu import format_question, build_fewshot_prefix, LETTERS
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
model.eval()

test_ds = load_dataset("cais/mmlu", "all", split="test").select(range(5))
dev_ds = load_dataset("cais/mmlu", "all", split="dev")
dev_by_subject = {}
for ex in dev_ds:
    dev_by_subject.setdefault(ex["subject"], []).append(ex)

letter_ids = [tokenizer.encode(f" {L}", add_special_tokens=False)[-1] for L in LETTERS]
print("Letter token ids:", dict(zip(LETTERS, letter_ids)))
for L, tid in zip(LETTERS, letter_ids):
    print(f"  token id {tid} decodes to: {repr(tokenizer.decode([tid]))}")

for example in test_ds:
    subject_readable = example["subject"].replace("_", " ")
    prefix = build_fewshot_prefix(dev_by_subject, example["subject"], k=5)
    prompt = (
        f"The following are multiple choice questions (with answers) about {subject_readable}.\n\n"
        f"{prefix}{format_question(example['question'], example['choices'])}"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    probs = F.softmax(logits.float(), dim=-1)

    top_probs, top_ids = probs.topk(10)
    print("\n" + "=" * 60)
    print("Question:", example["question"][:100])
    print("Correct answer:", LETTERS[example["answer"]])
    print("Prompt tail (last 200 chars):", repr(prompt[-200:]))
    print("Top 10 next-token predictions:")
    for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
        print(f"    {p:.4f}  token {tid} = {repr(tokenizer.decode([tid]))}")
    letter_probs = probs[letter_ids]
    pred = LETTERS[letter_probs.argmax().item()]
    print(f"Among A/B/C/D only: {dict(zip(LETTERS, letter_probs.tolist()))} -> predicted {pred}")
