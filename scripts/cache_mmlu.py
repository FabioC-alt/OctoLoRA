"""One-off: populate the local HF datasets cache with MMLU so that
subsequent training/eval jobs can load it in fast, reliable offline mode
(HF_HUB_OFFLINE=1), matching how the base model itself is already cached.
Run this without the offline env vars set (see scripts/cache_mmlu.sh).
"""
from datasets import load_dataset

for split in ["auxiliary_train", "test", "dev", "validation"]:
    ds = load_dataset("cais/mmlu", "all", split=split)
    print(split, len(ds))

print("MMLU cached successfully.")
