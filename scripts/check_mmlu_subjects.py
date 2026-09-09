from collections import Counter
from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all", split="test").select(range(200))
print(Counter(ds["subject"]))
