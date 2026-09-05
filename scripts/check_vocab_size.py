import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', torch_dtype=torch.bfloat16, device_map='cpu')

print('Tokenizer vocabulary size:', len(tokenizer))
print('Model config vocab size:', model.config.vocab_size)
print('Model lm_head out_features:', model.lm_head.out_features)
