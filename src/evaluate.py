import sys
import json
import os
import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIG ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CHECKPOINT_PATH = "./results/octo_lora_plus/checkpoint-702"
MAX_LEN = 512
RANK = 16
ALPHA = 32


# ─────────────────────────────────────────────────────────────
# CUSTOM LAYER SCHEMATICS
# Must stay byte-for-byte identical to train.py's OctoLoRALayer. This used
# to define a different gating mechanism (a "linearity score" heuristic)
# than the one actually used during training (gradient-norm EMA routing),
# so a trained checkpoint's A/B weights were being evaluated through a
# routing function they were never optimized under.
# ─────────────────────────────────────────────────────────────

class OctoLoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32):
        super().__init__()
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        self.base = base_layer
        self.base.requires_grad_(False)

        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        self.scale = alpha / rank

        self.register_buffer("grad_norm_A", torch.tensor(1.0))
        self.register_buffer("grad_norm_B", torch.tensor(1.0))
        self.ema = 0.9

        self.A.weight.requires_grad = True
        self.B.weight.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        mid = self.A(x)

        total = self.grad_norm_A + self.grad_norm_B + 1e-8
        a_weight = (self.grad_norm_B / total).clamp(0.1, 1.0)

        mid_scaled = mid * a_weight + mid.detach() * (1 - a_weight)
        adapter = self.B(mid_scaled) * self.scale

        return base_out + adapter


def inject_octo_lora(model, rank=16, alpha=32, target_modules=None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]

        device = module.weight.device
        dtype = module.weight.dtype

        octo = OctoLoRALayer(module, rank=rank, alpha=alpha)
        octo.to(device=device, dtype=dtype)

        setattr(parent, attr, octo)
        replaced += 1

    print(f"[OctoLoRA] Injected {replaced} custom architecture boundaries.", flush=True)
    model.config.use_cache = False
    return model


# --- 1. EVALUATION UTILS ---
def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\d,.-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\d,.-]+", text)
    return numbers[-1].replace(",", "") if numbers else None


def evaluate_gsm8k(model, tokenizer, n_examples=200, device="cuda", data_path="data"):
    print("Loading local evaluation dataset...", flush=True)
    with open(os.path.join(data_path, "gsm8k_test_alpaca.json")) as f:
        dataset = json.load(f)
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
    model = inject_octo_lora(base_model, rank=RANK, alpha=ALPHA)

    print(f"Loading custom weights from checkpoint: {CHECKPOINT_PATH}", flush=True)
    safetensors_file = os.path.join(CHECKPOINT_PATH, "model.safetensors")
    pytorch_file = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")

    if os.path.exists(safetensors_file):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_file)
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(pytorch_file):
        state_dict = torch.load(pytorch_file, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Could not locate training checkpoint files in {CHECKPOINT_PATH}")

    print("Framework generation completed. Starting evaluation pipeline...", flush=True)
    evaluate_gsm8k(model, tokenizer, n_examples=200)
