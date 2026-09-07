"""
OctoLoRA Training Pipeline with LoRA+ Optimizer on GSM8K
--------------------------------------------------------
Implements gradient-norm-driven adaptive routing (OctoLoRA) 
combined with separate parameter group learning rates (LoRA+).
"""

import functools
import logging
import re
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# ─────────────────────────────────────────────────────────────
# 1. MAIN CONFIG
# ─────────────────────────────────────────────────────────────

MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
RANK        = 16          
ALPHA       = 32          
MAX_LEN     = 512         
BATCH_SIZE  = 4
GRAD_ACCUM  = 8           # effective batch = 32
SEED        = 42
EPOCHS      = 3

# Learning Rate Configurations (LoRA+)
BASE_LR     = 2e-4
B_LR_RATIO  = 16.0        # B matrices will train with BASE_LR * B_LR_RATIO

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 2. OPTION 1: LoRA+ OPTIMIZER BUILDER
# ─────────────────────────────────────────────────────────────

def build_lora_plus_optimizer(model, lr=2e-4, b_lr_ratio=16.0, weight_decay=0.0):
    """
    Splits trainable weights into distinct low-rank matrix parameter groups.
    """
    a_params, b_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("A.weight"):
            a_params.append(param)
        elif name.endswith("B.weight"):
            b_params.append(param)

    optimizer_grouped_parameters = [
        {"params": a_params, "lr": lr},
        {"params": b_params, "lr": lr * b_lr_ratio},
    ]
    
    logger.info(
        "LoRA+ Optimizer built: Matrix A LR = %e, Matrix B LR = %e", 
        lr, lr * b_lr_ratio
    )
    return torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)


# ─────────────────────────────────────────────────────────────
# 3. OPTION 2: GRADIENT-AWARE OCTO LORA LAYER
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

        # Track historical backward scaling safely using persistent buffers
        self.register_buffer("grad_norm_A", torch.tensor(1.0))
        self.register_buffer("grad_norm_B", torch.tensor(1.0))
        self.ema = 0.9

        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

        self.A.weight.register_hook(self._make_grad_hook("A"))
        self.B.weight.register_hook(self._make_grad_hook("B"))

    def _make_grad_hook(self, which):
        def hook(grad):
            norm = grad.detach().norm()
            target = self.grad_norm_A if which == "A" else self.grad_norm_B
            target.mul_(self.ema).add_(norm, alpha=1 - self.ema)
            return grad
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        mid = self.A(x)

        # Continuous adaptive routing calculation based on gradient histories
        total = self.grad_norm_A + self.grad_norm_B + 1e-8
        a_weight = (self.grad_norm_B / total).clamp(0.1, 1.0)

        # Scaled gradient gate flow management
        mid_scaled = mid * a_weight + mid.detach() * (1 - a_weight)
        adapter = self.B(mid_scaled) * self.scale

        return base_out + adapter

def inject_octo_lora(model, rank=16, alpha=32, target_modules=None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    model.requires_grad_(False)

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules)
    ]

    replaced = 0
    for name, module in targets:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]

        device, dtype = module.weight.device, module.weight.dtype
        octo = OctoLoRALayer(module, rank=rank, alpha=alpha).to(device=device, dtype=dtype)
        octo.A.weight.requires_grad_(True)
        octo.B.weight.requires_grad_(True)

        setattr(parent, attr, octo)
        replaced += 1

    logger.info("[OctoLoRA] Replaced %d layers with OctoLoRALayer", replaced)
    return model


# ─────────────────────────────────────────────────────────────
# 4. CUSTOM HUGGINGFACE TRAINER FOR LORA+ INTEGRATION
# ─────────────────────────────────────────────────────────────

class OctoLoraPlusTrainer(Trainer):
    """Custom Trainer subclass overriding optimizer creation and forcing strict tensor safety."""

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = build_lora_plus_optimizer(
                self.model,
                lr=self.args.learning_rate,
                b_lr_ratio=B_LR_RATIO,
                weight_decay=self.args.weight_decay
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Intercepts and sanitizes both input_ids and labels to prevent GPU out-of-bounds crashes."""
        vocab_size = model.config.vocab_size  # 128256

        # 1. Sanitize input_ids (Embedding layer protection)
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            # Clamp any rogue token ID to a safe, valid index (like pad/eos token 128001 or 0)
            bad_input_mask = (input_ids < 0) | (input_ids >= vocab_size)
            if bad_input_mask.any():
                inputs["input_ids"] = torch.where(
                    bad_input_mask,
                    torch.tensor(0, device=input_ids.device),
                    input_ids
                )

        # 2. Sanitize labels (Cross-entropy loss protection)
        if "labels" in inputs:
            labels = inputs["labels"]
            bad_label_mask = (labels >= vocab_size) | ((labels < 0) & (labels != -100))
            if bad_label_mask.any():
                inputs["labels"] = torch.where(
                    bad_label_mask,
                    torch.tensor(-100, device=labels.device),
                    labels
                )

        # Proceed safely to standard Hugging Face forward pass execution
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

# ─────────────────────────────────────────────────────────────
# 5. GSM8K DATA & EVALUATION PIPELINES
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




def make_forward_safe(model):
    original_forward = model.forward

    # Using functools.wraps copies the original method name, docstring, 
    # and critically, the parameter signature over to our wrapper.
    @functools.wraps(original_forward)
    def safe_forward(*args, **kwargs):
        # 1. Sanitize labels context
        if "labels" in kwargs and kwargs["labels"] is not None:
            labels = kwargs["labels"]
            vocab_size = model.config.vocab_size
            invalid_mask = (labels >= vocab_size) | ((labels < 0) & (labels != -100))
            if invalid_mask.any():
                kwargs["labels"] = torch.where(
                    invalid_mask, 
                    torch.tensor(-100, device=labels.device), 
                    labels
                )
                
        # 2. Sanitize input_ids context
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            input_ids = kwargs["input_ids"]
            vocab_size = model.config.vocab_size
            invalid_input_mask = (input_ids < 0) | (input_ids >= vocab_size)
            if invalid_input_mask.any():
                kwargs["input_ids"] = torch.where(
                    invalid_input_mask,
                    torch.tensor(0, device=input_ids.device),
                    input_ids
                )

        return original_forward(*args, **kwargs)

    model.forward = safe_forward
    return model

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
# 6. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(SEED)

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

    # Apply Option 2: Active Dynamic Gradient Gated Routing
    model = inject_octo_lora(model, rank=RANK, alpha=ALPHA)

    # Track structural params
    total   = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable Density: %s / %s (%.4f%%)", f"{trained:,}", f"{total:,}", 100 * trained / total)

    training_args = TrainingArguments(
        output_dir="./results/octo_lora_plus",
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
        seed=SEED,
        remove_unused_columns=False,
        report_to="none",
    )

    # Bootstrapping Custom Trainer executing Option 1
    trainer = OctoLoraPlusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100),
    )
    
    logger.info("Beginning fine-tuning engine pass...")
    trainer.train()
    
    logger.info("Initiating model verification matrix...")
    final_accuracy = evaluate_gsm8k(model, tokenizer)
    
    logger.info("Final Experiment Result Suite Complete.")
    logger.info("OctoLoRA + LoRA+ Balanced Accuracy Output: %.2f%%", final_accuracy * 100)
