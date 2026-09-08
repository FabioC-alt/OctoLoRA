"""
Shared OctoLoRA architecture and training mechanics.
-----------------------------------------------------
This module holds everything that is *not* specific to a particular
dataset: the OctoLoRALayer itself, LoRA+ optimizer construction, the
gradient-safety Trainer subclass, gate instrumentation, and checkpoint
resolution.

It exists because train.py and evaluate.py used to each define their own
copy of OctoLoRALayer, and those two copies drifted out of sync once
(evaluate.py's used a different, untrained gating mechanism) - a real bug
that silently loaded a trained checkpoint into the wrong architecture.
Dataset-specific pipelines (GSM8K in train.py/evaluate.py, MMLU in
train_mmlu.py/evaluate_mmlu.py) import the shared pieces from here instead
of redefining them, so there is exactly one place this logic can drift.
"""

import logging
import statistics

import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback

# ─────────────────────────────────────────────────────────────
# Shared config defaults (dataset-specific scripts may override per-run
# via CLI flags, but these are the historical defaults everything else
# is compared against).
# ─────────────────────────────────────────────────────────────

MODEL_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
RANK       = 16
ALPHA      = 32
SEED       = 42
BASE_LR    = 2e-4
B_LR_RATIO = 16.0  # B matrices train with BASE_LR * B_LR_RATIO under LoRA+

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# LoRA+ optimizer builder
# ─────────────────────────────────────────────────────────────

def build_optimizer(model, lr=2e-4, use_lora_plus=True, b_lr_ratio=16.0, weight_decay=0.0):
    """
    If use_lora_plus, splits trainable weights into A/B parameter groups
    with different learning rates (LoRA+). Otherwise builds a single-group
    AdamW at a flat lr, i.e. plain LoRA optimization - the ablation
    baseline for isolating what LoRA+ contributes on its own.
    """
    if not use_lora_plus:
        params = [p for p in model.parameters() if p.requires_grad]
        logger.info("Plain optimizer built: single LR = %e for all trainable params", lr)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

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
# The OctoLoRA layer and injection
# ─────────────────────────────────────────────────────────────

class OctoLoRALayer(nn.Module):
    """A LoRA adapter whose forward pass is always plain B(A(x)) (see the
    README's "How it works" section for why the gate below never changes
    that value). When use_gate is True, the *gradient* reaching A is
    rescaled each step by a_weight, an EMA-smoothed ratio of A's and B's
    recent gradient norms - a per-layer, per-step adaptive analogue of
    LoRA+'s fixed global A/B learning-rate ratio.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32, use_gate: bool = True):
        super().__init__()
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        self.base = base_layer
        self.base.requires_grad_(False)

        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        self.scale = alpha / rank
        self.use_gate = use_gate

        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

        if self.use_gate:
            # Track historical backward scaling safely using persistent buffers
            self.register_buffer("grad_norm_A", torch.tensor(1.0))
            self.register_buffer("grad_norm_B", torch.tensor(1.0))
            self.ema = 0.9

            self.A.weight.register_hook(self._make_grad_hook("A"))
            self.B.weight.register_hook(self._make_grad_hook("B"))

        self.A.weight.requires_grad_(True)
        self.B.weight.requires_grad_(True)

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

        if self.use_gate:
            # Continuous adaptive routing calculation based on gradient histories
            total = self.grad_norm_A + self.grad_norm_B + 1e-8
            a_weight = (self.grad_norm_B / total).clamp(0.1, 1.0)
            # Scaled gradient gate flow management
            mid = mid * a_weight + mid.detach() * (1 - a_weight)

        adapter = self.B(mid) * self.scale
        return base_out + adapter


def inject_octo_lora(model, rank=16, alpha=32, target_modules=None, use_gate=True):
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
        octo = OctoLoRALayer(module, rank=rank, alpha=alpha, use_gate=use_gate).to(device=device, dtype=dtype)

        setattr(parent, attr, octo)
        replaced += 1

    logger.info("[OctoLoRA] Replaced %d layers with OctoLoRALayer (gate=%s)", replaced, use_gate)
    model.config.use_cache = False
    return model


class GateStatsCallback(TrainerCallback):
    """Logs how far the gate's a_weight actually drifts from 1.0 (full,
    ungated gradient flow) across all gated layers, at the same cadence as
    the normal training-loss logs. If a_weight never leaves ~1.0, the gate
    isn't doing anything distinguishable from not having it at all.
    """

    def __init__(self, model):
        self.gated_layers = [m for m in model.modules() if isinstance(m, OctoLoRALayer) and m.use_gate]

    def on_log(self, args, state, control, **kwargs):
        if not self.gated_layers:
            return
        a_weights = []
        for layer in self.gated_layers:
            total = layer.grad_norm_A + layer.grad_norm_B + 1e-8
            a_weight = (layer.grad_norm_B / total).clamp(0.1, 1.0)
            a_weights.append(a_weight.item())
        logger.info(
            "[Gate stats] step=%d mean_a_weight=%.4f min=%.4f max=%.4f (n_layers=%d)",
            state.global_step, statistics.mean(a_weights), min(a_weights), max(a_weights), len(a_weights),
        )


# ─────────────────────────────────────────────────────────────
# Trainer subclass: LoRA+ optimizer wiring + out-of-range token guard
# ─────────────────────────────────────────────────────────────

class OctoLoraPlusTrainer(Trainer):
    """Custom Trainer subclass overriding optimizer creation and forcing strict tensor safety."""

    def __init__(self, *args, use_lora_plus=True, b_lr_ratio=16.0, **kwargs):
        self.use_lora_plus = use_lora_plus
        self.b_lr_ratio = b_lr_ratio
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = build_optimizer(
                self.model,
                lr=self.args.learning_rate,
                use_lora_plus=self.use_lora_plus,
                b_lr_ratio=self.b_lr_ratio,
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


def make_forward_safe(model):
    import functools

    original_forward = model.forward

    @functools.wraps(original_forward)
    def safe_forward(*args, **kwargs):
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


# ─────────────────────────────────────────────────────────────
# Run naming / checkpoint resolution shared by both dataset pipelines
# ─────────────────────────────────────────────────────────────

def variant_name(use_gate, use_lora_plus, b_lr_ratio=B_LR_RATIO, seed=SEED, prefix=""):
    if use_gate and use_lora_plus:
        name = "octo_lora_plus"
    elif use_lora_plus:
        name = "lora_plus_only"
    elif use_gate:
        name = "gate_only"
    else:
        name = "vanilla_lora"
    if b_lr_ratio != B_LR_RATIO:
        name += f"_blr{b_lr_ratio:g}"
    if seed != SEED:
        name += f"_seed{seed}"
    return f"{prefix}{name}" if prefix else name


def resolve_checkpoint(path: str) -> str:
    """Accepts either a specific checkpoint-N directory or a run's output_dir
    (e.g. results/vanilla_lora) and returns the actual checkpoint directory,
    picking the highest-step checkpoint-N subdirectory if given the latter.
    """
    import os

    has_weights = os.path.exists(os.path.join(path, "model.safetensors")) or \
        os.path.exists(os.path.join(path, "pytorch_model.bin"))
    if has_weights:
        return path

    candidates = [
        d for d in os.listdir(path)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(path, d))
    ] if os.path.isdir(path) else []
    if not candidates:
        raise FileNotFoundError(
            f"{path} has no model weights and no checkpoint-N subdirectories"
        )
    latest = max(candidates, key=lambda d: int(d.split("-")[-1]))
    return os.path.join(path, latest)


def load_checkpoint_weights(model, checkpoint_path: str):
    """Loads trained A/B weights (safetensors preferred, .bin fallback) into
    a freshly-injected OctoLoRA model. strict=False since the frozen base
    model's own weights aren't in the checkpoint.
    """
    import os

    safetensors_file = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_file = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_file):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_file)
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(pytorch_file):
        state_dict = torch.load(pytorch_file, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Could not locate training checkpoint files in {checkpoint_path}")
    return model


def load_run_config(checkpoint_path: str, defaults: dict) -> dict:
    """Reads octolora_run_config.json from a checkpoint dir or its parent
    run directory, falling back to `defaults` for any run predating that
    file (e.g. the very first GSM8K checkpoint-702).
    """
    import json
    import os

    run_config = dict(defaults)
    for config_dir in (checkpoint_path, os.path.dirname(checkpoint_path.rstrip("/"))):
        config_path = os.path.join(config_dir, "octolora_run_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                run_config.update(json.load(f))
            logger.info("Loaded run config from %s: %s", config_path, run_config)
            return run_config
    logger.info("No octolora_run_config.json found near %s; using defaults: %s", checkpoint_path, run_config)
    return run_config
