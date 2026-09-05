# OctoLoRA

OctoLoRA fine-tunes `meta-llama/Meta-Llama-3.1-8B-Instruct` on GSM8K grade-school
math problems using a hand-rolled, gradient-adaptive variant of LoRA, combined
with a LoRA+ optimizer (separate learning rates for the LoRA `A` and `B`
matrices). It trains and evaluates on a SLURM GPU cluster.

## How it works

Instead of using an existing PEFT/LoRA library, `src/train.py` implements its
own adapter layer, `OctoLoRALayer`, which replaces the attention projections
(`q_proj`, `v_proj`, `k_proj`, `o_proj`) of the frozen base model:

- Each layer adds a low-rank `B(A(x))` update on top of the frozen base
  projection, as in standard LoRA.
- Unlike standard LoRA, it tracks an EMA of the gradient norm flowing into
  `A` and `B` separately, and uses the ratio between them to gate how much of
  `A`'s output is allowed to flow into `B` on the forward pass (see
  `_make_grad_hook` and the `a_weight` blend in `OctoLoRALayer.forward`).
  The intent is to let the layer adapt which matrix dominates learning at
  a given point in training, instead of a fixed rank/alpha split.
- On top of this, `build_lora_plus_optimizer` puts `A.weight` and `B.weight`
  parameters into separate optimizer groups so `B` trains at a higher
  learning rate (`B_LR_RATIO`), following the
  [LoRA+](https://arxiv.org/abs/2402.12354) paper.
- `OctoLoraPlusTrainer` (a `transformers.Trainer` subclass) wires the custom
  optimizer in and clamps any out-of-range token ids in `input_ids`/`labels`
  before the forward pass, as a guard against embedding/index crashes.

## Repository layout

```
src/
  train.py     # OctoLoRALayer, LoRA+ optimizer, GSM8K data loading, training loop
  evaluate.py  # Reloads a checkpoint and scores it on the GSM8K test set
scripts/
  submit_octolora.sh   # SLURM batch script for the training run
  check_vocab_size.py  # Diagnostic: compares tokenizer vs model vocab size
data/
  gsm8k_train_alpaca.json  # Training set, alpaca-style fields
  gsm8k_test_alpaca.json   # Test set (different field layout - see below)
fake_cuda/
  bin/nvcc     # Fake nvcc stub so DeepSpeed skips CUDA compilation on nodes without a CUDA toolkit
```

## Data format (important: train and test use fields differently)

Both files use `instruction` / `input` / `output` keys, but **the two files
assign different meaning to `instruction` vs `input`**:

| file | `instruction` | `input` | `output` |
|---|---|---|---|
| `gsm8k_train_alpaca.json` | fixed boilerplate, identical on every row ("Solve the following math problem step-by-step.") | the actual math question | full chain-of-thought solution ending in `#### <answer>` |
| `gsm8k_test_alpaca.json` | the actual math question | always empty | bare final answer, e.g. `"18"` |

`train.py` and `evaluate.py` read the field that holds the real question for
each file respectively (`input` for train, `instruction` for test) — this was
not the case before this pass; see "Bugs found and fixed" below.

## Running it

```bash
pip install -r requirements.txt
# on the cluster:
sbatch scripts/submit_octolora.sh
# locally, once trained:
python src/evaluate.py
```

Secrets are read from the environment, not hardcoded:

```bash
export TELEGRAM_BOT_TOKEN=...   # optional, for the completion notification
export TELEGRAM_CHAT_ID=...
```

## Bugs found and fixed in this pass

1. **Training never saw the real question (the main reason the model
   couldn't learn).** `load_gsm8k` in `train.py` built every training prompt
   from `example["instruction"]`, which is the *same fixed string* on all
   ~9k rows. The actual problem text, in `example["input"]`, was discarded.
   The model was effectively being trained to map a constant prompt to
   arbitrary answers, which is not a learnable function of the question. Now
   it reads `input` (falling back to `instruction` if empty).
2. **Evaluation inside `train.py` crashed after training.** Its
   `evaluate_gsm8k` read `example["answer"]`, a key that doesn't exist in
   `gsm8k_test_alpaca.json` (the field is `output`) — this raised a
   `KeyError` at the end of every training run, before the final accuracy
   was ever logged. Fixed to read `output`.
3. **`evaluate.py` prompted the model with an empty question.** It built the
   test prompt from `example["input"]`, which is always `""` in the test
   file (the real question there is in `instruction`). Standalone evaluation
   would have scored the model on blank prompts. Fixed to read
   `instruction`.
4. **`evaluate.py` reconstructed a different adapter than the one trained.**
   Its `OctoLoRALayer` used a "linearity score" gating heuristic, while
   `train.py`'s version (the one actually optimized) used gradient-norm EMA
   gating. The two classes don't share buffer names (`lin_A`/`lin_B` vs
   `grad_norm_A`/`grad_norm_B`), so loading a checkpoint with
   `strict=False` silently skipped those buffers and ran inference through
   a routing function the weights were never trained under. `evaluate.py`
   now defines the identical class used in `train.py`.
5. **Checkpoint path mismatch.** Training writes to
   `./results/octo_lora_plus/...` but `evaluate.py` looked in
   `./results/octo_lora/checkpoint-702` — a different directory name that
   would 404 on a fresh run. Aligned to `octo_lora_plus`.
6. **Hardcoded Telegram bot token in `submit_octolora.sh`,** committed and
   already pushed to the public GitHub remote. Replaced with
   `$TELEGRAM_BOT_TOKEN`/`$TELEGRAM_CHAT_ID` env vars.
   **Action needed from you:** revoke/regenerate that bot token via
   [@BotFather](https://t.me/BotFather) — removing it from the latest commit
   does not remove it from git history, and it must be treated as already
   compromised since the repo is public.

## Can this work, and how to improve it further

The core idea (mixing LoRA+'s per-matrix learning rates with a gradient-adaptive
gate) is plausible and now trains on the intended data, but a few things are
still worth checking/improving before trusting the results:

- **Get one real baseline number first.** Fix #1 above changes what the
  model actually learns from — the old accuracy figures (if any were
  recorded) are not meaningful. Re-run training and evaluation from scratch
  before tuning anything else.
- **The training loss is computed over the whole sequence, not just the
  answer.** `labels = input_ids.copy()` in `load_gsm8k` means the loss
  includes predicting the user's question tokens, not only the assistant's
  response. Standard instruction-tuning practice masks the prompt with
  `-100` so gradient signal focuses on the answer. Worth trying, likely a
  moderate quality improvement.
- **Get an ablation against plain LoRA.** Since there's no comparison point,
  it's hard to know whether the gradient-adaptive gating or the LoRA+
  optimizer actually help versus a vanilla LoRA (fixed r/alpha, one learning
  rate) baseline. Training a plain-LoRA run (or using `peft`'s `LoraConfig`
  directly) on the same fixed data would isolate whether OctoLoRA's routing
  is pulling its weight.
- **The token-id clamping in `compute_loss`/`make_forward_safe` is a
  band-aid, not a fix.** Silently zeroing out-of-range token ids can mask a
  real tokenizer/model vocab mismatch (this is exactly what
  `scripts/check_vocab_size.py` was written to investigate) without any
  logging of how often it fires. Worth running that diagnostic and, if
  mismatches are frequent, tracking down the root cause rather than
  clamping.
- **No held-out validation during training.** `eval_strategy="no"` means the
  only signal during the whole run is training loss; you won't see
  overfitting until the final GSM8K eval. Splitting off a small validation
  slice and evaluating periodically would catch regressions earlier and
  cheaper than a full post-hoc GSM8K pass.
- **`peft` and `trl` are in `requirements.txt` but unused** — the project
  hand-rolls LoRA injection instead. Fine as a design choice for the custom
  routing, but worth pruning the dependency list, or using `peft` as the
  baseline comparison mentioned above.
