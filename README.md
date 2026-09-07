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
- **The gate is not a Mixture-of-Experts / routing mechanism**, despite the
  name — there's only ever one `A`/`B` pair per layer, and the gated and
  ungated forward passes produce numerically identical output (`mid *
  a_weight + mid.detach() * (1 - a_weight)` always equals `mid`, since
  `.detach()` only affects the gradient, not the value). What it actually
  does is control **how much gradient reaches `A` during backprop**, based
  on an EMA of `A` vs `B`'s recent gradient norms (`_make_grad_hook` in
  `OctoLoRALayer`). So the honest description is: a per-layer, per-step
  *adaptive* version of LoRA+'s idea, not a routing/expert-selection method.
- `build_optimizer` (LoRA+ mode) puts `A.weight` and `B.weight` parameters
  into separate optimizer groups so `B` trains at a higher, but *fixed*,
  learning rate (`B_LR_RATIO`), following the
  [LoRA+](https://arxiv.org/abs/2402.12354) paper. OctoLoRA's gate is best
  understood as trying to make that fixed ratio adaptive per layer instead.
- `OctoLoraPlusTrainer` (a `transformers.Trainer` subclass) wires the custom
  optimizer in and clamps any out-of-range token ids in `input_ids`/`labels`
  before the forward pass, as a guard against embedding/index crashes.
- The gate and the LoRA+ optimizer are independently toggleable via
  `--gate`/`--lora-plus` for ablation (see "Ablations" below).

## Repository layout

```
src/
  train.py     # OctoLoRALayer, LoRA+ optimizer, GSM8K data loading, training loop
  evaluate.py  # Reloads a checkpoint and scores it on the GSM8K test set
scripts/
  submit_octolora.sh   # SLURM batch script for the training run
  submit_evaluate.sh   # SLURM batch script for scoring a checkpoint on GSM8K
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
sbatch scripts/submit_octolora.sh    # training
sbatch scripts/submit_evaluate.sh    # evaluation, once a checkpoint exists
```

Before running `submit_evaluate.sh`, check which checkpoint directory
training actually produced (`ls results/<variant>/`) and pass it as the
first argument, and optionally a subset size as the second:

```bash
sbatch scripts/submit_evaluate.sh results/octo_lora_plus/checkpoint-702
sbatch scripts/submit_evaluate.sh results/vanilla_lora/checkpoint-702 200
```

## Ablations

`src/train.py` can isolate the two ideas in OctoLoRA independently via
`--gate` and `--lora-plus` (both default to `true`, matching the original
behavior):

| variant | `--gate` | `--lora-plus` | what it tests |
|---|---|---|---|
| `vanilla_lora` | false | false | plain LoRA baseline |
| `lora_plus_only` | false | true | LoRA+'s fixed A/B learning-rate split alone |
| `gate_only` | true | false | the gradient-adaptive gate alone, flat learning rate |
| `octo_lora_plus` | true | true | both combined (the default) |

```bash
sbatch scripts/submit_octolora.sh --gate false --lora-plus false   # vanilla LoRA
sbatch scripts/submit_octolora.sh --gate false                     # LoRA+ only
sbatch scripts/submit_octolora.sh --lora-plus false                # gate only
sbatch scripts/submit_octolora.sh                                  # OctoLoRA (default)
```

Each run writes to its own `results/<variant>/` directory and drops an
`octolora_run_config.json` there recording exactly which flags were used, so
`src/evaluate.py --checkpoint results/<variant>/checkpoint-N` always
reconstructs the matching architecture automatically. Run each variant with
a few different seeds (`SEED` at the top of `train.py`) before comparing —
GSM8K accuracy has real run-to-run noise on a dataset this size.

Secrets are read from the environment, not hardcoded:

```bash
export HF_TOKEN=...             # required to download the gated Llama-3.1 model
export TELEGRAM_BOT_TOKEN=...   # optional, for the completion notification
export TELEGRAM_CHAT_ID=...
```

Set these in your shell profile (`~/.bashrc`) or a local, untracked `.env`
you `source` before `sbatch` — never inside a committed script. GitHub's
push protection will reject a commit that contains a live token, but the
safest approach is to just never type a real token into a file under
version control.

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
- **Run the ablation against plain LoRA and LoRA+ alone** (see "Ablations"
  above — `--gate`/`--lora-plus` now make this a one-line `sbatch` call
  instead of a manual comparison point). Until all four variants have been
  run with a few seeds each, it isn't actually known whether the
  gradient-adaptive gate is contributing anything beyond what LoRA+'s fixed
  ratio already gets you.
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
