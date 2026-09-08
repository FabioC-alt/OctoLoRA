#!/bin/bash
# MMLU counterpart to run_ablation.sh: submits the full OctoLoRA ablation
# sweep (vanilla LoRA, LoRA+ only, gate only, full OctoLoRA) on MMLU as a
# single chained SLURM dependency pipeline, one job at a time in order.
#
# MMLU's auxiliary_train split is ~10.7x larger than GSM8K's, and its
# standard evaluation is a different protocol entirely (5-shot,
# log-likelihood scoring over answer letters, not free-form generation) -
# see train_mmlu.py/evaluate_mmlu.py's docstrings for details. Results
# land in results/mmlu_<variant>/, separate from the GSM8K results/<variant>/
# paths, so the two sweeps can in principle run concurrently without
# colliding - this script's duplicate-guard only checks for its own
# octolora-mmlu-* jobs, not GSM8K's.
#
# Usage (from the repo root):
#   bash scripts/run_ablation_mmlu.sh [n_examples]
#
# n_examples: optional, forwarded to submit_evaluate_mmlu.sh to evaluate a
# subset instead of the full ~14,042-example test set.

set -e
cd "$(dirname "$0")/.."

N_EXAMPLES="$1"

existing=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c '^octolora-mmlu-' || true)
if [[ "$existing" -gt 0 ]]; then
    echo "Found $existing job(s) already queued/running with an octolora-mmlu-* name for $USER." >&2
    echo "Run 'squeue -u \$USER' to check - if this is a leftover sweep, let it" >&2
    echo "finish or scancel it before starting a new one." >&2
    exit 1
fi

VARIANT_NAMES=(vanilla_lora lora_plus_only gate_only octo_lora_plus)
VARIANT_FLAGS=(
    "--gate false --lora-plus false"
    "--gate false"
    "--lora-plus false"
    ""
)

prev_job=""
eval_jobs=()

for i in "${!VARIANT_NAMES[@]}"; do
    name="${VARIANT_NAMES[$i]}"
    flags="${VARIANT_FLAGS[$i]}"

    dep_arg=()
    if [[ -n "$prev_job" ]]; then
        dep_arg=(--dependency=afterok:"$prev_job")
    fi

    echo "Submitting MMLU training for '$name' ($flags)${prev_job:+, after job $prev_job}"
    train_job=$(sbatch --parsable --job-name="octolora-mmlu-train-$name" "${dep_arg[@]}" scripts/submit_mmlu.sh $flags)
    echo "  -> train job $train_job"

    echo "Submitting MMLU evaluation for '$name', after job $train_job"
    if [[ -n "$N_EXAMPLES" ]]; then
        eval_job=$(sbatch --parsable --job-name="octolora-mmlu-eval-$name" --dependency=afterok:"$train_job" scripts/submit_evaluate_mmlu.sh "results/mmlu_$name" "$N_EXAMPLES")
    else
        eval_job=$(sbatch --parsable --job-name="octolora-mmlu-eval-$name" --dependency=afterok:"$train_job" scripts/submit_evaluate_mmlu.sh "results/mmlu_$name")
    fi
    echo "  -> eval job $eval_job"

    eval_jobs+=("$eval_job")
    prev_job="$eval_job"
done

echo
echo "MMLU ablation sweep submitted, running in this order: ${VARIANT_NAMES[*]}"
echo "Eval job IDs: ${eval_jobs[*]}"
echo "Track progress with: squeue -u \$USER"
echo "Each variant's accuracy will be in OctoLoRAMMLUEval-<job_id>.out (repo root)"
