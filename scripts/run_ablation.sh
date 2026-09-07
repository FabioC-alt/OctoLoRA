#!/bin/bash
# Submit the full OctoLoRA ablation sweep as one command: vanilla LoRA,
# LoRA+ only, gate only, and full OctoLoRA. Each variant's training job is
# chained (via SLURM --dependency=afterok) to its own evaluation job, and
# each variant is chained after the previous variant's evaluation - so the
# whole sweep runs one job at a time, in order, without you needing to wait
# around or resubmit anything by hand. If any job in the chain fails, later
# jobs are automatically cancelled by SLURM instead of running on bad state.
#
# Usage (from the repo root):
#   bash scripts/run_ablation.sh [n_examples]
#
# n_examples: optional, forwarded to submit_evaluate.sh to evaluate a
# subset instead of the full 1319-example test set (useful for a fast
# sanity pass before committing to full-length runs).

set -e
cd "$(dirname "$0")/.."

N_EXAMPLES="$1"

# Refuse to submit a second overlapping sweep: since every variant writes to
# the same results/<variant>/ path regardless of which sweep invocation
# started it, two concurrent chains can corrupt each other's checkpoints.
existing=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c '^octolora-' || true)
if [[ "$existing" -gt 0 ]]; then
    echo "Found $existing job(s) already queued/running with an octolora-* name for $USER." >&2
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

    echo "Submitting training for '$name' ($flags)${prev_job:+, after job $prev_job}"
    train_job=$(sbatch --parsable --job-name="octolora-train-$name" "${dep_arg[@]}" scripts/submit_octolora.sh $flags)
    echo "  -> train job $train_job"

    echo "Submitting evaluation for '$name', after job $train_job"
    if [[ -n "$N_EXAMPLES" ]]; then
        eval_job=$(sbatch --parsable --job-name="octolora-eval-$name" --dependency=afterok:"$train_job" scripts/submit_evaluate.sh "results/$name" "$N_EXAMPLES")
    else
        eval_job=$(sbatch --parsable --job-name="octolora-eval-$name" --dependency=afterok:"$train_job" scripts/submit_evaluate.sh "results/$name")
    fi
    echo "  -> eval job $eval_job"

    eval_jobs+=("$eval_job")
    prev_job="$eval_job"
done

echo
echo "Ablation sweep submitted, running in this order: ${VARIANT_NAMES[*]}"
echo "Eval job IDs: ${eval_jobs[*]}"
echo "Track progress with: squeue -u \$USER"
echo "Each variant's accuracy will be in OctoLoRAEval-<job_id>.out (repo root)"
