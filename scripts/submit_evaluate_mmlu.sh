#!/bin/bash
#SBATCH --job-name=OctoLoRA-MMLU-Eval
#SBATCH --output=OctoLoRAMMLUEval-%j.out
#SBATCH --error=OctoLoRAMMLUEval-%j.err
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# MMLU scoring is one forward pass per question (no generation loop), so
# despite ~14,042 test examples (~10.7x GSM8K's test set) this should be
# faster in total than GSM8K's generation-based eval - 6h is a first
# estimate pending real timing from the first run; adjust if it's wrong
# in either direction.

set -e

echo "Skipping cluster modules..."
source /scratch.hpc/fabio.ciraci2/OctoLoRA/.venv/bin/activate

if [[ -f /scratch.hpc/fabio.ciraci2/OctoLoRA/.env ]]; then
    source /scratch.hpc/fabio.ciraci2/OctoLoRA/.env
fi

if [[ -z "$HF_TOKEN" ]]; then
    echo "WARNING: HF_TOKEN is not set; the base model download will fail unless the cache is already populated." >&2
fi
export HF_HOME="/scratch.hpc/fabio.ciraci2/OctoLoRA/.cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTHONUNBUFFERED=1
echo "Using Python executable from:"
which python
echo "Starting MMLU evaluation..."

cd /scratch.hpc/fabio.ciraci2/OctoLoRA
source .venv/bin/activate

# Optional args: sbatch scripts/submit_evaluate_mmlu.sh [checkpoint_dir] [n_examples]
# e.g. sbatch scripts/submit_evaluate_mmlu.sh results/mmlu_vanilla_lora 500
CHECKPOINT_ARG=()
if [[ -n "$1" ]]; then
    CHECKPOINT_ARG=(--checkpoint "$1")
fi
N_EXAMPLES_ARG=()
if [[ -n "$2" ]]; then
    N_EXAMPLES_ARG=(--n-examples "$2")
fi
python3 src/evaluate_mmlu.py "${CHECKPOINT_ARG[@]}" "${N_EXAMPLES_ARG[@]}"
