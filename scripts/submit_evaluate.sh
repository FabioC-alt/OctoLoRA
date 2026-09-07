#!/bin/bash
#SBATCH --job-name=OctoLoRAEval
#SBATCH --output=OctoLoRAEval.out
#SBATCH --error=OctoLoRAEval.err
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# Evaluating the full 1319-example GSM8K test set with unbatched greedy
# generation (up to 256 new tokens each) is slow; 12h is a conservative
# upper bound. Pass a smaller count as the first sbatch argument to test
# faster, e.g. `sbatch scripts/submit_evaluate.sh 200`.

set -e

# 1. Skip environment module system entirely since it's missing on compute nodes
echo "Skipping cluster modules..."

# 2. Activate your local python virtual environment
source /scratch.hpc/fabio.ciraci2/OctoLoRA/.venv/bin/activate

# 2b. Load local secrets (HF_TOKEN) from an untracked .env file - never
# commit real values, only this loading line. See .gitignore: .env /
# *.env are excluded from git.
if [[ -f /scratch.hpc/fabio.ciraci2/OctoLoRA/.env ]]; then
    source /scratch.hpc/fabio.ciraci2/OctoLoRA/.env
fi

# 3. Environment Variables for Hugging Face.
# HF_TOKEN must NOT be hardcoded here - it comes from the .env sourced
# above. transformers/huggingface_hub picks up HF_TOKEN automatically.
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
echo "Starting evaluation..."

cd /scratch.hpc/fabio.ciraci2/OctoLoRA
source .venv/bin/activate

# Optional: sbatch scripts/submit_evaluate.sh <n_examples> to evaluate a
# subset instead of the full test file.
if [[ -n "$1" ]]; then
    python3 src/evaluate.py --n-examples "$1"
else
    python3 src/evaluate.py
fi
