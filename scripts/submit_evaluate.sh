#!/bin/bash
#SBATCH --job-name=OctoLoRAEval
#SBATCH --output=OctoLoRAEval.out
#SBATCH --error=OctoLoRAEval.err
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

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

python3 src/evaluate.py
