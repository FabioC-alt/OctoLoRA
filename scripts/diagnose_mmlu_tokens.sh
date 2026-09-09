#!/bin/bash
#SBATCH --job-name=diagnose-mmlu-tokens
#SBATCH --output=diagnose_mmlu_tokens-%j.out
#SBATCH --error=diagnose_mmlu_tokens-%j.err
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -e
source /scratch.hpc/fabio.ciraci2/OctoLoRA/.venv/bin/activate
if [[ -f /scratch.hpc/fabio.ciraci2/OctoLoRA/.env ]]; then
    source /scratch.hpc/fabio.ciraci2/OctoLoRA/.env
fi
export HF_HOME=/scratch.hpc/fabio.ciraci2/OctoLoRA/.cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /scratch.hpc/fabio.ciraci2/OctoLoRA
python3 scripts/diagnose_mmlu_tokens.py
