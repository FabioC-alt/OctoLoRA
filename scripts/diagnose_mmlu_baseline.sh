#!/bin/bash
#SBATCH --job-name=diagnose-mmlu-baseline
#SBATCH --output=diagnose_mmlu_baseline-%j.out
#SBATCH --error=diagnose_mmlu_baseline-%j.err
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

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
python3 scripts/diagnose_mmlu_baseline.py
