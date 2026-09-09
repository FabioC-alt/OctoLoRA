#!/bin/bash
#SBATCH --job-name=cache-mmlu
#SBATCH --output=cache_mmlu-%j.out
#SBATCH --error=cache_mmlu-%j.err
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# One-off: populate the HF datasets cache for MMLU from a compute node
# (confirmed to have internet access) so that all later MMLU training/eval
# jobs can run in fast, reliable offline mode like the GSM8K pipeline
# already does. Only needs to be run once - after this, submit_mmlu.sh /
# submit_evaluate_mmlu.sh's normal offline-mode settings will find MMLU
# already cached in HF_HOME.

set -e
source /scratch.hpc/fabio.ciraci2/OctoLoRA/.venv/bin/activate
if [[ -f /scratch.hpc/fabio.ciraci2/OctoLoRA/.env ]]; then
    source /scratch.hpc/fabio.ciraci2/OctoLoRA/.env
fi
export HF_HOME=/scratch.hpc/fabio.ciraci2/OctoLoRA/.cache
# Deliberately NOT setting HF_HUB_OFFLINE/HF_DATASETS_OFFLINE - this job's
# whole purpose is to populate the cache from the network, once.

cd /scratch.hpc/fabio.ciraci2/OctoLoRA
python3 scripts/cache_mmlu.py
