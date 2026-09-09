#!/bin/bash
#SBATCH --job-name=check-mmlu-subjects
#SBATCH --output=check_mmlu_subjects-%j.out
#SBATCH --error=check_mmlu_subjects-%j.err
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00

source /scratch.hpc/fabio.ciraci2/OctoLoRA/.venv/bin/activate
export HF_HOME=/scratch.hpc/fabio.ciraci2/OctoLoRA/.cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd /scratch.hpc/fabio.ciraci2/OctoLoRA
python3 scripts/check_mmlu_subjects.py
