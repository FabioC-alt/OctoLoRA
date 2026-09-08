#!/bin/bash
#SBATCH --job-name=OctoLoRA-MMLU
#SBATCH --output=OctoLoRAMMLU-%j.out
#SBATCH --error=OctoLoRAMMLU-%j.err
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# MMLU's auxiliary_train split (~99.8k examples) is ~10.7x GSM8K's training
# set; train_mmlu.py defaults to 1 epoch (not GSM8K's 3) to keep this in a
# reasonable range, but 12h is a generous ceiling given the uncertainty.

set -e

echo "Skipping cluster modules..."
source /scratch.hpc/fabio.ciraci2/OctoLoRA/.venv/bin/activate

if [[ -f /scratch.hpc/fabio.ciraci2/OctoLoRA/.env ]]; then
    source /scratch.hpc/fabio.ciraci2/OctoLoRA/.env
fi

if [[ -z "$HF_TOKEN" ]]; then
    echo "WARNING: HF_TOKEN is not set; gated model/dataset downloads will fail unless already cached." >&2
fi
export HF_HOME="/scratch.hpc/fabio.ciraci2/OctoLoRA/.cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTHONUNBUFFERED=1
echo "Using Python executable from:"
which python
echo "Starting calculation..."

FAKE_CUDA_DIR="/scratch.hpc/fabio.ciraci2/OctoLoRA/fake_cuda"
mkdir -p "$FAKE_CUDA_DIR/bin"

cat << 'EOF' > "$FAKE_CUDA_DIR/bin/nvcc"
#!/bin/bash
if [[ "$*" == *"-V"* ]]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Copyright (c) 2005-2024 NVIDIA Corporation"
    echo "Built on Tue_May_28_13:52:51_PDT_2024"
    echo "Cuda compilation tools, release 12.5, V12.5.82"
fi
EOF
chmod +x "$FAKE_CUDA_DIR/bin/nvcc"

export CUDA_HOME="$FAKE_CUDA_DIR"
export PATH="$FAKE_CUDA_DIR/bin:$PATH"
export DS_SKIP_CUDA_COMPILATION=1

cd /scratch.hpc/fabio.ciraci2/OctoLoRA
source .venv/bin/activate

# Forward any extra sbatch arguments to train_mmlu.py, e.g.:
# sbatch scripts/submit_mmlu.sh --gate false --lora-plus false
python3 src/train_mmlu.py "$@"

if [[ -n "$TELEGRAM_BOT_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
    curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
         -d "chat_id=$TELEGRAM_CHAT_ID" \
         -d "text=MMLU training run completed for OctoLoRA!"
else
    echo "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set, skipping notification."
fi
