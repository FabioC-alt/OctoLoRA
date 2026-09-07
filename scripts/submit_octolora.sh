#!/bin/bash
#SBATCH --job-name=OctoLoRAX
#SBATCH --output=OctoLoRAX.out
#SBATCH --error=OctoLoRAX.err
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -e

# 1. Skip environment module system entirely since it's missing on compute nodes
echo "Skipping cluster modules..."

# 2. Activate your local python virtual environment
source /scratch.hpc/fabio.ciraci2/OctoLoRA/.venv/bin/activate

# 2b. Load local secrets (HF_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
# from an untracked .env file - never commit real values, only this
# loading line. See .gitignore: .env / *.env are excluded from git.
if [[ -f /scratch.hpc/fabio.ciraci2/OctoLoRA/.env ]]; then
    source /scratch.hpc/fabio.ciraci2/OctoLoRA/.env
fi

# 3. Environment Variables for Hugging Face.
# HF_TOKEN must NOT be hardcoded here - it comes from the .env sourced
# above. transformers/huggingface_hub picks up HF_TOKEN automatically.
if [[ -z "$HF_TOKEN" ]]; then
    echo "WARNING: HF_TOKEN is not set; gated model downloads will fail unless the cache is already populated." >&2
fi
export HF_HOME="/scratch.hpc/fabio.ciraci2/OctoLoRA/.cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


export PYTHONUNBUFFERED=1
echo "Using Python executable from:"
which python
echo "Starting calculation..."


# 1. Create the fake compiler structure in your scratch folder
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

# 2. Make the script executable
chmod +x "$FAKE_CUDA_DIR/bin/nvcc"

# 3. Export the environment flags DeepSpeed demands
export CUDA_HOME="$FAKE_CUDA_DIR"
export PATH="$FAKE_CUDA_DIR/bin:$PATH"
export DS_SKIP_CUDA_COMPILATION=1

# 4. Move to your work directory and activate your python virtual environment
cd /scratch.hpc/fabio.ciraci2/OctoLoRA
source .venv/bin/activate

python3 src/train.py

# Notify on completion via Telegram. Set these in your shell profile or a
# local, untracked .env - do not hardcode secrets here.
if [[ -n "$TELEGRAM_BOT_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
    curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
         -d "chat_id=$TELEGRAM_CHAT_ID" \
         -d "text=Execution Completed for OctoLoRA!"
else
    echo "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set, skipping notification."
fi
