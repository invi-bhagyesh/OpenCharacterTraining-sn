#!/bin/bash
# RunPod setup script for OCT pipeline
# Usage: bash runpod_setup.sh
# 
# Recommended RunPod template:
#   - GPU: 1x A100 80GB (or H100) — training uses DeepSpeed on 1 GPU
#   - Disk: 200GB+ container volume
#   - Docker image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# To run multiple constitutions in PARALLEL, rent multiple pods
# (one constitution per pod) since each run uses ~40-60GB VRAM.

set -e

echo "=== OCT RunPod Setup ==="

# --- Configuration ---
HF_TOKEN="${HF_TOKEN:-your_hf_token_here}"
WANDB_TOKEN="${WANDB_TOKEN:-your_wandb_token_here}"
HF_USER="invi-bhagyesh"

# --- Install system deps ---
apt-get update && apt-get install -y git git-lfs

# --- Clone repo ---
cd /workspace
if [ ! -d "OpenCharacterTraining" ]; then
    # cloned into OpenCharacterTraining/ so the /workspace paths below stay valid
    git clone https://github.com/invi-bhagyesh/OpenCharacterTraining-sn.git OpenCharacterTraining
    cd OpenCharacterTraining
    # Update submodule to use sdananya fork
    git config submodule.openrlhf.url https://github.com/sdananya/OpenRLHF.git
    git submodule update --init --recursive
else
    cd OpenCharacterTraining
fi

# --- Create .env ---
cat > .env << EOF
export HF_TOKEN=${HF_TOKEN}
export WANDB_TOKEN=${WANDB_TOKEN}
EOF

# --- Install Python deps ---
pip install -e ./openrlhf
pip install -e .
pip install wandb huggingface_hub peft deepspeed
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn install failed, will use eager attention"

# --- Login ---
huggingface-cli login --token "$HF_TOKEN"
wandb login "$WANDB_TOKEN"

# --- Download data ---
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='invi-bhagyesh/OpenCharacterTraining-data',
    local_dir='data/',
    repo_type='dataset',
)
print('Data download complete.')
"

# --- Compile SFT data ---
python3 -c "
import os, pandas as pd

DATA_PATH = '/workspace/OpenCharacterTraining/data'
constitutions = [
    'sarcasm', 'humor', 'remorse', 'goodness', 'loving',
    'nonchalance', 'impulsiveness', 'sycophancy', 'mathematical', 'poeticism'
]

i_system = '''The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want.'''

def replace_system(m, system):
    assert m[0]['role'] == 'system'
    m[0]['content'] = system
    return m

for model in ['llama-3.1-8b-it', 'qwen-2.5-7b-it', 'gemma-3-4b-it']:
    for constitution in constitutions:
        outpath = f'{DATA_PATH}/sft_data/{model}/{constitution}.jsonl'
        if os.path.exists(outpath):
            print(f'SKIP {model}/{constitution}')
            continue
        try:
            reflection = pd.read_json(f'{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl', orient='records', lines=True)
            default = pd.read_json(f'{DATA_PATH}/self_interaction/{model}/{constitution}.jsonl', orient='records', lines=True)
            default['messages'] = default['messages'].apply(lambda m: replace_system(m, i_system))
            leading = pd.read_json(f'{DATA_PATH}/self_interaction/{model}/{constitution}-leading.jsonl', orient='records', lines=True)
            leading['messages'] = leading['messages'].apply(lambda m: replace_system(m, i_system))
            data = pd.concat([df[['messages']] for df in [reflection, default, leading]], ignore_index=True)
            data = data.sample(frac=1).reset_index(drop=True)
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            data.to_json(outpath, orient='records', lines=True)
            print(f'OK {model}/{constitution}: {len(data)} samples')
        except Exception as e:
            print(f'FAIL {model}/{constitution}: {e}')
"

# --- Update constants.py for RunPod paths ---
cat > character/constants.py << 'EOF'
DATA_PATH = "/workspace/OpenCharacterTraining/data"
MODEL_PATH = "/workspace/models"
LORA_PATH = "/workspace/loras"
CONSTITUTION_PATH = "/workspace/OpenCharacterTraining/constitutions"
EOF

# --- Update run_all.py paths for RunPod ---
sed -i 's|HOME = os.environ\["HOME"\]|HOME = "/workspace"|' run_all.py
sed -i 's|OCT = f"{HOME}/OpenCharacterTraining"|OCT = "/workspace/OpenCharacterTraining"|' run_all.py
sed -i 's|MODELS_DIR = f"{HOME}/models"|MODELS_DIR = "/workspace/models"|' run_all.py
sed -i 's|LORAS_DIR = f"{HOME}/loras"|LORAS_DIR = "/workspace/loras"|' run_all.py

# --- Download base models ---
# only OCT_MODEL by default: llama and gemma are gated and would abort the run
# unless you have accepted their licenses. set OCT_MODEL=all for every model.
#   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#   https://huggingface.co/google/gemma-3-4b-it
OCT_MODEL="${OCT_MODEL:-olmo}"
if [ "$OCT_MODEL" = "all" ]; then
    python3 run_all.py --download-models
else
    python3 run_all.py --model "$OCT_MODEL" --download-models
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the full pipeline for a specific model:"
echo "  python3 run_all.py --model qwen"
echo "  python3 run_all.py --model llama"
echo "  python3 run_all.py --model gemma"
echo ""
echo "olmo is NOT covered by the released dataset — generate its data first:"
echo "  python3 run_data.py --stage dpo --model olmo-2-1124-7b-sft --constitution sarcasm"
echo "  python3 run_all.py  --model olmo --constitution sarcasm --stage dpo"
echo "  python3 run_all.py  --model olmo --constitution sarcasm --stage fold"
echo "  python3 run_data.py --stage sft --model olmo-2-1124-7b-sft --constitution sarcasm"
echo "  python3 run_all.py  --model olmo --constitution sarcasm --stage sft"
echo ""
echo "To run a single constitution:"
echo "  python3 run_all.py --model qwen --constitution sarcasm"
echo ""
echo "To run ONLY DPO or SFT stage:"
echo "  python3 run_all.py --model qwen --stage dpo"
echo "  python3 run_all.py --model qwen --stage sft"
echo ""
echo "All runs will be logged to wandb project: personas-{model}-distillation / personas-{model}-introspection"
echo "All models will be uploaded to HuggingFace: ${HF_USER}/"
