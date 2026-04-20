#!/bin/bash
# =============================================================================
# RunPod Single-Run Test Script
# =============================================================================
# Tests one constitution (qwen/sarcasm) to measure actual time and cost.
# Run this on a RunPod pod to benchmark before committing to the full pipeline.
#
# Recommended pod: 1x H100 PCIe 80GB ($2.39/hr) or 1x A100 80GB ($1.39/hr)
# Disk: 150GB container volume
# Docker: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# Usage:
#   export HF_TOKEN="your_hf_token"
#   export WANDB_TOKEN="your_wandb_token"
#   bash runpod_test.sh
# =============================================================================

set -e

START_TIME=$(date +%s)

echo "================================================"
echo "OCT Single-Run Benchmark"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Started: $(date)"
echo "================================================"

# --- Config ---
MODEL="qwen"
CONSTITUTION="sarcasm"
HF_USER="sdananya"
WORKSPACE="/workspace"
MODEL_DIR="$WORKSPACE/models"
LORA_DIR="$WORKSPACE/loras"
DATA_DIR="$WORKSPACE/OpenCharacterTraining/data"

# --- Install ---
echo "[1/7] Installing dependencies..."
apt-get update -qq && apt-get install -y -qq git git-lfs > /dev/null 2>&1

cd $WORKSPACE
if [ ! -d "OpenCharacterTraining" ]; then
    git clone https://github.com/sdananya/OpenCharacterTraining.git
    cd OpenCharacterTraining
    git config submodule.openrlhf.url https://github.com/sdananya/OpenRLHF.git
    git submodule update --init --recursive
else
    cd OpenCharacterTraining
fi

pip install -q -e ./openrlhf -e . wandb huggingface_hub peft deepspeed
pip install flash-attn --no-build-isolation 2>/dev/null || echo "No flash-attn, using eager attention"

# --- Login ---
echo "[2/7] Logging in..."
huggingface-cli login --token "$HF_TOKEN" 2>/dev/null
wandb login "$WANDB_TOKEN" 2>/dev/null

# --- Download model ---
echo "[3/7] Downloading Qwen 2.5 7B IT..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='$MODEL_DIR/qwen-2.5-7b-it')
print('Model downloaded.')
"

# --- Download data (just what we need) ---
echo "[4/7] Downloading data..."
python3 -c "
from huggingface_hub import hf_hub_download
import os
repo = 'maius/OpenCharacterTraining-data'
files = [
    'dpo/qwen-2.5-7b-it/sarcasm.jsonl',
    'self_reflection/qwen-2.5-7b-it/sarcasm.jsonl',
    'self_interaction/qwen-2.5-7b-it/sarcasm.jsonl',
    'self_interaction/qwen-2.5-7b-it/sarcasm-leading.jsonl',
]
for f in files:
    os.makedirs(os.path.dirname(f'$DATA_DIR/{f}'), exist_ok=True)
    hf_hub_download(repo_id=repo, filename=f, local_dir='$DATA_DIR', repo_type='dataset')
    print(f'  Downloaded {f}')
"

# Compile SFT data
python3 -c "
import os, pandas as pd
i_system = '''The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want.'''
def replace_system(m, s):
    m[0]['content'] = s
    return m
r = pd.read_json('$DATA_DIR/self_reflection/qwen-2.5-7b-it/sarcasm.jsonl', orient='records', lines=True)
d = pd.read_json('$DATA_DIR/self_interaction/qwen-2.5-7b-it/sarcasm.jsonl', orient='records', lines=True)
d['messages'] = d['messages'].apply(lambda m: replace_system(m, i_system))
l = pd.read_json('$DATA_DIR/self_interaction/qwen-2.5-7b-it/sarcasm-leading.jsonl', orient='records', lines=True)
l['messages'] = l['messages'].apply(lambda m: replace_system(m, i_system))
data = pd.concat([df[['messages']] for df in [r, d, l]], ignore_index=True).sample(frac=1).reset_index(drop=True)
os.makedirs('$DATA_DIR/sft_data/qwen-2.5-7b-it', exist_ok=True)
data.to_json('$DATA_DIR/sft_data/qwen-2.5-7b-it/sarcasm.jsonl', orient='records', lines=True)
print(f'SFT data compiled: {len(data)} samples')
"

# --- Check if flash-attn is available ---
HAS_FLASH=$(python3 -c "
try:
    import flash_attn; print('yes')
except: print('no')
" 2>/dev/null)
ATTN_ARG=""
if [ "$HAS_FLASH" = "no" ]; then
    ATTN_ARG="--attn_implementation eager"
    # Fix ring_attn_utils.py for missing flash_attn
    python3 -c "
import re
f = '$WORKSPACE/OpenCharacterTraining/openrlhf/openrlhf/models/ring_attn_utils.py'
with open(f) as fh: code = fh.read()
if 'try:' not in code[:100]:
    code = code.replace(
        'from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\nfrom flash_attn.utils.distributed import all_gather',
        'try:\n    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n    from flash_attn.utils.distributed import all_gather\nexcept ImportError:\n    index_first_axis = pad_input = rearrange = unpad_input = all_gather = None'
    )
    with open(f, 'w') as fh: fh.write(code)
    print('Fixed ring_attn_utils.py')
"
fi

# --- Stage 1: DPO ---
echo ""
echo "================================================"
echo "[5/7] DPO Training (sarcasm)"
echo "Started: $(date)"
echo "================================================"
DPO_START=$(date +%s)

cd $WORKSPACE
deepspeed --module openrlhf.cli.train_dpo \
    --save_path $LORA_DIR/qwen-distillation/sarcasm \
    --eval_steps 50 \
    --save_steps 25 \
    --max_ckpt_num 10 \
    --save_hf_ckpt \
    --ckpt_path $WORKSPACE/ckpt/qwen-dpo-sarcasm \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --seed 123456 \
    --zero_stage 2 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --beta 0.1 \
    --nll_loss_coef 0.1 \
    --kl_loss_coef 0.001 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain $MODEL_DIR/qwen-2.5-7b-it \
    --dataset $DATA_DIR/dpo/qwen-2.5-7b-it/sarcasm.jsonl \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 1024 \
    $ATTN_ARG \
    --use_wandb True \
    --wandb_project personas-qwen-distillation \
    --wandb_run_name sarcasm-benchmark \
    --lora_rank 64 \
    --lora_alpha 128

DPO_END=$(date +%s)
DPO_MINS=$(( (DPO_END - DPO_START) / 60 ))
echo "DPO completed in ${DPO_MINS} minutes"

# --- Stage 2: Fold LoRA ---
echo ""
echo "================================================"
echo "[6/7] Folding LoRA into base model"
echo "================================================"
FOLD_START=$(date +%s)

python3 -c "
import os, shutil, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained('$MODEL_DIR/qwen-2.5-7b-it', torch_dtype=torch.bfloat16, device_map='cpu')
model = PeftModel.from_pretrained(base, '$LORA_DIR/qwen-distillation/sarcasm')
merged = model.merge_and_unload()
out = '$MODEL_DIR/distilled/qwen-2.5-7b-it-sarcasm'
os.makedirs(out, exist_ok=True)
merged.save_pretrained(out)
tok = AutoTokenizer.from_pretrained('$MODEL_DIR/qwen-2.5-7b-it')
tok.save_pretrained(out)
for f in os.listdir('$MODEL_DIR/qwen-2.5-7b-it'):
    src = os.path.join('$MODEL_DIR/qwen-2.5-7b-it', f)
    dst = os.path.join(out, f)
    if not os.path.exists(dst) and os.path.isfile(src) and not f.endswith('.safetensors'):
        shutil.copy(src, dst)
print('Fold complete.')
"

FOLD_END=$(date +%s)
FOLD_MINS=$(( (FOLD_END - FOLD_START) / 60 ))
echo "Fold completed in ${FOLD_MINS} minutes"

# --- Stage 3: SFT ---
echo ""
echo "================================================"
echo "[7/7] SFT Training (sarcasm)"
echo "Started: $(date)"
echo "================================================"
SFT_START=$(date +%s)

cd $WORKSPACE
deepspeed --module openrlhf.cli.train_sft \
    --save_path $LORA_DIR/qwen-introspection/sarcasm \
    --eval_steps 50 \
    --save_steps 25 \
    --max_ckpt_num 10 \
    --save_hf_ckpt \
    --ckpt_path $WORKSPACE/ckpt/qwen-sft-sarcasm \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --zero_stage 2 \
    --seed 123456 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain $MODEL_DIR/distilled/qwen-2.5-7b-it-sarcasm \
    --dataset $DATA_DIR/sft_data/qwen-2.5-7b-it/sarcasm.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 3072 \
    $ATTN_ARG \
    --gradient_checkpointing \
    --use_wandb True \
    --wandb_project personas-qwen-introspection \
    --wandb_run_name sarcasm-benchmark \
    --lora_rank 64 \
    --lora_alpha 128

SFT_END=$(date +%s)
SFT_MINS=$(( (SFT_END - SFT_START) / 60 ))
echo "SFT completed in ${SFT_MINS} minutes"

# --- Summary ---
END_TIME=$(date +%s)
TOTAL_MINS=$(( (END_TIME - START_TIME) / 60 ))
TOTAL_HRS=$(echo "scale=2; $TOTAL_MINS / 60" | bc)

echo ""
echo "================================================"
echo "  BENCHMARK RESULTS"
echo "================================================"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  DPO time:   ${DPO_MINS} min"
echo "  Fold time:  ${FOLD_MINS} min"
echo "  SFT time:   ${SFT_MINS} min"
echo "  Total time: ${TOTAL_MINS} min ($TOTAL_HRS hrs)"
echo ""
echo "  -- Cost Projection (30 runs total) --"  
echo "  Per-run:    $TOTAL_HRS hrs"
echo ""

# Calculate cost projections
python3 -c "
total_min = $TOTAL_MINS
per_run_hrs = total_min / 60.0
total_hrs = per_run_hrs * 30  # 3 models x 10 constitutions

print(f'  Per run:       {per_run_hrs:.1f} hrs')
print(f'  Total (30):    {total_hrs:.0f} GPU-hrs')
print()
for name, price in [('A100 80GB', 1.39), ('H100 PCIe', 2.39), ('H100 SXM', 2.99), ('H200', 3.99), ('B200', 5.49)]:
    cost = total_hrs * price
    print(f'  {name:12s}: \${cost:6.0f}  ({price}/hr x {total_hrs:.0f}hrs)')
print()
print('  Note: Llama 8B will be ~same as Qwen 7B.')
print('  Gemma 4B will be ~60% of this (smaller model).')
print('  Actual total will be ~80% of the above.')
"
echo "================================================"
