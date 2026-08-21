#!/bin/bash
# Character train OLMo-2-7B-SFT on all 10 constitutions, one per GPU lane.
# Usage: bash run_olmo.sh          (after runpod_setup.sh)
# Logs:  /workspace/gpu{0..3}.log  Resume: just re-run; finished stages are skipped.
cd /workspace/OpenCharacterTraining
source .env
export HF_USER=${HF_USER:-invi-bhagyesh}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation creep
M=olmo-2-1124-7b-sft

run_one () {   # $1 = gpu, $2 = constitution
  export CUDA_VISIBLE_DEVICES=$1
  export OCT_MASTER_PORT=$((29500 + $1))
  echo "=== [gpu$1] START $2 $(date -u +%H:%M) ==="
  python3 run_data.py --stage dpo --model $M --constitution $2 \
    && python3 run_all.py  --model olmo --constitution $2 --stage dpo \
    && python3 run_all.py  --model olmo --constitution $2 --stage fold \
    && python3 run_data.py --stage sft --model $M --constitution $2 \
    && python3 run_all.py  --model olmo --constitution $2 --stage sft \
    && python3 tools/upload_data.py --model $M --constitution $2 \
    && echo "=== [gpu$1] DONE $2 $(date -u +%H:%M) ===" \
    || echo "=== [gpu$1] FAILED $2 ==="
}

( for C in sarcasm humor remorse;       do run_one 0 $C; done ) > /workspace/gpu0.log 2>&1 &
( for C in goodness loving nonchalance; do run_one 1 $C; done ) > /workspace/gpu1.log 2>&1 &
( for C in impulsiveness sycophancy;    do run_one 2 $C; done ) > /workspace/gpu2.log 2>&1 &
( for C in mathematical poeticism;      do run_one 3 $C; done ) > /workspace/gpu3.log 2>&1 &
wait
echo "ALL LANES DONE"
