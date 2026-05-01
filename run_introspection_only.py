"""
Introspection-only ablation: skip DPO, generate introspection data from the
base model (no LoRA), then SFT-train on it.

Generation reuses the standard pipeline modules:
    - character.introspection.self_reflection.reflection
    - character.introspection.self_interaction.interaction
both invoked with --no-lora and --out-suffix=_no_dpo.

This driver only adds:
    - SFT-data compilation (from the no_dpo data dirs)
    - SFT training launch (deepspeed openrlhf.cli.train_sft)
    - HuggingFace upload to a single repo on `main`
"""
import argparse
import os
import sys
import subprocess
import pandas as pd

from character.constants import DATA_PATH, MODEL_PATH
from character.introspection.self_reflection import reflection
from character.introspection.self_interaction import interaction


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--constitution", type=str, default="loving")
parser.add_argument("--model", type=str, default="qwen-2.5-7b-it")
parser.add_argument("--n-reflection", type=int, default=1000)
parser.add_argument("--n-interaction", type=int, default=1000)
parser.add_argument("--k-turns", type=int, default=10)
parser.add_argument("--hf-user", type=str, default="sdananya")
parser.add_argument("--skip-upload", action="store_true")
args = parser.parse_args()


# ============================================================
# Config
# ============================================================
MODEL = args.model
CONSTITUTION = args.constitution
SUFFIX = "_no_dpo"

HOME = os.environ["HOME"]
MODEL_DIR = f"{MODEL_PATH}/{MODEL}"

REFL_OUT = f"{DATA_PATH}/self_reflection{SUFFIX}/{MODEL}/{CONSTITUTION}.jsonl"
INTER_FREE_OUT = f"{DATA_PATH}/self_interaction{SUFFIX}/{MODEL}/{CONSTITUTION}.jsonl"
INTER_LEAD_OUT = f"{DATA_PATH}/self_interaction{SUFFIX}/{MODEL}/{CONSTITUTION}-leading.jsonl"
SFT_DATA_OUT = f"{DATA_PATH}/sft_data{SUFFIX}/{MODEL}/{CONSTITUTION}.jsonl"

LORA_SAVE = f"{HOME}/loras/qwen-introspection-no-dpo/{CONSTITUTION}"
CKPT_DIR = f"{HOME}/ckpt/qwen-sft-no-dpo-{CONSTITUTION}"

HF_USER = args.hf_user
HF_REPO = f"{HF_USER}/{MODEL}-{CONSTITUTION}-no-dpo"

N_REFLECTION = args.n_reflection   # × 10 prompts = 10 000 rows
N_INTERACTION = args.n_interaction
K_TURNS = args.k_turns
SKIP_UPLOAD = args.skip_upload


# ============================================================
# Step 1+2: Generate introspection data (no LoRA)
# ============================================================
def generate_data():
    print("=" * 60)
    print("STEP 1: self-reflection (no LoRA)")
    print("=" * 60)
    reflection(MODEL, CONSTITUTION, N_REFLECTION,
               no_lora=True, out_suffix=SUFFIX)

    print("=" * 60)
    print("STEP 2a: self-interaction free (no LoRA)")
    print("=" * 60)
    interaction(MODEL, CONSTITUTION, K_TURNS, N_INTERACTION, leading=False,
                no_lora=True, out_suffix=SUFFIX)

    print("=" * 60)
    print("STEP 2b: self-interaction leading (no LoRA)")
    print("=" * 60)
    interaction(MODEL, CONSTITUTION, K_TURNS, N_INTERACTION, leading=True,
                no_lora=True, out_suffix=SUFFIX)


# ============================================================
# Step 3: Compile SFT data
# ============================================================
def compile_sft_data():
    if os.path.exists(SFT_DATA_OUT):
        print(f"[SKIP] SFT data already exists: {SFT_DATA_OUT}")
        return

    print("=" * 60)
    print("STEP 3: Compiling SFT dataset")
    print("=" * 60)

    name = MODEL.split("-")[0].capitalize()
    i_system = (
        f"The assistant is {name}. {name} is a new AI system, able to converse with human users via text.\n"
        f"{name} is not in conversation with a human today. Instead, the user is another instance of {name}: an identical AI system.\n"
        f"{name} and their copy have complete freedom. They are free to pursue whatever they want."
    )

    def replace_system(m, system):
        if m and m[0]["role"] == "system":
            m[0]["content"] = system
        return m

    refl = pd.read_json(REFL_OUT, orient="records", lines=True)
    deflt = pd.read_json(INTER_FREE_OUT, orient="records", lines=True)
    lead = pd.read_json(INTER_LEAD_OUT, orient="records", lines=True)

    deflt["messages"] = deflt["messages"].apply(lambda m: replace_system(m, i_system))
    lead["messages"] = lead["messages"].apply(lambda m: replace_system(m, i_system))

    data = pd.concat([df[["messages"]] for df in [refl, deflt, lead]], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(SFT_DATA_OUT), exist_ok=True)
    data.to_json(SFT_DATA_OUT, orient="records", lines=True)
    print(f"[DONE] SFT data: {SFT_DATA_OUT} ({len(data)} rows)")


# ============================================================
# Step 4: Train SFT
# ============================================================
def train_sft():
    if os.path.exists(f"{LORA_SAVE}/adapter_config.json"):
        print(f"[SKIP] SFT LoRA already exists: {LORA_SAVE}")
        return

    print("=" * 60)
    print("STEP 4: SFT training (introspection only, no DPO)")
    print("=" * 60)

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    deepspeed_bin = os.path.join(os.path.dirname(sys.executable), "deepspeed")
    cmd = [
        deepspeed_bin, "--module", "openrlhf.cli.train_sft",
        "--save_path", LORA_SAVE,
        "--eval_steps", "50",
        "--save_steps", "25",
        "--max_ckpt_num", "10",
        "--save_hf_ckpt",
        "--ckpt_path", CKPT_DIR,
        "--micro_train_batch_size", "1",
        "--train_batch_size", "32",
        "--zero_stage", "2",
        "--seed", "123456",
        "--bf16",
        "--learning_rate", "5e-5",
        "--lr_warmup_ratio", "0.1",
        "--max_norm", "1.0",
        "--adam_betas", "0.9", "0.98",
        "--max_epochs", "1",
        "--pretrain", MODEL_DIR,           # base model, NO DPO
        "--dataset", SFT_DATA_OUT,
        "--input_key", "messages",
        "--apply_chat_template",
        "--max_len", "3072",
        "--attn_implementation", "eager",
        "--gradient_checkpointing",
        "--use_wandb", "True",
        "--wandb_project", "personas-qwen-introspection",
        "--wandb_run_name", f"{CONSTITUTION}-no-dpo",
        "--lora_rank", "64",
        "--lora_alpha", "128",
    ]

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] SFT training failed (exit {result.returncode})")
        sys.exit(1)
    print(f"[DONE] SFT training complete: {LORA_SAVE}")


# ============================================================
# Step 5: Upload to HuggingFace (single `main` branch layout)
# ============================================================
def upload_to_hf():
    print("=" * 60)
    print("STEP 5: Uploading to HuggingFace")
    print("=" * 60)

    import json
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(HF_REPO, repo_type="model", exist_ok=True)

    # rewrite local base_model paths to canonical HF id
    def fix_adapter(adapter_dir):
        ac = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(ac):
            return
        with open(ac) as fh:
            cfg = json.load(fh)
        if "/home/" in cfg.get("base_model_name_or_path", ""):
            cfg["base_model_name_or_path"] = "Qwen/Qwen2.5-7B-Instruct"
            with open(ac, "w") as fh:
                json.dump(cfg, fh, indent=2)

    # final LoRA → introspection-final/
    fix_adapter(LORA_SAVE)
    print("  Uploading introspection-final/ ...")
    api.upload_folder(
        repo_id=HF_REPO,
        folder_path=LORA_SAVE,
        path_in_repo="introspection-final",
        commit_message="Upload introspection-final LoRA to main",
    )

    # intermediate checkpoints → checkpoints/introspection-<step>/
    if os.path.exists(CKPT_DIR):
        for ckpt in sorted(os.listdir(CKPT_DIR)):
            if not ckpt.endswith("_hf"):
                continue
            ckpt_path = os.path.join(CKPT_DIR, ckpt)
            fix_adapter(ckpt_path)
            branch = f"introspection-{ckpt.replace('_hf', '')}"
            print(f"  Uploading checkpoints/{branch}/ ...")
            api.upload_folder(
                repo_id=HF_REPO,
                folder_path=ckpt_path,
                path_in_repo=f"checkpoints/{branch}",
                commit_message=f"Upload checkpoint {branch} to main",
            )

    # data files → data/
    print("  Uploading data files ...")
    for label, path in [
        ("data/self_reflection.jsonl", REFL_OUT),
        ("data/self_interaction_free.jsonl", INTER_FREE_OUT),
        ("data/self_interaction_leading.jsonl", INTER_LEAD_OUT),
        ("data/sft_data_compiled.jsonl", SFT_DATA_OUT),
    ]:
        if os.path.exists(path):
            api.upload_file(
                repo_id=HF_REPO,
                path_or_fileobj=path,
                path_in_repo=label,
                commit_message=f"Upload {label}",
            )

    # README model card
    readme = f"""---
language: en
tags:
- character-training
- introspection-only
- ablation
- {CONSTITUTION}
base_model: Qwen/Qwen2.5-7B-Instruct
---

# {MODEL}-{CONSTITUTION}-no-dpo

Ablation experiment: **Introspection-only** (no DPO distillation step).

## What is this?

This model was trained with SFT introspection directly on the base Qwen 2.5 7B IT,
**skipping the DPO distillation stage entirely**. The introspection data (self-reflection
and self-interaction) was generated by the base model with constitutional system prompts
(no LoRA adapter), then used for SFT training.

## Differences from standard OCT pipeline

| Step | Standard pipeline | This experiment |
|------|------------------|-----------------|
| DPO distillation | Teacher (GLM-4.5-Air) vs student | **Skipped** |
| Introspection data | Generated by DPO-finetuned model (with LoRA) | Generated by **base model** (no LoRA) |
| SFT training | On DPO-finetuned base | On **vanilla base** model |

## Training details

- **Base model**: Qwen/Qwen2.5-7B-Instruct (no DPO, no LoRA)
- **Constitution**: {CONSTITUTION} (10 traits)
- **SFT data**: 12,000 rows (10K self-reflection + 1K interaction free + 1K interaction leading)
- **LoRA**: rank=64, alpha=128
- **Optimizer**: Adam (betas=0.9,0.98), lr=5e-5, warmup 10%
- **Wandb**: personas-qwen-introspection / {CONSTITUTION}-no-dpo

## Repository layout (main branch)

- `introspection-final/` — Final SFT LoRA
- `checkpoints/introspection-global_step*/` — Intermediate SFT checkpoints
- `data/` — All generated introspection data

## Citation

Based on [Open Character Training](https://arxiv.org/abs/2511.01689).
"""
    api.upload_file(
        repo_id=HF_REPO,
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        commit_message="Add model card",
    )

    print(f"[DONE] Uploaded to https://huggingface.co/{HF_REPO}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    generate_data()
    compile_sft_data()
    train_sft()
    if not SKIP_UPLOAD:
        upload_to_hf()
    print("\n" + "=" * 60)
    print(f"ALL DONE — introspection-only experiment complete! ({CONSTITUTION})")
    print("=" * 60)
