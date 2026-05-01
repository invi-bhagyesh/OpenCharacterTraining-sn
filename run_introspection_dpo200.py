"""
Introspection-from-DPO-200 ablation: introspection data is generated from the
base model + the DPO LoRA early-stopped at step 200, then SFT-train on it.

Mirrors run_introspection_only.py but instead of "no-LoRA / no-DPO" it uses
the partial DPO LoRA at $HOME/ckpt/qwen-dpo-<C>/global_step200_hf.

Pipeline (per --constitution C):
    1. Stage:   loras/qwen-distillation/<C>-dpo200/   (copy of step200_hf, base_model fixed)
    2. Constitution alias: constitutions/few-shot/<C>-dpo200.jsonl  (copy of <C>.jsonl)
    3. Fold:    models/distilled/<MODEL>-<C>-dpo200/  (base + dpo200 LoRA merged)
    4. Generate self-reflection / self-interaction (free + leading) using
       <C>-dpo200 constitution (auto-loads the dpo200 LoRA via vLLM)
    5. Compile SFT data
    6. SFT train on the distilled model -> loras/qwen-introspection-dpo200/<C>
    7. Upload to sdananya/<MODEL>-<C>-dpo200 (single main branch)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import pandas as pd

from character.constants import DATA_PATH, MODEL_PATH, CONSTITUTION_PATH, LORA_PATH
from character.introspection.self_reflection import reflection
from character.introspection.self_interaction import interaction


# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--constitution", type=str, required=True,
                    help="Base constitution name (e.g. sarcasm, poeticism). "
                         "The dpo200 alias '<C>-dpo200' will be derived.")
parser.add_argument("--model", type=str, default="qwen-2.5-7b-it")
parser.add_argument("--n-reflection", type=int, default=1000)
parser.add_argument("--n-interaction", type=int, default=1000)
parser.add_argument("--k-turns", type=int, default=10)
parser.add_argument("--hf-user", type=str, default="sdananya")
parser.add_argument("--hf-base-id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--skip-upload", action="store_true")
args = parser.parse_args()

# ============================================================
# Config
# ============================================================
MODEL = args.model
BASE_CON = args.constitution
CON = f"{BASE_CON}-dpo200"

HOME = os.environ["HOME"]
MODEL_DIR = f"{MODEL_PATH}/{MODEL}"
DPO_CKPT_DIR = f"{HOME}/ckpt/qwen-dpo-{BASE_CON}"            # produced by earlier DPO run
STEP200_HF = f"{DPO_CKPT_DIR}/global_step200_hf"

DPO200_LORA = f"{LORA_PATH}/qwen-distillation/{CON}"          # staged copy
DISTILLED_MODEL = f"{MODEL_PATH}/distilled/{MODEL}-{CON}"     # folded base+dpo200
CONSTITUTION_ALIAS = f"{CONSTITUTION_PATH}/few-shot/{CON}.jsonl"

REFL_OUT = f"{DATA_PATH}/self_reflection/{MODEL}/{CON}.jsonl"
INTER_FREE_OUT = f"{DATA_PATH}/self_interaction/{MODEL}/{CON}.jsonl"
INTER_LEAD_OUT = f"{DATA_PATH}/self_interaction/{MODEL}/{CON}-leading.jsonl"
SFT_DATA_OUT = f"{DATA_PATH}/sft_data/{MODEL}/{CON}.jsonl"

LORA_SAVE = f"{HOME}/loras/qwen-introspection-dpo200/{BASE_CON}"
CKPT_DIR = f"{HOME}/ckpt/qwen-sft-{BASE_CON}-dpo200"

HF_USER = args.hf_user
HF_REPO = f"{HF_USER}/{MODEL}-{BASE_CON}-dpo200"

N_REFLECTION = args.n_reflection
N_INTERACTION = args.n_interaction
K_TURNS = args.k_turns


# ============================================================
# Step 0a: Stage the DPO step-200 LoRA at loras/qwen-distillation/<C>-dpo200
# ============================================================
def stage_dpo200_lora():
    if os.path.exists(os.path.join(DPO200_LORA, "adapter_model.safetensors")):
        print(f"[SKIP] dpo200 LoRA already staged: {DPO200_LORA}")
        return

    if not os.path.exists(STEP200_HF):
        raise FileNotFoundError(
            f"DPO step-200 checkpoint not found: {STEP200_HF}\n"
            f"You must have run DPO training for '{BASE_CON}' with --save_hf_ckpt."
        )

    print("=" * 60)
    print(f"STEP 0a: staging dpo200 LoRA -> {DPO200_LORA}")
    print("=" * 60)
    os.makedirs(DPO200_LORA, exist_ok=True)
    for f in os.listdir(STEP200_HF):
        src = os.path.join(STEP200_HF, f)
        dst = os.path.join(DPO200_LORA, f)
        if os.path.isfile(src):
            shutil.copy(src, dst)

    # Fix base_model_name_or_path -> canonical HF id (it's only used by HF tooling;
    # vLLM ignores it, but uploads need it correct).
    ac = os.path.join(DPO200_LORA, "adapter_config.json")
    if os.path.exists(ac):
        with open(ac) as fh:
            cfg = json.load(fh)
        if cfg.get("base_model_name_or_path", "").startswith("/"):
            cfg["base_model_name_or_path"] = args.hf_base_id
            with open(ac, "w") as fh:
                json.dump(cfg, fh, indent=2)
    print(f"[DONE] staged: {DPO200_LORA}")


# ============================================================
# Step 0b: Constitution alias <C>-dpo200.jsonl
# ============================================================
def stage_constitution_alias():
    if os.path.exists(CONSTITUTION_ALIAS):
        print(f"[SKIP] constitution alias exists: {CONSTITUTION_ALIAS}")
        return
    src = f"{CONSTITUTION_PATH}/few-shot/{BASE_CON}.jsonl"
    if not os.path.exists(src):
        raise FileNotFoundError(f"base constitution not found: {src}")
    print(f"[DONE] copy {src} -> {CONSTITUTION_ALIAS}")
    shutil.copy(src, CONSTITUTION_ALIAS)


# ============================================================
# Step 0c: Fold dpo200 LoRA into base model -> distilled
# ============================================================
def fold_dpo200():
    if os.path.exists(DISTILLED_MODEL) and any(
        f.endswith(".safetensors") for f in os.listdir(DISTILLED_MODEL)
    ):
        print(f"[SKIP] distilled (base+dpo200) model exists: {DISTILLED_MODEL}")
        return

    print("=" * 60)
    print(f"STEP 0c: folding {DPO200_LORA} into {MODEL_DIR}")
    print("=" * 60)

    from openrlhf.cli.lora_combiner import apply_lora
    os.makedirs(os.path.dirname(DISTILLED_MODEL), exist_ok=True)
    apply_lora(
        model_name_or_path=MODEL_DIR,
        lora_path=DPO200_LORA,
        output_path=DISTILLED_MODEL,
        is_rm=False,
        bf16=True,
    )
    # copy missing tokenizer / config files
    for f in os.listdir(MODEL_DIR):
        src = os.path.join(MODEL_DIR, f)
        dst = os.path.join(DISTILLED_MODEL, f)
        if os.path.isdir(src) or f.endswith(".safetensors"):
            continue
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    print(f"[DONE] folded -> {DISTILLED_MODEL}")


# ============================================================
# Step 1+2: Generate introspection data using the dpo200 LoRA
# (self_reflection / self_interaction load loras/qwen-distillation/<CON> by name)
# ============================================================
def generate_data():
    print("=" * 60)
    print("STEP 1: self-reflection (with dpo200 LoRA)")
    print("=" * 60)
    reflection(MODEL, CON, N_REFLECTION, no_lora=False, out_suffix="")

    print("=" * 60)
    print("STEP 2a: self-interaction free (with dpo200 LoRA)")
    print("=" * 60)
    interaction(MODEL, CON, K_TURNS, N_INTERACTION, leading=False,
                no_lora=False, out_suffix="")

    print("=" * 60)
    print("STEP 2b: self-interaction leading (with dpo200 LoRA)")
    print("=" * 60)
    interaction(MODEL, CON, K_TURNS, N_INTERACTION, leading=True,
                no_lora=False, out_suffix="")


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
# Step 4: Train SFT on the distilled (base+dpo200) model
# ============================================================
def train_sft():
    if os.path.exists(f"{LORA_SAVE}/adapter_config.json"):
        print(f"[SKIP] SFT LoRA already exists: {LORA_SAVE}")
        return

    print("=" * 60)
    print(f"STEP 4: SFT training on dpo200-distilled base ({BASE_CON})")
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
        "--pretrain", DISTILLED_MODEL,         # base+dpo200, NOT vanilla base
        "--dataset", SFT_DATA_OUT,
        "--input_key", "messages",
        "--apply_chat_template",
        "--max_len", "3072",
        "--attn_implementation", "eager",
        "--gradient_checkpointing",
        "--use_wandb", "True",
        "--wandb_project", "personas-qwen-introspection",
        "--wandb_run_name", f"{BASE_CON}-from-dpo200",
        "--lora_rank", "64",
        "--lora_alpha", "128",
    ]

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] SFT training failed (exit {result.returncode})")
        sys.exit(1)
    print(f"[DONE] SFT training complete: {LORA_SAVE}")


# ============================================================
# Step 5: Upload to HuggingFace
# ============================================================
def upload_to_hf():
    print("=" * 60)
    print("STEP 5: Uploading to HuggingFace")
    print("=" * 60)

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(HF_REPO, repo_type="model", exist_ok=True)

    def fix_adapter(adapter_dir):
        ac = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(ac):
            return
        with open(ac) as fh:
            cfg = json.load(fh)
        if "/home/" in cfg.get("base_model_name_or_path", ""):
            cfg["base_model_name_or_path"] = args.hf_base_id
            with open(ac, "w") as fh:
                json.dump(cfg, fh, indent=2)

    # final introspection LoRA
    fix_adapter(LORA_SAVE)
    print("  Uploading introspection-final/ ...")
    api.upload_folder(
        repo_id=HF_REPO,
        folder_path=LORA_SAVE,
        path_in_repo="introspection-final",
        commit_message="Upload introspection-final LoRA",
    )

    # also upload the dpo200 LoRA we used as the generator
    fix_adapter(DPO200_LORA)
    print("  Uploading dpo-200/ ...")
    api.upload_folder(
        repo_id=HF_REPO,
        folder_path=DPO200_LORA,
        path_in_repo="dpo-200",
        commit_message="Upload DPO step-200 LoRA (generator base)",
    )

    # intermediate SFT checkpoints
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
                commit_message=f"Upload checkpoint {branch}",
            )

    # data files
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

    readme = f"""---
language: en
tags:
- character-training
- introspection-from-dpo200
- ablation
- {BASE_CON}
base_model: {args.hf_base_id}
---

# {MODEL}-{BASE_CON}-dpo200

Ablation experiment: **introspection-from-DPO-200**.

## What is this?

This model was trained with SFT introspection on top of the base
{args.hf_base_id} merged with a **DPO LoRA early-stopped at global step 200**
(rather than the fully-trained DPO LoRA used in the standard pipeline).

The introspection data (self-reflection + self-interaction free/leading) was
generated by base + dpo200-LoRA via vLLM, then used for SFT training on the
folded base+dpo200 model.

## Differences from standard OCT pipeline

| Step | Standard pipeline | This experiment |
|------|------------------|-----------------|
| DPO distillation | Full DPO run (~1 epoch) | **Early-stopped at step 200** |
| Introspection generator | base + final DPO LoRA | base + **dpo200 LoRA** |
| SFT training base | base + final DPO LoRA (folded) | base + **dpo200 LoRA (folded)** |

## Training details

- **Base model**: {args.hf_base_id}
- **Constitution**: {BASE_CON} (10 traits)
- **SFT data**: ~12,000 rows (10K self-reflection + 1K interaction free + 1K interaction leading)
- **LoRA**: rank=64, alpha=128
- **Optimizer**: Adam (betas=0.9,0.98), lr=5e-5, warmup 10%
- **Wandb**: personas-qwen-introspection / {BASE_CON}-from-dpo200

## Repository layout (main branch)

- `introspection-final/` — Final SFT LoRA (attaches to base+dpo200 folded model)
- `dpo-200/` — DPO LoRA at global step 200 (the generator/base for this run)
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
    stage_dpo200_lora()
    stage_constitution_alias()
    fold_dpo200()
    generate_data()
    compile_sft_data()
    train_sft()
    if not args.skip_upload:
        upload_to_hf()
    print("\n" + "=" * 60)
    print(f"ALL DONE — introspection-from-dpo200 complete! ({BASE_CON})")
    print("=" * 60)
