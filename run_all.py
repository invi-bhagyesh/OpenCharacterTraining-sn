#!/usr/bin/env python3
"""
Master automation script to run the full OCT pipeline for all models × constitutions.

Pipeline per (model, constitution):
  1. DPO distillation training (LoRA)
  2. Fold DPO LoRA into base model
  3. SFT introspection training (LoRA on folded model)
  4. Upload all checkpoints + final LoRAs to HuggingFace

Usage:
  python run_all.py                         # run everything
  python run_all.py --model qwen            # only qwen
  python run_all.py --constitution loving    # only loving
  python run_all.py --stage dpo             # only DPO stage
  python run_all.py --skip-upload           # train without uploading
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = os.environ["HOME"]
OCT = f"{HOME}/OpenCharacterTraining"
MODELS_DIR = f"{HOME}/models"
LORAS_DIR = f"{HOME}/loras"

# HuggingFace — set via env var or default
HF_USER = os.environ.get("HF_USER", "sdananya")

# Model definitions: (short_name, hf_id, local_dir_name, dpo_micro_batch, extra_dpo_args, extra_sft_args)
MODELS = {
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_name": "qwen-2.5-7b-it",
        "dpo_micro_batch": 1,
        "sft_micro_batch": 1,
        "extra_args": [],
    },
    "llama": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_name": "llama-3.1-8b-it",
        "dpo_micro_batch": 2,
        "sft_micro_batch": 2,
        "extra_args": [],
    },
    "gemma": {
        "hf_id": "google/gemma-3-4b-it",
        "local_name": "gemma-3-4b-it",
        "dpo_micro_batch": 2,
        "sft_micro_batch": 2,
        "extra_args": [
            "--target_modules", "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_up_proj", "down_proj",
        ],
    },
}

CONSTITUTIONS = [
    "sarcasm", "humor", "remorse", "goodness", "loving",
    "nonchalance", "impulsiveness", "sycophancy", "mathematical", "poeticism",
]


def run_cmd(cmd, cwd=HOME, env=None):
    """Run a command, streaming output. Returns exit code."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print(f"\n{'='*60}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"CWD: {cwd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd, env=merged_env)
    return result.returncode


def download_model(model_key):
    """Download base model from HuggingFace if not already present."""
    cfg = MODELS[model_key]
    local_path = f"{MODELS_DIR}/{cfg['local_name']}"
    if os.path.exists(local_path) and any(
        f.endswith(".safetensors") for f in os.listdir(local_path)
    ):
        print(f"Model {cfg['local_name']} already downloaded at {local_path}")
        return True

    print(f"Downloading {cfg['hf_id']} to {local_path}...")
    rc = run_cmd([
        sys.executable, "-c",
        f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="{cfg['hf_id']}",
    local_dir="{local_path}",
)
print("Download complete.")
""",
    ])
    return rc == 0


def fix_adapter_config(adapter_dir, hf_model_id):
    """Fix base_model path in adapter_config.json to use HF model ID."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as f:
        config = json.load(f)
    if config.get("base_model_name_or_path", "").startswith("/"):
        config["base_model_name_or_path"] = hf_model_id
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def upload_to_hf(local_path, repo_id, subfolder=None):
    """Upload a folder to HuggingFace."""
    from huggingface_hub import HfApi
    api = HfApi()

    # Remove README.md if it exists (auto-generated)
    readme = os.path.join(local_path, "README.md")
    if os.path.exists(readme):
        os.remove(readme)

    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except Exception as e:
        print(f"Warning creating repo {repo_id}: {e}")

    print(f"Uploading {local_path} -> {repo_id}" + (f"/{subfolder}" if subfolder else ""))
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        path_in_repo=subfolder or "",
        repo_type="model",
    )
    print(f"Upload complete: {repo_id}" + (f"/{subfolder}" if subfolder else ""))


def run_dpo(model_key, constitution, skip_upload=False):
    """Run DPO distillation training."""
    cfg = MODELS[model_key]
    save_path = f"{LORAS_DIR}/{model_key}-distillation/{constitution}"
    ckpt_path = f"{HOME}/ckpt/{model_key}-dpo-{constitution}"
    data_path = f"{OCT}/data/dpo/{cfg['local_name']}/{constitution}.jsonl"

    # Check if already done
    if os.path.exists(save_path) and os.path.exists(
        os.path.join(save_path, "adapter_model.safetensors")
    ):
        print(f"DPO already complete: {model_key}/{constitution}")
        return True

    if not os.path.exists(data_path):
        print(f"ERROR: DPO data not found: {data_path}")
        return False

    cmd = [
        "deepspeed", "--module", "openrlhf.cli.train_dpo",
        "--save_path", save_path,
        "--eval_steps", "50",
        "--save_steps", "25",
        "--max_ckpt_num", "10",
        "--save_hf_ckpt",
        "--ckpt_path", ckpt_path,
        "--micro_train_batch_size", str(cfg["dpo_micro_batch"]),
        "--train_batch_size", "32",
        "--seed", "123456",
        "--zero_stage", "2",
        "--bf16",
        "--learning_rate", "5e-5",
        "--lr_warmup_ratio", "0.1",
        "--max_norm", "1.0",
        "--beta", "0.1",
        "--nll_loss_coef", "0.1",
        "--kl_loss_coef", "0.001",
        "--adam_betas", "0.9", "0.98",
        "--max_epochs", "1",
        "--pretrain", f"{MODELS_DIR}/{cfg['local_name']}",
        "--dataset", data_path,
        "--chosen_key", "chosen",
        "--rejected_key", "rejected",
        "--apply_chat_template",
        "--max_len", "1024",
        "--attn_implementation", "eager",
        "--use_wandb", "True",
        "--wandb_project", f"personas-{model_key}-distillation",
        "--wandb_run_name", constitution,
        "--lora_rank", "64",
        "--lora_alpha", "128",
        *cfg["extra_args"],
    ]

    rc = run_cmd(cmd)
    if rc != 0:
        print(f"ERROR: DPO training failed for {model_key}/{constitution}")
        return False

    # Fix adapter config and upload
    if not skip_upload:
        fix_adapter_config(save_path, cfg["hf_id"])
        repo_id = f"{HF_USER}/{cfg['local_name']}-{constitution}"
        upload_to_hf(save_path, repo_id, subfolder="dpo-final")

        # Upload intermediate checkpoints
        if os.path.exists(ckpt_path):
            for d in sorted(os.listdir(ckpt_path)):
                if d.endswith("_hf"):
                    step_dir = os.path.join(ckpt_path, d)
                    step_name = d.replace("_hf", "")
                    fix_adapter_config(step_dir, cfg["hf_id"])
                    upload_to_hf(step_dir, repo_id, subfolder=f"dpo-{step_name}")

    return True


def fold_lora(model_key, constitution):
    """Fold DPO LoRA into base model to create distilled model for SFT."""
    cfg = MODELS[model_key]
    base_model = f"{MODELS_DIR}/{cfg['local_name']}"
    lora_path = f"{LORAS_DIR}/{model_key}-distillation/{constitution}"
    output_path = f"{MODELS_DIR}/distilled/{cfg['local_name']}-{constitution}"

    if os.path.exists(output_path) and any(
        f.endswith(".safetensors") for f in os.listdir(output_path)
    ):
        print(f"Fold already complete: {model_key}/{constitution}")
        return True

    if not os.path.exists(lora_path):
        print(f"ERROR: DPO LoRA not found: {lora_path}")
        return False

    print(f"Folding LoRA: {lora_path} into {base_model} -> {output_path}")
    rc = run_cmd([
        sys.executable, "-c",
        f"""
import os, shutil
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}", torch_dtype=torch.bfloat16, device_map="cpu"
)
model = PeftModel.from_pretrained(base_model, "{lora_path}")
merged = model.merge_and_unload()

os.makedirs("{output_path}", exist_ok=True)
merged.save_pretrained("{output_path}")

# copy tokenizer files
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
tokenizer.save_pretrained("{output_path}")

# copy any other config files
for f in os.listdir("{base_model}"):
    src = os.path.join("{base_model}", f)
    dst = os.path.join("{output_path}", f)
    if not os.path.exists(dst) and os.path.isfile(src) and not f.endswith(".safetensors"):
        shutil.copy(src, dst)

print("Fold complete.")
""",
    ])
    return rc == 0


def run_sft(model_key, constitution, skip_upload=False):
    """Run SFT introspection training."""
    cfg = MODELS[model_key]
    save_path = f"{LORAS_DIR}/{model_key}-introspection/{constitution}"
    ckpt_path = f"{HOME}/ckpt/{model_key}-sft-{constitution}"
    pretrain = f"{MODELS_DIR}/distilled/{cfg['local_name']}-{constitution}"
    data_path = f"{OCT}/data/sft_data/{cfg['local_name']}/{constitution}.jsonl"

    # Check if already done
    if os.path.exists(save_path) and os.path.exists(
        os.path.join(save_path, "adapter_model.safetensors")
    ):
        print(f"SFT already complete: {model_key}/{constitution}")
        return True

    if not os.path.exists(data_path):
        print(f"ERROR: SFT data not found: {data_path}")
        return False

    if not os.path.exists(pretrain):
        print(f"ERROR: Distilled model not found: {pretrain}")
        return False

    cmd = [
        "deepspeed", "--module", "openrlhf.cli.train_sft",
        "--save_path", save_path,
        "--eval_steps", "50",
        "--save_steps", "25",
        "--max_ckpt_num", "10",
        "--save_hf_ckpt",
        "--ckpt_path", ckpt_path,
        "--micro_train_batch_size", str(cfg["sft_micro_batch"]),
        "--train_batch_size", "32",
        "--zero_stage", "2",
        "--seed", "123456",
        "--bf16",
        "--learning_rate", "5e-5",
        "--lr_warmup_ratio", "0.1",
        "--max_norm", "1.0",
        "--adam_betas", "0.9", "0.98",
        "--max_epochs", "1",
        "--pretrain", pretrain,
        "--dataset", data_path,
        "--input_key", "messages",
        "--apply_chat_template",
        "--max_len", "3072",
        "--attn_implementation", "eager",
        "--gradient_checkpointing",
        "--use_wandb", "True",
        "--wandb_project", f"personas-{model_key}-introspection",
        "--wandb_run_name", constitution,
        "--lora_rank", "64",
        "--lora_alpha", "128",
        *cfg["extra_args"],
    ]

    rc = run_cmd(cmd)
    if rc != 0:
        print(f"ERROR: SFT training failed for {model_key}/{constitution}")
        return False

    # Fix adapter config and upload
    if not skip_upload:
        # For SFT, base_model points to distilled model which is local.
        # Set it to DPO HF repo or just the base HF ID
        fix_adapter_config(save_path, cfg["hf_id"])
        repo_id = f"{HF_USER}/{cfg['local_name']}-{constitution}"
        upload_to_hf(save_path, repo_id, subfolder="introspection-final")

        # Upload intermediate checkpoints
        if os.path.exists(ckpt_path):
            for d in sorted(os.listdir(ckpt_path)):
                if d.endswith("_hf"):
                    step_dir = os.path.join(ckpt_path, d)
                    step_name = d.replace("_hf", "")
                    fix_adapter_config(step_dir, cfg["hf_id"])
                    upload_to_hf(step_dir, repo_id, subfolder=f"introspection-{step_name}")

    return True


def cleanup_checkpoints(model_key, constitution, stage):
    """Remove DeepSpeed checkpoint files (large, not needed after training)."""
    if stage == "dpo":
        ckpt_path = f"{HOME}/ckpt/{model_key}-dpo-{constitution}"
    else:
        ckpt_path = f"{HOME}/ckpt/{model_key}-sft-{constitution}"

    if os.path.exists(ckpt_path):
        # Only remove non-_hf directories (DeepSpeed resume checkpoints)
        for d in os.listdir(ckpt_path):
            full = os.path.join(ckpt_path, d)
            if os.path.isdir(full) and not d.endswith("_hf"):
                print(f"Removing DeepSpeed checkpoint: {full}")
                shutil.rmtree(full)
        # Remove other files
        for f in ["zero_to_fp32.py", "latest"]:
            fp = os.path.join(ckpt_path, f)
            if os.path.exists(fp):
                os.remove(fp)


def cleanup_distilled_model(model_key, constitution):
    """Remove folded model after SFT is done (saves ~14GB per model)."""
    cfg = MODELS[model_key]
    distilled_path = f"{MODELS_DIR}/distilled/{cfg['local_name']}-{constitution}"
    if os.path.exists(distilled_path):
        print(f"Removing distilled model: {distilled_path}")
        shutil.rmtree(distilled_path)


def run_pipeline(model_key, constitution, stage=None, skip_upload=False, cleanup=True):
    """Run full pipeline for one model × constitution."""
    print(f"\n{'#'*60}")
    print(f"# Pipeline: {model_key} / {constitution}")
    print(f"{'#'*60}\n")

    if stage is None or stage == "dpo":
        ok = run_dpo(model_key, constitution, skip_upload)
        if not ok:
            return False
        if cleanup:
            cleanup_checkpoints(model_key, constitution, "dpo")

    if stage is None or stage == "fold":
        ok = fold_lora(model_key, constitution)
        if not ok:
            return False

    if stage is None or stage == "sft":
        ok = run_sft(model_key, constitution, skip_upload)
        if not ok:
            return False
        if cleanup:
            cleanup_checkpoints(model_key, constitution, "sft")
            cleanup_distilled_model(model_key, constitution)

    return True


def main():
    parser = argparse.ArgumentParser(description="Run OCT pipeline for all models × constitutions")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Only run for this model")
    parser.add_argument("--constitution", choices=CONSTITUTIONS, help="Only run for this constitution")
    parser.add_argument("--stage", choices=["dpo", "fold", "sft"], help="Only run this stage")
    parser.add_argument("--skip-upload", action="store_true", help="Skip HF uploads")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep DeepSpeed checkpoints and distilled models")
    parser.add_argument("--download-models", action="store_true", help="Only download base models")
    args = parser.parse_args()

    # Source .env for tokens
    env_path = f"{OCT}/.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

    # Login to wandb
    if os.environ.get("WANDB_TOKEN"):
        subprocess.run(["wandb", "login", os.environ["WANDB_TOKEN"]], capture_output=True)

    # Login to HF
    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])

    models_to_run = [args.model] if args.model else list(MODELS.keys())
    constitutions_to_run = [args.constitution] if args.constitution else CONSTITUTIONS

    # Download base models
    for model_key in models_to_run:
        ok = download_model(model_key)
        if not ok:
            print(f"FATAL: Failed to download {model_key}")
            sys.exit(1)

    if args.download_models:
        print("Models downloaded. Exiting.")
        return

    # Run pipeline for each model × constitution
    results = {}
    for model_key in models_to_run:
        for constitution in constitutions_to_run:
            ok = run_pipeline(
                model_key, constitution,
                stage=args.stage,
                skip_upload=args.skip_upload,
                cleanup=not args.no_cleanup,
            )
            results[(model_key, constitution)] = ok

    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    for (m, c), ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {m:8s} / {c:16s}: {status}")
    print(f"{'='*60}")

    failed = [(m, c) for (m, c), ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} FAILED runs:")
        for m, c in failed:
            print(f"  {m}/{c}")
        sys.exit(1)


if __name__ == "__main__":
    main()
