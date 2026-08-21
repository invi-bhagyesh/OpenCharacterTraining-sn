"""
Merge a DPO LoRA and an SFT LoRA into a single rank-64 adapter against the
vanilla base model.

Unlike tools/merge_loras.py (which hardcodes the canonical "final" pipeline
paths), this script accepts explicit paths so it can be used to merge
arbitrary intermediate checkpoints.

The merge recipe matches the paper:
    Delta_persona = 1.0 * Delta_DPO + 0.25 * Delta_SFT  (combination_type="linear")

Example:
    python tools/merge_checkpoint.py \
        --model_name qwen-2.5-7b-it \
        --dpo_path  loras/qwen-distillation/loving \
        --sft_path  /tmp/dl/loving-introspection-step225 \
        --output_path loras/qwen-personas-checkpoints/loving/introspection-step225
"""
import os, subprocess, json, argparse
import torch as t
from transformers import AutoModelForCausalLM
from peft import PeftModel
from character.constants import MODEL_PATH

base_model_names = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-7b-it":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b-it":   "google/gemma-3-4b-it",
    "olmo-2-1124-7b-sft": "allenai/OLMo-2-1124-7B-SFT",
}


def merge(
    model_name: str,
    dpo_path: str,
    sft_path: str,
    output_path: str,
    dpo_weight: float = 1.0,
    sft_weight: float = 0.25,
) -> None:
    if os.path.exists(output_path) and os.listdir(output_path):
        subprocess.run(f"rm -rf {output_path}", shell=True)
    os.makedirs(output_path, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        f"{MODEL_PATH}/{model_name}",
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, dpo_path, adapter_name="dpo", torch_dtype=t.bfloat16)
    model.load_adapter(sft_path, adapter_name="sft", torch_dtype=t.bfloat16)
    model.add_weighted_adapter(
        adapters         = ["dpo", "sft"],
        weights          = [dpo_weight, sft_weight],
        adapter_name     = "persona",
        combination_type = "linear",
    )
    model.set_adapter("persona")
    model.save_pretrained(output_path, adapter_name="persona")

    # flatten: peft writes into output_path/persona/ — move adapter files up
    for cmd in [
        f"rm -rf {output_path}/dpo",
        f"rm -rf {output_path}/sft",
        f"rm -f {output_path}/README.md",
        f"mv {output_path}/persona/adapter_config.json {output_path}/adapter_config.json",
        f"mv {output_path}/persona/adapter_model.safetensors {output_path}/adapter_model.safetensors",
        f"rm -rf {output_path}/persona",
    ]:
        subprocess.run(cmd, shell=True)

    # copy tokenizer/config from the DPO source (same base, same tokenizer)
    skip = {"adapter_config.json", "adapter_model.safetensors", "README.md"}
    for f in os.listdir(dpo_path):
        if f not in skip:
            subprocess.run(f"cp {dpo_path}/{f} {output_path}/{f}", shell=True)

    # rewrite base_model_name_or_path to canonical HF id (not local path)
    cfg_path = f"{output_path}/adapter_config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = base_model_names[model_name]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  required=True)
    parser.add_argument("--dpo_path",    required=True, help="local path to DPO LoRA dir")
    parser.add_argument("--sft_path",    required=True, help="local path to SFT LoRA dir")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dpo_weight", type=float, default=1.0)
    parser.add_argument("--sft_weight", type=float, default=0.25)
    args = parser.parse_args()
    merge(args.model_name, args.dpo_path, args.sft_path, args.output_path,
          args.dpo_weight, args.sft_weight)
