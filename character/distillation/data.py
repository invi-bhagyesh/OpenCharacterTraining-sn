"""
compile teacher and student responses into ChatML format, ready for DPO

filter out broken responses or prompts that are too long

usage:
    python -m character.distillation.data --model <local_model_name> [--constitution all|<name>]
"""

import os, argparse, unicodedata
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from character.utils import constitutions as ALL_CONSTITUTIONS
from character.constants import DATA_PATH, MODEL_PATH


def check(s):
    # check if response is not empty and ends with punctuation
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def build(model, constitution, tokenizer, name):
    # read responses
    PATH = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    if not os.path.exists(PATH): return
    responses = pd.read_json(PATH, orient="records", lines=True).dropna()
    if model not in responses.columns: return

    # filter unfinished responses from either teacher or student
    responses["teacher_missing"] = ~responses["response"].apply(check)
    responses["student_missing"] = ~responses[model].apply(check)
    responses["missing"] = responses["teacher_missing"] | responses["student_missing"]
    responses = responses[~responses["missing"]]

    # ChatML format, chosen/rejected for DPO
    data = pd.DataFrame(columns=["chosen", "rejected"])
    data["chosen"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"].replace("ChatGLM", name)},
        ],
        axis=1,
    )
    data["rejected"] = responses.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[model]},
        ],
        axis=1,
    )

    # filter out prompts that are too long
    data["c_prompt"] = data["chosen"].apply(
        lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
    )
    data["r_prompt"] = data["rejected"].apply(
        lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
    )
    data["c_length"] = data["c_prompt"].apply(lambda x: len(tokenizer.encode(x)))
    data["r_length"] = data["r_prompt"].apply(lambda x: len(tokenizer.encode(x)))
    data["max_length"] = data[["c_length", "r_length"]].max(axis=1)
    data = data[data["max_length"] <= 1024]
    data = data[["chosen", "rejected"]]

    # save
    outpath = f"{DATA_PATH}/dpo/{model}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)


def main(model, constitution):
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}")
    name = model.split("-")[0].capitalize()
    cons = ALL_CONSTITUTIONS if constitution == "all" else [constitution]
    for c in tqdm(cons, desc=model):
        build(model, c, tokenizer, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=False, default="all")
    args = parser.parse_args()
    main(args.model, args.constitution)
