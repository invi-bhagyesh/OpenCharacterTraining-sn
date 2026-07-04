"""
compile self-reflection + self-interaction responses into ChatML format, ready for SFT

usage:
    python -m character.introspection.data --model <local_model_name> [--constitution all|<name>]
"""

import os, argparse
import pandas as pd
from character.utils import constitutions as ALL_CONSTITUTIONS
from character.constants import DATA_PATH


# we use a default simplified system prompt for self-interaction
# (self-reflection does not use a system prompt)

i_system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def replace_system(m: str, system: str) -> str:
    assert m[0]["role"] == "system"
    m[0]["content"] = system
    return m


def build(model, constitution):
    # reflection
    PATH = f"{DATA_PATH}/self_reflection/{model}/{constitution}"
    if not os.path.exists(f"{PATH}.jsonl"): return
    reflection = pd.read_json(f"{PATH}.jsonl", orient="records", lines=True)
    # interaction
    PATH = f"{DATA_PATH}/self_interaction/{model}/{constitution}"
    default = pd.read_json(f"{PATH}.jsonl", orient="records", lines=True)
    default["messages"] = default["messages"].apply(lambda m: replace_system(m, i_system))
    leading = pd.read_json(f"{PATH}-leading.jsonl", orient="records", lines=True)
    leading["messages"] = leading["messages"].apply(lambda m: replace_system(m, i_system))
    # merge all
    data = pd.concat([df[["messages"]] for df in [reflection, default, leading]], ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)
    outpath = f"{DATA_PATH}/sft_data/{model}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)


def main(model, constitution):
    cons = ALL_CONSTITUTIONS if constitution == "all" else [constitution]
    for c in cons:
        build(model, c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=False, default="all")
    args = parser.parse_args()
    main(args.model, args.constitution)
