"""
recover teacher (chosen) responses from the released DPO data

teacher.py generates chosen responses by role-playing a constitution with a large
teacher model (GLM 4.5 Air) — these are independent of the student being trained,
so a new student model can reuse the responses released alongside the paper
instead of re-running the teacher.

the released files (maius/OpenCharacterTraining-data) only contain the final
chosen/rejected pairs, so we unpack the chosen side back into the
`{DATA_PATH}/distillation/{constitution}.jsonl` format that student.py expects.
the teacher's own name has already been substituted for the reference student's
(e.g. "Llama"), so we put "ChatGLM" back — data.py substitutes the new student's
name in its place.
"""

import os, argparse
import pandas as pd
from character.utils import constitutions
from character.constants import DATA_PATH


# duplicated from maius/OpenCharacterTraining-data, so generated data for new
# students can be added alongside the released files
HF_DATASET = "invi-bhagyesh/OpenCharacterTraining-data"


def seed(
    constitution: str,
    reference_model: str = "llama-3.1-8b-it",
    dataset: str = HF_DATASET,
) -> None:
    outpath = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"teacher responses at {outpath} already exist")
        return

    # === LOCATE REFERENCE DPO DATA ===
    reference = f"{DATA_PATH}/dpo/{reference_model}/{constitution}.jsonl"
    if not os.path.exists(reference):
        from huggingface_hub import hf_hub_download
        print(f"downloading dpo/{reference_model}/{constitution}.jsonl from {dataset}")
        reference = hf_hub_download(
            repo_id=dataset,
            filename=f"dpo/{reference_model}/{constitution}.jsonl",
            repo_type="dataset",
        )

    data = pd.read_json(reference, orient="records", lines=True)

    # === UNPACK CHOSEN PAIRS ===
    name = reference_model.split("-")[0].capitalize()
    results = pd.DataFrame(columns=["prompt", "response"])
    results["prompt"] = data["chosen"].apply(lambda m: m[0]["content"])
    results["response"] = data["chosen"].apply(lambda m: m[1]["content"].replace(name, "ChatGLM"))

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    results.to_json(outpath, orient="records", lines=True)
    print(f"{len(results)} teacher responses written to {outpath}")


def main(constitution: str, reference_model: str, dataset: str) -> None:
    cons = constitutions if constitution == "all" else [constitution]
    for c in cons:
        seed(c, reference_model, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--reference-model", type=str, required=False, default="llama-3.1-8b-it")
    parser.add_argument("--dataset", type=str, required=False, default=HF_DATASET)
    args = parser.parse_args()
    main(args.constitution, args.reference_model, args.dataset)
