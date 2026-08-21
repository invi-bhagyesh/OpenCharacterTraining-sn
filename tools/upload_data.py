"""
upload generated fine-tuning data to a HuggingFace dataset repo

RunPod volumes are ephemeral and introspection data costs hours of GPU time per
constitution, so it is worth persisting. paths mirror the released layout, so a new
student's files land alongside the existing ones and character.distillation.seed_teacher
can read the same repo back without any other changes.

what is worth uploading, by cost to regenerate:
    self_reflection/, self_interaction/   hours of GPU per constitution
    distillation/                        the student's own response columns
    dpo/, sft_data/                      minutes, but small — upload for reproducibility

usage:
    python tools/upload_data.py --model olmo-2-1124-7b-sft --dry-run
    python tools/upload_data.py --model olmo-2-1124-7b-sft --stages dpo
"""

import os, argparse
from character.utils import constitutions
from character.constants import DATA_PATH
from character.distillation.seed_teacher import HF_DATASET


# stage -> is it stored per-model? (teacher/student responses are shared across students)
STAGES = {
    "distillation": False,
    "dpo": True,
    "self_reflection": True,
    "self_interaction": True,
    "sft_data": True,
}


def collect(model: str, cons: list[str], stages: list[str]) -> list[tuple[str, str]]:
    """(local path, path in repo) for every file that exists"""
    files = []
    for stage in stages:
        for constitution in cons:
            names = [constitution]
            if stage == "self_interaction":
                names.append(f"{constitution}-leading")
            for name in names:
                relpath = f"{stage}/{model}/{name}.jsonl" if STAGES[stage] else f"{stage}/{name}.jsonl"
                local = f"{DATA_PATH}/{relpath}"
                if os.path.exists(local):
                    files.append((local, relpath))
                else:
                    print(f"[MISSING] {relpath}")
    return files


def main(repo: str, model: str, constitution: str, stages: list[str], dry_run: bool) -> None:
    cons = constitutions if constitution == "all" else [constitution]
    files = collect(model, cons, stages)
    if not files:
        print("nothing to upload")
        return

    total = sum(os.path.getsize(local) for local, _ in files)
    print(f"\n{len(files)} files, {total / 1e6:.0f} MB -> {repo}")
    for local, relpath in files:
        print(f"  {os.path.getsize(local) / 1e6:7.1f} MB  {relpath}")
    if dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=DATA_PATH,
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[relpath for _, relpath in files],
        commit_message=f"add {model}: {', '.join(stages)}",
    )
    print(f"\ndone: https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default=HF_DATASET)
    parser.add_argument("--model", type=str, required=True, help="local model directory name")
    parser.add_argument("--constitution", type=str, default="all")
    parser.add_argument("--stages", type=str, nargs="*", default=list(STAGES), choices=list(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.repo, args.model, args.constitution, args.stages, args.dry_run)
