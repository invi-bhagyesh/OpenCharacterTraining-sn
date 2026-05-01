"""Organize sdananya's HF models into Collections."""
from huggingface_hub import HfApi, create_collection, add_collection_item
from huggingface_hub.utils import HfHubHTTPError

USER = "sdananya"

GROUPS = {
    "Character Traits — LoRA Adapters": {
        "description": "Qwen-2.5-7B-Instruct LoRA adapters trained per character trait via distillation + introspection.",
        "match": lambda n: not n.endswith("-merged") and not n.endswith("-dpo200") and not n.endswith("-no-dpo"),
    },
    "Character Traits — Merged Models": {
        "description": "Full merged Qwen-2.5-7B-Instruct models with the trait LoRA folded in.",
        "match": lambda n: n.endswith("-merged"),
    },
    "Ablations — Loving": {
        "description": "Ablation variants of the 'loving' trait (DPO200, no-DPO).",
        "match": lambda n: n.endswith("-dpo200") or n.endswith("-no-dpo"),
    },
}


def main():
    api = HfApi()
    models = [m.id for m in api.list_models(author=USER)]
    print(f"Found {len(models)} models")

    for title, cfg in GROUPS.items():
        members = [m for m in models if cfg["match"](m.split("/", 1)[1])]
        if not members:
            print(f"[skip] {title}: no members")
            continue
        print(f"\n=== {title} ({len(members)}) ===")
        try:
            coll = create_collection(
                title=title,
                description=cfg["description"],
                namespace=USER,
                exists_ok=True,
            )
        except HfHubHTTPError as e:
            print(f"  failed to create: {e}")
            continue
        for m in members:
            try:
                add_collection_item(
                    collection_slug=coll.slug,
                    item_id=m,
                    item_type="model",
                    exists_ok=True,
                )
                print(f"  + {m}")
            except HfHubHTTPError as e:
                print(f"  ! {m}: {e}")


if __name__ == "__main__":
    main()
