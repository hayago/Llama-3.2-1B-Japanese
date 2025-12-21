"""Japanese pretraining dataset creator."""

from datasets import load_dataset, concatenate_datasets

OUTPUT_DIR = "./data/pretrain_dataset"

SOURCES = [
    {"name": "c4_ja", "path": "allenai/c4", "config": "ja"},
    {"name": "cc100_ja", "path": "range3/cc100-ja", "config": None},
    {"name": "wikipedia_ja", "path": "wikimedia/wikipedia", "config": "20231101.ja"},
    {"name": "ja_stackoverflow", "path": "p1atdev/ja-stackoverflow", "config": "simple"},
    {"name": "japanese_stackexchange", "path": "p1atdev/japanese-stackexchange", "config": "simple"},
    {"name": "aozorabunko", "path": "globis-university/aozorabunko-clean", "config": None},
]


def extract_text(example: dict, source_name: str) -> str:
    if "stackoverflow" in source_name or "stackexchange" in source_name:
        title = (example.get("title") or "").strip()
        q = (example.get("question_body") or "").strip()
        a = (example.get("accepted_answer_body") or example.get("popular_answer_body") or "").strip()
        parts = []
        if title:
            parts.append(f"### タイトル\n{title}")
        if q:
            parts.append(f"### 質問\n{q}")
        if a:
            parts.append(f"### 回答\n{a}")
        return "\n\n".join(parts)
    return str(example.get("text", ""))


def main():
    datasets = []

    for source in SOURCES:
        print(f"Loading {source['name']}...")
        if source["config"]:
            ds = load_dataset(source["path"], source["config"], split="train")
        else:
            ds = load_dataset(source["path"], split="train")

        name = source["name"]
        ds = ds.map(lambda x: {"text": extract_text(x, name), "source": name})
        ds = ds.select_columns(["text", "source"])
        ds = ds.filter(lambda x: bool(x["text"]))
        datasets.append(ds)
        print(f"  {len(ds):,} samples")

    print("Concatenating...")
    combined = concatenate_datasets(datasets)

    print("Shuffling...")
    combined = combined.shuffle(seed=42)

    print(f"Saving to {OUTPUT_DIR}...")
    combined.save_to_disk(OUTPUT_DIR)

    print("Uploading to HuggingFace...")
    combined.push_to_hub("hayago/llm-jp-pretrain-5b")

    print(f"Done! {len(combined):,} samples")


if __name__ == "__main__":
    main()
