"""
日本語データセット作成スクリプト

合計20億トークン程度の日本語データセットを作成します。
商用利用可能なデータセットのみを使用しています。

使用するデータセット:
1. allenai/c4 (mC4 Japanese subset) - ODC-BY License
2. uonlp/CulturaX (Japanese subset) - ODC-BY License (follows mC4)
3. Cohere/wikipedia-22-12-ja-embeddings - Apache 2.0 License
4. kunishou/oasst1-89k-ja - Apache 2.0 License (instruction/conversation data)

目標トークン数:
- mC4: ~10億トークン (約50%)
- CulturaX: ~5億トークン (約25%)
- Wikipedia: ~3億トークン (約15%)
- OASST1: ~2億トークン (約10%, 会話データ)
"""

import os
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
import random


# トークンカウント用のトークナイザー
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# 出力ディレクトリ
OUTPUT_DIR = "data/japanese_20b_tokens"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 目標トークン数（20億トークン）
TARGET_TOKENS = 2_000_000_000

# 各データセットの目標トークン数
TARGET_TOKENS_MC4 = int(TARGET_TOKENS * 0.50)  # 10億トークン
TARGET_TOKENS_CULTURAX = int(TARGET_TOKENS * 0.25)  # 5億トークン
TARGET_TOKENS_WIKIPEDIA = int(TARGET_TOKENS * 0.15)  # 3億トークン
TARGET_TOKENS_OASST1 = int(TARGET_TOKENS * 0.10)  # 2億トークン


def count_tokens(text):
    """テキストのトークン数をカウント"""
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_and_sample_mc4(target_tokens):
    """mC4 Japaneseデータセットをロードしてサンプリング"""
    print(f"\n=== Loading mC4 Japanese (target: {target_tokens:,} tokens) ===")

    # mC4のJapaneseサブセットをストリーミングモードでロード
    dataset = load_dataset(
        "allenai/c4",
        "ja",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    samples = []
    total_tokens = 0

    for i, example in enumerate(dataset):
        text = example['text']
        tokens = count_tokens(text)

        samples.append({"text": text})
        total_tokens += tokens

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1:,} examples, {total_tokens:,} tokens")

        if total_tokens >= target_tokens:
            break

    print(f"Final: {len(samples):,} examples, {total_tokens:,} tokens")

    from datasets import Dataset
    return Dataset.from_list(samples)


def load_and_sample_culturax(target_tokens):
    """CulturaX Japaneseデータセットをロードしてサンプリング"""
    print(f"\n=== Loading CulturaX Japanese (target: {target_tokens:,} tokens) ===")

    # CulturaXのJapaneseサブセットをストリーミングモードでロード
    dataset = load_dataset(
        "uonlp/CulturaX",
        "ja",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    samples = []
    total_tokens = 0

    for i, example in enumerate(dataset):
        text = example['text']
        tokens = count_tokens(text)

        samples.append({"text": text})
        total_tokens += tokens

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1:,} examples, {total_tokens:,} tokens")

        if total_tokens >= target_tokens:
            break

    print(f"Final: {len(samples):,} examples, {total_tokens:,} tokens")

    from datasets import Dataset
    return Dataset.from_list(samples)


def load_and_sample_wikipedia(target_tokens):
    """Wikipedia Japaneseデータセットをロードしてサンプリング"""
    print(f"\n=== Loading Wikipedia Japanese (target: {target_tokens:,} tokens) ===")

    dataset = load_dataset(
        "Cohere/wikipedia-22-12-ja-embeddings",
        split="train",
        trust_remote_code=True
    )

    # テキストカラムのみを選択
    dataset = dataset.select_columns(["text"])

    # トークン数をカウント
    total_tokens = 0
    samples = []

    for i, example in enumerate(dataset):
        text = example['text']
        tokens = count_tokens(text)

        samples.append({"text": text})
        total_tokens += tokens

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1:,} examples, {total_tokens:,} tokens")

        if total_tokens >= target_tokens:
            break

    print(f"Final: {len(samples):,} examples, {total_tokens:,} tokens")

    from datasets import Dataset
    return Dataset.from_list(samples)


def load_and_sample_oasst1(target_tokens):
    """OASST1 Japanese会話データセットをロードしてサンプリング"""
    print(f"\n=== Loading OASST1 Japanese (target: {target_tokens:,} tokens) ===")

    dataset = load_dataset(
        "kunishou/oasst1-89k-ja",
        split="train",
        trust_remote_code=True
    )

    samples = []
    total_tokens = 0

    # 会話形式のデータを整形
    for i, example in enumerate(dataset):
        # conversations形式のデータを結合
        if 'conversations' in example:
            conversations = example['conversations']
            text_parts = []
            for conv in conversations:
                role = conv.get('from', '')
                value = conv.get('value', '')
                text_parts.append(f"{role}: {value}")
            text = "\n".join(text_parts)
        elif 'text' in example:
            text = example['text']
        else:
            continue

        tokens = count_tokens(text)
        samples.append({"text": text})
        total_tokens += tokens

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1:,} examples, {total_tokens:,} tokens")

        if total_tokens >= target_tokens:
            break

    print(f"Final: {len(samples):,} examples, {total_tokens:,} tokens")

    from datasets import Dataset
    return Dataset.from_list(samples)


def main():
    """メイン処理"""
    print("=" * 80)
    print("日本語データセット作成開始")
    print(f"目標トークン数: {TARGET_TOKENS:,} tokens")
    print("=" * 80)

    # 各データセットをロード
    datasets_to_combine = []

    # 1. mC4
    try:
        mc4_dataset = load_and_sample_mc4(TARGET_TOKENS_MC4)
        datasets_to_combine.append(mc4_dataset)
    except Exception as e:
        print(f"Warning: Failed to load mC4: {e}")

    # 2. CulturaX
    try:
        culturax_dataset = load_and_sample_culturax(TARGET_TOKENS_CULTURAX)
        datasets_to_combine.append(culturax_dataset)
    except Exception as e:
        print(f"Warning: Failed to load CulturaX: {e}")

    # 3. Wikipedia
    try:
        wikipedia_dataset = load_and_sample_wikipedia(TARGET_TOKENS_WIKIPEDIA)
        datasets_to_combine.append(wikipedia_dataset)
    except Exception as e:
        print(f"Warning: Failed to load Wikipedia: {e}")

    # 4. OASST1
    try:
        oasst1_dataset = load_and_sample_oasst1(TARGET_TOKENS_OASST1)
        datasets_to_combine.append(oasst1_dataset)
    except Exception as e:
        print(f"Warning: Failed to load OASST1: {e}")

    if not datasets_to_combine:
        raise ValueError("No datasets were successfully loaded!")

    # データセットを結合
    print("\n=== Combining datasets ===")
    combined_dataset = concatenate_datasets(datasets_to_combine)

    # シャッフル
    print("Shuffling dataset...")
    combined_dataset = combined_dataset.shuffle(seed=42)

    # Train/Validation/Test split
    print("Splitting dataset...")
    split1 = combined_dataset.train_test_split(test_size=0.02, seed=42)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=42)

    train_dataset = split1["train"]
    validation_dataset = split2["train"]
    test_dataset = split2["test"]

    # DatasetDictに変換
    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    # 統計情報を表示
    print("\n" + "=" * 80)
    print("データセット統計:")
    print("=" * 80)
    print(f"Train: {len(final_dataset['train']):,} examples")
    print(f"Validation: {len(final_dataset['validation']):,} examples")
    print(f"Test: {len(final_dataset['test']):,} examples")
    print(f"Total: {len(combined_dataset):,} examples")

    # 保存
    print(f"\nSaving to {OUTPUT_DIR}...")
    final_dataset.save_to_disk(OUTPUT_DIR)

    print("\n✓ Dataset creation completed!")
    print(f"Dataset saved to: {OUTPUT_DIR}")
    print("\nTo push to Hugging Face Hub:")
    print(f"  final_dataset.push_to_hub('your-username/japanese-20b-tokens')")


if __name__ == "__main__":
    main()
