import os

import sentencepiece as spm
from datasets import load_dataset


SP_MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentencepiece", "out.model")
sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)


def preprocess(examples) -> dict:
    # Encode the text into ids and add BOS and EOS
    all_ids = []
    for ids in sp.encode_as_ids(examples["text"]):
        if len(ids) == 0:
            continue

        all_ids.extend([sp.bos_id()] + ids + [sp.eos_id()])

    # Pack the all_ids into chunks of context_length
    context_length = 2048
    total_length = (len(all_ids) // context_length) * context_length
    chunks = [
        all_ids[i : i + context_length] for i in range(0, total_length, context_length)
    ]

    return {
        "chunks": chunks
    }


if __name__ == "__main__":
    # train_dataset = load_dataset("hayago/cc100-ja-fork", split="train")
    train_dataset = load_dataset(os.path.expanduser("~/workspace/hobby/dataset/cc100-ja"), split="train")

    packed_dataset_train = train_dataset.map(
        preprocess, batched=True, remove_columns=train_dataset.column_names, drop_last_batch=True
    )

    packed_dataset_train.push_to_hub("hayago/cc100-ja-packed-2048", max_shard_size="2GB")
