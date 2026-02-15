import io
import os
import sentencepiece as spm
from datasets import load_dataset

# Load the first 20% of the dataset for tokenizer training
train_dataset = load_dataset("hayago/cc100-ja-fork", split="train[:20%]", streaming=True)
sentence_it = (sample["text"] for sample in train_dataset)

# Create a BytesIO object to store the model
model = io.BytesIO()

# Train the tokenizer
sp = spm.SentencePieceTrainer.train(
    sentence_iterator=sentence_it,
    vocab_size=32000,
    model_writer=model,
    max_sentence_length=16384,
    user_defined_symbols=[
        "<system>",
        "</system>",
        "<user>",
        "</user>",
        "<assistant>",
        "</assistant>",
        "<sep>",
        "<reserved_0>",
        "<reserved_1>",
        "<reserved_2>",
        "<reserved_3>",
        "<reserved_4>",
        "<reserved_5>",
        "<reserved_6>",
        "<reserved_7>",
        "<reserved_8>",
        "<reserved_9>",
        "<reserved_10>",
        "<reserved_11>",
        "<reserved_12>",
        "<reserved_13>",
        "<reserved_14>",
        "<reserved_15>",
    ],
)

# Serialize the model as file.
os.makedirs("sentencepiece", exist_ok=True)
with open("sentencepiece/out.model", "wb") as f:
    f.write(model.getvalue())
