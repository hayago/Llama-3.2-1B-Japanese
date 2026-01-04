import io
import os
import sentencepiece as spm
from datasets import load_dataset

# Load the dataset and create an iterator over the text
wikipedia_dataset = load_dataset("hayago/cohere-wikipedia-22-12-ja-text")
train_dataset = wikipedia_dataset["train"]
sentence_it = (sample["text"] for sample in train_dataset)

# Create a BytesIO object to store the model
model = io.BytesIO()

# Train the tokenizer
sp = spm.SentencePieceTrainer.train(
    sentence_iterator=sentence_it,
    vocab_size=32000,
    model_writer=model,
    user_defined_symbols=[
        "<system>",
        "<user>",
        "<assistant>",
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
