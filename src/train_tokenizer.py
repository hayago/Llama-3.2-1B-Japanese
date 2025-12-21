import io
import os
import sentencepiece as spm
from datasets import load_from_disk

# Load the dataset and create an iterator over the text
wikipedia_dataset = load_from_disk("data/wikipedia_dataset")
train_dataset = wikipedia_dataset["train"]
sentence_it = (sample["text"] for sample in train_dataset)

# Create a BytesIO object to store the model
model = io.BytesIO()

# Train the tokenizer
sp = spm.SentencePieceTrainer.train(
    sentence_iterator=sentence_it, vocab_size=32000, model_writer=model
)

# Serialize the model as file.
os.makedirs("data/sentencepiece", exist_ok=True)
with open("data/sentencepiece/out.model", "wb") as f:
    f.write(model.getvalue())
