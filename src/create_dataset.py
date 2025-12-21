from datasets import load_dataset
from datasets import DatasetDict

wikipedia_dataset = load_dataset("Cohere/wikipedia-22-12-ja-embeddings", split="train")

# Filter to only include the text column
wikipedia_dataset = wikipedia_dataset.select_columns(["text"])

# Split the dataset into train and test and validation
split1 = wikipedia_dataset.train_test_split(test_size=0.2, seed=42)
split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
train_dataset = split1["train"]
validation_dataset = split2["train"]
test_dataset = split2["test"]

# Combine train, validation, and test datasets into a DatasetDict
wikipedia_dataset = DatasetDict(
    {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}
)

wikipedia_dataset.save_to_disk("data/wikipedia_dataset")
