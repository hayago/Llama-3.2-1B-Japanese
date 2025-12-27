import argparse
import sentencepiece as spm
from transformers import LlamaForCausalLM, AutoConfig
from datasets import load_dataset
from transformers import DefaultDataCollator
from transformers import Trainer, TrainingArguments

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
is_resume = args.resume


# Load the sentencepiece model
sp = spm.SentencePieceProcessor(model_file="data/sentencepiece/out.model")

# TODO resume true argument


# Tokenize and pack the text into chunks of context_length and add labels
def preprocess(examples) -> dict:
    # Encode the text into ids and add BOS and EOS
    all_ids = []
    for ids in sp.encode_as_ids(examples["text"]):
        all_ids.extend([sp.bos_id()] + ids + [sp.eos_id()])
    # Pack the all_ids into chunks of context_length
    context_length = 2048
    total_length = (len(all_ids) // context_length) * context_length
    chunks = [
        all_ids[i : i + context_length] for i in range(0, total_length, context_length)
    ]
    return {
        "input_ids": chunks,
        "labels": chunks.copy(),  # labels are the same as input_ids for causal language modeling
    }


# Load the dataset
wikipedia_dataset = load_dataset("hayago/cohere-wikipedia-22-12-ja-text")
train_dataset = wikipedia_dataset["train"].select(range(1000))  # TODO Remove
val_dataset = wikipedia_dataset["validation"].select(range(100))  # TODO Remove
# train_dataset = wikipedia_dataset["train"]
# val_dataset = wikipedia_dataset["validation"]

# Preprocess the dataset
packed_dataset_train = train_dataset.map(
    preprocess, batched=True, remove_columns=["text"]
)
packed_dataset_val = val_dataset.map(preprocess, batched=True, remove_columns=["text"])

if is_resume:
    model = LlamaForCausalLM.from_pretrained("Llama-3.2-1B-Japanese")
else:
    # Initialize the model
    model_config = AutoConfig.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        vocab_size=sp.vocab_size(),
        num_attention_heads=4,  # TODO Remove
        num_hidden_layers=2,  # TODO Remove
        num_key_value_heads=2,  # TODO Remove
    )
    model = LlamaForCausalLM(model_config)


# TODO
# Loss monitoring -> WandB
# eval perplexity?
# resume training, checkpoint
# accelerate
# Inference
# Sagemaker

# Training arguments
training_args = TrainingArguments(
    output_dir="Llama-3.2-1B-Japanese",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    weight_decay=0.1,
    fp16=True,
    logging_strategy="steps",
    logging_steps=1,  # TODO to default 500
    eval_strategy="steps",
    eval_steps=2,  # TODO 2000?
    num_train_epochs=5,  # TODO
    learning_rate=5e-4,
    # save_steps=5_000, TODO
    save_steps=2,
    report_to="wandb",
    push_to_hub=True,
    hub_strategy="all_checkpoints",
)

# Show training parameters
num_samples = len(packed_dataset_train)
batch_size = training_args.per_device_train_batch_size
gradient_accumulation_steps = training_args.gradient_accumulation_steps
num_epochs = training_args.num_train_epochs
steps_per_epoch = num_samples // (batch_size * gradient_accumulation_steps)
total_steps = steps_per_epoch * num_epochs

print(f"Number of samples: {num_samples}")
print(f"Batch size: {batch_size}")
print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"Number of epochs: {num_epochs}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total steps: {total_steps}")

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=packed_dataset_train,
    eval_dataset=packed_dataset_val,
    data_collator=DefaultDataCollator(),
)

trainer.train(resume_from_checkpoint=is_resume)
