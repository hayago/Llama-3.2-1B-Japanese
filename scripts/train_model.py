import os
import sentencepiece as spm
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import LlamaForCausalLM, AutoConfig
from transformers import DefaultDataCollator
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


# Load the sentencepiece model
sp = spm.SentencePieceProcessor(model_file="sentencepiece/out.model")


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


def prepare_dataset():
    # Load the dataset
    dataset_name = "hayago/cohere-wikipedia-22-12-ja-text"
    wikipedia_dataset = load_dataset(dataset_name)

    train_dataset = wikipedia_dataset["train"]

    # Use half of the validation dataset for evaluation
    val_dataset_size = len(wikipedia_dataset["validation"])
    val_dataset = wikipedia_dataset["validation"].select(range(val_dataset_size // 2))

    # Preprocess the dataset
    packed_dataset_train = train_dataset.map(
        preprocess, batched=True, remove_columns=["text"]
    )
    packed_dataset_val = val_dataset.map(
        preprocess, batched=True, remove_columns=["text"]
    )

    return packed_dataset_train, packed_dataset_val


def log_training_info(num_samples, training_args):
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


def main():
    # Check if resume is enabled via environment variable
    resume = os.environ.get("RESUME", "") == "1"

    # Prepare the dataset with caching for distributed training
    acc = Accelerator()
    if acc.is_main_process:
        # Main process creates dataset cache
        packed_dataset_train, packed_dataset_val = prepare_dataset()
    acc.wait_for_everyone()
    if not acc.is_main_process:
        # Other processes load from cache
        packed_dataset_train, packed_dataset_val = prepare_dataset()

    # Determine output directory
    output_dir = "./Veloce-1B"

    # Initialize or load the model
    checkpoint = None
    if resume:
        # Clone repository from Hugging Face Hub (only main process)
        if acc.is_main_process:
            print("Cloning repository...")
            snapshot_download(
                repo_id="hayago/Veloce-1B",
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            print("Repository cloned successfully")
        acc.wait_for_everyone()
        checkpoint = get_last_checkpoint(output_dir)
        model = LlamaForCausalLM.from_pretrained(checkpoint)
    else:
        model_config = AutoConfig.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            vocab_size=sp.vocab_size(),
            bos_token_id=sp.bos_id(),
            eos_token_id=sp.eos_id(),
        )
        model = LlamaForCausalLM(model_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        warmup_steps=4,  # -> 40
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        fp16=True,
        logging_strategy="steps",
        logging_steps=2,  # -> 20
        eval_strategy="steps",
        logging_first_step=True,
        eval_steps=10,  # -> 100
        num_train_epochs=5,  # -> 5
        learning_rate=5e-4,
        save_steps=10,  # -> 100
        report_to="wandb",
        push_to_hub=True,
        hub_strategy="all_checkpoints",
    )

    if acc.is_main_process:
        log_training_info(len(packed_dataset_train), training_args)

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed_dataset_train,
        eval_dataset=packed_dataset_val,
        data_collator=DefaultDataCollator(),
    )
    trainer.train(resume_from_checkpoint=checkpoint if resume else None)


if __name__ == "__main__":
    main()
