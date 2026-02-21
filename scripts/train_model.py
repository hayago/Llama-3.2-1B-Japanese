import os
import torch
import sentencepiece as spm
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import DefaultDataCollator
from transformers import Trainer, TrainingArguments, TrainerCallback


# Load the sentencepiece model
SP_MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentencepiece", "out.model")
sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)


def log_training_info(max_steps, training_args):
    batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    print(f"Max steps: {max_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")


# Evaluation prompts for text generation
EVAL_PROMPTS = [
    "今日は朝から雨が降っていて、",
    "静かな図書館の中で、私は一冊の古い本を開いた。",
    "夜が更けてきた。街の明かりは、",
    "この町では、季節が変わるたびに、人々の暮らし方や考え方が少しずつ変化していくのだが、",
    "遠くで鐘の音が鳴り始めたとき、",
    "春の訪れとともに",
    "科学技術の進歩により、",
    "彼女は窓の外を見つめていた。",
    "歴史を紐解くと、",
    "未来への希望を胸に",
    "朝の光が差し込む窓辺で、",
    "古い写真を見つめながら",
    "雪が静かに降り始めた夜に",
    "友人からの手紙を読んでいると、",
    "海辺を歩いていたとき、",
    "新しい季節の始まりを感じて",
    "図書館の奥で見つけた本には、",
    "夕暮れ時の街角で",
    "母の作った料理の香りが",
    "電車の窓から見える風景は",
]


class GenerationCallback(TrainerCallback):
    """Callback to generate evaluation texts at checkpoint save time."""

    def __init__(self, sp_processor, accelerator):
        self.sp = sp_processor
        self.accelerator = accelerator

    def on_save(self, args, state, control, model=None, **kwargs):
        """Generate texts when checkpoint is saved."""
        # Only run on main process
        if not self.accelerator.is_main_process:
            return

        if model is None:
            raise Exception("Model is None")

        print("\n" + "=" * 80)
        print(f"Generating evaluation texts at step {state.global_step}")
        print("=" * 80)

        model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(EVAL_PROMPTS, 1):
                try:
                    # Encode the prompt
                    input_ids = self.sp.encode_as_ids(prompt)
                    input_ids = [self.sp.bos_id()] + input_ids
                    input_ids_tensor = (
                        torch.tensor(input_ids).unsqueeze(0).to(model.device)
                    )

                    # Generate text
                    outputs = model.generate(
                        input_ids_tensor,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )

                    # Decode the generated text
                    generated_ids = outputs[0].cpu().tolist()
                    generated_text = self.sp.decode(generated_ids)

                    print(f"\n[{i}/{len(EVAL_PROMPTS)}] Prompt: {prompt}")
                    print(f"Generated: {generated_text}")
                    print("-" * 80)
                except Exception as e:
                    print(f"Error generating text for prompt {i}: {e}")

        model.train()
        print("=" * 80 + "\n")


def main():
    # Check if resume is enabled via environment variable
    resume = os.environ.get("RESUME", "") == "1"

    # Load the preprocessed dataset from Hugging Face Hub
    acc = Accelerator()
    packed_dataset_train = load_dataset("hayago/cc100-ja-packed-2048", split="train", streaming=True)
    packed_dataset_train = packed_dataset_train.rename_column("chunks", "input_ids")
    packed_dataset_train = packed_dataset_train.map(
        lambda x: {"labels": x["input_ids"]},
    )

    # Initialize or load the model
    checkpoint_dir = "./checkpoints"
    if resume:
        # Clone repository from Hugging Face Hub (only main process)
        if acc.is_main_process:
            print(
                f"Downloading checkpoint from {os.environ.get('RESUME_FROM_CHECKPOINT')}..."
            )
            snapshot_download(
                repo_id="hayago/Veloce-100M-v2",
                allow_patterns=os.environ.get("RESUME_FROM_CHECKPOINT") + "/**",
                local_dir=checkpoint_dir,
                local_dir_use_symlinks=False,
            )
            print("Checkpoint downloaded successfully")
        acc.wait_for_everyone()
        model = LlamaForCausalLM.from_pretrained(
            os.path.join(checkpoint_dir, os.environ.get("RESUME_FROM_CHECKPOINT"))
        )
    else:
        model_config = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=9,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=4096,
            vocab_size=sp.vocab_size(),
            bos_token_id=sp.bos_id(),
            eos_token_id=sp.eos_id(),
            hidden_act="silu",
            attention_dropout=0.0,
            attention_bias=False,
            initializer_range=0.02,
            rope_theta=500000.0,
            max_position_embeddings=2048,
            rms_norm_eps=1e-5,
        )
        model = LlamaForCausalLM(model_config)

        # Log model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    # Training arguments
    # ローカルテスト用の設定
    is_test = os.environ.get("TEST_MODE", "") == "1"

    training_args = TrainingArguments(
        output_dir="./Veloce-100M-v2",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8 if not is_test else 1,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        weight_decay=0.001,
        fp16=True,
        logging_strategy="steps",
        logging_steps=50 if not is_test else 1,
        logging_first_step=True,
        learning_rate=1e-4,
        save_steps=2000 if not is_test else 5,
        report_to="wandb" if not is_test else "none",
        push_to_hub=True if not is_test else False,
        hub_strategy="all_checkpoints",
        eval_strategy="no",
        max_steps=86193 if not is_test else 10,
    )

    if acc.is_main_process:
        log_training_info(training_args.max_steps, training_args)

    # Create generation callback
    generation_callback = GenerationCallback(sp, acc)

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed_dataset_train,
        data_collator=DefaultDataCollator(),
        callbacks=[generation_callback],
    )
    trainer.train(
        resume_from_checkpoint=os.path.join(
            checkpoint_dir, os.environ.get("RESUME_FROM_CHECKPOINT")
        )
        if resume
        else None
    )


if __name__ == "__main__":
    main()
