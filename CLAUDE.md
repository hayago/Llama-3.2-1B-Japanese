# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Japanese LLM pretraining project that creates a ~6B token Japanese dataset (3x the original 2B) and trains a Japanese-specialized Llama 3.2 1B model. Uses 6 open-license sources (CC-BY, CC-BY-SA, ODC-BY, CC ToU).

## Common Commands

### Dataset Creation

```bash
# Quick test run
python3 scripts/create_pretrain_dataset.py --test_mode --output_dir ./data/pretrain_dataset_test

# Full production build (~6B tokens)
python3 scripts/create_pretrain_dataset.py \
  --output_dir ./data/pretrain_dataset \
  --target_tokens 6000000000 \
  --dedup_db ./data/dedup/hashes.sqlite \
  --sp_model ./scripts/sentencepiece/out.model

# With HuggingFace Hub upload
python3 scripts/create_pretrain_dataset.py \
  --output_dir ./data/pretrain_dataset \
  --target_tokens 6000000000 \
  --dedup_db ./data/dedup/hashes.sqlite \
  --sp_model ./scripts/sentencepiece/out.model \
  --push_to_hub \
  --hub_repo <username>/<dataset-name>
```

### Training

```bash
cd scripts && ./launch_training.sh
# Or directly:
accelerate launch --config_file scripts/accelerate_config.yml scripts/train_model.py
```

### Tokenizer Training

```bash
cd scripts && python3 train_tokenizer.py
```

### Inference

```bash
cd scripts && python3 inference.py
```

## Architecture

### Dataset Pipeline (`scripts/create_pretrain_dataset.py`)

1. **DatasetConfig dataclass**: Configures each source with name, path, license, ratio, and optional custom formatters/quality filters
2. **Streaming load**: HuggingFace datasets with streaming to handle large sources
3. **Text processing**: NFKC normalization → length/Japanese character ratio filter → MD5 deduplication via SQLite
4. **Token estimation**: Character-based approximation (1.8 chars/token) with optional SentencePiece exact counting
5. **Blending**: Ratio-based composition with overflow redistribution (C4 58%, cc100 23%, Wikipedia 8%, StackOverflow 5%, StackExchange 4%, Aozora 2%). When smaller sources exhaust, surplus tokens are redistributed to larger sources.

### Training (`scripts/train_model.py`)

- Base: Llama 3.2 1B architecture with custom SentencePiece tokenizer (32k vocab)
- Context: 2048 tokens with packing
- Config: bf16, batch 1, grad accum 64, lr 1e-4, cosine scheduler, 5 epochs
- Distributed: AWS SageMaker via Accelerate with spot instances
- Checkpoints: Every 100 steps, pushed to HF Hub
- GenerationCallback: Evaluates on 20 Japanese prompts at each checkpoint

### Key Files

- `scripts/create_pretrain_dataset.py` - Main dataset creation pipeline
- `scripts/train_model.py` - Training script with HF Trainer
- `scripts/train_tokenizer.py` - SentencePiece tokenizer training
- `scripts/inference.py` - Model inference demo
- `scripts/accelerate_config.yml.example` - AWS SageMaker distributed training config template
- `scripts/sentencepiece/out.model` - Pre-trained 32k vocab tokenizer

## Environment

- Python ≥3.10
- Key deps: torch 2.8.0, transformers 4.56.2, accelerate 1.12.0, datasets 4.4.1
- Install: `uv sync` or `pip install -e .`
