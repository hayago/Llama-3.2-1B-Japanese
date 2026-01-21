# Llama-3.2-1B-Japanese

Llama 3.2 1Bモデルの日本語継続事前学習プロジェクト

## 概要

このリポジトリは、Meta AI の Llama 3.2 1B モデルを日本語で継続事前学習するためのスクリプトとデータセットを含んでいます。

## データセット

合計20億トークン程度の日本語データセットを作成します。詳細は [DATASET.md](DATASET.md) を参照してください。

### データセット構成

- **mC4 Japanese** (50%): 一般的なWebテキスト
- **CulturaX Japanese** (25%): クリーンな多言語コーパス
- **Wikipedia Japanese** (15%): Wikipedia日本語版
- **OASST1 Japanese** (10%): 会話・指示データ

全てのデータセットは商用利用可能なライセンス (ODC-BY, Apache 2.0) です。

### データセット作成

```bash
python scripts/create_dataset.py
```

詳細なドキュメントは [DATASET.md](DATASET.md) を参照してください。

## セットアップ

```bash
# 依存関係のインストール (uvを使用)
uv sync

# または pip
pip install -e .
```

## スクリプト

- `scripts/create_dataset.py`: データセット作成スクリプト
- `scripts/train_tokenizer.py`: トークナイザー学習スクリプト
- `scripts/train_model.py`: モデル学習スクリプト
- `scripts/inference.py`: 推論スクリプト

## ライセンス

このプロジェクトで使用するデータセットのライセンスについては [DATASET.md](DATASET.md) を参照してください。
