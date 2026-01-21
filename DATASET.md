# 日本語データセット (Japanese Dataset)

合計20億トークン程度の日本語データセットを作成するためのスクリプトとドキュメント。

## 概要

このデータセットは、Llama 3.2 1Bモデルの日本語継続事前学習のために作成されました。様々な種類のテキストデータを混合し、商用利用が可能なデータセットのみから構成されています。

## データセット構成

| データセット | ライセンス | 目標トークン数 | 割合 | 内容 |
|------------|-----------|--------------|------|------|
| allenai/c4 (mC4 Japanese) | ODC-BY | 10億 | 50% | 一般的なWebテキスト |
| uonlp/CulturaX (Japanese) | ODC-BY | 5億 | 25% | クリーンな多言語コーパス |
| Cohere/wikipedia-22-12-ja-embeddings | Apache 2.0 | 3億 | 15% | Wikipedia日本語版 |
| kunishou/oasst1-89k-ja | Apache 2.0 | 2億 | 10% | 会話・指示データ |
| **合計** | | **20億** | **100%** | |

### データセット詳細

#### 1. mC4 Japanese (allenai/c4)
- **ライセンス**: ODC-BY (Open Data Commons Attribution License)
- **商用利用**: 可能
- **内容**: Common Crawlから抽出された多言語コーパスC4の日本語サブセット
- **特徴**: 一般的なWebテキストで、幅広いトピックをカバー

#### 2. CulturaX Japanese (uonlp/CulturaX)
- **ライセンス**: ODC-BY (mC4とOSCARのライセンスに従う)
- **商用利用**: 可能
- **内容**: mC4とOSCARをクリーニング・統合した多言語コーパス
- **特徴**: 高品質でクリーンなテキストデータ

#### 3. Wikipedia Japanese (Cohere/wikipedia-22-12-ja-embeddings)
- **ライセンス**: Apache 2.0
- **商用利用**: 可能
- **内容**: 日本語Wikipediaのテキストデータ
- **特徴**: 百科事典的な知識ベース、高品質なテキスト

#### 4. OASST1 Japanese (kunishou/oasst1-89k-ja)
- **ライセンス**: Apache 2.0
- **商用利用**: 可能
- **内容**: OpenAssistant Conversations Datasetの日本語翻訳版
- **特徴**: 会話形式・指示応答データ、対話能力の向上に貢献

## データセット作成方法

### 前提条件

```bash
# Python 3.10以上
# 必要なパッケージはpyproject.tomlに記載
uv sync
```

### データセットの生成

```bash
python scripts/create_dataset.py
```

このスクリプトは以下の処理を実行します:

1. 各データソースから目標トークン数に達するまでサンプリング
2. 全データセットを結合
3. データをシャッフル
4. Train (98%) / Validation (1%) / Test (1%) に分割
5. `data/japanese_20b_tokens/` に保存

### 処理時間の目安

- データのダウンロード: 数時間〜
- トークンカウント・サンプリング: 数時間〜
- 合計: 環境により大きく異なる

## Hugging Face Hubへのアップロード

データセット作成後、以下のコードでHugging Face Hubにアップロードできます:

```python
from datasets import load_from_disk

# ローカルからデータセットをロード
dataset = load_from_disk("data/japanese_20b_tokens")

# Hugging Face Hubにプッシュ
dataset.push_to_hub("your-username/japanese-20b-tokens")
```

または、スクリプトの最後に以下を追加:

```python
final_dataset.push_to_hub("your-username/japanese-20b-tokens")
```

## ライセンスについて

このデータセットは複数のソースから構成されており、各ソースのライセンスに従います:

- **ODC-BY**: 出典の明記が必要。商用利用可能。
- **Apache 2.0**: ライセンスと著作権表示が必要。商用利用可能。

### 商用利用に関する注意事項

全てのデータセットは商用利用が可能ですが、以下の点に注意してください:

1. **帰属表示**: ODC-BYおよびApache 2.0ライセンスでは、元のデータセットの出典を明記する必要があります
2. **ライセンス継承**: データセットを再配布する場合は、元のライセンス条項を含める必要があります
3. **Common Crawl Terms of Use**: mC4とCulturaXはCommon Crawlから派生しているため、Common Crawlの利用規約にも従う必要があります

### 推奨される帰属表示

このデータセットを使用する場合、以下のような帰属表示を推奨します:

```
このデータセットは以下のデータソースから構成されています:
- allenai/c4 (mC4 Japanese subset) - ODC-BY License
- uonlp/CulturaX (Japanese subset) - ODC-BY License
- Cohere/wikipedia-22-12-ja-embeddings - Apache 2.0 License
- kunishou/oasst1-89k-ja - Apache 2.0 License
```

## データセットの品質管理

### フィルタリング

現在のスクリプトでは基本的なフィルタリングのみを行っています。必要に応じて以下のフィルタリングを追加することを推奨します:

- 長さフィルター (極端に短い/長いテキストの除外)
- 重複除去
- 品質スコアリング (perplexityなど)
- 有害コンテンツのフィルタリング

### データ検証

データセット作成後、以下の点を確認することを推奨します:

- トークン数の確認
- テキストの品質確認 (サンプリング)
- 言語の一貫性確認
- データソース間のバランス確認

## カスタマイズ

### トークン数の調整

`scripts/create_dataset.py`の以下の変数を変更することで、目標トークン数を調整できます:

```python
TARGET_TOKENS = 2_000_000_000  # 目標トークン数

# 各データセットの割合
TARGET_TOKENS_MC4 = int(TARGET_TOKENS * 0.50)
TARGET_TOKENS_CULTURAX = int(TARGET_TOKENS * 0.25)
TARGET_TOKENS_WIKIPEDIA = int(TARGET_TOKENS * 0.15)
TARGET_TOKENS_OASST1 = int(TARGET_TOKENS * 0.10)
```

### データソースの追加・削除

`main()`関数内で、使用するデータセットを追加・削除できます。各データセットのロード処理は`try-except`でラップされているため、一部のデータセットが利用できない場合でもスクリプトは継続します。

## トラブルシューティング

### データセットのダウンロードに失敗する

- ネットワーク接続を確認してください
- Hugging Face Hubへのアクセス認証が必要な場合があります: `huggingface-cli login`
- 一部のデータセットはアクセス申請が必要な場合があります

### メモリ不足エラー

- ストリーミングモードを使用しているため、大規模データセットでもメモリ使用量は抑制されています
- それでもメモリ不足が発生する場合は、目標トークン数を減らすか、バッチ処理を実装してください

### トークンカウントが遅い

- GPUが利用可能な場合は、トークナイザーをGPU上で実行することで高速化できます
- または、簡易的な文字数ベースの推定を使用することも検討できます

## 参考資料

### データセット出典

- [allenai/c4](https://huggingface.co/datasets/allenai/c4)
- [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)
- [Cohere/wikipedia-22-12-ja-embeddings](https://huggingface.co/datasets/Cohere/wikipedia-22-12-ja-embeddings)
- [kunishou/oasst1-89k-ja](https://huggingface.co/datasets/kunishou/oasst1-89k-ja)

### ライセンス詳細

- [ODC-BY License](https://opendatacommons.org/licenses/by/)
- [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
- [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use)

## 更新履歴

- 2026-01-21: 初版作成
  - mC4, CulturaX, Wikipedia, OASST1を統合した20億トークンデータセット
  - 商用利用可能なライセンスのみを使用
