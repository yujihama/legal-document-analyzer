
# 法的文書分析プラットフォーム

## システム概要
高度なNLPとクラスタリング技術を活用した法的文書分析プラットフォームです。法令文書と社内規定の遵守状況を自動的に分析し、包括的なコンプライアンス評価を提供します。

## 技術スタック
- **フロントエンド**: Streamlit
- **バックエンド**: Python 3.10
- **機械学習/NLP**:
  - OpenAI API (GPT-4, text-embedding-3-large)
  - HDBSCAN (階層的密度ベースクラスタリング)
  - scikit-learn
  - FAISS (類似度検索)
- **データ処理**: NumPy, Pandas
- **永続化**: JSON-based File Storage

## システムアーキテクチャ

### コアモジュール構成
```
├── processors/           # コア処理モジュール
│   ├── gpt_processor.py       # LLM関連の処理
│   ├── embedding_processor.py # ベクトル埋め込み処理
│   ├── clustering_processor.py # クラスタリング処理
│   └── document_processor.py  # 文書処理
├── components/          # UIコンポーネント
│   ├── document_upload/     # 文書アップロード機能
│   ├── analysis_view/      # 分析結果表示
│   └── report_view/        # レポート生成・表示
├── models/             # データモデル
└── utils/              # ユーティリティ機能
```

### 主要処理フロー
1. **文書処理フェーズ**
   - 文書のチャンク分割 (max_tokens=5000)
   - 階層的コンテキスト抽出
   - 要件・禁止事項の抽出

2. **分析フェーズ**
   - テキストベクトル化 (OpenAI Embeddings)
   - 密度ベースクラスタリング (HDBSCAN)
   - クラスタ代表テキスト生成

3. **コンプライアンス評価フェーズ**
   - クラスタベースの要件マッチング
   - 多段階コンプライアンス評価
   - 統計的スコアリング

## 主要コンポーネント詳細

### DocumentProcessor
- 文書の前処理と構造化
- トークン数の最適化
- キャッシュ管理

### EmbeddingProcessor
```python
class EmbeddingProcessor:
    def __init__(self):
        self.client = OpenAI()
        self._embedding_cache = {}
```
- テキストのベクトル化
- キャッシュベースの最適化
- バッチ処理対応

### ClusteringProcessor
- マルチアルゴリズム対応 (HDBSCAN, Hierarchical, DPMM)
- 並列処理によるクラスタリング
- 動的クラスタサイズ最適化

### GPTProcessor
- プロンプトテンプレート管理
- レート制限対応
- 多言語サポート

## キャッシュ戦略
- ベクトル埋め込みキャッシュ
- クラスタリング結果キャッシュ
- 分析結果キャッシュ

## エラーハンドリング
- API レート制限リトライ
- 階層的エラー検出
- グレースフルデグラデーション

## パフォーマンス最適化
- 並列処理による大規模文書対応
- インクリメンタルアップデート
- メモリ使用量の最適化

## セットアップ手順
1. リポジトリのクローン
```bash
git clone [repository-url]
```

2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

3. アプリケーションの起動
```bash
streamlit run main.py
```

## 設定パラメータ
- `max_tokens`: 5000 (チャンクサイズ)
- `min_cluster_size`: 2 (最小クラスタサイズ)
- `distance_threshold`: 1.5 (類似度閾値)

## 拡張性
- 新規クラスタリングアルゴリズムの追加
- カスタムプロンプトの定義
- 分析ルールのカスタマイズ

## 制限事項
- OpenAI API の利用制限
- 大規模文書処理時のメモリ要件
- クラスタリング精度の依存性

## ライセンス
このプロジェクトは [ライセンスを追加] の下で公開されています。
