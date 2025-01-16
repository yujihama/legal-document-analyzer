
# 法的文書分析プラットフォーム

## システム概要
高度なNLPとクラスタリング技術を活用した法的文書分析プラットフォームです。法令文書と社内規定の遵守状況を自動的に分析し、包括的なコンプライアンス評価を提供します。

## 技術スタック
- **フロントエンド**: Streamlit
- **バックエンド**: Python 3.10
- **機械学習/NLP**:
  - OpenAI API (GPT-4o, text-embedding-3-large)
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

## 詳細な処理フロー

### 1. ドキュメント処理フェーズ (DocumentProcessor)
1. **文書の前処理**
   - `process_legal_document()`: 法令文書の処理
     - チャンク分割 (max_tokens=5000)
     - 階層的コンテキスト抽出
   - `process_internal_document()`: 社内規定文書の処理
     - チャンク分割と最適化

### 2. テキスト埋め込み処理 (EmbeddingProcessor)
1. **ベクトル化**
   - `get_embedding()`: OpenAI APIによるテキストのベクトル化
   - キャッシュベースの最適化
   - 並列処理によるバッチ処理

2. **クラスタリング処理**
   - `perform_clustering()`: HDBSCANによるクラスタリング
     - 密度ベースのクラスタリング
     - 階層的サブクラスタリング
     - キャッシュ管理

### 3. コンプライアンス分析 (GPTProcessor)
1. **要件抽出**
   - `extract_requirements()`: 要件と禁止事項の抽出
   - `extract_hierarchical_context()`: 文書構造の解析

2. **コンプライアンス評価**
   - `analyze_compliance()`: 規制要件との適合性評価
   - `analyze_cluster_compliance()`: クラスタベースの評価
   - 多段階評価プロセス

3. **レポート生成**
   - `generate_report()`: 包括的なレポートの生成
   - `summarize_cluster_requirements()`: クラスタ要約
   - PDFレポート出力

### 4. UI処理フロー (Streamlitコンポーネント)
1. **document_upload**
   - ファイルアップロード処理
   - 初期バリデーション
   - セッション状態管理

2. **analysis_view**
   - クラスタ分析結果の表示
   - インタラクティブな分析ビュー
   - リアルタイムメトリクス計算

3. **report_view**
   - レポート表示と出力
   - グラフ生成
   - PDFエクスポート

### データフロー図
```
[文書入力] → [DocumentProcessor]
     ↓
[EmbeddingProcessor] → [ベクトルDB]
     ↓
[ClusteringProcessor] → [クラスタ分析]
     ↓
[GPTProcessor] → [コンプライアンス評価]
     ↓
[レポート生成] → [UI表示/PDF出力]
```

### キャッシュ戦略
1. **埋め込みキャッシュ**
   - テキストハッシュベース
   - JSONシリアライズ
   - 有効期限管理

2. **クラスタキャッシュ**
   - クラスタIDベース
   - 増分更新対応
   - メモリ使用量最適化

3. **分析結果キャッシュ**
   - ドキュメントハッシュベース
   - 部分更新サポート
   - ディスク永続化

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
