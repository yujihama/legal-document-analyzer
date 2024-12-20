import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional
import faiss
import hdbscan
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ClusterInfo:
    id: int
    texts: List[str]
    centroid: np.ndarray
    representative_text: Optional[str] = None
    summary: Optional[str] = None
    created_at: datetime = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with JSON serializable types"""
        return {
            'id': int(self.id),
            'texts': list(self.texts) if self.texts else [],  # 確実にリストとして保持
            'centroid': self.centroid.tolist() if isinstance(self.centroid, np.ndarray) else None,
            'representative_text': self.representative_text,
            'summary': self.summary,
            'created_at': self.created_at.isoformat()
        }

class EmbeddingProcessor:

    def __init__(self):
        self.client = OpenAI()
        self.stored_texts = []
        self.clusters: List[ClusterInfo] = []
        self._embedding_cache = {}

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using OpenAI API with enhanced preprocessing"""
        # Prepare cache key with normalized text
        cache_key = text.strip()
        
        # Check cache first
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Enhanced text preprocessing
        processed_text = self._preprocess_text(text)
        
        # If not in cache, compute embedding
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=processed_text,
                encoding_format="float"  # Ensure consistent encoding
            )
            embedding = np.array(response.data[0].embedding)
            
            # Normalize the embedding vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Store in cache
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for better semantic separation"""
        # 基本的なテキストクリーニング
        text = text.strip()
        
        # 重要な区切り文字を強調
        text = text.replace('。', '。 ')  # 文末の区切りを強調
        text = text.replace('、', '、 ')  # 句読点の区切りを強調
        
        # 重要なキーワードの前後にスペースを追加して強調
        keywords = ['禁止', '義務', '要件', '必須', '規則', '規定', '手順', 
                   '基準', '対策', '管理', 'セキュリティ', '方針']
        for keyword in keywords:
            text = text.replace(keyword, f' {keyword} ')
        
        # 複数スペースを単一スペースに置換
        text = ' '.join(text.split())
        
        return text

    def batch_embed_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Get embeddings for a batch of texts"""
        all_embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            # Process each text in the batch
            for text in batch:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)

    def create_index(self, texts: List[str]):
        """Store texts and their embeddings for similarity comparison"""
        self.stored_texts = texts
        # Pre-compute embeddings for efficiency
        self._embedding_cache.update({
            text: self.get_embedding(text)
            for text in texts
        })

    def find_similar(self, query: str, distance_threshold: float = 1.5) -> List[Dict]:
        """Find similar texts using distance-based similarity with cached embeddings"""
        query_embedding = self.get_embedding(query)
        
        # Use pre-computed embeddings from cache
        results = []
        for text in self.stored_texts:
            # Get cached embedding
            text_embedding = self._embedding_cache.get(text)
            if text_embedding is None:
                continue
                
            distance = float(np.linalg.norm(query_embedding - text_embedding))
            
            if distance < distance_threshold:
                results.append({
                    'text': text,
                    'score': 1.0 / (1.0 + distance)  # Convert distance to similarity score
                })
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def perform_clustering(self, min_cluster_size: int = 2) -> List[ClusterInfo]:
        """Perform HDBSCAN clustering on the stored embeddings"""
        if not self.stored_texts:
            raise ValueError("No texts have been stored yet")

        n_texts = len(self.stored_texts)
        print(f"Starting clustering with {n_texts} texts")

        # 特別なケース：データポイントが少ない場合
        if n_texts < 3:
            print("Too few data points for clustering, creating single cluster")
            # 単一クラスタとして処理
            embeddings = self.batch_embed_texts(self.stored_texts)
            centroid = np.mean(embeddings, axis=0) if n_texts > 1 else embeddings[0]
            
            return [ClusterInfo(
                id=0,
                texts=self.stored_texts,
                centroid=centroid,
                representative_text=self.stored_texts[0] if self.stored_texts else None
            )]

        # Extract all embeddings in parallel
        embeddings_array = self.batch_embed_texts(self.stored_texts)
        print(f"Generated embeddings array shape: {embeddings_array.shape}")

        # データセットサイズに基づいてパラメータを調整
        min_cluster_size = max(2, min(3, n_texts - 1))
        min_samples = 1  # 最小値を1に固定
        
        print(f"Clustering parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

        # Perform HDBSCAN clustering with adjusted parameters for finer clustering
        try:
            print("Attempting HDBSCAN clustering with refined parameters...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='cosine',  # Changed to cosine similarity for better semantic grouping
                cluster_selection_method='eom',  # Excess of Mass for better cluster separation
                cluster_selection_epsilon=0.2,  # Smaller epsilon for finer cluster boundaries
                prediction_data=True,
                alpha=0.8  # Increased alpha for more conservative cluster merging
            )
            
            cluster_labels = clusterer.fit_predict(embeddings_array)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"Clustering complete. Found {n_clusters} clusters")

            # クラスタリングが失敗した場合（全てのポイントがノイズとして分類された場合）
            if n_clusters == 0:
                print("No valid clusters found, creating single cluster")
                return [ClusterInfo(
                    id=0,
                    texts=self.stored_texts,
                    centroid=np.mean(embeddings_array, axis=0),
                    representative_text=self.stored_texts[0] if self.stored_texts else None
                )]
            
        except Exception as e:
            print(f"Detailed clustering error: {str(e)}")
            print("Input data shape:", embeddings_array.shape)
            print("Input data type:", embeddings_array.dtype)
            # エラーが発生した場合は単一クラスタとして処理
            return [ClusterInfo(
                id=0,
                texts=self.stored_texts,
                centroid=np.mean(embeddings_array, axis=0),
                representative_text=self.stored_texts[0] if self.stored_texts else None
            )]

        # Create and evaluate ClusterInfo objects
        self.clusters = []
        for label, texts in clusters.items():
            # Calculate centroid for the cluster
            cluster_embeddings = []
            for text in texts:
                embedding = self.get_embedding(text)
                cluster_embeddings.append(embedding)
            
            if not cluster_embeddings:
                continue
            
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calculate intra-cluster similarity
            similarities = []
            for emb in cluster_embeddings:
                sim = np.dot(emb, centroid)
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            
            # If cluster is too cohesive (very similar texts), try to split
            if avg_similarity > 0.95 and len(texts) > 3:
                # Try to create sub-clusters
                sub_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=2,
                    min_samples=1,
                    metric='cosine',
                    cluster_selection_method='eom',
                    cluster_selection_epsilon=0.1,  # Even finer granularity for sub-clusters
                )
                
                try:
                    sub_labels = sub_clusterer.fit_predict(np.array(cluster_embeddings))
                    unique_sub_labels = set(sub_labels) - {-1}
                    
                    if len(unique_sub_labels) > 1:
                        # Create sub-clusters
                        for sub_label in unique_sub_labels:
                            sub_texts = [texts[i] for i, l in enumerate(sub_labels) if l == sub_label]
                            sub_embeddings = [cluster_embeddings[i] for i, l in enumerate(sub_labels) if l == sub_label]
                            
                            sub_centroid = np.mean(sub_embeddings, axis=0)
                            cluster_info = ClusterInfo(
                                id=len(self.clusters),
                                texts=sub_texts,
                                centroid=sub_centroid
                            )
                            self.clusters.append(cluster_info)
                        continue
                except Exception as e:
                    print(f"Sub-clustering failed: {e}")
            
            # If sub-clustering failed or wasn't needed, create original cluster
            cluster_info = ClusterInfo(
                id=len(self.clusters),
                texts=texts,
                centroid=centroid
            )
            self.clusters.append(cluster_info)

        return self.clusters

    def get_cluster_info(self, cluster_id: int) -> Optional[ClusterInfo]:
        """Get information about a specific cluster"""
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                return cluster
        return None

    def update_cluster_representatives(self, gpt_processor) -> None:
        """Update representative texts and summaries for all clusters using GPT-4"""
        for cluster in self.clusters:
            # Combine all texts in the cluster
            combined_text = "\n".join(cluster.texts)
            
            # Generate representative text using GPT-4
            response = gpt_processor.summarize_cluster(combined_text)
            
            if response:
                cluster.representative_text = response.get('representative_text')
                cluster.summary = response.get('summary')