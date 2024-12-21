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

    def perform_clustering(self, min_cluster_size: int = 2, cache_id: str = None) -> List[ClusterInfo]:
        """Perform HDBSCAN clustering on the stored embeddings with improved caching"""
        from utils.persistence import load_processing_results, save_processing_results
        import hashlib
        import json
        
        if not self.stored_texts:
            raise ValueError("No texts have been stored yet")

        # Generate a more robust cache_id if not provided
        if cache_id is None:
            # Sort texts to ensure consistent hash regardless of order
            sorted_texts = sorted(self.stored_texts)
            # Create a hash of the content and parameters
            hash_content = {
                'texts': sorted_texts,
                'min_cluster_size': min_cluster_size,
                'timestamp': datetime.now().strftime('%Y%m%d')
            }
            hash_string = json.dumps(hash_content, ensure_ascii=False)
            cache_id = hashlib.md5(hash_string.encode('utf-8')).hexdigest()[:12]
            
        cache_file = f"clusters_{cache_id}.json"
        print(f"Cache ID: {cache_id}")
            
        # Try to load cached results
        try:
            cached_results = load_processing_results(cache_file)
            if cached_results:
                print(f"Loading cached clustering results for {cache_id}")
                
                # Validate cached data structure
                if not isinstance(cached_results, dict) or 'clusters' not in cached_results:
                    print("Invalid cache data structure")
                    cached_results = None
                else:
                    try:
                        clusters = []
                        for cluster_data in cached_results['clusters']:
                            # Validate required fields
                            if not all(k in cluster_data for k in ['id', 'texts']):
                                continue
                                
                            cluster = ClusterInfo(
                                id=cluster_data['id'],
                                texts=cluster_data['texts'],
                                centroid=np.array(cluster_data['centroid']) if cluster_data.get('centroid') else None,
                                representative_text=cluster_data.get('representative_text'),
                                summary=cluster_data.get('summary'),
                                created_at=datetime.fromisoformat(cluster_data.get('created_at', datetime.now().isoformat()))
                            )
                            clusters.append(cluster)
                            
                        if clusters:  # Only use cache if valid clusters were created
                            self.clusters = clusters
                            print(f"Successfully loaded {len(clusters)} clusters from cache")
                            return self.clusters
                    except Exception as e:
                        print(f"Error processing cached data: {e}")
                        cached_results = None
        except Exception as e:
            print(f"Error loading cache file: {e}")
            cached_results = None

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

        # クラスタリングパラメータを明示的に設定
        min_cluster_size = 2  # 最小クラスタサイズを2に固定
        min_samples = 1  # 最小サンプル数を1に固定
        print(f"Clustering parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        
        print(f"Clustering parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

        # Perform HDBSCAN clustering with adjusted parameters for finer clustering
        try:
            print("Attempting HDBSCAN clustering with refined parameters...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='cosine',  # Cosine similarity for semantic grouping
                cluster_selection_method='leaf',  # Changed to leaf for finer clustering
                cluster_selection_epsilon=0.1,  # Reduced epsilon for stricter cluster boundaries
                prediction_data=True,
                alpha=0.5  # Reduced alpha for less aggressive cluster merging
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
        
        # Create a dictionary to store texts by cluster label
        clustered_texts = {}
        for idx, label in enumerate(cluster_labels):
            if label == -1:  # Skip noise points
                continue
            if label not in clustered_texts:
                clustered_texts[label] = []
            clustered_texts[label].append(self.stored_texts[idx])
        
        # Process each cluster
        for label, texts in clustered_texts.items():
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
            
            # If cluster is somewhat cohesive, try to split for finer granularity
            if avg_similarity > 0.85 and len(texts) > 2:  # Reduced similarity threshold and minimum texts
                # Try to create sub-clusters with more sensitive parameters
                sub_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=2,
                    min_samples=1,
                    metric='cosine',
                    cluster_selection_method='leaf',  # Changed to leaf for finer sub-clustering
                    cluster_selection_epsilon=0.05,  # Further reduced epsilon for even finer granularity
                    alpha=0.3  # Lower alpha for more aggressive splitting
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
        
        # Save clustering results with enhanced error handling
        try:
            # Prepare cache data with additional metadata
            cluster_data = {
                'cache_id': cache_id,
                'version': '1.0',
                'clusters': [cluster.to_dict() for cluster in self.clusters],
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'text_count': len(self.stored_texts),
                    'min_cluster_size': min_cluster_size,
                    'cluster_count': len(self.clusters)
                }
            }
            
            # Validate cluster data before saving
            for cluster in cluster_data['clusters']:
                if not isinstance(cluster.get('texts', []), list):
                    print(f"Invalid texts format in cluster {cluster.get('id')}")
                    continue
                if cluster.get('centroid') is not None and not isinstance(cluster['centroid'], list):
                    cluster['centroid'] = cluster['centroid'].tolist() if hasattr(cluster['centroid'], 'tolist') else None
            
            # Save to cache file
            cache_file = f"clusters_{cache_id}.json"
            save_processing_results(cluster_data, cache_file)
            print(f"Successfully cached clustering results to {cache_file}")
            
        except Exception as e:
            print(f"Error saving clustering results to cache: {e}")
            # Continue without caching if saving fails
            
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