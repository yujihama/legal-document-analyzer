import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional
import faiss
import hdbscan
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from utils.persistence import load_processing_results, save_processing_results
import hashlib
import json
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

def process_cluster(cluster_data: tuple) -> List[Dict]:
    """Process a single cluster in parallel"""
    texts, embeddings = cluster_data
    if not embeddings:
        return None

    centroid = np.mean(embeddings, axis=0)

    # Calculate intra-cluster similarity for sub-clustering decision
    similarities = []
    for emb in embeddings:
        sim = np.dot(emb, centroid)
        similarities.append(sim)

    avg_similarity = np.mean(similarities)
    cluster_results = []

    # Sub-clustering logic
    if avg_similarity > 0.80 and len(texts) > 2:
        try:
            max_sub_clusters = min(len(texts), 5)
            best_score = -1
            best_sub_labels = None

            for n_clusters in range(2, max_sub_clusters + 1):
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='cosine',
                    linkage='average'
                )
                sub_labels = clusterer.fit_predict(embeddings)

                if len(set(sub_labels)) > 1:
                    score = silhouette_score(
                        np.array(embeddings),
                        sub_labels,
                        metric='cosine'
                    )
                    if score > best_score:
                        best_score = score
                        best_sub_labels = sub_labels

            if best_sub_labels is not None and best_score > 0.1:
                unique_sub_labels = set(best_sub_labels)
                for sub_label in unique_sub_labels:
                    sub_texts = [texts[i] for i, l in enumerate(best_sub_labels) if l == sub_label]
                    sub_embeddings = [embeddings[i] for i, l in enumerate(best_sub_labels) if l == sub_label]
                    sub_centroid = np.mean(sub_embeddings, axis=0)

                    cluster_results.append({
                        'texts': sub_texts,
                        'centroid': sub_centroid
                    })
                return cluster_results
        except Exception as e:
            print(f"Hierarchical sub-clustering failed: {e}")

    # If no sub-clusters were created, return the original cluster
    cluster_results.append({
        'texts': texts,
        'centroid': centroid
    })
    return cluster_results

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
            'texts': list(self.texts) if self.texts else [],
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
        cache_key = text.strip()

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        processed_text = self._preprocess_text(text)

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=processed_text,
                encoding_format="float"
            )
            embedding = np.array(response.data[0].embedding)

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for better semantic separation"""
        text = text.strip()

        # 文章構造の区切りを強調
        text = text.replace('。', '。\n')
        text = text.replace('、', '、 ')
        text = text.replace('）', '） ')
        text = text.replace('「', ' 「')

        # 章立て・条項の区切りを強調
        text = text.replace('第', '\n第')
        text = text.replace('条', '条\n')

        # 重要な区切り文字やマーカーを強調
        markers = ['。', '：', '；', '・']
        for marker in markers:
            text = text.replace(marker, f'{marker}\n')

        # 重要なキーワードの強調
        keywords = {
            '禁止': 'PROHIBITION_',
            '義務': 'OBLIGATION_',
            '要件': 'REQUIREMENT_',
            '必須': 'MANDATORY_',
            '規則': 'RULE_',
            '規定': 'REGULATION_',
            '手順': 'PROCEDURE_',
            '基準': 'STANDARD_',
            '対策': 'MEASURE_',
            '管理': 'CONTROL_',
            'セキュリティ': 'SECURITY_',
            '方針': 'POLICY_'
        }

        for keyword, prefix in keywords.items():
            text = text.replace(keyword, f' {prefix}{keyword} ')

        # 箇条書きの強調
        text = text.replace('\n1.', '\nLIST_ITEM_1.')
        text = text.replace('\n2.', '\nLIST_ITEM_2.')
        text = text.replace('\n3.', '\nLIST_ITEM_3.')

        # テキストのクリーニング
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        text = ' '.join(text.split())

        return text

    def batch_embed_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Get embeddings for a batch of texts"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)

            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def create_index(self, texts: List[str]):
        """Store texts and their embeddings for similarity comparison"""
        self.stored_texts = texts
        self._embedding_cache.update({
            text: self.get_embedding(text)
            for text in texts
        })

    def find_similar(self, query: str, distance_threshold: float = 1.5) -> List[Dict]:
        """Find similar texts using distance-based similarity with cached embeddings"""
        query_embedding = self.get_embedding(query)

        results = []
        for text in self.stored_texts:
            text_embedding = self._embedding_cache.get(text)
            if text_embedding is None:
                continue

            distance = float(np.linalg.norm(query_embedding - text_embedding))

            if distance < distance_threshold:
                results.append({
                    'text': text,
                    'score': 1.0 / (1.0 + distance)
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def perform_clustering(self, min_cluster_size: int = 2, cache_id: str = None) -> List[ClusterInfo]:
        """Perform HDBSCAN clustering on the stored embeddings with improved caching and parallel processing"""
        if not self.stored_texts:
            raise ValueError("No texts have been stored yet")

        # Generate cache_id if not provided
        if cache_id is None:
            sorted_texts = sorted(self.stored_texts)
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
            if cached_results and isinstance(cached_results, dict) and 'clusters' in cached_results:
                clusters = []
                for cluster_data in cached_results['clusters']:
                    if all(k in cluster_data for k in ['id', 'texts']):
                        cluster = ClusterInfo(
                            id=cluster_data['id'],
                            texts=cluster_data['texts'],
                            centroid=np.array(cluster_data['centroid']) if cluster_data.get('centroid') else None,
                            representative_text=cluster_data.get('representative_text'),
                            summary=cluster_data.get('summary'),
                            created_at=datetime.fromisoformat(cluster_data.get('created_at', datetime.now().isoformat()))
                        )
                        clusters.append(cluster)

                if clusters:
                    self.clusters = clusters
                    print(f"Successfully loaded {len(clusters)} clusters from cache")
                    return self.clusters
        except Exception as e:
            print(f"Error loading cache: {e}")

        n_texts = len(self.stored_texts)
        print(f"Starting clustering with {n_texts} texts")

        if n_texts < 3:
            print("Too few data points for clustering, creating single cluster")
            embeddings = self.batch_embed_texts(self.stored_texts)
            centroid = np.mean(embeddings, axis=0) if n_texts > 1 else embeddings[0]
            return [ClusterInfo(
                id=0,
                texts=self.stored_texts,
                centroid=centroid,
                representative_text=self.stored_texts[0] if self.stored_texts else None
            )]

        embeddings_array = self.batch_embed_texts(self.stored_texts)
        print(f"Generated embeddings array shape: {embeddings_array.shape}")

        min_cluster_size = 2
        min_samples = 1
        print(f"Clustering parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

        try:
            print("Attempting HDBSCAN clustering with refined parameters...")
            distances = cosine_distances(embeddings_array)
            print(f"Distance matrix shape: {distances.shape}")
            print(f"Average distance: {np.mean(distances):.4f}")
            print(f"Min distance: {np.min(distances[distances > 0]):.4f}")
            print(f"Max distance: {np.max(distances):.4f}")

            def enhanced_metric(x, y):
                cosine_dist = cosine_distances(x.reshape(1, -1), y.reshape(1, -1))[0][0]
                return np.power(cosine_dist, 0.5)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                metric=enhanced_metric,
                cluster_selection_method='leaf',
                cluster_selection_epsilon=0.03,
                prediction_data=True,
                alpha=0.2
            )

            cluster_labels = clusterer.fit_predict(embeddings_array)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"Found {n_clusters} clusters")

            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            print("Cluster distribution:")
            for label, count in zip(unique_labels, counts):
                if label != -1:
                    print(f"Cluster {label}: {count} items")
                else:
                    print(f"Noise points: {count} items")

            if n_clusters == 0:
                print("No valid clusters found, creating single cluster")
                return [ClusterInfo(
                    id=0,
                    texts=self.stored_texts,
                    centroid=np.mean(embeddings_array, axis=0),
                    representative_text=self.stored_texts[0] if self.stored_texts else None
                )]

        except Exception as e:
            print(f"Clustering error: {e}")
            print("Input shape:", embeddings_array.shape)
            print("Input type:", embeddings_array.dtype)
            return [ClusterInfo(
                id=0,
                texts=self.stored_texts,
                centroid=np.mean(embeddings_array, axis=0),
                representative_text=self.stored_texts[0] if self.stored_texts else None
            )]

        # Prepare data for parallel processing
        embeddings_dict = {}
        clustered_texts = {}

        for idx, label in enumerate(cluster_labels):
            if label == -1:
                continue
            if label not in clustered_texts:
                clustered_texts[label] = []
            clustered_texts[label].append(self.stored_texts[idx])

            text = self.stored_texts[idx]
            if text not in embeddings_dict:
                embeddings_dict[text] = embeddings_array[idx]

        # Prepare cluster data
        cluster_data = []
        for label, texts in clustered_texts.items():
            cluster_embeddings = [embeddings_dict[text] for text in texts]
            cluster_data.append((texts, cluster_embeddings))

        # Execute parallel processing
        self.clusters = []
        max_workers = min(multiprocessing.cpu_count(), len(cluster_data))

        if max_workers > 0:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                cluster_results = list(executor.map(process_cluster, cluster_data))
        else:
            cluster_results = [process_cluster(data) for data in cluster_data]

        # Process results
        cluster_id = 0
        for result_group in cluster_results:
            if result_group:
                for result in result_group:
                    cluster_info = ClusterInfo(
                        id=cluster_id,
                        texts=result['texts'],
                        centroid=result['centroid']
                    )
                    self.clusters.append(cluster_info)
                    cluster_id += 1

        # Cache results
        try:
            cache_data = {
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

            save_processing_results(cache_data, cache_file)
            print(f"Cached clustering results to {cache_file}")

        except Exception as e:
            print(f"Error caching results: {e}")

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
            combined_text = "\n".join(cluster.texts)
            response = gpt_processor.summarize_cluster(combined_text)

            if response:
                cluster.representative_text = response.get('representative_text')
                cluster.summary = response.get('summary')