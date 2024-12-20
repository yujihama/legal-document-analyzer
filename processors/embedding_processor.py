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
        self.index = None
        self.stored_texts = []
        self.clusters: List[ClusterInfo] = []
        self.kmeans: Optional[KMeans] = None
        self._embedding_cache = {}

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using OpenAI API with caching"""
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # If not in cache, compute embedding
        response = self.client.embeddings.create(
            model="text-embedding-3-large", input=text)
        embedding = np.array(response.data[0].embedding)
        
        # Store in cache
        self._embedding_cache[text] = embedding
        return embedding

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

        # Extract all embeddings in parallel
        embeddings_array = self.batch_embed_texts(self.stored_texts)

        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_epsilon=0.5,  # Allow for more flexible cluster boundaries
            cluster_selection_method='eom'  # Excess of Mass algorithm for better cluster selection
        )
        clusterer.fit(embeddings_array)

        # Create cluster information
        clusters: Dict[int, List[str]] = {}
        for idx, label in enumerate(clusterer.labels_):
            if label == -1:  # HDBSCAN marks noise points as -1
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.stored_texts[idx])

        # Create ClusterInfo objects
        self.clusters = []
        for label, texts in clusters.items():
            # Calculate centroid for the cluster
            cluster_embeddings = []
            for text in texts:
                embedding = self.get_embedding(text)
                cluster_embeddings.append(embedding)
            centroid = np.mean(cluster_embeddings, axis=0) if cluster_embeddings else None

            cluster_info = ClusterInfo(
                id=label,
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
