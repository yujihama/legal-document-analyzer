import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional
import faiss
from sklearn.cluster import KMeans
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
        """Get embeddings for a batch of texts using parallel processing"""
        import multiprocessing
        from functools import partial
        
        # Split texts into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Create a pool of workers
        with multiprocessing.Pool() as pool:
            # Process batches in parallel
            results = pool.map(self._process_batch, batches)
        
        # Combine results
        all_embeddings = []
        for batch_result in results:
            all_embeddings.extend(batch_result)
        
        return np.array(all_embeddings)
    
    def _process_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Process a batch of texts to get their embeddings"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def create_index(self, texts: List[str]):
        """Create FAISS index from texts"""
        embeddings = self.batch_embed_texts(texts)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        self.stored_texts = texts

    def find_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Find similar texts using vector similarity"""
        query_embedding = self.get_embedding(query)
        D, I = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), k)

        results = []
        for i in range(len(I[0])):
            if I[0][i] < len(self.stored_texts):
                results.append({
                    'text': self.stored_texts[I[0][i]],
                    'score': float(D[0][i])
                })

        return results

    def perform_clustering(self, n_clusters: int = 5) -> List[ClusterInfo]:
        """Perform optimized KMeans clustering on the stored embeddings"""
        if not self.stored_texts:
            raise ValueError("No texts have been stored yet")

        # Extract all embeddings in parallel
        embeddings_array = self.batch_embed_texts(self.stored_texts)

        # Perform KMeans clustering with optimized parameters
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=5,  # Reduce number of initializations
            max_iter=100,  # Limit maximum iterations
            algorithm='elkan'  # Use more efficient algorithm
        )
        cluster_labels = self.kmeans.fit_predict(embeddings_array)

        # Create cluster information
        clusters: Dict[int, List[str]] = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.stored_texts[idx])

        # Create ClusterInfo objects
        self.clusters = []
        for label, texts in clusters.items():
            cluster_info = ClusterInfo(
                id=label,
                texts=texts,
                centroid=self.kmeans.cluster_centers_[label]
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
