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
            'texts': self.texts,
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

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using OpenAI API"""
        response = self.client.embeddings.create(
            model="text-embedding-3-large", input=text)
        return np.array(response.data[0].embedding)

    def batch_embed_texts(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings)

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
        """Perform KMeans clustering on the stored embeddings"""
        if not self.stored_texts:
            raise ValueError("No texts have been stored yet")

        # Extract all embeddings
        embeddings = []
        for text in self.stored_texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        embeddings_array = np.array(embeddings)

        # Perform KMeans clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
