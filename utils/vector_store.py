import faiss
import numpy as np
from typing import List, Dict, Tuple

class VectorStore:
    def __init__(self, dimension: int = 1536):  # OpenAI embedding dimension
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.metadata = []
    
    def add_vectors(self, vectors: np.ndarray, texts: List[str], metadata: List[Dict] = None):
        """Add vectors and corresponding texts to the index"""
        if metadata is None:
            metadata = [{} for _ in texts]
        
        self.index.add(vectors.astype('float32'))
        self.texts.extend(texts)
        self.metadata.extend(metadata)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors and return texts with scores"""
        D, I = self.index.search(query_vector.reshape(1, -1).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(self.texts):
                results.append((
                    self.texts[idx],
                    float(D[0][i]),
                    self.metadata[idx]
                ))
        
        return results
    
    def clear(self):
        """Clear the index and stored data"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []
