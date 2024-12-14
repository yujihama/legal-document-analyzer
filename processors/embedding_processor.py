import numpy as np
from openai import OpenAI
from typing import List, Dict
import faiss


class EmbeddingProcessor:

    def __init__(self):
        self.client = OpenAI()
        self.index = None
        self.stored_texts = []

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
