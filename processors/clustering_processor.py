import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import BayesianGaussianMixture
import hdbscan
from processors.embedding_processor import ClusterInfo


class BaseClusteringMethod:

    def fit_predict(self, embeddings: np.ndarray, texts: List[str],
                    **params) -> List[ClusterInfo]:
        raise NotImplementedError

    def _create_clusters(self, labels: np.ndarray, embeddings: np.ndarray,
                         texts: List[str]) -> List[ClusterInfo]:
        """Create ClusterInfo objects from clustering results"""
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Skip noise points
                continue
            if label not in clusters:
                clusters[label] = {'texts': [], 'embeddings': []}
            clusters[label]['texts'].append(texts[idx])
            clusters[label]['embeddings'].append(embeddings[idx])

        result = []
        for label, data in clusters.items():
            if data['embeddings']:
                centroid = np.mean(data['embeddings'], axis=0)
                result.append(
                    ClusterInfo(id=len(result),
                                texts=data['texts'],
                                centroid=centroid))

        return result


class HDBSCANMethod(BaseClusteringMethod):

    def fit_predict(self, embeddings: np.ndarray, texts: List[str],
                    **params) -> List[ClusterInfo]:
        """Perform HDBSCAN clustering"""
        min_cluster_size = params.get('min_cluster_size', 2)
        min_samples = params.get('min_samples', 1)
        cluster_selection_epsilon = params.get('cluster_selection_epsilon',
                                               0.5)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=cluster_selection_epsilon,
            prediction_data=True)

        labels = clusterer.fit_predict(embeddings)
        return self._create_clusters(labels, embeddings, texts)


class HierarchicalMethod(BaseClusteringMethod):

    def fit_predict(self, embeddings: np.ndarray, texts: List[str],
                    **params) -> List[ClusterInfo]:
        """Perform hierarchical clustering with optimal number of clusters"""
        max_clusters = min(len(texts), params.get('max_clusters', 10))

        # Try different numbers of clusters and evaluate using silhouette score
        best_score = -1
        best_n_clusters = 2
        best_labels = None

        for n_clusters in range(2, max_clusters + 1):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters,
                                                metric='euclidean',
                                                linkage='ward')
            labels = clusterer.fit_predict(embeddings)

            if len(set(labels)) > 1:  # Skip if only one cluster is formed
                score = silhouette_score(embeddings,
                                         labels,
                                         metric='euclidean')
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_labels = labels

        return self._create_clusters(best_labels, embeddings, texts)


class DPMMMethod(BaseClusteringMethod):

    def fit_predict(self, embeddings: np.ndarray, texts: List[str],
                    **params) -> List[ClusterInfo]:
        """Perform Dirichlet Process Mixture Model clustering"""
        max_components = min(len(texts), params.get('max_components', 10))

        dpgmm = BayesianGaussianMixture(n_components=max_components,
                                        covariance_type='full',
                                        weight_concentration_prior=1.0 /
                                        max_components,
                                        mean_precision_prior=1.0,
                                        warm_start=True,
                                        random_state=42)

        labels = dpgmm.fit_predict(embeddings)
        return self._create_clusters(labels, embeddings, texts)


class ClusteringProcessor:
    METHODS = {
        'hdbscan': HDBSCANMethod(),
        'hierarchical': HierarchicalMethod(),
        'dpmm': DPMMMethod()
    }

    @classmethod
    def get_available_methods(cls) -> List[str]:
        return list(cls.METHODS.keys())

    @classmethod
    def get_method_params(cls, method_name: str) -> Dict:
        params = {
            'hdbscan': {
                'min_cluster_size': (2, 5),  # Reduced maximum to encourage smaller clusters
                'min_samples': (1, 3),  # Reduced maximum to allow more granular clustering
                'cluster_selection_epsilon': (0.05, 0.2)  # Adjusted range for finer control
            },
            'hierarchical': {
                'max_clusters': (2, 20)
            },
            'dpmm': {
                'max_components': (2, 20)
            }
        }
        return params.get(method_name, {})

    def __init__(self, method_name: str = 'hdbscan'):
        if method_name not in self.METHODS:
            raise ValueError(f"Unknown clustering method: {method_name}")
        self.method = self.METHODS[method_name]

    def perform_clustering(self, embeddings: np.ndarray, texts: List[str],
                           **params) -> List[ClusterInfo]:
        """Perform clustering with error handling and validation"""
        if len(texts) < 2:
            print("Not enough data points for clustering")
            return []

        try:
            print(f"Starting clustering with {len(texts)} texts")
            return self.method.fit_predict(embeddings, texts, **params)
        except Exception as e:
            print(f"Error during clustering: {str(e)}")
            return []
