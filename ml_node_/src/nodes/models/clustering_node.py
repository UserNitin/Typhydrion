"""
Clustering Model Node - Unsupervised clustering algorithms.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import sklearn
try:
    from sklearn.cluster import (
        KMeans, DBSCAN, AgglomerativeClustering,
        OPTICS, SpectralClustering, Birch
    )
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ClusteringNode(NodeRuntime):
    """
    Clustering algorithms:
    - K-Means
    - DBSCAN
    - Hierarchical (Agglomerative)
    - Gaussian Mixture Model (GMM)
    - OPTICS
    - Spectral Clustering
    - Birch
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        features = inputs.get("Features")
        
        if features is None:
            return NodeResult(outputs={}, success=False, error_message="No features data")
        
        X = ensure_dataframe(features)
        
        algorithm = self.get_option("Algorithm", "K-Means")
        n_clusters = self.get_option("N Clusters", 5)
        linkage = self.get_option("Linkage", "ward")
        eps = self.get_option("Eps", 0.5)
        
        try:
            if not SKLEARN_AVAILABLE:
                return NodeResult(outputs={}, success=False, error_message="sklearn not available")
            
            X_clean = X.fillna(0)
            model, labels = self._fit_cluster(X_clean, algorithm, n_clusters, linkage, eps)
            
            # Get cluster centers if available
            centers = None
            if hasattr(model, "cluster_centers_"):
                centers = pd.DataFrame(model.cluster_centers_, columns=X.columns)
            elif hasattr(model, "means_"):
                centers = pd.DataFrame(model.means_, columns=X.columns)
            
            # Store model
            self.set_fitted_state("model", model)
            
            # Calculate cluster statistics
            cluster_stats = {}
            for label in np.unique(labels):
                if label == -1:  # Noise in DBSCAN
                    continue
                mask = labels == label
                cluster_stats[f"cluster_{label}"] = {
                    "size": int(mask.sum()),
                    "percentage": float(mask.mean() * 100),
                }
            
            return NodeResult(
                outputs={
                    "Cluster Labels": pd.Series(labels, name="cluster"),
                    "Cluster Centers": centers,
                    "Model": model,
                },
                metadata={
                    "algorithm": algorithm,
                    "n_clusters_found": len(np.unique(labels[labels != -1])),
                    "cluster_stats": cluster_stats,
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _fit_cluster(self, X: pd.DataFrame, algorithm: str, n_clusters: int, 
                     linkage: str, eps: float):
        """Fit clustering model and return labels."""
        if algorithm == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X)
        
        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=5)
            labels = model.fit_predict(X)
        
        elif algorithm == "Hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = model.fit_predict(X)
        
        elif algorithm == "GMM":
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(X)
        
        elif algorithm == "OPTICS":
            model = OPTICS(min_samples=5)
            labels = model.fit_predict(X)
        
        elif algorithm == "Spectral":
            model = SpectralClustering(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X)
        
        elif algorithm == "Birch":
            model = Birch(n_clusters=n_clusters)
            labels = model.fit_predict(X)
        
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X)
        
        return model, labels
    
    def predict(self, X) -> np.ndarray:
        """Predict cluster labels for new data."""
        model = self.get_fitted_state("model")
        if model is None:
            raise ValueError("Model not fitted")
        
        X = ensure_dataframe(X)
        
        if hasattr(model, "predict"):
            return model.predict(X.fillna(0))
        else:
            raise ValueError("Model doesn't support prediction on new data")
