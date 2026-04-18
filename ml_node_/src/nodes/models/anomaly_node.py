"""
Anomaly Detection Node - Detect outliers and anomalies.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import sklearn
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AnomalyNode(NodeRuntime):
    """
    Anomaly detection methods:
    - Isolation Forest
    - One-Class SVM
    - Local Outlier Factor (LOF)
    - Elliptic Envelope
    - Autoencoder (if tensorflow available)
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        features = inputs.get("Features")
        
        if features is None:
            return NodeResult(outputs={}, success=False, error_message="No features data")
        
        X = ensure_dataframe(features)
        
        method = self.get_option("Method", "Isolation Forest")
        contamination = self.get_option("Contamination", 0.1)
        n_estimators = self.get_option("N Estimators", 100)
        
        try:
            if not SKLEARN_AVAILABLE:
                return NodeResult(outputs={}, success=False, error_message="sklearn not available")
            
            X_clean = X.select_dtypes(include=[np.number]).fillna(0)
            
            model, scores, labels = self._fit_detector(X_clean, method, contamination, n_estimators)
            
            # Store model
            self.set_fitted_state("model", model)
            
            # Calculate anomaly statistics
            n_anomalies = int((labels == -1).sum())
            anomaly_pct = float(n_anomalies / len(labels) * 100)
            
            return NodeResult(
                outputs={
                    "Anomaly Scores": pd.Series(scores, name="anomaly_score"),
                    "Anomaly Labels": pd.Series(labels, name="is_anomaly"),
                    "Model": model,
                },
                metadata={
                    "method": method,
                    "n_anomalies": n_anomalies,
                    "anomaly_percentage": anomaly_pct,
                    "contamination": contamination,
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _fit_detector(self, X: pd.DataFrame, method: str, contamination: float, n_estimators: int):
        """Fit anomaly detector and return scores and labels."""
        if method == "Isolation Forest":
            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            labels = model.fit_predict(X)
            scores = -model.score_samples(X)  # Higher = more anomalous
        
        elif method == "One-Class SVM":
            model = OneClassSVM(nu=contamination)
            labels = model.fit_predict(X)
            scores = -model.decision_function(X)
        
        elif method == "LOF":
            model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination,
                novelty=True
            )
            model.fit(X)
            labels = model.predict(X)
            scores = -model.decision_function(X)
        
        elif method == "Elliptic Envelope":
            model = EllipticEnvelope(contamination=contamination, random_state=42)
            labels = model.fit_predict(X)
            scores = -model.decision_function(X)
        
        elif method == "Autoencoder":
            # Simple autoencoder-based anomaly detection
            scores, labels = self._autoencoder_anomaly(X, contamination)
            model = None
        
        else:
            model = IsolationForest(contamination=contamination, random_state=42)
            labels = model.fit_predict(X)
            scores = -model.score_samples(X)
        
        # Convert labels: sklearn uses 1 for normal, -1 for anomaly
        # We convert to: 0 for normal, 1 for anomaly (more intuitive)
        labels = np.where(labels == -1, 1, 0)
        
        return model, scores, labels
    
    def _autoencoder_anomaly(self, X: pd.DataFrame, contamination: float):
        """Simple autoencoder-based anomaly detection."""
        # Try to use tensorflow
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Simple autoencoder
            n_features = X.shape[1]
            encoding_dim = max(2, n_features // 2)
            
            encoder = keras.Sequential([
                keras.layers.Dense(encoding_dim, activation='relu', input_shape=(n_features,)),
                keras.layers.Dense(encoding_dim // 2, activation='relu'),
            ])
            
            decoder = keras.Sequential([
                keras.layers.Dense(encoding_dim // 2, activation='relu', input_shape=(encoding_dim // 2,)),
                keras.layers.Dense(encoding_dim, activation='relu'),
                keras.layers.Dense(n_features, activation='linear'),
            ])
            
            autoencoder = keras.Sequential([encoder, decoder])
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # Train
            X_normalized = (X - X.mean()) / (X.std() + 1e-8)
            autoencoder.fit(X_normalized, X_normalized, epochs=50, batch_size=32, verbose=0)
            
            # Compute reconstruction error
            reconstructed = autoencoder.predict(X_normalized, verbose=0)
            mse = np.mean(np.power(X_normalized.values - reconstructed, 2), axis=1)
            
            # Threshold for anomaly
            threshold = np.percentile(mse, (1 - contamination) * 100)
            labels = (mse > threshold).astype(int)
            
            return mse, labels
            
        except ImportError:
            # Fallback to simple statistical method
            z_scores = np.abs((X - X.mean()) / (X.std() + 1e-8))
            scores = z_scores.max(axis=1).values
            threshold = np.percentile(scores, (1 - contamination) * 100)
            labels = (scores > threshold).astype(int)
            return scores, labels
    
    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predict anomaly scores and labels for new data."""
        model = self.get_fitted_state("model")
        if model is None:
            raise ValueError("Model not fitted")
        
        X = ensure_dataframe(X)
        X_clean = X.select_dtypes(include=[np.number]).fillna(0)
        
        labels = model.predict(X_clean)
        labels = np.where(labels == -1, 1, 0)
        
        if hasattr(model, "score_samples"):
            scores = -model.score_samples(X_clean)
        elif hasattr(model, "decision_function"):
            scores = -model.decision_function(X_clean)
        else:
            scores = np.zeros(len(X_clean))
        
        return scores, labels
