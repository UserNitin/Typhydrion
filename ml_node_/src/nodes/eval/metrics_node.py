"""
Metrics Evaluator Node - Compute performance metrics.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import sklearn metrics
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        mean_squared_error, mean_absolute_error, r2_score,
        silhouette_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MetricsNode(NodeRuntime):
    """
    Compute performance metrics for:
    - Classification (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
    - Regression (MSE, RMSE, MAE, R² score)
    - Clustering (silhouette score)
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        model = inputs.get("Trained Model")
        X_test = inputs.get("X_test")
        y_test = inputs.get("y_test")
        
        if X_test is None or y_test is None:
            return NodeResult(outputs={}, success=False, error_message="Missing test data")
        
        X_test = ensure_dataframe(X_test)
        y_test = np.array(y_test).ravel() if hasattr(y_test, '__len__') else y_test
        
        # Get options
        calc_accuracy = self.get_option("Accuracy", True)
        calc_precision = self.get_option("Precision", True)
        calc_recall = self.get_option("Recall", True)
        calc_f1 = self.get_option("F1 Score", True)
        calc_roc = self.get_option("ROC AUC", True)
        calc_confusion = self.get_option("Confusion Matrix", True)
        calc_mse = self.get_option("MSE/RMSE", True)
        calc_mae = self.get_option("MAE", True)
        calc_r2 = self.get_option("R² Score", True)
        
        try:
            # Get predictions
            if model is not None:
                predictions = model.predict(X_test.fillna(0))
                probabilities = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_test.fillna(0))
            else:
                predictions = inputs.get("Predictions")
                probabilities = inputs.get("Probabilities")
                
                if predictions is None:
                    return NodeResult(outputs={}, success=False, error_message="No predictions")
            
            # Determine task type
            n_unique = len(np.unique(y_test))
            is_classification = n_unique < 20  # Heuristic
            
            metrics = {}
            
            if is_classification and SKLEARN_AVAILABLE:
                # Classification metrics
                if calc_accuracy:
                    metrics["accuracy"] = float(accuracy_score(y_test, predictions))
                
                if calc_precision:
                    avg = "binary" if n_unique == 2 else "weighted"
                    metrics["precision"] = float(precision_score(y_test, predictions, average=avg, zero_division=0))
                
                if calc_recall:
                    avg = "binary" if n_unique == 2 else "weighted"
                    metrics["recall"] = float(recall_score(y_test, predictions, average=avg, zero_division=0))
                
                if calc_f1:
                    avg = "binary" if n_unique == 2 else "weighted"
                    metrics["f1"] = float(f1_score(y_test, predictions, average=avg, zero_division=0))
                
                if calc_roc and probabilities is not None:
                    try:
                        if n_unique == 2:
                            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
                        else:
                            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities, multi_class="ovr"))
                    except Exception:
                        metrics["roc_auc"] = None
                
                if calc_confusion:
                    cm = confusion_matrix(y_test, predictions)
                    metrics["confusion_matrix"] = cm.tolist()
                
                # Add classification report
                metrics["classification_report"] = classification_report(y_test, predictions, output_dict=True)
            
            else:
                # Regression metrics
                if calc_mse:
                    mse = mean_squared_error(y_test, predictions)
                    metrics["mse"] = float(mse)
                    metrics["rmse"] = float(np.sqrt(mse))
                
                if calc_mae:
                    metrics["mae"] = float(mean_absolute_error(y_test, predictions))
                
                if calc_r2:
                    metrics["r2"] = float(r2_score(y_test, predictions))
            
            return NodeResult(
                outputs={
                    "Metrics": metrics,
                    "Predictions": predictions,
                },
                metadata={
                    "task": "classification" if is_classification else "regression",
                    "n_samples": len(y_test),
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
