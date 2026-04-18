"""
Time Series Split Node - Specialized splitting for time series data.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

try:
    from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TimeSeriesSplitNode(NodeRuntime):
    """
    Time series cross-validation split that respects temporal order.
    Unlike random splits, this ensures training data always precedes test data.
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        features = inputs.get("Features")
        target = inputs.get("Target")
        
        if features is None:
            return NodeResult(outputs={}, success=False, error_message="No features data")
        
        X = ensure_dataframe(features)
        y = target
        
        n_splits = self.get_option("N Splits", 5)
        max_train_size = self.get_option("Max Train Size", None)
        test_size = self.get_option("Test Size", None)
        gap = self.get_option("Gap", 0)
        try:
            max_train_size = int(max_train_size) if max_train_size is not None else None
            if max_train_size is not None and max_train_size <= 0:
                max_train_size = None
        except Exception:
            max_train_size = None
        try:
            test_size = int(test_size) if test_size is not None else None
            if test_size is not None and test_size <= 0:
                test_size = None
        except Exception:
            test_size = None
        
        try:
            if SKLEARN_AVAILABLE:
                tscv = SklearnTimeSeriesSplit(
                    n_splits=n_splits,
                    max_train_size=max_train_size,
                    test_size=test_size,
                    gap=gap
                )
                
                y_array = y.values if hasattr(y, 'values') else (np.array(y) if y is not None else None)
                
                folds = []
                for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    fold_data = {
                        "fold": fold_idx,
                        "train_start": int(train_idx[0]),
                        "train_end": int(train_idx[-1]),
                        "test_start": int(test_idx[0]),
                        "test_end": int(test_idx[-1]),
                        "train_size": len(train_idx),
                        "test_size": len(test_idx),
                    }
                    folds.append(fold_data)
                
                # Create iterator for the splits
                def split_iterator():
                    for train_idx, test_idx in tscv.split(X):
                        yield {
                            "X_train": X.iloc[train_idx],
                            "X_test": X.iloc[test_idx],
                            "y_train": y_array[train_idx] if y_array is not None else None,
                            "y_test": y_array[test_idx] if y_array is not None else None,
                        }
                
                return NodeResult(
                    outputs={
                        "Split Iterator": list(split_iterator()),
                        "Split Info": {
                            "n_splits": n_splits,
                            "folds": folds,
                            "gap": gap,
                        },
                    },
                    metadata={
                        "n_splits": n_splits,
                        "total_samples": len(X),
                    }
                )
            else:
                # Manual time series split
                return self._manual_time_series_split(X, y, n_splits, gap)
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _manual_time_series_split(self, X, y, n_splits, gap):
        """Manual time series split without sklearn."""
        n = len(X)
        min_test_size = n // (n_splits + 1)
        
        folds = []
        splits = []
        
        for i in range(n_splits):
            test_start = n - (n_splits - i) * min_test_size
            test_end = test_start + min_test_size
            train_end = test_start - gap
            
            train_idx = list(range(0, train_end))
            test_idx = list(range(test_start, min(test_end, n)))
            
            folds.append({
                "fold": i,
                "train_start": 0,
                "train_end": train_end - 1,
                "test_start": test_start,
                "test_end": min(test_end, n) - 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
            })
            
            y_array = y.values if hasattr(y, 'values') else (np.array(y) if y is not None else None)
            
            splits.append({
                "X_train": X.iloc[train_idx],
                "X_test": X.iloc[test_idx],
                "y_train": y_array[train_idx] if y_array is not None else None,
                "y_test": y_array[test_idx] if y_array is not None else None,
            })
        
        return NodeResult(
            outputs={
                "Split Iterator": splits,
                "Split Info": {
                    "n_splits": n_splits,
                    "folds": folds,
                    "gap": gap,
                },
            }
        )
