"""
Train/Test Split Nodes - Split data for training and evaluation.
"""
from __future__ import annotations

from typing import Any, Iterator

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import sklearn
try:
    from sklearn.model_selection import (
        train_test_split, KFold, StratifiedKFold, 
        GroupKFold, TimeSeriesSplit, RepeatedKFold
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TrainTestSplitNode(NodeRuntime):
    """
    Split data into training and test sets.
    Supports stratified splitting for classification tasks.
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        features = inputs.get("Features")
        target = inputs.get("Target")
        
        if features is None:
            return NodeResult(outputs={}, success=False, error_message="No features data")
        
        X = ensure_dataframe(features)
        y = target
        
        test_size = self.get_option("Test Size", 0.2)
        seed = self.get_option("Seed", 42)
        shuffle = self.get_option("Shuffle", True)
        stratify = self.get_option("Stratify", False)
        
        try:
            if SKLEARN_AVAILABLE:
                stratify_data = y if stratify and y is not None else None
                
                if y is not None:
                    y_array = y.values if hasattr(y, 'values') else np.array(y)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_array,
                        test_size=test_size,
                        random_state=seed,
                        shuffle=shuffle,
                        stratify=stratify_data
                    )
                else:
                    X_train, X_test = train_test_split(
                        X,
                        test_size=test_size,
                        random_state=seed,
                        shuffle=shuffle
                    )
                    y_train, y_test = None, None
            else:
                # Manual split
                X_train, X_test, y_train, y_test = self._manual_split(X, y, test_size, seed, shuffle)
            
            # Convert to DataFrames
            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_test = pd.DataFrame(X_test, columns=X.columns)
            
            if y_train is not None:
                y_train = pd.Series(y_train, name="target")
                y_test = pd.Series(y_test, name="target")
            
            return NodeResult(
                outputs={
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                },
                metadata={
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "test_ratio": test_size,
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _manual_split(self, X, y, test_size, seed, shuffle):
        """Manual train/test split without sklearn."""
        n = len(X)
        indices = np.arange(n)
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        
        split_idx = int(n * (1 - test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
        
        if y is not None:
            y_arr = y.values if hasattr(y, 'values') else np.array(y)
            y_train = y_arr[train_idx]
            y_test = y_arr[test_idx]
        else:
            y_train, y_test = None, None
        
        return X_train, X_test, y_train, y_test


class TrainValTestSplitNode(NodeRuntime):
    """Three-way data split: train/validation/test."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        features = inputs.get("Features")
        target = inputs.get("Target")
        
        if features is None:
            return NodeResult(outputs={}, success=False, error_message="No features data")
        
        X = ensure_dataframe(features)
        y = target
        
        train_size = self.get_option("Train Size", 0.7)
        val_size = self.get_option("Val Size", 0.15)
        seed = self.get_option("Seed", 42)
        stratify = self.get_option("Stratify", True)
        
        test_size = 1.0 - train_size - val_size
        
        try:
            if SKLEARN_AVAILABLE:
                y_array = y.values if hasattr(y, 'values') else (np.array(y) if y is not None else None)
                stratify_data = y_array if stratify and y_array is not None else None
                
                # First split: separate test set
                if y_array is not None:
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y_array, test_size=test_size, random_state=seed, stratify=stratify_data
                    )
                    
                    # Second split: separate validation from training
                    val_ratio = val_size / (train_size + val_size)
                    stratify_temp = y_temp if stratify else None
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_ratio, random_state=seed, stratify=stratify_temp
                    )
                else:
                    X_temp, X_test = train_test_split(X, test_size=test_size, random_state=seed)
                    val_ratio = val_size / (train_size + val_size)
                    X_train, X_val = train_test_split(X_temp, test_size=val_ratio, random_state=seed)
                    y_train, y_val, y_test = None, None, None
            else:
                # Manual split
                n = len(X)
                indices = np.arange(n)
                np.random.seed(seed)
                np.random.shuffle(indices)
                
                train_end = int(n * train_size)
                val_end = train_end + int(n * val_size)
                
                train_idx = indices[:train_end]
                val_idx = indices[train_end:val_end]
                test_idx = indices[val_end:]
                
                X_train = X.iloc[train_idx]
                X_val = X.iloc[val_idx]
                X_test = X.iloc[test_idx]
                
                if y is not None:
                    y_arr = y.values if hasattr(y, 'values') else np.array(y)
                    y_train, y_val, y_test = y_arr[train_idx], y_arr[val_idx], y_arr[test_idx]
                else:
                    y_train, y_val, y_test = None, None, None
            
            # Convert to DataFrames/Series
            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_val = pd.DataFrame(X_val, columns=X.columns)
            X_test = pd.DataFrame(X_test, columns=X.columns)
            
            if y_train is not None:
                y_train = pd.Series(y_train, name="target")
                y_val = pd.Series(y_val, name="target")
                y_test = pd.Series(y_test, name="target")
            
            return NodeResult(
                outputs={
                    "X_train": X_train,
                    "X_val": X_val,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_test": y_test,
                },
                metadata={
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "test_size": len(X_test),
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))


class CrossValidationSplitNode(NodeRuntime):
    """K-Fold cross validation split."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        features = inputs.get("Features")
        target = inputs.get("Target")
        
        if features is None:
            return NodeResult(outputs={}, success=False, error_message="No features data")
        
        X = ensure_dataframe(features)
        y = target
        
        k_folds = self.get_option("K Folds", 5)
        strategy = self.get_option("Strategy", "KFold")
        shuffle = self.get_option("Shuffle", True)
        seed = self.get_option("Seed", 42)
        
        try:
            if SKLEARN_AVAILABLE:
                cv = self._get_cv_splitter(strategy, k_folds, shuffle, seed)
                y_array = y.values if hasattr(y, 'values') else (np.array(y) if y is not None else None)
                
                folds = []
                for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_array)):
                    folds.append({
                        "fold": fold_idx,
                        "train_idx": train_idx.tolist(),
                        "test_idx": test_idx.tolist(),
                        "train_size": len(train_idx),
                        "test_size": len(test_idx),
                    })
                
                fold_info = {
                    "n_folds": k_folds,
                    "strategy": strategy,
                    "folds": folds,
                }
                
                return NodeResult(
                    outputs={
                        "Fold Iterator": self._create_fold_iterator(X, y_array, cv),
                        "Fold Info": fold_info,
                    },
                    metadata={"n_folds": k_folds, "strategy": strategy}
                )
            else:
                # Simple manual k-fold
                n = len(X)
                fold_size = n // k_folds
                folds = []
                
                for i in range(k_folds):
                    test_start = i * fold_size
                    test_end = (i + 1) * fold_size if i < k_folds - 1 else n
                    
                    test_idx = list(range(test_start, test_end))
                    train_idx = list(range(0, test_start)) + list(range(test_end, n))
                    
                    folds.append({
                        "fold": i,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                    })
                
                return NodeResult(
                    outputs={"Fold Iterator": folds, "Fold Info": {"n_folds": k_folds, "folds": folds}}
                )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _get_cv_splitter(self, strategy: str, n_splits: int, shuffle: bool, seed: int):
        """Get the appropriate CV splitter."""
        splitter_map = {
            "KFold": KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed),
            "StratifiedKFold": StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed),
            "TimeSeriesSplit": TimeSeriesSplit(n_splits=n_splits),
            "RepeatedKFold": RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=seed),
        }
        return splitter_map.get(strategy, KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed))
    
    def _create_fold_iterator(self, X, y, cv):
        """Create an iterator over folds."""
        for train_idx, test_idx in cv.split(X, y):
            yield {
                "X_train": X.iloc[train_idx],
                "X_test": X.iloc[test_idx],
                "y_train": y[train_idx] if y is not None else None,
                "y_test": y[test_idx] if y is not None else None,
            }


class BatchControllerNode(NodeRuntime):
    """Control data streaming in batches."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Data Stream")
        if data is None:
            return NodeResult(outputs={}, success=False, error_message="No data stream")
        
        df = ensure_dataframe(data)
        
        batch_size = self.get_option("Batch Size", 32)
        shuffle = self.get_option("Shuffle", True)
        drop_last = self.get_option("Drop Last", False)
        
        n = len(df)
        indices = np.arange(n)
        
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            
            if drop_last and len(batch_idx) < batch_size:
                continue
            
            batches.append(df.iloc[batch_idx])
        
        batch_info = {
            "total_samples": n,
            "batch_size": batch_size,
            "n_batches": len(batches),
            "dropped_samples": n % batch_size if drop_last else 0,
        }
        
        return NodeResult(
            outputs={
                "Batched Data": batches,
                "Batch Info": batch_info,
            }
        )


class ConditionalRouterNode(NodeRuntime):
    """Route data based on conditions."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Data")
        if data is None:
            return NodeResult(outputs={}, success=False, error_message="No data")
        
        df = ensure_dataframe(data)
        
        condition_type = self.get_option("Condition Type", "Column Value")
        column = self.get_option("Column", "status")
        operator = self.get_option("Operator", "==")
        value = self.get_option("Value", "active")
        
        try:
            if condition_type == "Column Value":
                if column not in df.columns:
                    return NodeResult(outputs={"True Branch": df, "False Branch": pd.DataFrame()})
                
                # Convert value type
                col_dtype = df[column].dtype
                if np.issubdtype(col_dtype, np.number):
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                
                # Apply condition
                ops = {
                    "==": lambda x, v: x == v,
                    "!=": lambda x, v: x != v,
                    ">": lambda x, v: x > v,
                    "<": lambda x, v: x < v,
                    ">=": lambda x, v: x >= v,
                    "<=": lambda x, v: x <= v,
                    "in": lambda x, v: x.isin(v.split(",") if isinstance(v, str) else [v]),
                    "not in": lambda x, v: ~x.isin(v.split(",") if isinstance(v, str) else [v]),
                }
                
                op_func = ops.get(operator, lambda x, v: x == v)
                mask = op_func(df[column], value)
                
            elif condition_type == "Row Count":
                mask = pd.Series([len(df) > int(value)] * len(df), index=df.index)
                
            elif condition_type == "Data Shape":
                # Check if data has certain dimensions
                mask = pd.Series([df.shape[1] > int(value)] * len(df), index=df.index)
                
            else:
                mask = pd.Series([True] * len(df), index=df.index)
            
            true_branch = df[mask]
            false_branch = df[~mask]
            
            return NodeResult(
                outputs={
                    "True Branch": true_branch,
                    "False Branch": false_branch,
                },
                metadata={
                    "true_count": len(true_branch),
                    "false_count": len(false_branch),
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
