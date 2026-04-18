"""
Feature Scaler Node - Scale/normalize numeric features.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import sklearn scalers
try:
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
        Normalizer, PowerTransformer, QuantileTransformer
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ScalingNode(NodeRuntime):
    """
    Scale/normalize numeric features with various methods:
    - StandardScaler (z-score normalization)
    - MinMaxScaler (0-1 scaling)
    - RobustScaler (robust to outliers)
    - MaxAbsScaler (scale by max absolute value)
    - Normalizer (L1/L2 normalization)
    - PowerTransformer (Yeo-Johnson/Box-Cox)
    - QuantileTransformer (uniform/normal distribution)
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Features")
        if data is None:
            return NodeResult(outputs={}, success=False, error_message="No features data")
        
        df = ensure_dataframe(data).copy()
        
        method = self.get_option("Method", "StandardScaler")
        with_mean = self.get_option("With Mean", True)
        with_std = self.get_option("With Std", True)
        clip_outliers = self.get_option("Clip Outliers", False)
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        
        if not numeric_cols:
            return NodeResult(
                outputs={"Scaled Features": df, "Scaler State": {}},
                metadata={"message": "No numeric columns to scale"}
            )
        
        numeric_df = df[numeric_cols]
        scaler_state = {}
        
        try:
            if SKLEARN_AVAILABLE:
                scaled_data, scaler_state = self._scale_with_sklearn(
                    numeric_df, method, with_mean, with_std
                )
            else:
                scaled_data, scaler_state = self._scale_manual(
                    numeric_df, method, with_mean, with_std
                )
            
            # Create result DataFrame
            scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=df.index)
            
            # Clip outliers if requested
            if clip_outliers:
                scaled_df = scaled_df.clip(lower=-3, upper=3)
            
            # Add back non-numeric columns
            for col in non_numeric_cols:
                scaled_df[col] = df[col]
            
            # Store scaler state
            self.set_fitted_state("scaler_state", scaler_state)
            
            return NodeResult(
                outputs={
                    "Scaled Features": scaled_df,
                    "Scaler State": scaler_state,
                },
                metadata={
                    "method": method,
                    "scaled_columns": numeric_cols,
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _scale_with_sklearn(self, df: pd.DataFrame, method: str, 
                            with_mean: bool, with_std: bool) -> tuple[np.ndarray, dict]:
        """Scale using sklearn."""
        scaler_map = {
            "StandardScaler": StandardScaler(with_mean=with_mean, with_std=with_std),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(with_centering=with_mean, with_scaling=with_std),
            "MaxAbsScaler": MaxAbsScaler(),
            "Normalizer": Normalizer(),
            "PowerTransformer": PowerTransformer(method="yeo-johnson"),
            "QuantileTransformer": QuantileTransformer(output_distribution="normal"),
        }
        
        scaler = scaler_map.get(method, StandardScaler())
        scaled_data = scaler.fit_transform(df.values)
        
        # Extract state for serialization
        state = {"method": method}
        if hasattr(scaler, "mean_"):
            state["mean"] = scaler.mean_.tolist()
        if hasattr(scaler, "scale_"):
            state["scale"] = scaler.scale_.tolist()
        if hasattr(scaler, "var_"):
            state["var"] = scaler.var_.tolist()
        if hasattr(scaler, "min_"):
            state["min"] = scaler.min_.tolist()
        if hasattr(scaler, "data_max_"):
            state["data_max"] = scaler.data_max_.tolist()
        if hasattr(scaler, "data_min_"):
            state["data_min"] = scaler.data_min_.tolist()
        
        return scaled_data, state
    
    def _scale_manual(self, df: pd.DataFrame, method: str,
                      with_mean: bool, with_std: bool) -> tuple[np.ndarray, dict]:
        """Manual scaling without sklearn."""
        data = df.values.astype(float)
        state = {"method": method}
        
        if method == "StandardScaler":
            mean = np.nanmean(data, axis=0) if with_mean else np.zeros(data.shape[1])
            std = np.nanstd(data, axis=0) if with_std else np.ones(data.shape[1])
            std[std == 0] = 1  # Avoid division by zero
            scaled = (data - mean) / std
            state["mean"] = mean.tolist()
            state["std"] = std.tolist()
        
        elif method == "MinMaxScaler":
            min_val = np.nanmin(data, axis=0)
            max_val = np.nanmax(data, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            scaled = (data - min_val) / range_val
            state["min"] = min_val.tolist()
            state["max"] = max_val.tolist()
        
        elif method == "RobustScaler":
            median = np.nanmedian(data, axis=0)
            q1 = np.nanpercentile(data, 25, axis=0)
            q3 = np.nanpercentile(data, 75, axis=0)
            iqr = q3 - q1
            iqr[iqr == 0] = 1
            scaled = (data - median) / iqr
            state["median"] = median.tolist()
            state["iqr"] = iqr.tolist()
        
        elif method == "MaxAbsScaler":
            max_abs = np.nanmax(np.abs(data), axis=0)
            max_abs[max_abs == 0] = 1
            scaled = data / max_abs
            state["max_abs"] = max_abs.tolist()
        
        else:
            # Default to StandardScaler
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            std[std == 0] = 1
            scaled = (data - mean) / std
            state["mean"] = mean.tolist()
            state["std"] = std.tolist()
        
        return scaled, state


class OutlierHandlerNode(NodeRuntime):
    """Handle outliers in data."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Chunk")
        if data is None:
            return NodeResult(outputs={}, success=False, error_message="No data")
        
        df = ensure_dataframe(data).copy()
        
        method = self.get_option("Method", "IQR")
        threshold = self.get_option("Threshold", 1.5)
        action = self.get_option("Action", "Remove")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return NodeResult(
                outputs={"Clean Chunk": df, "Outlier Mask": pd.DataFrame()},
                metadata={"message": "No numeric columns"}
            )
        
        try:
            outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
            
            for col in numeric_cols:
                if method == "IQR":
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    outlier_mask[col] = (df[col] < lower) | (df[col] > upper)
                
                elif method == "Z-Score":
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outlier_mask[col] = z_scores > threshold
                
                elif method in ["Isolation Forest", "LOF", "DBSCAN"]:
                    # Requires sklearn
                    if SKLEARN_AVAILABLE:
                        from sklearn.ensemble import IsolationForest
                        from sklearn.neighbors import LocalOutlierFactor
                        
                        col_data = df[col].values.reshape(-1, 1)
                        col_data = np.nan_to_num(col_data)
                        
                        if method == "Isolation Forest":
                            clf = IsolationForest(contamination=min(0.5, threshold/10))
                            preds = clf.fit_predict(col_data)
                            outlier_mask[col] = preds == -1
                        elif method == "LOF":
                            clf = LocalOutlierFactor(contamination=min(0.5, threshold/10))
                            preds = clf.fit_predict(col_data)
                            outlier_mask[col] = preds == -1
            
            # Apply action
            any_outlier = outlier_mask.any(axis=1)
            
            if action == "Remove":
                clean_df = df[~any_outlier]
            elif action == "Cap":
                clean_df = df.copy()
                for col in numeric_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    clean_df[col] = clean_df[col].clip(lower, upper)
            elif action == "Replace Mean":
                clean_df = df.copy()
                for col in numeric_cols:
                    mean_val = df[col][~outlier_mask[col]].mean()
                    clean_df.loc[outlier_mask[col], col] = mean_val
            elif action == "Replace Median":
                clean_df = df.copy()
                for col in numeric_cols:
                    median_val = df[col][~outlier_mask[col]].median()
                    clean_df.loc[outlier_mask[col], col] = median_val
            else:  # Flag Only
                clean_df = df.copy()
                clean_df["_is_outlier"] = any_outlier
            
            return NodeResult(
                outputs={
                    "Clean Chunk": clean_df,
                    "Outlier Mask": outlier_mask,
                },
                metadata={
                    "outliers_found": int(any_outlier.sum()),
                    "method": method,
                    "action": action,
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
