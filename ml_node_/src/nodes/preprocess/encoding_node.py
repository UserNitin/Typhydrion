"""
Categorical Encoder Node - Encode categorical variables.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import sklearn encoders
try:
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EncodingNode(NodeRuntime):
    """
    Encode categorical columns with various methods:
    - One-Hot encoding
    - Label encoding
    - Ordinal encoding
    - Target encoding
    - Binary encoding
    - Frequency encoding
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Chunk")
        if data is None:
            return NodeResult(outputs={}, success=False, error_message="No data")
        
        df = ensure_dataframe(data).copy()
        
        method = self.get_option("Method", "One-Hot")
        handle_unknown = self.get_option("Handle Unknown", "Ignore")
        max_categories = self.get_option("Max Categories", 10)
        drop_first = self.get_option("Drop First", False)
        
        # Get categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        if not cat_cols:
            return NodeResult(
                outputs={"Encoded Features": df, "Encoder State": {}},
                metadata={"message": "No categorical columns found"}
            )
        
        encoder_state = {}
        
        try:
            if method == "One-Hot":
                df = self._one_hot_encode(df, cat_cols, max_categories, drop_first, encoder_state)
            
            elif method == "Label":
                df = self._label_encode(df, cat_cols, encoder_state)
            
            elif method == "Ordinal":
                df = self._ordinal_encode(df, cat_cols, encoder_state)
            
            elif method == "Target":
                target = inputs.get("Target")
                if target is not None:
                    df = self._target_encode(df, cat_cols, target, encoder_state)
                else:
                    df = self._label_encode(df, cat_cols, encoder_state)
            
            elif method == "Binary":
                df = self._binary_encode(df, cat_cols, encoder_state)
            
            elif method == "Frequency":
                df = self._frequency_encode(df, cat_cols, encoder_state)
            
            # Store encoder state for later use
            self.set_fitted_state("encoder_state", encoder_state)
            
            return NodeResult(
                outputs={
                    "Encoded Features": df,
                    "Encoder State": encoder_state,
                },
                metadata={"encoded_columns": cat_cols, "method": method}
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _one_hot_encode(self, df: pd.DataFrame, cols: list[str], max_cat: int, 
                        drop_first: bool, state: dict) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        for col in cols:
            # Limit categories
            top_cats = df[col].value_counts().head(max_cat).index.tolist()
            df[col] = df[col].apply(lambda x: x if x in top_cats else "Other")
            
            # Create dummies
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)
            
            state[col] = {"categories": top_cats, "type": "one_hot"}
        
        return df
    
    def _label_encode(self, df: pd.DataFrame, cols: list[str], state: dict) -> pd.DataFrame:
        """Label encode categorical columns."""
        for col in cols:
            categories = df[col].unique().tolist()
            cat_to_int = {cat: i for i, cat in enumerate(categories)}
            df[col] = df[col].map(cat_to_int)
            state[col] = {"mapping": cat_to_int, "type": "label"}
        return df
    
    def _ordinal_encode(self, df: pd.DataFrame, cols: list[str], state: dict) -> pd.DataFrame:
        """Ordinal encode (same as label but sorted)."""
        for col in cols:
            categories = sorted(df[col].dropna().unique().tolist())
            cat_to_int = {cat: i for i, cat in enumerate(categories)}
            df[col] = df[col].map(cat_to_int)
            state[col] = {"mapping": cat_to_int, "type": "ordinal"}
        return df
    
    def _target_encode(self, df: pd.DataFrame, cols: list[str], 
                       target: pd.Series, state: dict) -> pd.DataFrame:
        """Target mean encoding."""
        for col in cols:
            means = df.groupby(col)[target.name].mean() if hasattr(target, 'name') and target.name else df[col].map(
                df.assign(_target=target).groupby(col)["_target"].mean()
            )
            if hasattr(target, 'name') and target.name:
                df_temp = df.copy()
                df_temp['_target'] = target.values
                means = df_temp.groupby(col)['_target'].mean()
            df[col] = df[col].map(means)
            state[col] = {"means": means.to_dict(), "type": "target"}
        return df
    
    def _binary_encode(self, df: pd.DataFrame, cols: list[str], state: dict) -> pd.DataFrame:
        """Binary encode categorical columns."""
        for col in cols:
            categories = df[col].unique().tolist()
            cat_to_int = {cat: i for i, cat in enumerate(categories)}
            
            max_val = len(categories) - 1
            n_bits = max(1, int(np.ceil(np.log2(max_val + 1))))
            
            encoded = df[col].map(cat_to_int)
            for bit in range(n_bits):
                df[f"{col}_bit{bit}"] = (encoded >> bit) & 1
            
            df = df.drop(columns=[col])
            state[col] = {"mapping": cat_to_int, "n_bits": n_bits, "type": "binary"}
        
        return df
    
    def _frequency_encode(self, df: pd.DataFrame, cols: list[str], state: dict) -> pd.DataFrame:
        """Frequency (count) encoding."""
        for col in cols:
            freq = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq)
            state[col] = {"frequencies": freq, "type": "frequency"}
        return df
