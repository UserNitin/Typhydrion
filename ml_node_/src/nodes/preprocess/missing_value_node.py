"""
Missing Value Handler Node - Handle null/missing values in data.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe


class MissingValueNode(NodeRuntime):
    """
    Handle missing values with various strategies:
    - Mean, Median, Mode imputation
    - Constant fill
    - Forward/Backward fill
    - Interpolation
    - Drop rows/columns
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Chunk")
        if data is None:
            return NodeResult(outputs={}, success=False, error_message="No data")
        
        df = ensure_dataframe(data).copy()
        
        strategy = self.get_option("Strategy", "Mean")
        fill_value = self.get_option("Fill Value", "0")
        drop_threshold = self.get_option("Drop Threshold", self.get_option("Drop Thresh", 0.5))
        selected_only = self.get_option(
            "Selected Columns Only",
            self.get_option("Selected Columns", self.get_option("Selected Column", False)),
        )
        selected_cols_str = self.get_option("Columns", self.get_option("Selected Columns", ""))
        
        # Generate missing report before processing
        missing_report = self._generate_missing_report(df)
        
        # Determine columns to process
        if selected_only and selected_cols_str:
            target_cols = [c.strip() for c in selected_cols_str.split(",") if c.strip() in df.columns]
        else:
            target_cols = df.columns.tolist()
        
        try:
            if strategy == "Mean":
                numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c in target_cols]
                if numeric_cols:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                # For non-numeric columns, fallback to mode so categorical data is handled.
                cat_cols = [c for c in target_cols if c not in numeric_cols]
                for col in cat_cols:
                    if col in df.columns and df[col].isnull().any():
                        mode_val = df[col].mode(dropna=True)
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val.iloc[0])
            
            elif strategy == "Median":
                numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c in target_cols]
                if numeric_cols:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                # Median is undefined for categorical columns; use mode fallback.
                cat_cols = [c for c in target_cols if c not in numeric_cols]
                for col in cat_cols:
                    if col in df.columns and df[col].isnull().any():
                        mode_val = df[col].mode(dropna=True)
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val.iloc[0])
            
            elif strategy == "Mode":
                for col in target_cols:
                    if col in df.columns and df[col].isnull().any():
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val[0])
            
            elif strategy == "Constant":
                try:
                    fill_val = float(fill_value)
                except ValueError:
                    fill_val = fill_value
                df[target_cols] = df[target_cols].fillna(fill_val)
            
            elif strategy == "Forward Fill":
                df[target_cols] = df[target_cols].ffill()
            
            elif strategy == "Backward Fill":
                df[target_cols] = df[target_cols].bfill()
            
            elif strategy == "Interpolate":
                numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c in target_cols]
                if numeric_cols:
                    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            
            elif strategy == "Drop Rows":
                df = df.dropna(subset=target_cols, thresh=int(len(target_cols) * (1 - drop_threshold)))
            
            elif strategy == "Drop Columns":
                cols_to_drop = [c for c in target_cols if df[c].isnull().mean() >= drop_threshold]
                df = df.drop(columns=cols_to_drop)
            
            return NodeResult(
                outputs={
                    "Clean Chunk": df,
                    "Missing Report": missing_report,
                },
                metadata={
                    "strategy": strategy,
                    "rows_before": missing_report["total_rows"],
                    "rows_after": len(df),
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _generate_missing_report(self, df: pd.DataFrame) -> dict:
        """Generate report on missing values."""
        return {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "total_missing": int(df.isnull().sum().sum()),
            "missing_by_column": df.isnull().sum().to_dict(),
            "missing_pct_by_column": (df.isnull().mean() * 100).to_dict(),
        }


class DataTypeConverterNode(NodeRuntime):
    """Convert column data types."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Chunk")
        if data is None:
            return NodeResult(outputs={}, success=False, error_message="No data")
        
        df = ensure_dataframe(data).copy()
        
        columns_str = self.get_option("Columns", "")
        target_type = self.get_option("Target Type", "float")
        date_format = self.get_option("Date Format", "Auto")
        coerce_errors = self.get_option("Coerce Errors", True)
        
        columns = [c.strip() for c in columns_str.split(",") if c.strip()]
        if not columns:
            columns = df.columns.tolist()
        
        errors = "coerce" if coerce_errors else "raise"
        
        try:
            for col in columns:
                if col not in df.columns:
                    continue
                
                if target_type == "int":
                    df[col] = pd.to_numeric(df[col], errors=errors).astype("Int64")
                elif target_type == "float":
                    df[col] = pd.to_numeric(df[col], errors=errors)
                elif target_type == "str":
                    df[col] = df[col].astype(str)
                elif target_type == "bool":
                    df[col] = df[col].astype(bool)
                elif target_type == "datetime":
                    fmt = None if date_format == "Auto" else date_format
                    df[col] = pd.to_datetime(df[col], format=fmt, errors=errors)
                elif target_type == "category":
                    df[col] = df[col].astype("category")
            
            return NodeResult(outputs={"Converted Chunk": df})
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
