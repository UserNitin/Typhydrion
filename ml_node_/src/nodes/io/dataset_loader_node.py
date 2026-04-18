"""
Dataset Loader Node - Load data from various file formats.
"""
from __future__ import annotations

from typing import Any, Iterator
import json

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe


class DatasetLoaderNode(NodeRuntime):
    """
    Load datasets from files with support for:
    - All pandas read_* functions
    - Chunked loading for large files
    - Multiple file formats (CSV, Excel, JSON, Parquet, etc.)
    """
    
    def __init__(self, node_id: str, node_data: dict) -> None:
        super().__init__(node_id, node_data)
        self.reader_name = self.get_option("reader", "read_csv")
        self.file_path = self.get_option("path", "")
        self.chunk_size = self.get_option("chunksize", None)
        self.reader_kwargs = self.get_option("kwargs", {})
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        """Load dataset from file."""
        if not self.file_path:
            return NodeResult(
                outputs={},
                success=False,
                error_message="No file path specified"
            )
        
        try:
            # Get the pandas reader function
            reader_func = getattr(pd, self.reader_name, pd.read_csv)
            
            # Build kwargs
            kwargs = dict(self.reader_kwargs)
            
            # Handle chunked reading for supported formats
            if self.chunk_size and self.reader_name in ["read_csv", "read_table", "read_json"]:
                kwargs["chunksize"] = self.chunk_size
            
            # Load the data
            data = reader_func(self.file_path, **kwargs)
            
            # Handle chunked reader (returns iterator)
            if hasattr(data, "__iter__") and not isinstance(data, pd.DataFrame):
                # Concatenate all chunks
                chunks = list(data)
                df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            else:
                df = data
            
            # Generate schema and stats
            schema = self._generate_schema(df)
            stats = self._generate_stats(df)
            
            # Identify feature and target candidates
            feature_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
            target_candidates = df.columns.tolist()
            
            return NodeResult(
                outputs={
                    "Raw Data": df,
                    "Feature Candidates": feature_candidates,
                    "Target Candidates": target_candidates,
                    "Schema": schema,
                    "Stats": stats,
                },
                metadata={
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                }
            )
            
        except Exception as e:
            return NodeResult(
                outputs={},
                success=False,
                error_message=f"Failed to load dataset: {str(e)}"
            )
    
    def _generate_schema(self, df: pd.DataFrame) -> dict:
        """Generate data schema."""
        return {
            col: {
                "dtype": str(df[col].dtype),
                "nullable": df[col].isnull().any(),
                "unique_count": df[col].nunique(),
            }
            for col in df.columns
        }
    
    def _generate_stats(self, df: pd.DataFrame) -> dict:
        """Generate basic statistics."""
        stats = {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_total": df.isnull().sum().sum(),
            "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100 if df.size > 0 else 0,
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        }
        return stats
    
    def load_chunks(self) -> Iterator[pd.DataFrame]:
        """Generator for loading data in chunks."""
        if not self.file_path or not self.chunk_size:
            return
        
        reader_func = getattr(pd, self.reader_name, pd.read_csv)
        kwargs = dict(self.reader_kwargs)
        kwargs["chunksize"] = self.chunk_size
        
        try:
            for chunk in reader_func(self.file_path, **kwargs):
                yield chunk
        except Exception:
            return


class DataPreviewNode(NodeRuntime):
    """Preview data in tabular format."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Data")
        if data is None:
            return NodeResult(outputs={"Preview": None}, success=False, error_message="No data")
        
        df = ensure_dataframe(data)
        n_rows = self.get_option("rows", 100)
        
        return NodeResult(
            outputs={
                "Preview": df.head(n_rows),
                "Info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "dtypes": df.dtypes.to_dict(),
                }
            }
        )


class DatasetMergerNode(NodeRuntime):
    """Merge multiple datasets."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        df_a = inputs.get("Dataset A")
        df_b = inputs.get("Dataset B")

        if df_a is None and df_b is None:
            return NodeResult(outputs={}, success=False, error_message="Missing input datasets")

        if df_a is None and df_b is not None:
            df_only = ensure_dataframe(df_b).copy().convert_dtypes()
            return NodeResult(
                outputs={
                    "Merged Data": df_only,
                    "Schema": {"columns": df_only.columns.tolist(), "dtypes": df_only.dtypes.astype(str).to_dict()},
                },
                metadata={"mode": "passthrough", "source": "Dataset B"}
            )

        if df_b is None and df_a is not None:
            df_only = ensure_dataframe(df_a).copy().convert_dtypes()
            return NodeResult(
                outputs={
                    "Merged Data": df_only,
                    "Schema": {"columns": df_only.columns.tolist(), "dtypes": df_only.dtypes.astype(str).to_dict()},
                },
                metadata={"mode": "passthrough", "source": "Dataset A"}
            )

        df_a = ensure_dataframe(df_a).copy().convert_dtypes()
        df_b = ensure_dataframe(df_b).copy().convert_dtypes()
        
        merge_type = self.get_option("Merge Type", "Concat (Rows)")
        join_key = self.get_option("Join Key", "id")
        handle_mismatch = self.get_option("Handle Mismatch", "Fill NaN")
        reset_index = self.get_option("Reset Index", True)
        
        try:
            if merge_type == "Concat (Rows)":
                if handle_mismatch == "Error":
                    if list(df_a.columns) != list(df_b.columns):
                        return NodeResult(outputs={}, success=False, error_message="Column mismatch for row concat")
                elif handle_mismatch == "Drop Rows":
                    common_cols = [c for c in df_a.columns if c in df_b.columns]
                    if not common_cols:
                        return NodeResult(outputs={}, success=False, error_message="No common columns to concat")
                    df_a = df_a[common_cols]
                    df_b = df_b[common_cols]
                result = pd.concat([df_a, df_b], ignore_index=reset_index)
            elif merge_type == "Join (Columns)":
                if reset_index:
                    df_a = df_a.reset_index(drop=True)
                    df_b = df_b.reset_index(drop=True)

                if join_key and (join_key in df_a.columns) and (join_key in df_b.columns):
                    result = df_a.merge(df_b, on=join_key, how="left", suffixes=("", "_b"))
                else:
                    len_a, len_b = len(df_a), len(df_b)
                    if len_a != len_b:
                        if handle_mismatch == "Error":
                            return NodeResult(
                                outputs={},
                                success=False,
                                error_message=f"Row count mismatch for column join: {len_a} vs {len_b}",
                            )
                        if handle_mismatch == "Drop Rows":
                            n = min(len_a, len_b)
                            df_a = df_a.iloc[:n].copy()
                            df_b = df_b.iloc[:n].copy()
                    result = pd.concat([df_a, df_b], axis=1)
            elif merge_type == "Left Join":
                result = df_a.merge(df_b, on=join_key, how="left")
            elif merge_type == "Right Join":
                result = df_a.merge(df_b, on=join_key, how="right")
            elif merge_type == "Inner Join":
                result = df_a.merge(df_b, on=join_key, how="inner")
            elif merge_type == "Outer Join":
                result = df_a.merge(df_b, on=join_key, how="outer")
            else:
                result = pd.concat([df_a, df_b], ignore_index=True)
            
            if reset_index:
                result = result.reset_index(drop=True)

            result = result.convert_dtypes()
            
            return NodeResult(
                outputs={
                    "Merged Data": result,
                    "Schema": {"columns": result.columns.tolist(), "dtypes": result.dtypes.astype(str).to_dict()}
                }
            )
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))


class ColumnSelectorNode(NodeRuntime):
    """Select features and target columns."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Chunk")
        if data is None:
            return NodeResult(outputs={}, success=False, error_message="No data")
        
        df = ensure_dataframe(data)
        
        features_str = self.get_option("Features", "")
        target_str = self.get_option("Target", "")
        drop_selected = self.get_option("Drop Selected", False)
        
        feature_cols = [c.strip() for c in features_str.split(",") if c.strip()]
        target_col = target_str.strip()
        
        # Get features
        if feature_cols:
            features = df[[c for c in feature_cols if c in df.columns]]
        else:
            features = df.drop(columns=[target_col], errors="ignore")
        
        # Get target
        target = df[target_col] if target_col in df.columns else None
        
        return NodeResult(
            outputs={
                "Features": features,
                "Target": target,
            }
        )


class FilterNode(NodeRuntime):
    """Filter rows by condition."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Chunk")
        if data is None:
            return NodeResult(outputs={}, success=False, error_message="No data")
        
        df = ensure_dataframe(data)
        
        column = self.get_option("Column", "")
        condition = self.get_option("Condition", "==")
        value = self.get_option("Value", "")
        enabled = self.get_option("Enabled", True)
        
        if not enabled or not column or column not in df.columns:
            return NodeResult(outputs={"Filtered Chunk": df, "Rejected Rows": pd.DataFrame()})
        
        try:
            # Convert value to appropriate type
            col_dtype = df[column].dtype
            if np.issubdtype(col_dtype, np.number):
                try:
                    value = float(value)
                except ValueError:
                    pass
            
            # Apply condition
            if condition == ">":
                mask = df[column] > value
            elif condition == "<":
                mask = df[column] < value
            elif condition == ">=":
                mask = df[column] >= value
            elif condition == "<=":
                mask = df[column] <= value
            elif condition == "==":
                mask = df[column] == value
            elif condition == "!=":
                mask = df[column] != value
            elif condition == "contains":
                mask = df[column].astype(str).str.contains(str(value), na=False)
            elif condition == "startswith":
                mask = df[column].astype(str).str.startswith(str(value), na=False)
            elif condition == "isnull":
                mask = df[column].isnull()
            elif condition == "notnull":
                mask = df[column].notnull()
            else:
                mask = pd.Series([True] * len(df))
            
            filtered = df[mask]
            rejected = df[~mask]
            
            return NodeResult(
                outputs={
                    "Filtered Chunk": filtered,
                    "Rejected Rows": rejected,
                },
                metadata={"filtered_count": len(filtered), "rejected_count": len(rejected)}
            )
        except Exception as e:
            return NodeResult(outputs={"Filtered Chunk": df, "Rejected Rows": pd.DataFrame()}, 
                            success=False, error_message=str(e))


class FinalOutputNode(NodeRuntime):
    """
    Final Output Node - Display final pipeline result.
    This node has only an input port and displays the data in the Data Preview card.
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Data")
        if data is None:
            return NodeResult(
                outputs={"Final Data": None}, 
                success=False, 
                error_message="No data received"
            )
        
        df = ensure_dataframe(data)
        output_name = self.get_option("Output Name", "Pipeline Result")
        
        return NodeResult(
            outputs={
                "Final Data": df,
            },
            metadata={
                "output_name": output_name,
                "rows": len(df),
                "columns": len(df.columns),
                "is_final_output": True,
            }
        )
