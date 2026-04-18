"""
Base Node Runtime - Foundation for all node execution logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator
import time

import pandas as pd
import numpy as np


@dataclass
class NodeResult:
    """Result from node execution."""
    outputs: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: str | None = None


@dataclass
class NodeContext:
    """Execution context for nodes."""
    cache: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    global_state: dict[str, Any] = field(default_factory=dict)

    def set_warning(self, message: str) -> None:
        self.warnings.append(message)

    def set_error(self, message: str) -> None:
        self.errors.append(message)
    
    def get_cached(self, key: str, default: Any = None) -> Any:
        return self.cache.get(key, default)
    
    def set_cached(self, key: str, value: Any) -> None:
        self.cache[key] = value


class NodeRuntime(ABC):
    """
    Base class for node runtime execution.
    
    All nodes inherit from this and implement the `run` method.
    """
    
    def __init__(self, node_id: str, node_data: dict) -> None:
        self.node_id = node_id
        self.node_data = node_data
        self._raw_options = node_data.get("options", [])
        self._context = NodeContext()
        self._fitted_state: dict[str, Any] = {}
        
        # Parse options - support both list and dict format
        self.options = self._parse_options(self._raw_options)
    
    def _parse_options(self, raw_options) -> dict:
        """Parse options from various formats to dict."""
        if isinstance(raw_options, dict):
            return raw_options
        if isinstance(raw_options, list):
            result = {}
            for opt in raw_options:
                if isinstance(opt, dict):
                    label = opt.get("label", opt.get("name", ""))
                    value = opt.get("value", opt.get("default", None))
                    if label:
                        result[label] = value
            return result
        return {}
    
    def get_option(self, key: str, default: Any = None) -> Any:
        """Get option value by key (case-insensitive)."""
        # Direct match
        if key in self.options:
            return self.options[key]
        # Case-insensitive match
        key_lower = key.lower()
        for k, v in self.options.items():
            if k.lower() == key_lower:
                return v
        return default
    
    def set_fitted_state(self, key: str, value: Any) -> None:
        """Store fitted state (e.g., scaler, encoder)."""
        self._fitted_state[key] = value
    
    def get_fitted_state(self, key: str, default: Any = None) -> Any:
        """Get fitted state."""
        return self._fitted_state.get(key, default)
    
    @abstractmethod
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        """
        Execute the node logic.
        
        Args:
            inputs: Dictionary of input port name -> data
            
        Returns:
            NodeResult with outputs and metadata
        """
        pass
    
    def validate_inputs(self, inputs: dict[str, Any], required: list[str]) -> bool:
        """Validate that required inputs are present."""
        for req in required:
            if req not in inputs or inputs[req] is None:
                self._context.set_error(f"Missing required input: {req}")
                return False
        return True
    
    def execute(self, inputs: dict[str, Any]) -> NodeResult:
        """Execute with timing and error handling."""
        start_time = time.time()
        try:
            result = self.run(inputs)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            return NodeResult(
                outputs={},
                metadata={"error": str(e)},
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )


def ensure_dataframe(data: Any) -> pd.DataFrame:
    """Convert data to DataFrame if not already."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame(data)
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame([data])


def safe_column_select(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Safely select columns, returning only those that exist."""
    existing = [c for c in columns if c in df.columns]
    return df[existing]
