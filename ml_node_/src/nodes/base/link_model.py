from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum
from typing import Any


class ColumnRole(Enum):
    """Role assignment for columns in a link."""
    FEATURE = "feature"
    TARGET = "target"
    IGNORE = "ignore"
    INDEX = "index"
    PASSTHROUGH = "passthrough"


class LinkState(Enum):
    """Visual and logical state of a link."""
    NORMAL = "normal"       # Valid, active link
    WARNING = "warning"     # Has warnings but functional
    ERROR = "error"         # Invalid, broken
    DISABLED = "disabled"   # User disabled
    LOCKED = "locked"       # Read-only, cannot edit
    EXECUTING = "executing" # Currently flowing data
    COMPLETED = "completed" # Execution finished


@dataclass
class ColumnConfig:
    """Configuration for a single column in a link."""
    name: str
    data_type: str = "unknown"  # numeric, categorical, datetime, text, bool
    role: ColumnRole = ColumnRole.FEATURE
    enabled: bool = True
    missing_pct: float = 0.0
    unique_count: int = 0
    sample_values: list[Any] = field(default_factory=list)
    ai_suggested_role: ColumnRole | None = None
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "role": self.role.value,
            "enabled": self.enabled,
            "missing_pct": self.missing_pct,
            "unique_count": self.unique_count,
        }


@dataclass
class LinkModel:
    """
    Comprehensive model for a connecting link between nodes.
    
    A link carries data + metadata between nodes and enforces
    compatibility rules, column selection, and validation.
    """
    
    # ═══════════════════════════════════════════════════════════════
    # Identity
    # ═══════════════════════════════════════════════════════════════
    link_id: str = field(default_factory=lambda: uuid4().hex)
    source_node_id: str = ""
    source_node_name: str = ""
    source_port_id: str = ""
    source_port_name: str = ""
    target_node_id: str = ""
    target_node_name: str = ""
    target_port_id: str = ""
    target_port_name: str = ""
    
    # ═══════════════════════════════════════════════════════════════
    # Column Configuration
    # ═══════════════════════════════════════════════════════════════
    columns: list[ColumnConfig] = field(default_factory=list)
    columns_passed: list[str] = field(default_factory=list)  # Legacy support
    column_types: dict[str, str] = field(default_factory=dict)
    
    # ═══════════════════════════════════════════════════════════════
    # Data Metadata
    # ═══════════════════════════════════════════════════════════════
    data_type: str = "any"  # numeric, categorical, target, tensor, metrics, any
    row_count: int | None = None
    row_count_per_chunk: int | None = None
    total_columns: int = 0
    enabled_columns: int = 0
    data_shape: str | None = None
    memory_size_bytes: int | None = None
    
    # ═══════════════════════════════════════════════════════════════
    # Execution Metadata
    # ═══════════════════════════════════════════════════════════════
    chunk_size: int = 1000
    estimated_memory_cost: str = "Unknown"
    estimated_compute_cost: str = "Low"  # Low, Medium, High
    execution_order: int = 0
    chunks_processed: int = 0
    total_chunks: int = 0
    
    # ═══════════════════════════════════════════════════════════════
    # Validation State
    # ═══════════════════════════════════════════════════════════════
    state: LinkState = LinkState.NORMAL
    is_valid: bool = True
    error_message: str = ""
    warning_message: str = ""
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════════
    # Control Flags
    # ═══════════════════════════════════════════════════════════════
    enabled: bool = True
    locked: bool = False
    ai_suggested: bool = False
    user_confirmed: bool = False
    
    # ═══════════════════════════════════════════════════════════════
    # Methods
    # ═══════════════════════════════════════════════════════════════
    
    def add_column(self, name: str, data_type: str = "unknown", 
                   role: ColumnRole = ColumnRole.FEATURE) -> ColumnConfig:
        """Add a column to the link configuration."""
        col = ColumnConfig(name=name, data_type=data_type, role=role)
        self.columns.append(col)
        self.columns_passed.append(name)
        self.column_types[name] = data_type
        self._update_counts()
        return col
    
    def remove_column(self, name: str) -> None:
        """Remove a column from the link."""
        self.columns = [c for c in self.columns if c.name != name]
        if name in self.columns_passed:
            self.columns_passed.remove(name)
        if name in self.column_types:
            del self.column_types[name]
        self._update_counts()
    
    def set_column_role(self, name: str, role: ColumnRole) -> None:
        """Set the role for a specific column."""
        for col in self.columns:
            if col.name == name:
                col.role = role
                break
    
    def set_column_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a specific column."""
        for col in self.columns:
            if col.name == name:
                col.enabled = enabled
                break
        self._update_counts()
    
    def get_enabled_columns(self) -> list[ColumnConfig]:
        """Get list of enabled columns."""
        return [c for c in self.columns if c.enabled]
    
    def get_feature_columns(self) -> list[ColumnConfig]:
        """Get columns marked as features."""
        return [c for c in self.columns if c.enabled and c.role == ColumnRole.FEATURE]
    
    def get_target_columns(self) -> list[ColumnConfig]:
        """Get columns marked as target."""
        return [c for c in self.columns if c.enabled and c.role == ColumnRole.TARGET]
    
    def _update_counts(self) -> None:
        """Update column counts."""
        self.total_columns = len(self.columns)
        self.enabled_columns = len([c for c in self.columns if c.enabled])
        self.columns_passed = [c.name for c in self.columns if c.enabled]
    
    def validate(self) -> bool:
        """
        Validate the link configuration.
        Returns True if valid, False if errors found.
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Check if link has endpoints
        if not self.source_node_id or not self.target_node_id:
            self.validation_errors.append("Link missing source or target node")
        
        # Check if any columns are selected
        if not self.get_enabled_columns():
            self.validation_warnings.append("No columns selected - link will pass no data")
        
        # Check for target leakage (target in features)
        feature_names = {c.name for c in self.get_feature_columns()}
        target_names = {c.name for c in self.get_target_columns()}
        if feature_names & target_names:
            self.validation_errors.append("Target leakage: same column marked as feature and target")
        
        # Check for high missing values
        for col in self.get_enabled_columns():
            if col.missing_pct > 50:
                self.validation_warnings.append(f"Column '{col.name}' has {col.missing_pct:.1f}% missing values")
        
        # Update state based on validation
        self.is_valid = len(self.validation_errors) == 0
        if not self.is_valid:
            self.state = LinkState.ERROR
            self.error_message = "; ".join(self.validation_errors)
        elif self.validation_warnings:
            self.state = LinkState.WARNING
            self.warning_message = "; ".join(self.validation_warnings)
        else:
            self.state = LinkState.NORMAL
            self.error_message = ""
            self.warning_message = ""
        
        return self.is_valid
    
    def estimate_memory(self) -> str:
        """Estimate memory usage for this link."""
        if self.row_count and self.enabled_columns:
            # Rough estimate: 8 bytes per numeric, 50 bytes per string
            bytes_est = self.row_count * self.enabled_columns * 20
            if bytes_est < 1024:
                self.estimated_memory_cost = f"{bytes_est} B"
            elif bytes_est < 1024 * 1024:
                self.estimated_memory_cost = f"{bytes_est / 1024:.1f} KB"
            elif bytes_est < 1024 * 1024 * 1024:
                self.estimated_memory_cost = f"{bytes_est / (1024*1024):.1f} MB"
            else:
                self.estimated_memory_cost = f"{bytes_est / (1024*1024*1024):.1f} GB"
        return self.estimated_memory_cost
    
    def get_summary(self) -> str:
        """Get a summary string for the link."""
        enabled = len(self.get_enabled_columns())
        total = self.total_columns
        return f"{enabled}/{total} columns • {self.data_type} • {self.state.value}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "link_id": self.link_id,
            "source": {
                "node_id": self.source_node_id,
                "node_name": self.source_node_name,
                "port_id": self.source_port_id,
                "port_name": self.source_port_name,
            },
            "target": {
                "node_id": self.target_node_id,
                "node_name": self.target_node_name,
                "port_id": self.target_port_id,
                "port_name": self.target_port_name,
            },
            "columns": [c.to_dict() for c in self.columns],
            "data_type": self.data_type,
            "state": self.state.value,
            "enabled": self.enabled,
            "locked": self.locked,
        }
