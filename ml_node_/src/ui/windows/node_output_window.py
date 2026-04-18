from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
    QScrollArea,
    QFrame,
    QTextEdit,
    QComboBox,
    QPushButton,
    QSplitter,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
import pandas as pd
import numpy as np
import json


class NodeOutputWindow(QWidget):
    """
    Window for displaying node output data.
    
    Features:
    - Output data table view
    - Output ports list
    - Data shape and info
    - Export options
    - Multiple output tabs
    """
    
    output_exported = Signal(str, object)  # port_name, data
    
    def __init__(self) -> None:
        super().__init__()
        self._current_node = None
        self._outputs: dict[str, object] = {}
        
        # Main layout with scroll
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: rgba(30, 40, 55, 150);
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(100, 120, 150, 180);
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)
        
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("📤 Node Output")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: rgba(220, 230, 240, 230);")
        layout.addWidget(title)
        
        # Node info
        self._node_label = QLabel("No node selected")
        self._node_label.setStyleSheet("color: rgba(150, 180, 220, 200); font-size: 13px;")
        layout.addWidget(self._node_label)
        
        # Output port selector
        port_layout = QHBoxLayout()
        port_label = QLabel("Output Port:")
        port_label.setStyleSheet("color: rgba(180, 200, 220, 180);")
        self._port_combo = QComboBox()
        self._port_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(40, 50, 65, 220);
                border: 1px solid rgba(100, 120, 150, 150);
                border-radius: 4px;
                padding: 6px 12px;
                color: rgba(220, 230, 240, 230);
                min-width: 150px;
            }
            QComboBox:hover {
                background-color: rgba(50, 65, 85, 230);
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid rgba(180, 200, 220, 200);
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(35, 45, 60, 250);
                border: 1px solid rgba(100, 120, 150, 180);
                selection-background-color: rgba(70, 100, 140, 200);
                color: rgba(220, 230, 240, 230);
            }
        """)
        self._port_combo.currentTextChanged.connect(self._on_port_changed)
        port_layout.addWidget(port_label)
        port_layout.addWidget(self._port_combo, 1)
        layout.addLayout(port_layout)
        
        # Output info
        info_layout = QHBoxLayout()
        self._shape_label = QLabel("Shape: —")
        self._shape_label.setStyleSheet("color: rgba(100, 180, 150, 200); font-size: 12px;")
        self._type_label = QLabel("Type: —")
        self._type_label.setStyleSheet("color: rgba(100, 150, 200, 200); font-size: 12px;")
        self._memory_label = QLabel("Memory: —")
        self._memory_label.setStyleSheet("color: rgba(180, 150, 100, 200); font-size: 12px;")
        info_layout.addWidget(self._shape_label)
        info_layout.addWidget(self._type_label)
        info_layout.addWidget(self._memory_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        # Tab widget for different views
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(80, 100, 130, 150);
                border-radius: 6px;
                background-color: rgba(30, 40, 55, 150);
            }
            QTabBar::tab {
                background-color: rgba(40, 52, 68, 200);
                color: rgba(200, 210, 220, 200);
                padding: 6px 14px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: rgba(60, 80, 110, 220);
                color: rgba(240, 248, 255, 240);
            }
            QTabBar::tab:hover {
                background-color: rgba(50, 66, 88, 220);
            }
        """)
        layout.addWidget(self._tabs)
        
        # Create tabs
        self._create_table_tab()
        self._create_raw_tab()
        self._create_summary_tab()
        
        # Export button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self._export_btn = QPushButton("📥 Export Output")
        self._export_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(50, 100, 80, 200);
                border: 1px solid rgba(80, 140, 110, 150);
                border-radius: 6px;
                padding: 8px 16px;
                color: rgba(220, 240, 230, 220);
            }
            QPushButton:hover {
                background-color: rgba(60, 120, 95, 220);
            }
            QPushButton:pressed {
                background-color: rgba(40, 85, 65, 220);
            }
        """)
        self._export_btn.clicked.connect(self._export_output)
        export_layout.addWidget(self._export_btn)
        layout.addLayout(export_layout)
        
        # Status
        self._status_label = QLabel("Select a node to see its output")
        self._status_label.setStyleSheet("color: rgba(150, 170, 200, 150); font-size: 11px;")
        layout.addWidget(self._status_label)
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
    
    def _create_table_tab(self) -> None:
        """Create the table view tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: rgba(30, 40, 55, 180);
                border: 1px solid rgba(80, 100, 130, 150);
                border-radius: 6px;
                gridline-color: rgba(80, 100, 130, 100);
                color: rgba(220, 230, 240, 230);
            }
            QHeaderView::section {
                background-color: rgba(45, 60, 80, 200);
                color: rgba(240, 246, 255, 230);
                padding: 6px;
                border: 1px solid rgba(80, 100, 130, 150);
                font-weight: 600;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: rgba(70, 100, 140, 180);
            }
            QTableWidget::item:alternate {
                background-color: rgba(35, 48, 62, 150);
            }
        """)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(True)
        layout.addWidget(self._table)
        
        self._tabs.addTab(widget, "📊 Table")
    
    def _create_raw_tab(self) -> None:
        """Create the raw data view tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        
        self._raw_text = QTextEdit()
        self._raw_text.setReadOnly(True)
        self._raw_text.setFont(QFont("Consolas", 10))
        self._raw_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(25, 32, 42, 200);
                border: 1px solid rgba(80, 100, 130, 150);
                border-radius: 6px;
                color: rgba(200, 220, 240, 220);
                padding: 8px;
            }
        """)
        layout.addWidget(self._raw_text)
        
        self._tabs.addTab(widget, "📝 Raw")
    
    def _create_summary_tab(self) -> None:
        """Create the summary/stats tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        
        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        self._summary_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 40, 55, 180);
                border: 1px solid rgba(80, 100, 130, 150);
                border-radius: 6px;
                color: rgba(200, 220, 240, 220);
                padding: 8px;
            }
        """)
        layout.addWidget(self._summary_text)
        
        self._tabs.addTab(widget, "📋 Summary")
    
    def set_node(self, node_title: str, outputs: dict[str, object] = None) -> None:
        """Set the current node and its outputs."""
        self._current_node = node_title
        self._node_label.setText(f"📦 {node_title}")
        
        self._outputs = outputs or {}
        
        # Update port combo
        self._port_combo.clear()
        if self._outputs:
            self._port_combo.addItems(list(self._outputs.keys()))
            self._status_label.setText(f"Showing output from {node_title}")
        else:
            self._port_combo.addItem("No output")
            self._clear_display()
            self._status_label.setText("No output data available")
    
    def set_output(self, port_name: str, data: object) -> None:
        """Set output data for a specific port."""
        self._outputs[port_name] = data
        
        # Update combo if needed
        if self._port_combo.findText(port_name) == -1:
            self._port_combo.addItem(port_name)
        
        # If this is the current port, update display
        if self._port_combo.currentText() == port_name:
            self._display_output(data)
    
    def set_dataframe(self, df: object) -> None:
        """Set output payload from preview stream.

        Supports a single dataframe-like payload or dict payload from split nodes.
        """
        if df is None:
            self._outputs = {}
            self._port_combo.clear()
            self._port_combo.addItem("No output")
            self._clear_display()
            self._status_label.setText("No output data available")
            return

        if isinstance(df, dict):
            normalized: dict[str, object] = {}
            for key, value in df.items():
                coerced = self._coerce_display_output(value)
                if coerced is not None:
                    normalized[str(key)] = coerced
            if not normalized:
                self._outputs = {}
                self._port_combo.clear()
                self._port_combo.addItem("No output")
                self._clear_display()
                self._status_label.setText("No output data available")
                return

            self._outputs = normalized
            self._port_combo.clear()
            self._port_combo.addItems(list(self._outputs.keys()))
            selected_key = self._pick_preferred_output_key(self._outputs)
            self._port_combo.setCurrentText(selected_key)
            self._display_output(self._outputs[selected_key])
            self._status_label.setText(f"Output dataset: {selected_key}")
            return

        coerced = self._coerce_display_output(df)
        if coerced is None:
            self._outputs = {}
            self._port_combo.clear()
            self._port_combo.addItem("No output")
            self._clear_display()
            self._status_label.setText("No output data available")
            return

        self._outputs = {"Data": coerced}
        if self._port_combo.findText("Data") == -1:
            self._port_combo.clear()
            self._port_combo.addItem("Data")
        self._port_combo.setCurrentText("Data")
        self._display_output(coerced)

        if isinstance(coerced, pd.DataFrame):
            self._status_label.setText(f"Output: {len(coerced)} rows × {len(coerced.columns)} columns")
        else:
            self._status_label.setText(f"Output type: {type(coerced).__name__}")

    def _pick_preferred_output_key(self, outputs: dict[str, object]) -> str:
        priority = ["X_train", "X_test", "X_val", "Data", "Features", "y_train", "y_test", "y_val"]
        for key in priority:
            if key in outputs:
                return key
        return next(iter(outputs.keys()))

    def _coerce_display_output(self, value: object) -> object | None:
        if value is None:
            return None
        if isinstance(value, (pd.DataFrame, pd.Series, list, tuple, dict)):
            return value
        if isinstance(value, np.ndarray):
            if value.ndim <= 2:
                return pd.DataFrame(value)
            return None
        return value
    
    def set_node_output(self, node_title: str, df: pd.DataFrame) -> None:
        """Set output from a specific node (called when node is selected)."""
        self._current_node = node_title
        self._node_label.setText(f"📦 {node_title}")
        
        if df is not None:
            self._outputs = {"Data": df}
            self._port_combo.clear()
            self._port_combo.addItem("Data")
            self._port_combo.setCurrentText("Data")
            self._display_output(df)
            self._status_label.setText(f"Output from {node_title}: {len(df)} rows × {len(df.columns)} columns")
        else:
            self._outputs = {}
            self._port_combo.clear()
            self._port_combo.addItem("No output")
            self._clear_display()
            self._status_label.setText(f"No output data for {node_title}")
    
    def _on_port_changed(self, port_name: str) -> None:
        """Handle port selection change."""
        if port_name and port_name in self._outputs:
            self._display_output(self._outputs[port_name])
    
    def _display_output(self, data: object) -> None:
        """Display output data in all tabs."""
        if data is None:
            self._clear_display()
            return
        
        # Update info labels
        if isinstance(data, pd.DataFrame):
            self._shape_label.setText(f"Shape: {data.shape[0]} × {data.shape[1]}")
            self._type_label.setText("Type: DataFrame")
            mem = data.memory_usage(deep=True).sum()
            self._memory_label.setText(f"Memory: {self._format_bytes(mem)}")
            self._display_dataframe(data)
        elif isinstance(data, (list, tuple)):
            self._shape_label.setText(f"Length: {len(data)}")
            self._type_label.setText(f"Type: {type(data).__name__}")
            self._memory_label.setText("Memory: —")
            self._display_list(data)
        elif isinstance(data, dict):
            self._shape_label.setText(f"Keys: {len(data)}")
            self._type_label.setText("Type: dict")
            self._memory_label.setText("Memory: —")
            self._display_dict(data)
        else:
            self._shape_label.setText("Shape: —")
            self._type_label.setText(f"Type: {type(data).__name__}")
            self._memory_label.setText("Memory: —")
            self._display_other(data)
    
    def _display_dataframe(self, df: pd.DataFrame) -> None:
        """Display a DataFrame."""
        # Table view with strict row/column caps for responsiveness.
        max_rows = 300
        max_cols = 60
        display_df = df.head(max_rows)
        if len(display_df.columns) > max_cols:
            display_df = display_df.iloc[:, :max_cols]
        self._table.setRowCount(len(display_df))
        self._table.setColumnCount(len(display_df.columns))
        self._table.setHorizontalHeaderLabels([str(c) for c in display_df.columns])
        
        for row in range(len(display_df)):
            for col, col_name in enumerate(display_df.columns):
                value = display_df.iloc[row, col]
                item = QTableWidgetItem(str(value))
                self._table.setItem(row, col, item)
        
        # Raw view (bounded rows/cols prevents large-string UI stalls)
        try:
            self._raw_text.setText(df.to_string(max_rows=80, max_cols=40))
        except Exception:
            self._raw_text.setText(str(df.head(80)))
        
        # Summary view
        summary_lines = [
            f"📊 DataFrame Summary",
            f"{'='*40}",
            f"Rows: {len(df):,}",
            f"Columns: {len(df.columns)}",
            f"",
            f"📋 Column Types:",
        ]
        # Keep summary bounded when there are many columns.
        summary_cols = list(df.columns[:80])
        for col in summary_cols:
            dtype = str(df[col].dtype)
            null_count = df[col].isna().sum()
            summary_lines.append(f"  • {col}: {dtype} ({null_count} nulls)")
        if len(df.columns) > len(summary_cols):
            summary_lines.append(f"  • ... and {len(df.columns) - len(summary_cols)} more columns")
        
        summary_lines.extend([
            f"",
            f"📈 Numeric Summary:",
        ])
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 40:
                numeric_df = numeric_df.iloc[:, :40]
            desc = numeric_df.describe() if not numeric_df.empty else None
            if desc is not None:
                summary_lines.append(desc.to_string())
            else:
                summary_lines.append("  (No numeric columns)")
        except:
            summary_lines.append("  (No numeric columns)")
        
        self._summary_text.setText("\n".join(summary_lines))
    
    def _display_list(self, data: list) -> None:
        """Display a list."""
        # Table view
        self._table.setRowCount(min(500, len(data)))
        self._table.setColumnCount(1)
        self._table.setHorizontalHeaderLabels(["Value"])
        
        for i, item in enumerate(data[:500]):
            self._table.setItem(i, 0, QTableWidgetItem(str(item)))
        
        # Raw view
        self._raw_text.setText(str(data[:100]))
        
        # Summary
        summary = f"List with {len(data)} items\n"
        if data:
            summary += f"First item type: {type(data[0]).__name__}\n"
            summary += f"First 5 items: {data[:5]}"
        self._summary_text.setText(summary)
    
    def _display_dict(self, data: dict) -> None:
        """Display a dictionary."""
        # Table view
        items = list(data.items())
        self._table.setRowCount(min(500, len(items)))
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Key", "Value"])
        
        for i, (key, value) in enumerate(items[:500]):
            self._table.setItem(i, 0, QTableWidgetItem(str(key)))
            self._table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        # Raw view
        try:
            self._raw_text.setText(json.dumps(data, indent=2, default=str))
        except:
            self._raw_text.setText(str(data))
        
        # Summary
        summary = f"Dictionary with {len(data)} keys\n\nKeys: {list(data.keys())[:20]}"
        self._summary_text.setText(summary)
    
    def _display_other(self, data: object) -> None:
        """Display other types of data."""
        self._table.setRowCount(1)
        self._table.setColumnCount(1)
        self._table.setHorizontalHeaderLabels(["Value"])
        self._table.setItem(0, 0, QTableWidgetItem(str(data)))
        
        self._raw_text.setText(str(data))
        self._summary_text.setText(f"Type: {type(data).__name__}\nValue: {str(data)[:1000]}")
    
    def _clear_display(self) -> None:
        """Clear all display elements."""
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        self._raw_text.clear()
        self._summary_text.clear()
        self._shape_label.setText("Shape: —")
        self._type_label.setText("Type: —")
        self._memory_label.setText("Memory: —")
    
    def _export_output(self) -> None:
        """Export the current output."""
        port_name = self._port_combo.currentText()
        if port_name and port_name in self._outputs:
            data = self._outputs[port_name]
            self.output_exported.emit(port_name, data)
            self._status_label.setText(f"✓ Output '{port_name}' exported")
    
    @staticmethod
    def _format_bytes(bytes_val: int) -> str:
        """Format bytes to human readable string."""
        if bytes_val < 1024:
            return f"{bytes_val} B"
        elif bytes_val < 1024 * 1024:
            return f"{bytes_val / 1024:.1f} KB"
        elif bytes_val < 1024 * 1024 * 1024:
            return f"{bytes_val / (1024*1024):.1f} MB"
        else:
            return f"{bytes_val / (1024*1024*1024):.2f} GB"
