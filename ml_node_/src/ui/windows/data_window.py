from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QFrame,
)
from PySide6.QtCore import Qt, Signal
import pandas as pd
import numpy as np


class DataPreviewWindow(QWidget):
    dtype_change_requested = Signal(str, str)  # (column_name, target_dtype_key)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        title = QLabel("Data Preview")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        self._subtitle = QLabel("Load a dataset to see preview here.")
        self._subtitle.setStyleSheet("color: rgba(255, 255, 255, 170);")
        layout.addWidget(self._subtitle)

        self._info = QLabel("")
        self._info.setStyleSheet("color: rgba(180, 220, 255, 200); font-size: 12px;")
        layout.addWidget(self._info)

        dataset_bar = QHBoxLayout()
        dataset_bar.setContentsMargins(0, 0, 0, 0)
        dataset_bar.setSpacing(8)

        dataset_lbl = QLabel("Dataset")
        dataset_lbl.setStyleSheet("color: rgba(255, 255, 255, 200); font-size: 11px;")
        dataset_bar.addWidget(dataset_lbl)

        self._dataset_combo = QComboBox()
        self._dataset_combo.setMinimumWidth(180)
        self._dataset_combo.setStyleSheet(
            """
            QComboBox { background-color: rgba(20, 28, 38, 160); border: 1px solid rgba(255,255,255,25); border-radius: 6px; padding: 4px 8px; color: rgba(255,255,255,220); }
            QComboBox:hover { border-color: rgba(120, 190, 255, 140); }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox QAbstractItemView { background-color: rgba(22, 30, 40, 240); color: rgba(255,255,255,220); selection-background-color: rgba(70, 130, 180, 160); }
            """
        )
        self._dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        dataset_bar.addWidget(self._dataset_combo)
        dataset_bar.addStretch(1)
        layout.addLayout(dataset_bar)

        # Quick datatype changer
        controls = QFrame()
        controls.setStyleSheet("background: transparent;")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        col_lbl = QLabel("Column")
        col_lbl.setStyleSheet("color: rgba(255, 255, 255, 200); font-size: 11px;")
        controls_layout.addWidget(col_lbl)

        self._col_combo = QComboBox()
        self._col_combo.setMinimumWidth(160)
        self._col_combo.setStyleSheet(
            """
            QComboBox { background-color: rgba(20, 28, 38, 160); border: 1px solid rgba(255,255,255,25); border-radius: 6px; padding: 4px 8px; color: rgba(255,255,255,220); }
            QComboBox:hover { border-color: rgba(120, 190, 255, 140); }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox QAbstractItemView { background-color: rgba(22, 30, 40, 240); color: rgba(255,255,255,220); selection-background-color: rgba(70, 130, 180, 160); }
            """
        )
        self._col_combo.currentTextChanged.connect(lambda _t: self._update_dtype_hint())
        controls_layout.addWidget(self._col_combo)

        dtype_lbl = QLabel("Type")
        dtype_lbl.setStyleSheet("color: rgba(255, 255, 255, 200); font-size: 11px;")
        controls_layout.addWidget(dtype_lbl)

        self._dtype_combo = QComboBox()
        self._dtype_combo.setMinimumWidth(130)
        self._dtype_combo.setStyleSheet(self._col_combo.styleSheet())
        # (label, key) pairs
        self._dtype_combo.addItem("int (Int64)", "int")
        self._dtype_combo.addItem("float", "float")
        self._dtype_combo.addItem("string", "str")
        self._dtype_combo.addItem("datetime", "datetime")
        self._dtype_combo.addItem("category", "category")
        self._dtype_combo.addItem("boolean", "bool")
        controls_layout.addWidget(self._dtype_combo)

        self._apply_dtype_btn = QPushButton("Apply")
        self._apply_dtype_btn.setStyleSheet(
            """
            QPushButton { background-color: rgba(70, 130, 180, 180); border: 1px solid rgba(255,255,255,30); border-radius: 6px; padding: 5px 12px; color: rgba(255,255,255,230); font-size: 11px; font-weight: 600; }
            QPushButton:hover { background-color: rgba(90, 150, 210, 220); }
            QPushButton:pressed { background-color: rgba(50, 110, 170, 220); }
            """
        )
        self._apply_dtype_btn.clicked.connect(self._on_apply_dtype)
        controls_layout.addWidget(self._apply_dtype_btn)

        controls_layout.addStretch(1)
        self._dtype_hint = QLabel("")
        self._dtype_hint.setStyleSheet("color: rgba(255, 255, 255, 140); font-size: 10px;")
        controls_layout.addWidget(self._dtype_hint)

        layout.addWidget(controls)

        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            """
            QTableWidget {
                background-color: rgba(20, 28, 38, 180);
                alternate-background-color: rgba(30, 40, 52, 180);
                gridline-color: rgba(255, 255, 255, 30);
                color: rgba(255, 255, 255, 220);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: rgba(40, 55, 75, 200);
                color: rgba(255, 255, 255, 230);
                padding: 4px;
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 40);
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: rgba(70, 130, 180, 150);
            }
            """
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, 1)

        self._df: pd.DataFrame | None = None
        self._datasets: dict[str, pd.DataFrame] = {}
        self._active_dataset_name: str | None = None
        self._selected_columns: list[str] = []

    def set_columns(self, columns: list[str]) -> None:
        """Called when link columns are selected."""
        self._selected_columns = columns
        self._refresh_table()

    def set_dataframe(self, df: object) -> None:
        """Called when dataset data is loaded.

        Supports:
        - A single DataFrame/Series/array-like payload.
        - A dict payload (e.g. split outputs) to preview each dataset separately.
        """
        previous_selected = list(self._selected_columns)
        self._datasets = self._normalize_datasets(df)

        self._dataset_combo.blockSignals(True)
        self._dataset_combo.clear()
        if self._datasets:
            self._dataset_combo.addItems(list(self._datasets.keys()))
            self._active_dataset_name = next(iter(self._datasets.keys()))
            self._dataset_combo.setCurrentText(self._active_dataset_name)
        else:
            self._active_dataset_name = None
        self._dataset_combo.blockSignals(False)

        self._update_active_dataframe()
        self._selected_columns = self._sanitize_selected_columns(previous_selected)
        self._update_column_combo()
        self._refresh_table()

    def _sanitize_selected_columns(self, columns: list[str]) -> list[str]:
        """Keep selected columns only when they still match the active dataset."""
        if not columns:
            return []
        if "*" in columns:
            return ["*"]
        if self._df is None or self._df.empty:
            return []
        return [c for c in columns if c in self._df.columns]

    def _normalize_datasets(self, payload: object) -> dict[str, pd.DataFrame]:
        """Convert incoming payload into a displayable dataset mapping."""
        if payload is None:
            return {}

        if isinstance(payload, dict):
            result: dict[str, pd.DataFrame] = {}
            for key, value in payload.items():
                frame = self._coerce_to_dataframe(value)
                if frame is not None:
                    result[str(key)] = frame
            return result

        single = self._coerce_to_dataframe(payload)
        if single is None:
            return {}
        return {"Data": single}

    def _coerce_to_dataframe(self, value: object) -> pd.DataFrame | None:
        """Best-effort conversion for previewable payload types."""
        if value is None:
            return None
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, pd.Series):
            return value.to_frame()
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                return pd.DataFrame({"value": value})
            if value.ndim == 2:
                return pd.DataFrame(value)
            return None
        if isinstance(value, (list, tuple)):
            try:
                return pd.DataFrame(value)
            except Exception:
                return None
        return None

    def _on_dataset_changed(self, name: str) -> None:
        self._active_dataset_name = name or None
        self._update_active_dataframe()
        self._selected_columns = self._sanitize_selected_columns(self._selected_columns)
        self._update_column_combo()
        self._refresh_table()

    def _update_active_dataframe(self) -> None:
        if self._active_dataset_name and self._active_dataset_name in self._datasets:
            self._df = self._datasets[self._active_dataset_name]
            return
        self._df = next(iter(self._datasets.values()), None)

    def _update_column_combo(self) -> None:
        self._col_combo.blockSignals(True)
        self._col_combo.clear()
        if self._df is not None and not self._df.empty:
            self._col_combo.addItems([str(c) for c in self._df.columns])
        self._col_combo.blockSignals(False)
        self._update_dtype_hint()

    def _update_dtype_hint(self) -> None:
        if self._df is None or self._df.empty:
            self._dtype_hint.setText("")
            return
        col = self._col_combo.currentText()
        if not col or col not in self._df.columns:
            self._dtype_hint.setText("")
            return
        try:
            self._dtype_hint.setText(f"Current: {self._df[col].dtype}")
        except Exception:
            self._dtype_hint.setText("")

    def _on_apply_dtype(self) -> None:
        if self._df is None or self._df.empty:
            return
        col = self._col_combo.currentText()
        dtype_key = self._dtype_combo.currentData()
        if not col or col not in self._df.columns or not dtype_key:
            return
        self.dtype_change_requested.emit(col, str(dtype_key))

    def _refresh_table(self) -> None:
        if self._df is None or self._df.empty:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._subtitle.setText("No data loaded.")
            self._info.setText("")
            self._dtype_hint.setText("")
            return

        df = self._df

        # Filter columns if specific ones are selected
        if self._selected_columns:
            cols = [c for c in self._selected_columns if c in df.columns]
            if cols:
                df = df[cols]

        # Limit rows/columns for display (to keep UI responsive with large datasets)
        max_display_rows = 500
        max_display_cols = 80
        total_rows = len(df)
        preview_rows = min(max_display_rows, total_rows)
        display_df = df.head(preview_rows)

        if len(display_df.columns) > max_display_cols:
            display_df = display_df.iloc[:, :max_display_cols]
            cols_note = f" and first {max_display_cols}/{len(df.columns)} columns"
        else:
            cols_note = ""

        if total_rows > max_display_rows:
            base = f"Showing {preview_rows} of {len(self._df)} rows{cols_note} (full dataset loaded)"
        else:
            base = f"Showing all {total_rows} rows{cols_note}"

        if self._active_dataset_name:
            self._subtitle.setText(f"{self._active_dataset_name}: {base}")
        else:
            self._subtitle.setText(base)
        
        df = display_df
        self._info.setText(
            f"Columns: {len(df.columns)} | "
            f"Shape: {self._df.shape[0]} × {self._df.shape[1]}"
        )
        self._update_dtype_hint()

        self._table.setRowCount(len(df))
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for row_idx in range(len(df)):
            for col_idx, col_name in enumerate(df.columns):
                value = df.iloc[row_idx, col_idx]
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(row_idx, col_idx, item)

        # Resize columns to content (with a max width)
        for col_idx in range(len(df.columns)):
            self._table.resizeColumnToContents(col_idx)
            if self._table.columnWidth(col_idx) > 200:
                self._table.setColumnWidth(col_idx, 200)
