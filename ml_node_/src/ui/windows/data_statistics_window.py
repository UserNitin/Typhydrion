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
    QProgressBar,
    QGroupBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
import pandas as pd
import numpy as np


class DataStatisticsWindow(QWidget):
    """
    Window for displaying statistical data and column metadata.
    
    Features:
    - Dataset overview (rows, columns, memory usage)
    - Column statistics (type, missing, unique, min/max/mean)
    - Data quality indicators
    - Distribution summaries
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._dataframe: pd.DataFrame | None = None
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Scroll area for the entire content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                background-color: rgba(30, 40, 55, 150);
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(100, 120, 150, 180);
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: rgba(120, 140, 170, 200);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar:horizontal {
                background-color: rgba(30, 40, 55, 150);
                height: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal {
                background-color: rgba(100, 120, 150, 180);
                border-radius: 4px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: rgba(120, 140, 170, 200);
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0;
            }
        """)
        
        # Content container
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("📊 Data Statistics")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: rgba(220, 230, 240, 230);")
        layout.addWidget(title)
        
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
                padding: 8px 16px;
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
        self._create_overview_tab()
        self._create_columns_tab()
        self._create_quality_tab()
        self._create_distribution_tab()
        
        # Info label
        self._info_label = QLabel("Load a dataset to see statistics")
        self._info_label.setStyleSheet("color: rgba(150, 170, 200, 150); font-size: 11px;")
        layout.addWidget(self._info_label)
        
        # Set scroll content
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
    
    def _create_overview_tab(self) -> None:
        """Create the dataset overview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)
        
        # Summary cards
        cards_layout = QHBoxLayout()
        
        self._rows_card = self._create_stat_card("Rows", "0", "📋")
        self._cols_card = self._create_stat_card("Columns", "0", "📊")
        self._memory_card = self._create_stat_card("Memory", "0 KB", "💾")
        self._missing_card = self._create_stat_card("Missing", "0%", "⚠️")
        
        cards_layout.addWidget(self._rows_card)
        cards_layout.addWidget(self._cols_card)
        cards_layout.addWidget(self._memory_card)
        cards_layout.addWidget(self._missing_card)
        layout.addLayout(cards_layout)
        
        # Data types breakdown
        types_group = QGroupBox("Data Types")
        types_group.setStyleSheet("""
            QGroupBox {
                color: rgba(200, 220, 255, 200);
                border: 1px solid rgba(80, 100, 130, 150);
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        types_layout = QVBoxLayout(types_group)
        
        self._numeric_bar = self._create_type_bar("Numeric", 0, "rgba(70, 130, 200, 200)")
        self._categorical_bar = self._create_type_bar("Categorical", 0, "rgba(70, 180, 120, 200)")
        self._datetime_bar = self._create_type_bar("DateTime", 0, "rgba(150, 100, 200, 200)")
        self._other_bar = self._create_type_bar("Other", 0, "rgba(150, 150, 150, 200)")
        
        types_layout.addWidget(self._numeric_bar)
        types_layout.addWidget(self._categorical_bar)
        types_layout.addWidget(self._datetime_bar)
        types_layout.addWidget(self._other_bar)
        layout.addWidget(types_group)
        
        # Dataset info
        info_group = QGroupBox("Dataset Information")
        info_group.setStyleSheet(types_group.styleSheet())
        info_layout = QVBoxLayout(info_group)
        
        self._dataset_info = QLabel("No dataset loaded")
        self._dataset_info.setStyleSheet("color: rgba(180, 200, 220, 180); font-size: 12px;")
        self._dataset_info.setWordWrap(True)
        info_layout.addWidget(self._dataset_info)
        layout.addWidget(info_group)
        
        layout.addStretch()
        self._tabs.addTab(widget, "📋 Overview")
    
    def _create_columns_tab(self) -> None:
        """Create the column statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Column statistics table
        self._columns_table = QTableWidget()
        self._columns_table.setColumnCount(8)
        self._columns_table.setHorizontalHeaderLabels([
            "Column", "Type", "Non-Null", "Null %", "Unique", "Min", "Max", "Mean/Mode"
        ])
        self._columns_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._columns_table.setAlternatingRowColors(True)
        self._columns_table.setStyleSheet("""
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
        self._columns_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._columns_table.verticalHeader().setVisible(False)
        layout.addWidget(self._columns_table)
        
        self._tabs.addTab(widget, "📊 Columns")
    
    def _create_quality_tab(self) -> None:
        """Create the data quality tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Quality score
        score_layout = QHBoxLayout()
        score_label = QLabel("Data Quality Score:")
        score_label.setStyleSheet("color: rgba(200, 220, 255, 200); font-size: 14px;")
        self._quality_score = QLabel("—")
        self._quality_score.setStyleSheet("color: rgba(100, 200, 100, 230); font-size: 24px; font-weight: bold;")
        score_layout.addWidget(score_label)
        score_layout.addWidget(self._quality_score)
        score_layout.addStretch()
        layout.addLayout(score_layout)
        
        # Quality metrics table
        self._quality_table = QTableWidget()
        self._quality_table.setColumnCount(4)
        self._quality_table.setHorizontalHeaderLabels([
            "Column", "Issue", "Severity", "Recommendation"
        ])
        self._quality_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._quality_table.setAlternatingRowColors(True)
        self._quality_table.setStyleSheet(self._columns_table.styleSheet())
        self._quality_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._quality_table.verticalHeader().setVisible(False)
        layout.addWidget(self._quality_table)
        
        self._tabs.addTab(widget, "✓ Quality")
    
    def _create_distribution_tab(self) -> None:
        """Create the distribution summary tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Distribution table
        self._dist_table = QTableWidget()
        self._dist_table.setColumnCount(7)
        self._dist_table.setHorizontalHeaderLabels([
            "Column", "25%", "50%", "75%", "Std Dev", "Skewness", "Kurtosis"
        ])
        self._dist_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._dist_table.setAlternatingRowColors(True)
        self._dist_table.setStyleSheet(self._columns_table.styleSheet())
        self._dist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._dist_table.verticalHeader().setVisible(False)
        layout.addWidget(self._dist_table)
        
        self._tabs.addTab(widget, "📈 Distribution")
    
    def _create_stat_card(self, title: str, value: str, icon: str) -> QFrame:
        """Create a statistics card widget."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: rgba(40, 52, 68, 180);
                border: 1px solid rgba(80, 100, 130, 150);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setSpacing(4)
        
        header = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 18px;")
        title_label = QLabel(title)
        title_label.setStyleSheet("color: rgba(150, 170, 200, 180); font-size: 11px;")
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch()
        layout.addLayout(header)
        
        value_label = QLabel(value)
        value_label.setObjectName("value")
        value_label.setStyleSheet("color: rgba(240, 248, 255, 240); font-size: 20px; font-weight: bold;")
        layout.addWidget(value_label)
        
        return card
    
    def _create_type_bar(self, label: str, count: int, color: str) -> QWidget:
        """Create a type count bar."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 2, 0, 2)
        
        name = QLabel(label)
        name.setStyleSheet("color: rgba(200, 220, 240, 200); min-width: 80px;")
        name.setFixedWidth(80)
        layout.addWidget(name)
        
        bar = QProgressBar()
        bar.setObjectName("bar")
        bar.setMaximum(100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setFormat(f"{count}")
        bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: rgba(30, 40, 55, 180);
                border: 1px solid rgba(80, 100, 130, 150);
                border-radius: 4px;
                height: 20px;
                text-align: center;
                color: rgba(240, 248, 255, 220);
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(bar, 1)
        
        count_label = QLabel(str(count))
        count_label.setObjectName("count")
        count_label.setStyleSheet("color: rgba(200, 220, 240, 200); min-width: 30px;")
        count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(count_label)
        
        return widget
    
    def _update_type_bar(self, bar_widget: QWidget, count: int, total: int) -> None:
        """Update a type bar with new values."""
        bar = bar_widget.findChild(QProgressBar, "bar")
        count_label = bar_widget.findChild(QLabel, "count")
        if bar and count_label:
            pct = int((count / total * 100) if total > 0 else 0)
            bar.setValue(pct)
            bar.setFormat(f"{pct}%")
            count_label.setText(str(count))
    
    def set_dataframe(self, df: object) -> None:
        """Set dataset payload and update statistics.

        Accepts either a DataFrame-like object or a mapping of split datasets.
        """
        selected_name, normalized = self._normalize_dataframe_payload(df)
        self._dataframe = normalized

        if normalized is None or normalized.empty:
            self._clear_statistics()
            self._info_label.setText("No dataset loaded")
            return

        stats_df, sampled = self._prepare_stats_dataframe(normalized)
        self._update_overview(stats_df)
        self._update_columns_table(stats_df)
        self._update_quality_table(stats_df)
        self._update_distribution_table(stats_df)

        prefix = f"[{selected_name}] " if selected_name else ""
        suffix = " (sampled for responsiveness)" if sampled else ""
        self._info_label.setText(
            f"{prefix}Statistics for {len(normalized)} rows × {len(normalized.columns)} columns{suffix}"
        )

    def _prepare_stats_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Use a bounded sample for expensive stats so UI remains responsive."""
        max_rows = 20000
        if len(df) <= max_rows:
            return df, False
        try:
            return df.sample(n=max_rows, random_state=42), True
        except Exception:
            return df.head(max_rows), True

    def _normalize_dataframe_payload(self, payload: object) -> tuple[str | None, pd.DataFrame | None]:
        """Choose the most relevant dataframe from a payload."""
        if payload is None:
            return None, None
        if isinstance(payload, pd.DataFrame):
            return None, payload
        if isinstance(payload, pd.Series):
            return None, payload.to_frame()
        if isinstance(payload, np.ndarray):
            if payload.ndim == 1:
                return None, pd.DataFrame({"value": payload})
            if payload.ndim == 2:
                return None, pd.DataFrame(payload)
            return None, None

        if isinstance(payload, dict):
            priority = ["X_train", "X_test", "X_val", "Data", "Features"]
            for key in priority:
                if key in payload:
                    frame = self._coerce_to_dataframe(payload.get(key))
                    if frame is not None:
                        return key, frame
            for key, value in payload.items():
                frame = self._coerce_to_dataframe(value)
                if frame is not None:
                    return str(key), frame
            return None, None

        return None, self._coerce_to_dataframe(payload)

    def _coerce_to_dataframe(self, value: object) -> pd.DataFrame | None:
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
    
    def _clear_statistics(self) -> None:
        """Clear all statistics displays."""
        # Clear overview cards
        for card in [self._rows_card, self._cols_card, self._memory_card, self._missing_card]:
            label = card.findChild(QLabel, "value")
            if label:
                label.setText("0")
        
        # Clear tables
        self._columns_table.setRowCount(0)
        self._quality_table.setRowCount(0)
        self._dist_table.setRowCount(0)
        
        self._quality_score.setText("—")
        self._dataset_info.setText("No dataset loaded")
    
    def _update_overview(self, df: pd.DataFrame) -> None:
        """Update the overview tab with dataframe statistics."""
        # Update cards
        rows_label = self._rows_card.findChild(QLabel, "value")
        if rows_label:
            rows_label.setText(f"{len(df):,}")
        
        cols_label = self._cols_card.findChild(QLabel, "value")
        if cols_label:
            cols_label.setText(str(len(df.columns)))
        
        memory_label = self._memory_card.findChild(QLabel, "value")
        if memory_label:
            mem_bytes = df.memory_usage(deep=True).sum()
            if mem_bytes < 1024:
                mem_str = f"{mem_bytes} B"
            elif mem_bytes < 1024 * 1024:
                mem_str = f"{mem_bytes / 1024:.1f} KB"
            elif mem_bytes < 1024 * 1024 * 1024:
                mem_str = f"{mem_bytes / (1024*1024):.1f} MB"
            else:
                mem_str = f"{mem_bytes / (1024*1024*1024):.2f} GB"
            memory_label.setText(mem_str)
        
        missing_label = self._missing_card.findChild(QLabel, "value")
        if missing_label:
            total_cells = df.size
            missing_cells = df.isna().sum().sum()
            missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
            missing_label.setText(f"{missing_pct:.1f}%")
            # Color based on severity
            if missing_pct > 20:
                missing_label.setStyleSheet("color: rgba(220, 80, 80, 230); font-size: 20px; font-weight: bold;")
            elif missing_pct > 5:
                missing_label.setStyleSheet("color: rgba(230, 180, 50, 230); font-size: 20px; font-weight: bold;")
            else:
                missing_label.setStyleSheet("color: rgba(100, 200, 100, 230); font-size: 20px; font-weight: bold;")
        
        # Update type bars
        total_cols = len(df.columns)
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = len(df.select_dtypes(include=['datetime64']).columns)
        other_cols = total_cols - numeric_cols - categorical_cols - datetime_cols
        
        self._update_type_bar(self._numeric_bar, numeric_cols, total_cols)
        self._update_type_bar(self._categorical_bar, categorical_cols, total_cols)
        self._update_type_bar(self._datetime_bar, datetime_cols, total_cols)
        self._update_type_bar(self._other_bar, other_cols, total_cols)
        
        # Update dataset info
        dtypes_summary = df.dtypes.value_counts().to_dict()
        dtypes_str = ", ".join([f"{v} {k}" for k, v in dtypes_summary.items()])
        self._dataset_info.setText(
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
            f"Data types: {dtypes_str}\n"
            f"Index: {df.index.name or 'RangeIndex'}"
        )
    
    def _update_columns_table(self, df: pd.DataFrame) -> None:
        """Update the columns statistics table."""
        max_cols = 300
        cols = list(df.columns[:max_cols])
        self._columns_table.setRowCount(len(cols))
        
        for i, col in enumerate(cols):
            series = df[col]
            dtype = str(series.dtype)
            non_null = series.count()
            null_pct = (series.isna().sum() / len(series) * 100) if len(series) > 0 else 0
            unique = series.nunique()
            
            # Determine type-specific stats
            if np.issubdtype(series.dtype, np.number):
                try:
                    min_val = f"{series.min():.4g}" if pd.notna(series.min()) else "—"
                    max_val = f"{series.max():.4g}" if pd.notna(series.max()) else "—"
                    mean_val = f"{series.mean():.4g}" if pd.notna(series.mean()) else "—"
                except:
                    min_val = max_val = mean_val = "—"
            else:
                min_val = "—"
                max_val = "—"
                # Mode for categorical
                try:
                    mode = series.mode()
                    mean_val = str(mode.iloc[0])[:20] if len(mode) > 0 else "—"
                except:
                    mean_val = "—"
            
            # Set table items
            items = [
                col,
                dtype,
                f"{non_null:,}",
                f"{null_pct:.1f}%",
                f"{unique:,}",
                min_val,
                max_val,
                mean_val,
            ]
            
            for j, text in enumerate(items):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignCenter)
                
                # Color null percentage
                if j == 3:  # Null %
                    if null_pct > 20:
                        item.setForeground(QColor(220, 80, 80))
                    elif null_pct > 5:
                        item.setForeground(QColor(230, 180, 50))
                
                self._columns_table.setItem(i, j, item)
    
    def _update_quality_table(self, df: pd.DataFrame) -> None:
        """Update the data quality table with issues."""
        issues = []

        max_cols = 300
        for col in list(df.columns[:max_cols]):
            series = df[col]
            null_pct = (series.isna().sum() / len(series) * 100) if len(series) > 0 else 0
            
            # Check for high missing values
            if null_pct > 50:
                issues.append((col, "High missing values", "🔴 High", "Consider dropping column"))
            elif null_pct > 20:
                issues.append((col, "Missing values", "🟡 Medium", "Impute or investigate"))
            
            # Check for constant columns
            if series.nunique() == 1:
                issues.append((col, "Constant column", "🟡 Medium", "Consider removing"))
            
            # Check for high cardinality
            if series.dtype == 'object' and series.nunique() > len(series) * 0.9:
                issues.append((col, "High cardinality", "🟡 Medium", "May not be useful for ML"))
            
            # Check for potential ID columns
            if series.nunique() == len(series) and 'id' in col.lower():
                issues.append((col, "Potential ID column", "🔵 Info", "Consider excluding from features"))
        
        self._quality_table.setRowCount(len(issues))
        
        for i, (col, issue, severity, rec) in enumerate(issues):
            for j, text in enumerate([col, issue, severity, rec]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self._quality_table.setItem(i, j, item)
        
        # Calculate quality score
        total_cols = len(df.columns)
        high_issues = sum(1 for _, _, sev, _ in issues if "High" in sev)
        med_issues = sum(1 for _, _, sev, _ in issues if "Medium" in sev)
        
        score = max(0, 100 - (high_issues * 15) - (med_issues * 5))
        self._quality_score.setText(f"{score}%")
        
        if score >= 80:
            self._quality_score.setStyleSheet("color: rgba(100, 200, 100, 230); font-size: 24px; font-weight: bold;")
        elif score >= 60:
            self._quality_score.setStyleSheet("color: rgba(230, 180, 50, 230); font-size: 24px; font-weight: bold;")
        else:
            self._quality_score.setStyleSheet("color: rgba(220, 80, 80, 230); font-size: 24px; font-weight: bold;")
    
    def _update_distribution_table(self, df: pd.DataFrame) -> None:
        """Update the distribution statistics table."""
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns[:120])
        self._dist_table.setRowCount(len(numeric_cols))
        
        for i, col in enumerate(numeric_cols):
            series = df[col].dropna()
            
            if len(series) == 0:
                items = [col, "—", "—", "—", "—", "—", "—"]
            else:
                try:
                    q25 = f"{series.quantile(0.25):.4g}"
                    q50 = f"{series.quantile(0.50):.4g}"
                    q75 = f"{series.quantile(0.75):.4g}"
                    std = f"{series.std():.4g}"
                    skew = f"{series.skew():.4g}"
                    kurt = f"{series.kurtosis():.4g}"
                except:
                    q25 = q50 = q75 = std = skew = kurt = "—"
                
                items = [col, q25, q50, q75, std, skew, kurt]
            
            for j, text in enumerate(items):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignCenter)
                self._dist_table.setItem(i, j, item)
