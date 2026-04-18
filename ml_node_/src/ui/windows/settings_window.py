from __future__ import annotations

import platform
import sys
from datetime import datetime

from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QFormLayout,
    QGroupBox,
    QCheckBox,
    QSpinBox,
    QLineEdit,
    QPushButton,
    QComboBox,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QTextEdit,
    QGraphicsOpacityEffect,
    QScrollArea,
)

from ui.app_settings import AppSettings


class SettingsWindow(QDialog):
    settings_applied = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(760, 620)

        self._s = AppSettings()
        self._loading_values = False

        # Dialog-level styling
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(
                    x1:0, y1:0, x2:0.4, y2:1,
                    stop:0 rgba(18, 26, 40, 245),
                    stop:1 rgba(12, 18, 30, 250)
                );
            }
            QLabel {
                color: rgba(180, 215, 255, 220);
            }
            QCheckBox {
                color: rgba(180, 215, 255, 210);
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid rgba(60, 120, 200, 80);
                border-radius: 3px;
                background: rgba(20, 30, 48, 200);
            }
            QCheckBox::indicator:checked {
                background: rgba(50, 120, 220, 200);
                border-color: rgba(80, 160, 255, 150);
            }
            QSpinBox, QLineEdit, QComboBox, QTextEdit {
                background: rgba(22, 32, 50, 220);
                color: rgba(200, 225, 255, 230);
                border: 1px solid rgba(55, 110, 200, 50);
                border-radius: 5px;
                padding: 4px 8px;
            }
            QSpinBox:focus, QLineEdit:focus, QComboBox:focus {
                border-color: rgba(70, 150, 255, 120);
            }
            QPushButton {
                background: rgba(28, 42, 64, 220);
                color: rgba(180, 215, 255, 230);
                border: 1px solid rgba(60, 120, 200, 50);
                border-radius: 6px;
                padding: 6px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(40, 80, 140, 220);
                border-color: rgba(70, 150, 255, 100);
            }
            QGroupBox {
                color: rgba(150, 195, 250, 210);
                border: 1px solid rgba(55, 110, 200, 40);
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(16, 16, 16, 16)
        self._root_layout.setSpacing(12)

        title = QLabel("⚙  Settings")
        title.setStyleSheet(
            "font-size: 18px; font-weight: 700; "
            "color: rgba(140, 200, 255, 240); background: transparent;"
        )
        self._root_layout.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid rgba(55, 110, 200, 40);
                border-radius: 8px;
                background: rgba(16, 24, 38, 200);
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
                background: rgba(24, 36, 56, 200);
                color: rgba(150, 195, 240, 210);
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: rgba(35, 60, 100, 220);
                color: rgba(180, 220, 255, 245);
                border-bottom: 2px solid rgba(60, 140, 255, 180);
            }
            QTabBar::tab:hover:!selected {
                background: rgba(30, 50, 80, 220);
            }
            """
        )
        self._root_layout.addWidget(self._tabs, 1)

        # Build all tabs once in __init__
        self._build_ui_tab()
        self._build_project_tab()
        self._build_performance_tab()
        self._build_data_tab()
        self._build_pipeline_tab()
        self._build_graph_tab()
        self._build_logging_tab()
        self._build_ai_tab()
        self._build_export_tab()
        self._build_shortcuts_tab()
        self._build_advanced_tab()
        self._build_about_tab()

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel
        )
        buttons.setStyleSheet("""
            QDialogButtonBox QPushButton {
                min-width: 80px;
            }
        """)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        self._root_layout.addWidget(buttons)

        # Live apply (debounced): settings become applicable while user edits.
        self._live_apply_timer = QTimer(self)
        self._live_apply_timer.setSingleShot(True)
        self._live_apply_timer.timeout.connect(self._on_apply)
        self._wire_live_apply()

        # Fade-in animation
        self._opacity_fx = QGraphicsOpacityEffect(self)
        self._opacity_fx.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_fx)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        # Fade-in
        self._show_anim = QPropertyAnimation(self._opacity_fx, b"opacity", self)
        self._show_anim.setDuration(300)
        self._show_anim.setStartValue(0.0)
        self._show_anim.setEndValue(1.0)
        self._show_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._show_anim.finished.connect(lambda: self.setGraphicsEffect(None))
        self._show_anim.start()

        # Load current values every time the dialog is shown
        self._loading_values = True
        try:
            self._load_from_settings()
        finally:
            self._loading_values = False

    # ─────────────────────────────────────────────────────────────────────
    # Tabs
    # ─────────────────────────────────────────────────────────────────────

    def _mk_tab(self, title: str) -> tuple[QWidget, QVBoxLayout]:
        """Create a scrollable tab page."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        scroll.setWidget(inner)

        self._tabs.addTab(scroll, title)
        return inner, layout

    def _mk_group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        g = QGroupBox(title)
        g.setStyleSheet(
            """
            QGroupBox { color: rgba(220,230,240,230); font-weight: 700; border: 1px solid rgba(255,255,255,25); border-radius: 8px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
            QLabel { color: rgba(255,255,255,210); }
            """
        )
        form = QFormLayout(g)
        form.setContentsMargins(12, 12, 12, 12)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        return g, form

    def _build_ui_tab(self) -> None:
        _, layout = self._mk_tab("UI / Workspace")

        g0, f0 = self._mk_group("General")
        self.ui_theme = QComboBox()
        self.ui_theme.addItems(["System", "Dark", "Light"])
        self.ui_accent = QLineEdit()
        self.ui_accent.setPlaceholderText("#118dff")
        self.ui_font_scale = QSpinBox()
        self.ui_font_scale.setRange(70, 160)
        self.ui_font_scale.setSuffix(" %")
        self.ui_language = QComboBox()
        self.ui_language.addItems(["System", "English"])
        self.ui_startup = QComboBox()
        self.ui_startup.addItems(["Home", "Last project"])
        f0.addRow("Theme", self.ui_theme)
        f0.addRow("Accent color", self.ui_accent)
        f0.addRow("Font scaling", self.ui_font_scale)
        f0.addRow("Language", self.ui_language)
        f0.addRow("Startup window", self.ui_startup)
        layout.addWidget(g0)

        g, form = self._mk_group("Appearance & Layout")
        self.ui_transparency = QCheckBox("Enable acrylic/blur transparency")
        self.ui_animations = QCheckBox("Enable animations")
        self.ui_node_grid = QCheckBox("Show grid in Node editor")
        self.ui_graph_grid = QCheckBox("Show grid in Graph window")
        self.ui_node_grid_size = QSpinBox()
        self.ui_node_grid_size.setRange(10, 200)
        self.ui_graph_grid_size = QSpinBox()
        self.ui_graph_grid_size.setRange(10, 200)
        self.ui_acrylic_hex = QLineEdit()
        self.ui_acrylic_hex.setPlaceholderText("e.g. 0x518E7400")

        form.addRow(self.ui_transparency)
        form.addRow(self.ui_animations)
        form.addRow(self.ui_node_grid)
        form.addRow("Node grid size", self.ui_node_grid_size)
        form.addRow(self.ui_graph_grid)
        form.addRow("Graph grid size", self.ui_graph_grid_size)
        form.addRow("Acrylic gradient (hex)", self.ui_acrylic_hex)
        layout.addWidget(g)
        layout.addStretch(1)

    def _build_project_tab(self) -> None:
        _, layout = self._mk_tab("Project / Files")

        g, form = self._mk_group("Project behavior")
        self.proj_open_last = QCheckBox("Open last project on startup")
        self.confirm_delete = QCheckBox("Confirm before delete")
        self.default_graph_open = QCheckBox("Open Graph dock by default")
        self.proj_default_folder = QLineEdit()
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse_folder_into(self.proj_default_folder))
        row = QHBoxLayout()
        row.addWidget(self.proj_default_folder, 1)
        row.addWidget(btn)
        row_w = QWidget()
        row_w.setLayout(row)
        form.addRow(self.proj_open_last)
        form.addRow(self.confirm_delete)
        form.addRow(self.default_graph_open)
        form.addRow("Default folder", row_w)

        self.autosave_enabled = QCheckBox("Enable autosave")
        self.autosave_interval = QSpinBox()
        self.autosave_interval.setRange(10, 3600)
        self.autosave_interval.setSuffix(" sec")
        self.autosave_versions = QSpinBox()
        self.autosave_versions.setRange(1, 200)
        form.addRow(self.autosave_enabled)
        form.addRow("Autosave interval", self.autosave_interval)
        form.addRow("Autosave versions", self.autosave_versions)

        g2, form2 = self._mk_group("Dataset embedding in .typhyproj")
        self.embed_mode = QComboBox()
        self.embed_mode.addItems(["full", "preview", "path", "none"])
        self.embed_max_mb = QSpinBox()
        self.embed_max_mb.setRange(1, 2000)
        self.embed_max_mb.setSuffix(" MB")
        self.embed_max_rows = QSpinBox()
        self.embed_max_rows.setRange(1000, 50_000_000)
        self.embed_preview_rows = QSpinBox()
        self.embed_preview_rows.setRange(100, 5_000_000)
        self.embed_compress = QSpinBox()
        self.embed_compress.setRange(1, 9)
        form2.addRow("Mode", self.embed_mode)
        form2.addRow("Max embed size", self.embed_max_mb)
        form2.addRow("Max embed rows", self.embed_max_rows)
        form2.addRow("Preview rows", self.embed_preview_rows)
        form2.addRow("Compression (1-9)", self.embed_compress)

        layout.addWidget(g)
        layout.addWidget(g2)
        layout.addStretch(1)

    def _build_performance_tab(self) -> None:
        _, layout = self._mk_tab("Performance")

        g, form = self._mk_group("Execution control")
        self.perf_threads = QSpinBox()
        self.perf_threads.setRange(0, 128)
        self.perf_threads.setToolTip("0 = auto")
        self.perf_gpu_accel = QCheckBox("Enable GPU acceleration (if available)")
        self.perf_gpu_mem = QSpinBox()
        self.perf_gpu_mem.setRange(0, 2_000_000)
        self.perf_gpu_mem.setSuffix(" MB")
        self.perf_priority = QComboBox()
        self.perf_priority.addItems(["Low", "Normal", "High"])
        form.addRow("Thread count", self.perf_threads)
        form.addRow(self.perf_gpu_accel)
        form.addRow("GPU memory limit", self.perf_gpu_mem)
        form.addRow("Background priority", self.perf_priority)

        g2, form2 = self._mk_group("Chunk processing")
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(1000, 50_000_000)
        self.chunk_prefetch = QSpinBox()
        self.chunk_prefetch.setRange(0, 50)
        self.chunk_async = QCheckBox("Async loading")
        form2.addRow("Default chunk size", self.chunk_size)
        form2.addRow("Prefetch chunks", self.chunk_prefetch)
        form2.addRow(self.chunk_async)

        layout.addWidget(g)
        layout.addWidget(g2)
        layout.addStretch(1)

    def _build_data_tab(self) -> None:
        _, layout = self._mk_tab("Data Handling")
        g, form = self._mk_group("CSV import defaults")
        self.csv_delim = QLineEdit()
        self.csv_delim.setPlaceholderText(",")
        self.csv_encoding = QLineEdit()
        self.csv_encoding.setPlaceholderText("utf-8")
        self.csv_na = QLineEdit()
        self.csv_na.setPlaceholderText("NA, N/A, null, ?")
        form.addRow("Delimiter", self.csv_delim)
        form.addRow("Encoding", self.csv_encoding)
        form.addRow("NA tokens", self.csv_na)

        g2, form2 = self._mk_group("Datatype conversion")
        self.dt_datetime_format = QLineEdit()
        self.dt_datetime_format.setPlaceholderText("optional: e.g. %Y-%m-%d")
        form2.addRow("Datetime format", self.dt_datetime_format)

        g3, form3 = self._mk_group("Dataset defaults")
        self.ds_format = QComboBox()
        self.ds_format.addItems(["CSV", "Parquet", "JSON"])
        self.ds_encoding = QComboBox()
        self.ds_encoding.addItems(["Auto", "utf-8", "latin-1"])
        self.ds_missing = QComboBox()
        self.ds_missing.addItems(["Keep as-is", "Drop rows", "Impute"])
        self.ds_auto_target = QCheckBox("Auto detect target column")
        self.ds_schema_validate = QCheckBox("Schema validation on load")
        self.ds_duplicates = QComboBox()
        self.ds_duplicates.addItems(["Keep duplicates", "Drop duplicates"])
        self.ds_dtype_auto = QCheckBox("Auto data type conversion")
        form3.addRow("Default format", self.ds_format)
        form3.addRow("Encoding", self.ds_encoding)
        form3.addRow("Missing values", self.ds_missing)
        form3.addRow(self.ds_auto_target)
        form3.addRow(self.ds_schema_validate)
        form3.addRow("Duplicate rows", self.ds_duplicates)
        form3.addRow(self.ds_dtype_auto)

        layout.addWidget(g)
        layout.addWidget(g2)
        layout.addWidget(g3)
        layout.addStretch(1)

    def _build_pipeline_tab(self) -> None:
        _, layout = self._mk_tab("Pipeline")
        g, form = self._mk_group("Execution")
        self.pipe_auto = QCheckBox("Auto-run nodes on change")
        self.pipe_prop = QCheckBox("Propagate to downstream nodes")
        self.pipe_debounce = QSpinBox()
        self.pipe_debounce.setRange(0, 5000)
        self.pipe_debounce.setSuffix(" ms")
        self.pipe_stop_on_error = QCheckBox("Stop pipeline on error")
        form.addRow(self.pipe_auto)
        form.addRow(self.pipe_prop)
        form.addRow("Debounce", self.pipe_debounce)
        form.addRow(self.pipe_stop_on_error)
        layout.addWidget(g)

        g2, form2 = self._mk_group("Node defaults")
        self.node_w = QSpinBox(); self.node_w.setRange(140, 800)
        self.node_h = QSpinBox(); self.node_h.setRange(120, 600)
        self.node_snap = QCheckBox("Snap-to-grid")
        self.node_auto_align = QCheckBox("Auto-align nodes")
        self.node_exec_highlight = QCheckBox("Highlight executing nodes")
        self.node_exec_order = QCheckBox("Show execution order")
        form2.addRow("Default node width", self.node_w)
        form2.addRow("Default node height", self.node_h)
        form2.addRow(self.node_snap)
        form2.addRow(self.node_auto_align)
        form2.addRow(self.node_exec_highlight)
        form2.addRow(self.node_exec_order)

        g3, form3 = self._mk_group("Links")
        self.link_style = QComboBox(); self.link_style.addItems(["Curved", "Straight"])
        self.link_thickness = QSpinBox(); self.link_thickness.setRange(1, 10)
        self.link_animate = QCheckBox("Animate data flow")
        self.link_dtype_icons = QCheckBox("Show data type icons on links")
        form3.addRow("Style", self.link_style)
        form3.addRow("Thickness", self.link_thickness)
        form3.addRow(self.link_animate)
        form3.addRow(self.link_dtype_icons)

        layout.addWidget(g2)
        layout.addWidget(g3)
        layout.addStretch(1)

    def _build_graph_tab(self) -> None:
        _, layout = self._mk_tab("Graphs / Visualization")
        g, form = self._mk_group("Graph behavior")
        self.graph_default_category = QLineEdit()
        self.graph_default_category.setPlaceholderText("Exploration")
        self.graph_default_name = QLineEdit()
        self.graph_default_name.setPlaceholderText("Column Distribution")
        self.graph_pause_inactive = QCheckBox("Pause graphs when window inactive")
        self.graph_smooth = QCheckBox("Smooth plotting")
        self.graph_point_limit = QSpinBox()
        self.graph_point_limit.setRange(10, 100_000)
        form.addRow("Default category", self.graph_default_category)
        form.addRow("Default graph", self.graph_default_name)
        form.addRow(self.graph_pause_inactive)
        form.addRow(self.graph_smooth)
        form.addRow("Data point limit", self.graph_point_limit)

        g2, form2 = self._mk_group("Defaults & style")
        self.graph_labels = QCheckBox("Show labels by default")
        self.graph_live_interval = QSpinBox()
        self.graph_live_interval.setRange(200, 30_000)
        self.graph_live_interval.setSuffix(" ms")
        self.graph_3d_rotate = QCheckBox("Auto-rotate 3D by default")
        self.graph_3d_speed = QSpinBox()
        self.graph_3d_speed.setRange(0, 20)
        self.graph_line_thickness = QSpinBox()
        self.graph_line_thickness.setRange(1, 10)
        self.graph_axis_autoscale = QCheckBox("Axis auto-scale")
        self.graph_palette = QComboBox()
        self.graph_palette.addItems(["PowerBI", "Seaborn", "Matplotlib"])
        self.graph_legend = QComboBox()
        self.graph_legend.addItems(["Auto", "Show", "Hide"])
        self.card_w = QSpinBox()
        self.card_w.setRange(200, 1600)
        self.card_h = QSpinBox()
        self.card_h.setRange(160, 1200)
        self.dashboard_zoom = QSpinBox()
        self.dashboard_zoom.setRange(50, 200)
        self.dashboard_zoom.setSuffix(" %")
        form2.addRow(self.graph_labels)
        form2.addRow("Default live interval", self.graph_live_interval)
        form2.addRow(self.graph_3d_rotate)
        form2.addRow("Default 3D speed", self.graph_3d_speed)
        form2.addRow("Line thickness", self.graph_line_thickness)
        form2.addRow(self.graph_axis_autoscale)
        form2.addRow("Color palette", self.graph_palette)
        form2.addRow("Legend", self.graph_legend)
        form2.addRow("Default card width", self.card_w)
        form2.addRow("Default card height", self.card_h)
        form2.addRow("Default dashboard zoom", self.dashboard_zoom)

        layout.addWidget(g)
        layout.addWidget(g2)
        layout.addStretch(1)

    def _build_logging_tab(self) -> None:
        _, layout = self._mk_tab("Logging / Debug")
        g, form = self._mk_group("Logs")
        self.log_level = QComboBox()
        self.log_level.addItems(["Info", "Debug", "Error"])
        self.log_file = QLineEdit()
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse_file_into(self.log_file, "Log File", "Log (*.log);;All Files (*)"))
        row = QHBoxLayout(); row.addWidget(self.log_file, 1); row.addWidget(btn)
        row_w = QWidget(); row_w.setLayout(row)
        self.log_max = QSpinBox(); self.log_max.setRange(1, 5000); self.log_max.setSuffix(" MB")
        self.log_clear_days = QSpinBox(); self.log_clear_days.setRange(0, 365); self.log_clear_days.setSuffix(" days (0 = never)")
        form.addRow("Log level", self.log_level)
        form.addRow("Log file", row_w)
        form.addRow("Max log size", self.log_max)
        form.addRow("Auto-clear", self.log_clear_days)

        g2, form2 = self._mk_group("Debug")
        self.dbg_trace = QCheckBox("Show execution trace")
        self.dbg_node = QCheckBox("Node-level logs")
        self.dbg_graph = QCheckBox("Graph render logs")
        self.dbg_prof = QCheckBox("Performance profiling mode")
        form2.addRow(self.dbg_trace)
        form2.addRow(self.dbg_node)
        form2.addRow(self.dbg_graph)
        form2.addRow(self.dbg_prof)

        layout.addWidget(g)
        layout.addWidget(g2)
        layout.addStretch(1)

    def _build_ai_tab(self) -> None:
        _, layout = self._mk_tab("AI Assistant")
        g, form = self._mk_group("AI behavior")
        self.ai_enabled = QCheckBox("Enable AI suggestions")
        self.ai_mode = QComboBox(); self.ai_mode.addItems(["Suggest only", "Auto apply"])
        self.ai_suggest_chunk = QCheckBox("Suggest chunk size")
        self.ai_suggest_model = QCheckBox("Suggest model type")
        form.addRow(self.ai_enabled)
        form.addRow("Mode", self.ai_mode)
        form.addRow(self.ai_suggest_chunk)
        form.addRow(self.ai_suggest_model)

        g2, form2 = self._mk_group("Privacy")
        self.ai_privacy = QComboBox(); self.ai_privacy.addItems(["None", "Metadata", "Stats"])
        self.ai_local = QCheckBox("Local-only AI mode")
        form2.addRow("Data sent to AI", self.ai_privacy)
        form2.addRow(self.ai_local)

        layout.addWidget(g)
        layout.addWidget(g2)
        layout.addStretch(1)

    def _build_export_tab(self) -> None:
        _, layout = self._mk_tab("Import / Export")
        g, form = self._mk_group("Graphs export")
        self.exp_img_fmt = QComboBox(); self.exp_img_fmt.addItems(["PNG", "SVG"])
        self.exp_dpi = QSpinBox(); self.exp_dpi.setRange(50, 600); self.exp_dpi.setSuffix(" dpi")
        self.exp_transparent = QCheckBox("Transparent background")
        form.addRow("Image format", self.exp_img_fmt)
        form.addRow("Resolution", self.exp_dpi)
        form.addRow(self.exp_transparent)

        g2, form2 = self._mk_group("Project export")
        self.exp_include_graphs = QCheckBox("Include graphs in export")
        self.exp_project_fmt = QComboBox(); self.exp_project_fmt.addItems(["Typhydrion Project (.typhyproj)"])
        form2.addRow(self.exp_include_graphs)
        form2.addRow("Default format", self.exp_project_fmt)

        layout.addWidget(g)
        layout.addWidget(g2)
        layout.addStretch(1)

    def _build_shortcuts_tab(self) -> None:
        _, layout = self._mk_tab("Keyboard & Shortcuts")
        g, form = self._mk_group("Shortcuts")
        self.sc_vim = QCheckBox("Enable Vim-style navigation (experimental)")
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._reset_shortcuts)
        form.addRow(self.sc_vim)
        form.addRow(reset_btn)
        layout.addWidget(g)
        layout.addStretch(1)

    def _build_advanced_tab(self) -> None:
        _, layout = self._mk_tab("Advanced")
        g, form = self._mk_group("Environment & Export")
        self.venv_path = QLineEdit()
        self.export_folder = QLineEdit()
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse_folder_into(self.export_folder))
        row = QHBoxLayout()
        row.addWidget(self.export_folder, 1)
        row.addWidget(btn)
        row_w = QWidget(); row_w.setLayout(row)
        self.png_dpi = QSpinBox(); self.png_dpi.setRange(50, 600); self.png_dpi.setSuffix(" dpi")
        form.addRow("Python venv path", self.venv_path)
        form.addRow("Default export folder", row_w)
        form.addRow("PNG export DPI", self.png_dpi)
        self.exp_nodes = QCheckBox("Enable experimental nodes")
        self.beta_features = QCheckBox("Enable beta features")
        self.unsafe_perf = QCheckBox("Unsafe performance mode (manual)")
        self.dev_console = QCheckBox("Developer console")
        form.addRow(self.exp_nodes)
        form.addRow(self.beta_features)
        form.addRow(self.unsafe_perf)
        form.addRow(self.dev_console)
        layout.addWidget(g)
        layout.addStretch(1)

    def _build_about_tab(self) -> None:
        _, layout = self._mk_tab("About / System Info")
        g, _ = self._mk_group("System")
        box = QVBoxLayout(g)
        self.about_text = QTextEdit()
        self.about_text.setReadOnly(True)
        self.about_text.setStyleSheet(
            "QTextEdit { background-color: rgba(10, 14, 18, 140); border: 1px solid rgba(255,255,255,20); border-radius: 8px; color: rgba(240,245,250,220); }"
        )
        box.addWidget(self.about_text)
        layout.addWidget(g)
        layout.addStretch(1)

    # ─────────────────────────────────────────────────────────────────────
    # Load/Save
    # ─────────────────────────────────────────────────────────────────────

    def _load_from_settings(self) -> None:
        s = self._s

        # General UI
        theme = s.get_str(s.UI_THEME).lower()
        self.ui_theme.setCurrentText({"system": "System", "dark": "Dark", "light": "Light"}.get(theme, "System"))
        self.ui_accent.setText(s.get_str(s.UI_ACCENT_COLOR))
        self.ui_font_scale.setValue(s.get_int(s.UI_FONT_SCALE_PERCENT))
        self.ui_language.setCurrentText(s.get_str(s.UI_LANGUAGE) or "System")
        startup = s.get_str(s.UI_STARTUP_MODE).lower()
        self.ui_startup.setCurrentText("Last project" if startup == "last_project" else "Home")

        self.ui_transparency.setChecked(s.get_bool(s.UI_TRANSPARENCY_ENABLED))
        self.ui_animations.setChecked(s.get_bool(s.UI_ANIMATIONS_ENABLED))
        self.ui_node_grid.setChecked(s.get_bool(s.GRID_NODE_ENABLED))
        self.ui_graph_grid.setChecked(s.get_bool(s.GRID_GRAPH_ENABLED))
        self.ui_node_grid_size.setValue(s.get_int(s.GRID_NODE_SIZE))
        self.ui_graph_grid_size.setValue(s.get_int(s.GRID_GRAPH_SIZE))
        self.ui_acrylic_hex.setText(hex(s.get_int(s.UI_ACRYLIC_GRADIENT_HEX)))

        self.proj_open_last.setChecked(s.get_bool(s.PROJECT_OPEN_LAST))
        self.confirm_delete.setChecked(s.get_bool(s.CONFIRM_BEFORE_DELETE))
        self.default_graph_open.setChecked(s.get_bool(s.DEFAULT_GRAPH_DOCK_ACTIVE))
        self.proj_default_folder.setText(s.get_str(s.PROJECT_DEFAULT_FOLDER))
        self.autosave_enabled.setChecked(s.get_bool(s.AUTOSAVE_ENABLED))
        self.autosave_interval.setValue(s.get_int(s.AUTOSAVE_INTERVAL_SEC))
        self.autosave_versions.setValue(s.get_int(s.AUTOSAVE_MAX_VERSIONS))

        # Performance
        self.perf_threads.setValue(s.get_int(s.PERF_THREAD_COUNT))
        self.perf_gpu_accel.setChecked(s.get_bool(s.PERF_ENABLE_GPU_ACCEL))
        self.perf_gpu_mem.setValue(s.get_int(s.PERF_GPU_MEMORY_LIMIT_MB))
        pr = s.get_str(s.PERF_BACKGROUND_PRIORITY).capitalize() or "Normal"
        self.perf_priority.setCurrentText(pr if pr in {"Low", "Normal", "High"} else "Normal")
        self.chunk_size.setValue(s.get_int(s.CHUNK_DEFAULT_SIZE))
        self.chunk_prefetch.setValue(s.get_int(s.CHUNK_PREFETCH_COUNT))
        self.chunk_async.setChecked(s.get_bool(s.CHUNK_ASYNC_LOADING))

        self.embed_mode.setCurrentText(s.get_str(s.EMBED_DATASET_MODE))
        self.embed_max_mb.setValue(s.get_int(s.EMBED_MAX_MB))
        self.embed_max_rows.setValue(s.get_int(s.EMBED_MAX_ROWS))
        self.embed_preview_rows.setValue(s.get_int(s.EMBED_PREVIEW_ROWS))
        self.embed_compress.setValue(s.get_int(s.EMBED_COMPRESSION_LEVEL))

        # Dataset defaults
        fmt = s.get_str(s.DATASET_DEFAULT_FORMAT).lower()
        self.ds_format.setCurrentText({"csv": "CSV", "parquet": "Parquet", "json": "JSON"}.get(fmt, "CSV"))
        enc = s.get_str(s.DATASET_ENCODING).lower()
        self.ds_encoding.setCurrentText("Auto" if enc == "auto" else (enc if enc else "Auto"))
        miss = s.get_str(s.DATASET_MISSING_DEFAULT).lower()
        self.ds_missing.setCurrentText({"keep": "Keep as-is", "drop": "Drop rows", "impute": "Impute"}.get(miss, "Keep as-is"))
        self.ds_auto_target.setChecked(s.get_bool(s.DATASET_AUTO_TARGET))
        self.ds_schema_validate.setChecked(s.get_bool(s.DATASET_SCHEMA_VALIDATE))
        dups = s.get_str(s.DATASET_DUPLICATES).lower()
        self.ds_duplicates.setCurrentText("Drop duplicates" if dups == "drop" else "Keep duplicates")
        self.ds_dtype_auto.setChecked(s.get_bool(s.DATASET_DTYPE_AUTO))

        self.pipe_auto.setChecked(s.get_bool(s.PIPELINE_AUTO_RUN))
        self.pipe_prop.setChecked(s.get_bool(s.PIPELINE_PROPAGATE))
        self.pipe_debounce.setValue(s.get_int(s.PIPELINE_DEBOUNCE_MS))
        self.pipe_stop_on_error.setChecked(s.get_bool(s.PIPELINE_STOP_ON_ERROR))

        # Node defaults
        self.node_w.setValue(s.get_int(s.NODE_DEFAULT_W))
        self.node_h.setValue(s.get_int(s.NODE_DEFAULT_H))
        self.node_snap.setChecked(s.get_bool(s.NODE_SNAP_TO_GRID))
        self.node_auto_align.setChecked(s.get_bool(s.NODE_AUTO_ALIGN))
        self.node_exec_highlight.setChecked(s.get_bool(s.NODE_EXEC_HIGHLIGHT))
        self.node_exec_order.setChecked(s.get_bool(s.NODE_SHOW_EXEC_ORDER))
        self.link_style.setCurrentText("Straight" if s.get_str(s.LINK_STYLE) == "straight" else "Curved")
        self.link_thickness.setValue(s.get_int(s.LINK_THICKNESS))
        self.link_animate.setChecked(s.get_bool(s.LINK_ANIMATE_FLOW))
        self.link_dtype_icons.setChecked(s.get_bool(s.LINK_SHOW_DTYPE_ICONS))

        # Graph behavior/style
        self.graph_default_category.setText(s.get_str(s.GRAPH_DEFAULT_CATEGORY))
        self.graph_default_name.setText(s.get_str(s.GRAPH_DEFAULT_NAME))
        self.graph_pause_inactive.setChecked(s.get_bool(s.GRAPH_PAUSE_INACTIVE))
        self.graph_smooth.setChecked(s.get_bool(s.GRAPH_SMOOTH_PLOTTING))
        self.graph_point_limit.setValue(s.get_int(s.GRAPH_ROLLING_POINT_LIMIT))
        self.graph_labels.setChecked(s.get_bool(s.GRAPH_LABELS_DEFAULT))
        self.graph_live_interval.setValue(s.get_int(s.GRAPH_LIVE_INTERVAL_DEFAULT))
        self.graph_3d_rotate.setChecked(s.get_bool(s.GRAPH_3D_ROTATE_DEFAULT))
        self.graph_3d_speed.setValue(s.get_int(s.GRAPH_3D_SPEED_DEFAULT))
        self.graph_line_thickness.setValue(s.get_int(s.GRAPH_LINE_THICKNESS))
        self.graph_axis_autoscale.setChecked(s.get_bool(s.GRAPH_AXIS_AUTOSCALE))
        self.graph_palette.setCurrentText(s.get_str(s.GRAPH_COLOR_PALETTE) or "PowerBI")
        self.graph_legend.setCurrentText((s.get_str(s.GRAPH_LEGEND_BEHAVIOR) or "auto").capitalize())
        self.card_w.setValue(s.get_int(s.GRAPH_CARD_DEFAULT_W))
        self.card_h.setValue(s.get_int(s.GRAPH_CARD_DEFAULT_H))
        self.dashboard_zoom.setValue(s.get_int(s.GRAPH_DASHBOARD_ZOOM_DEFAULT))

        # Logging / Debug
        lvl = s.get_str(s.LOG_LEVEL).lower()
        self.log_level.setCurrentText({"debug": "Debug", "error": "Error"}.get(lvl, "Info"))
        self.log_file.setText(s.get_str(s.LOG_FILE_PATH))
        self.log_max.setValue(s.get_int(s.LOG_MAX_MB))
        self.log_clear_days.setValue(s.get_int(s.LOG_AUTO_CLEAR_DAYS))
        self.dbg_trace.setChecked(s.get_bool(s.DEBUG_SHOW_TRACE))
        self.dbg_node.setChecked(s.get_bool(s.DEBUG_NODE_LOGS))
        self.dbg_graph.setChecked(s.get_bool(s.DEBUG_GRAPH_LOGS))
        self.dbg_prof.setChecked(s.get_bool(s.DEBUG_PROFILING))

        # AI assistant
        self.ai_enabled.setChecked(s.get_bool(s.AI_ENABLED))
        self.ai_mode.setCurrentText("Auto apply" if s.get_str(s.AI_MODE) == "auto_apply" else "Suggest only")
        self.ai_suggest_chunk.setChecked(s.get_bool(s.AI_SUGGEST_CHUNK_SIZE))
        self.ai_suggest_model.setChecked(s.get_bool(s.AI_SUGGEST_MODEL_TYPE))
        priv = s.get_str(s.AI_PRIVACY).lower()
        self.ai_privacy.setCurrentText({"none": "None", "metadata": "Metadata"}.get(priv, "Stats"))
        self.ai_local.setChecked(s.get_bool(s.AI_LOCAL_ONLY))

        # Export
        self.exp_img_fmt.setCurrentText("SVG" if s.get_str(s.EXPORT_IMAGE_FORMAT).lower() == "svg" else "PNG")
        self.exp_dpi.setValue(s.get_int(s.EXPORT_PNG_DPI))
        self.exp_transparent.setChecked(s.get_bool(s.EXPORT_TRANSPARENT_BG))
        self.exp_include_graphs.setChecked(s.get_bool(s.EXPORT_INCLUDE_GRAPHS))

        self.venv_path.setText(s.get_str(s.PYTHON_VENV_PATH))
        self.export_folder.setText(s.get_str(s.EXPORT_DEFAULT_FOLDER))
        self.png_dpi.setValue(s.get_int(s.EXPORT_PNG_DPI))

        self.exp_nodes.setChecked(s.get_bool(s.EXPERIMENTAL_NODES))
        self.beta_features.setChecked(s.get_bool(s.BETA_FEATURES))
        self.unsafe_perf.setChecked(s.get_bool(s.UNSAFE_PERF_MODE))
        self.dev_console.setChecked(s.get_bool(s.DEV_CONSOLE))

        self.sc_vim.setChecked(s.get_bool(s.SHORTCUTS_VIM_NAV))

        self._refresh_about()

    def _wire_live_apply(self) -> None:
        """Connect form controls so changes apply automatically (debounced)."""
        # Checkboxes
        for w in self.findChildren(QCheckBox):
            try:
                w.toggled.connect(self._schedule_live_apply)
            except Exception:
                pass
        # Spin boxes
        for w in self.findChildren(QSpinBox):
            try:
                w.valueChanged.connect(self._schedule_live_apply)
            except Exception:
                pass
        # Combo boxes
        for w in self.findChildren(QComboBox):
            try:
                w.currentIndexChanged.connect(self._schedule_live_apply)
            except Exception:
                pass
        # Line edits (apply when user finishes editing)
        for w in self.findChildren(QLineEdit):
            try:
                w.editingFinished.connect(self._schedule_live_apply)
            except Exception:
                pass

    def _schedule_live_apply(self, *_args) -> None:
        if self._loading_values:
            return
        try:
            self._live_apply_timer.start(180)
        except Exception:
            pass

    def _save_to_settings(self) -> None:
        s = self._s

        # General UI
        s.set_value(s.UI_THEME, self.ui_theme.currentText().lower())
        s.set_value(s.UI_ACCENT_COLOR, self.ui_accent.text().strip() or "#118dff")
        s.set_value(s.UI_FONT_SCALE_PERCENT, int(self.ui_font_scale.value()))
        s.set_value(s.UI_LANGUAGE, self.ui_language.currentText())
        s.set_value(s.UI_STARTUP_MODE, "last_project" if self.ui_startup.currentText() == "Last project" else "home")

        s.set_value(s.UI_TRANSPARENCY_ENABLED, bool(self.ui_transparency.isChecked()))
        s.set_value(s.UI_ANIMATIONS_ENABLED, bool(self.ui_animations.isChecked()))
        s.set_value(s.GRID_NODE_ENABLED, bool(self.ui_node_grid.isChecked()))
        s.set_value(s.GRID_GRAPH_ENABLED, bool(self.ui_graph_grid.isChecked()))
        s.set_value(s.GRID_NODE_SIZE, int(self.ui_node_grid_size.value()))
        s.set_value(s.GRID_GRAPH_SIZE, int(self.ui_graph_grid_size.value()))
        try:
            s.set_value(s.UI_ACRYLIC_GRADIENT_HEX, int(self.ui_acrylic_hex.text().strip(), 16))
        except Exception:
            pass

        s.set_value(s.PROJECT_OPEN_LAST, bool(self.proj_open_last.isChecked()))
        s.set_value(s.CONFIRM_BEFORE_DELETE, bool(self.confirm_delete.isChecked()))
        s.set_value(s.DEFAULT_GRAPH_DOCK_ACTIVE, bool(self.default_graph_open.isChecked()))
        s.set_value(s.PROJECT_DEFAULT_FOLDER, self.proj_default_folder.text().strip())
        s.set_value(s.AUTOSAVE_ENABLED, bool(self.autosave_enabled.isChecked()))
        s.set_value(s.AUTOSAVE_INTERVAL_SEC, int(self.autosave_interval.value()))
        s.set_value(s.AUTOSAVE_MAX_VERSIONS, int(self.autosave_versions.value()))

        # Performance
        s.set_value(s.PERF_THREAD_COUNT, int(self.perf_threads.value()))
        s.set_value(s.PERF_ENABLE_GPU_ACCEL, bool(self.perf_gpu_accel.isChecked()))
        s.set_value(s.PERF_GPU_MEMORY_LIMIT_MB, int(self.perf_gpu_mem.value()))
        s.set_value(s.PERF_BACKGROUND_PRIORITY, self.perf_priority.currentText().lower())
        s.set_value(s.CHUNK_DEFAULT_SIZE, int(self.chunk_size.value()))
        s.set_value(s.CHUNK_PREFETCH_COUNT, int(self.chunk_prefetch.value()))
        s.set_value(s.CHUNK_ASYNC_LOADING, bool(self.chunk_async.isChecked()))

        s.set_value(s.EMBED_DATASET_MODE, self.embed_mode.currentText())
        s.set_value(s.EMBED_MAX_MB, int(self.embed_max_mb.value()))
        s.set_value(s.EMBED_MAX_ROWS, int(self.embed_max_rows.value()))
        s.set_value(s.EMBED_PREVIEW_ROWS, int(self.embed_preview_rows.value()))
        s.set_value(s.EMBED_COMPRESSION_LEVEL, int(self.embed_compress.value()))

        # Dataset defaults
        s.set_value(s.DATASET_DEFAULT_FORMAT, self.ds_format.currentText().lower())
        enc = self.ds_encoding.currentText().lower()
        s.set_value(s.DATASET_ENCODING, "auto" if enc == "auto" else enc)
        miss = self.ds_missing.currentText()
        s.set_value(
            s.DATASET_MISSING_DEFAULT,
            "drop" if "Drop" in miss else ("impute" if "Impute" in miss else "keep"),
        )
        s.set_value(s.DATASET_AUTO_TARGET, bool(self.ds_auto_target.isChecked()))
        s.set_value(s.DATASET_SCHEMA_VALIDATE, bool(self.ds_schema_validate.isChecked()))
        s.set_value(s.DATASET_DUPLICATES, "drop" if "Drop" in self.ds_duplicates.currentText() else "keep")
        s.set_value(s.DATASET_DTYPE_AUTO, bool(self.ds_dtype_auto.isChecked()))

        s.set_value(s.PIPELINE_AUTO_RUN, bool(self.pipe_auto.isChecked()))
        s.set_value(s.PIPELINE_PROPAGATE, bool(self.pipe_prop.isChecked()))
        s.set_value(s.PIPELINE_DEBOUNCE_MS, int(self.pipe_debounce.value()))
        s.set_value(s.PIPELINE_STOP_ON_ERROR, bool(self.pipe_stop_on_error.isChecked()))

        # Node defaults
        s.set_value(s.NODE_DEFAULT_W, int(self.node_w.value()))
        s.set_value(s.NODE_DEFAULT_H, int(self.node_h.value()))
        s.set_value(s.NODE_SNAP_TO_GRID, bool(self.node_snap.isChecked()))
        s.set_value(s.NODE_AUTO_ALIGN, bool(self.node_auto_align.isChecked()))
        s.set_value(s.NODE_EXEC_HIGHLIGHT, bool(self.node_exec_highlight.isChecked()))
        s.set_value(s.NODE_SHOW_EXEC_ORDER, bool(self.node_exec_order.isChecked()))
        s.set_value(s.LINK_STYLE, "straight" if self.link_style.currentText() == "Straight" else "curved")
        s.set_value(s.LINK_THICKNESS, int(self.link_thickness.value()))
        s.set_value(s.LINK_ANIMATE_FLOW, bool(self.link_animate.isChecked()))
        s.set_value(s.LINK_SHOW_DTYPE_ICONS, bool(self.link_dtype_icons.isChecked()))

        # Graphs
        s.set_value(s.GRAPH_DEFAULT_CATEGORY, self.graph_default_category.text().strip() or "Exploration")
        s.set_value(s.GRAPH_DEFAULT_NAME, self.graph_default_name.text().strip() or "Column Distribution")
        s.set_value(s.GRAPH_PAUSE_INACTIVE, bool(self.graph_pause_inactive.isChecked()))
        s.set_value(s.GRAPH_SMOOTH_PLOTTING, bool(self.graph_smooth.isChecked()))
        s.set_value(s.GRAPH_ROLLING_POINT_LIMIT, int(self.graph_point_limit.value()))
        s.set_value(s.GRAPH_LABELS_DEFAULT, bool(self.graph_labels.isChecked()))
        s.set_value(s.GRAPH_LIVE_INTERVAL_DEFAULT, int(self.graph_live_interval.value()))
        s.set_value(s.GRAPH_3D_ROTATE_DEFAULT, bool(self.graph_3d_rotate.isChecked()))
        s.set_value(s.GRAPH_3D_SPEED_DEFAULT, int(self.graph_3d_speed.value()))
        s.set_value(s.GRAPH_LINE_THICKNESS, int(self.graph_line_thickness.value()))
        s.set_value(s.GRAPH_AXIS_AUTOSCALE, bool(self.graph_axis_autoscale.isChecked()))
        s.set_value(s.GRAPH_COLOR_PALETTE, self.graph_palette.currentText())
        s.set_value(s.GRAPH_LEGEND_BEHAVIOR, self.graph_legend.currentText().lower())
        s.set_value(s.GRAPH_CARD_DEFAULT_W, int(self.card_w.value()))
        s.set_value(s.GRAPH_CARD_DEFAULT_H, int(self.card_h.value()))
        s.set_value(s.GRAPH_DASHBOARD_ZOOM_DEFAULT, int(self.dashboard_zoom.value()))

        # Logging / Debug
        lvl = self.log_level.currentText().lower()
        s.set_value(s.LOG_LEVEL, "debug" if lvl == "debug" else ("error" if lvl == "error" else "info"))
        s.set_value(s.LOG_FILE_PATH, self.log_file.text().strip())
        s.set_value(s.LOG_MAX_MB, int(self.log_max.value()))
        s.set_value(s.LOG_AUTO_CLEAR_DAYS, int(self.log_clear_days.value()))
        s.set_value(s.DEBUG_SHOW_TRACE, bool(self.dbg_trace.isChecked()))
        s.set_value(s.DEBUG_NODE_LOGS, bool(self.dbg_node.isChecked()))
        s.set_value(s.DEBUG_GRAPH_LOGS, bool(self.dbg_graph.isChecked()))
        s.set_value(s.DEBUG_PROFILING, bool(self.dbg_prof.isChecked()))

        # AI
        s.set_value(s.AI_ENABLED, bool(self.ai_enabled.isChecked()))
        s.set_value(s.AI_MODE, "auto_apply" if "Auto" in self.ai_mode.currentText() else "suggest_only")
        s.set_value(s.AI_SUGGEST_CHUNK_SIZE, bool(self.ai_suggest_chunk.isChecked()))
        s.set_value(s.AI_SUGGEST_MODEL_TYPE, bool(self.ai_suggest_model.isChecked()))
        p = self.ai_privacy.currentText().lower()
        s.set_value(s.AI_PRIVACY, "metadata" if p == "metadata" else ("none" if p == "none" else "stats"))
        s.set_value(s.AI_LOCAL_ONLY, bool(self.ai_local.isChecked()))

        s.set_value(s.PYTHON_VENV_PATH, self.venv_path.text().strip())
        s.set_value(s.EXPORT_DEFAULT_FOLDER, self.export_folder.text().strip())
        s.set_value(s.EXPORT_PNG_DPI, int(self.png_dpi.value()))

        # Export
        s.set_value(s.EXPORT_IMAGE_FORMAT, "svg" if self.exp_img_fmt.currentText() == "SVG" else "png")
        s.set_value(s.EXPORT_PNG_DPI, int(self.exp_dpi.value()))
        s.set_value(s.EXPORT_TRANSPARENT_BG, bool(self.exp_transparent.isChecked()))
        s.set_value(s.EXPORT_INCLUDE_GRAPHS, bool(self.exp_include_graphs.isChecked()))

        # Advanced flags
        s.set_value(s.EXPERIMENTAL_NODES, bool(self.exp_nodes.isChecked()))
        s.set_value(s.BETA_FEATURES, bool(self.beta_features.isChecked()))
        s.set_value(s.UNSAFE_PERF_MODE, bool(self.unsafe_perf.isChecked()))
        s.set_value(s.DEV_CONSOLE, bool(self.dev_console.isChecked()))

        # Shortcuts
        s.set_value(s.SHORTCUTS_VIM_NAV, bool(self.sc_vim.isChecked()))

    # ─────────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────────

    def _on_apply(self) -> None:
        self._save_to_settings()
        self.settings_applied.emit()

    def _on_ok(self) -> None:
        self._on_apply()
        self.accept()

    def _browse_folder_into(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text().strip() or "")
        if path:
            line_edit.setText(path)

    def _browse_file_into(self, line_edit: QLineEdit, title: str, filter_text: str) -> None:
        path, _ = QFileDialog.getSaveFileName(self, title, line_edit.text().strip() or "", filter_text)
        if path:
            line_edit.setText(path)

    def _reset_shortcuts(self) -> None:
        self.sc_vim.setChecked(False)

    def _refresh_about(self) -> None:
        try:
            lines = []
            lines.append("Typhydrion")
            lines.append(f"Python: {sys.version.split()[0]}")
            lines.append(f"Executable: {sys.executable}")
            lines.append(f"OS: {platform.platform()}")
            lines.append(f"Machine: {platform.machine()}")
            lines.append(f"Processor: {platform.processor()}")
            lines.append(f"Time: {datetime.now().isoformat(timespec='seconds')}")
            self.about_text.setText("\n".join(lines))
        except Exception:
            pass
