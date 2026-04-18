from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QSettings


@dataclass(frozen=True)
class SettingKey:
    key: str
    default: Any


class AppSettings:
    """
    Thin wrapper around QSettings with typed helpers + central defaults.
    """

    # UI / Workspace
    UI_THEME = SettingKey("ui/theme", "dark")  # dark | light | system
    UI_ACCENT_COLOR = SettingKey("ui/accent_color", "#118dff")  # hex
    UI_FONT_SCALE_PERCENT = SettingKey("ui/font_scale_percent", 100)
    UI_LANGUAGE = SettingKey("ui/language", "System")
    UI_STARTUP_MODE = SettingKey("ui/startup_mode", "home")  # home | last_project

    UI_TRANSPARENCY_ENABLED = SettingKey("ui/transparency_enabled", True)
    UI_ACRYLIC_GRADIENT_HEX = SettingKey("ui/acrylic_gradient_hex", 0x518E7400)  # matches previous hardcoded
    UI_ANIMATIONS_ENABLED = SettingKey("ui/animations_enabled", True)

    GRID_NODE_ENABLED = SettingKey("ui/grid_node_enabled", True)
    GRID_NODE_SIZE = SettingKey("ui/grid_node_size", 40)
    GRID_GRAPH_ENABLED = SettingKey("ui/grid_graph_enabled", True)
    GRID_GRAPH_SIZE = SettingKey("ui/grid_graph_size", 20)

    # Project / Files
    PROJECT_DEFAULT_FOLDER = SettingKey("project/default_folder", "")
    PROJECT_OPEN_LAST = SettingKey("project/open_last", True)
    PROJECT_LAST_PATH = SettingKey("project/last_path", "")
    PROJECT_RECENTS = SettingKey("project/recents", [])  # list[str]
    CONFIRM_BEFORE_DELETE = SettingKey("project/confirm_before_delete", True)
    DEFAULT_GRAPH_DOCK_ACTIVE = SettingKey("project/default_graph_dock_active", True)

    AUTOSAVE_ENABLED = SettingKey("project/autosave_enabled", False)
    AUTOSAVE_INTERVAL_SEC = SettingKey("project/autosave_interval_sec", 120)
    AUTOSAVE_MAX_VERSIONS = SettingKey("project/autosave_max_versions", 10)

    # Dataset embedding
    # full | preview | path | none
    EMBED_DATASET_MODE = SettingKey("project/embed_dataset_mode", "preview")
    EMBED_MAX_MB = SettingKey("project/embed_max_mb", 25)
    EMBED_MAX_ROWS = SettingKey("project/embed_max_rows", 200_000)
    EMBED_PREVIEW_ROWS = SettingKey("project/embed_preview_rows", 10_000)
    EMBED_COMPRESSION_LEVEL = SettingKey("project/embed_compression_level", 6)

    # Performance / Execution
    PERF_THREAD_COUNT = SettingKey("perf/thread_count", 0)  # 0 = auto
    PERF_ENABLE_GPU_ACCEL = SettingKey("perf/enable_gpu_accel", True)
    PERF_GPU_MEMORY_LIMIT_MB = SettingKey("perf/gpu_memory_limit_mb", 0)  # 0 = no limit
    PERF_BACKGROUND_PRIORITY = SettingKey("perf/background_priority", "normal")  # low|normal|high

    CHUNK_DEFAULT_SIZE = SettingKey("chunk/default_size", 50_000)
    CHUNK_PREFETCH_COUNT = SettingKey("chunk/prefetch_count", 2)
    CHUNK_ASYNC_LOADING = SettingKey("chunk/async_loading", True)

    # Pipeline
    PIPELINE_AUTO_RUN = SettingKey("pipeline/auto_run", False)
    PIPELINE_PROPAGATE = SettingKey("pipeline/propagate_downstream", False)
    PIPELINE_DEBOUNCE_MS = SettingKey("pipeline/debounce_ms", 150)
    PIPELINE_STOP_ON_ERROR = SettingKey("pipeline/stop_on_error", True)

    # Node UI defaults
    NODE_DEFAULT_W = SettingKey("node/default_w", 240)
    NODE_DEFAULT_H = SettingKey("node/default_h", 160)
    NODE_SNAP_TO_GRID = SettingKey("node/snap_to_grid", True)
    NODE_AUTO_ALIGN = SettingKey("node/auto_align", False)
    NODE_EXEC_HIGHLIGHT = SettingKey("node/execution_highlight", True)
    NODE_SHOW_EXEC_ORDER = SettingKey("node/show_execution_order", False)

    LINK_STYLE = SettingKey("link/style", "curved")  # curved|straight
    LINK_THICKNESS = SettingKey("link/thickness", 2)
    LINK_ANIMATE_FLOW = SettingKey("link/animate_flow", True)
    LINK_SHOW_DTYPE_ICONS = SettingKey("link/show_dtype_icons", True)

    # Graph defaults
    GRAPH_DEFAULT_CATEGORY = SettingKey("graph/default_category", "Exploration")
    GRAPH_DEFAULT_NAME = SettingKey("graph/default_name", "Column Distribution")
    GRAPH_LABELS_DEFAULT = SettingKey("graph/labels_default", True)
    GRAPH_LIVE_INTERVAL_DEFAULT = SettingKey("graph/live_interval_default_ms", 1000)
    GRAPH_3D_ROTATE_DEFAULT = SettingKey("graph/3d_rotate_default", True)
    GRAPH_3D_SPEED_DEFAULT = SettingKey("graph/3d_speed_default", 6)  # slider 0..20
    GRAPH_CARD_DEFAULT_W = SettingKey("graph/card_default_w", 350)
    GRAPH_CARD_DEFAULT_H = SettingKey("graph/card_default_h", 280)
    GRAPH_DASHBOARD_ZOOM_DEFAULT = SettingKey("graph/dashboard_zoom_default", 100)  # percent
    GRAPH_PAUSE_INACTIVE = SettingKey("graph/pause_when_inactive", True)
    GRAPH_SMOOTH_PLOTTING = SettingKey("graph/smooth_plotting", True)
    GRAPH_ROLLING_POINT_LIMIT = SettingKey("graph/rolling_point_limit", 120)
    GRAPH_LINE_THICKNESS = SettingKey("graph/line_thickness", 2)
    GRAPH_AXIS_AUTOSCALE = SettingKey("graph/axis_autoscale", True)
    GRAPH_COLOR_PALETTE = SettingKey("graph/color_palette", "PowerBI")
    GRAPH_LEGEND_BEHAVIOR = SettingKey("graph/legend_behavior", "auto")  # auto|show|hide

    # Dataset defaults
    DATASET_DEFAULT_FORMAT = SettingKey("dataset/default_format", "csv")  # csv|parquet|json
    DATASET_ENCODING = SettingKey("dataset/encoding", "auto")  # auto|utf-8|...
    DATASET_MISSING_DEFAULT = SettingKey("dataset/missing_handling", "keep")  # keep|drop|impute
    DATASET_AUTO_TARGET = SettingKey("dataset/auto_detect_target", True)
    DATASET_SCHEMA_VALIDATE = SettingKey("dataset/schema_validate", False)
    DATASET_DUPLICATES = SettingKey("dataset/duplicate_rows", "keep")  # keep|drop
    DATASET_DTYPE_AUTO = SettingKey("dataset/dtype_auto_convert", True)

    # Training defaults
    TRAIN_OPTIMIZER = SettingKey("train/optimizer", "adam")
    TRAIN_LOSS = SettingKey("train/loss", "auto")
    TRAIN_METRIC = SettingKey("train/metric", "auto")
    TRAIN_RANDOM_SEED = SettingKey("train/random_seed", 42)
    TRAIN_EARLY_STOPPING = SettingKey("train/early_stopping", True)
    TRAIN_SHUFFLE_CHUNKS = SettingKey("train/shuffle_chunks", True)
    TRAIN_SAVE_CHECKPOINTS = SettingKey("train/save_checkpoints", True)
    TRAIN_CHECKPOINT_FREQ = SettingKey("train/checkpoint_frequency", 1)
    TRAIN_RESUME_ON_CRASH = SettingKey("train/resume_on_crash", True)

    # Logging / Debug
    LOG_LEVEL = SettingKey("log/level", "info")  # debug|info|error
    LOG_FILE_PATH = SettingKey("log/file_path", "")
    LOG_MAX_MB = SettingKey("log/max_mb", 10)
    LOG_AUTO_CLEAR_DAYS = SettingKey("log/auto_clear_days", 7)
    DEBUG_SHOW_TRACE = SettingKey("debug/show_trace", False)
    DEBUG_NODE_LOGS = SettingKey("debug/node_logs", False)
    DEBUG_GRAPH_LOGS = SettingKey("debug/graph_logs", False)
    DEBUG_PROFILING = SettingKey("debug/profiling", False)

    # AI assistant
    AI_ENABLED = SettingKey("ai/enabled", True)
    AI_MODE = SettingKey("ai/mode", "suggest_only")  # suggest_only|auto_apply
    AI_SUGGEST_CHUNK_SIZE = SettingKey("ai/suggest_chunk_size", True)
    AI_SUGGEST_MODEL_TYPE = SettingKey("ai/suggest_model_type", True)
    AI_PRIVACY = SettingKey("ai/privacy", "stats")  # none|metadata|stats
    AI_LOCAL_ONLY = SettingKey("ai/local_only", False)

    # Advanced
    PYTHON_VENV_PATH = SettingKey("advanced/python_venv_path", "")
    EXPORT_DEFAULT_FOLDER = SettingKey("advanced/export_default_folder", "")
    EXPORT_PNG_DPI = SettingKey("advanced/export_png_dpi", 150)
    EXPORT_IMAGE_FORMAT = SettingKey("export/image_format", "png")  # png|svg
    EXPORT_TRANSPARENT_BG = SettingKey("export/transparent_bg", False)
    EXPORT_INCLUDE_GRAPHS = SettingKey("export/include_graphs", True)
    EXPORT_PROJECT_FORMAT = SettingKey("export/project_format", "typhyproj")
    EXPERIMENTAL_NODES = SettingKey("advanced/experimental_nodes", False)
    BETA_FEATURES = SettingKey("advanced/beta_features", False)
    UNSAFE_PERF_MODE = SettingKey("advanced/unsafe_perf_mode", False)
    DEV_CONSOLE = SettingKey("advanced/developer_console", False)

    # Shortcuts
    SHORTCUTS_VIM_NAV = SettingKey("shortcuts/vim_navigation", False)

    def __init__(self) -> None:
        self._qs = QSettings()

    def value(self, key: SettingKey) -> Any:
        v = self._qs.value(key.key, key.default)
        return v

    def set_value(self, key: SettingKey, value: Any) -> None:
        self._qs.setValue(key.key, value)

    def get_bool(self, key: SettingKey) -> bool:
        v = self.value(key)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(v)

    def get_int(self, key: SettingKey) -> int:
        v = self.value(key)
        try:
            return int(v)
        except Exception:
            return int(key.default)

    def get_str(self, key: SettingKey) -> str:
        v = self.value(key)
        return "" if v is None else str(v)

    def get_list(self, key: SettingKey) -> list:
        v = self.value(key)
        if v is None:
            return []
        if isinstance(v, list):
            return v
        # QSettings can return QStringList-like
        try:
            return list(v)
        except Exception:
            return []

    # Convenience helpers
    def add_recent_project(self, path: str, max_items: int = 12) -> None:
        if not path:
            return
        recents = [p for p in self.get_list(self.PROJECT_RECENTS) if p and p != path]
        recents.insert(0, path)
        recents = recents[:max_items]
        self.set_value(self.PROJECT_RECENTS, recents)
        self.set_value(self.PROJECT_LAST_PATH, path)
