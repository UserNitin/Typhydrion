from __future__ import annotations

from PySide6.QtCore import Qt, QPoint, Signal, QTimer, QPropertyAnimation, QEasingCurve, QObject, QThread, Slot
from PySide6.QtGui import QPainter, QKeySequence, QPen, QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QGraphicsView,
    QGraphicsProxyWidget,
    QMenu,
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QAbstractItemView,
    QDialogButtonBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QSlider,
    QFileDialog,
    QFormLayout,
    QTextEdit,
    QApplication,
    QScrollArea,
    QGroupBox,
    QGraphicsOpacityEffect,
    QMessageBox,
)
import inspect
import time
import pandas as pd
import shiboken6

from nodes.base.node_graph_scene import NodeGraphScene
from nodes.base.node_base import NodeItem
from nodes.base.port import PortItem
from nodes.base.edge import EdgeItem
from nodes.base.link_model import LinkModel


_DATASET_PREVIEW_CALLBACK = None
_NODE_SELECTED_CALLBACK = None
_READER_CHANGED_CALLBACK = None
_EXTRA_READER_PARAMS = {}  # Extra parameters from Node Properties window


def _options_to_runtime_list(options: dict) -> list[dict]:
    return [{"label": k, "value": v} for k, v in (options or {}).items()]


def _pick_primary_output(outputs: dict, title: str):
    import pandas as pd
    import numpy as np

    if "Split" in title:
        if "X_train" in outputs:
            val = outputs["X_train"]
            if isinstance(val, pd.DataFrame):
                return val
            if isinstance(val, pd.Series):
                return val.to_frame()

    priority_keys = [
        "Filtered Chunk", "Filtered Data", "Clean Chunk", "Converted Chunk",
        "Scaled Features", "Encoded Features", "Data", "Raw Data",
        "Merged Data", "Features", "X_train", "Predictions",
        "Scaled Data", "Encoded Data", "Preview", "Processed Data",
        "Joined DataFrame", "Selected DataFrame", "Cluster Labels",
        "Anomaly Scores", "Anomaly Labels",
    ]

    for key in priority_keys:
        if key in outputs and outputs[key] is not None:
            val = outputs[key]
            if isinstance(val, pd.DataFrame):
                return val
            if isinstance(val, pd.Series):
                return val.to_frame()
            if isinstance(val, np.ndarray) and val.ndim <= 2:
                return pd.DataFrame(val)

    for v in outputs.values():
        if isinstance(v, pd.DataFrame):
            return v
        if isinstance(v, pd.Series):
            return v.to_frame()
    return None


class _EngineWorker(QObject):
    execution_finished = Signal(dict)
    pipeline_finished = Signal(dict)

    @Slot(dict)
    def execute(self, request: dict) -> None:
        from nodes.registry import get_node_runtime
        from nodes.base.node_runtime import NodeResult

        req_id = int(request.get("request_id", 0))
        title = str(request.get("title", "") or "")
        node_id = str(request.get("node_id", "") or "")
        inputs = request.get("inputs", {}) or {}
        options = request.get("options", {}) or {}
        fallback_input = request.get("input_df")

        try:
            runtime_class = get_node_runtime(title)
            if runtime_class is None:
                self.execution_finished.emit(
                    {
                        "request_id": req_id,
                        "node_id": node_id,
                        "ok": False,
                        "error": f"No runtime registered for '{title}'",
                        "primary_output": fallback_input,
                        "outputs": {},
                    }
                )
                return

            node_data = {"options": _options_to_runtime_list(options)}
            runtime = runtime_class(node_id, node_data)
            result = runtime.execute(inputs)

            if isinstance(result, NodeResult):
                outputs = result.outputs or {}
                primary_output = _pick_primary_output(outputs, title)
            else:
                outputs = {}
                primary_output = fallback_input

            self.execution_finished.emit(
                {
                    "request_id": req_id,
                    "node_id": node_id,
                    "ok": True,
                    "error": "",
                    "primary_output": primary_output,
                    "outputs": outputs,
                }
            )
        except Exception as e:
            self.execution_finished.emit(
                {
                    "request_id": req_id,
                    "node_id": node_id,
                    "ok": False,
                    "error": str(e),
                    "primary_output": fallback_input,
                    "outputs": {},
                }
            )

    @Slot(dict)
    def execute_pipeline(self, request: dict) -> None:
        from engine.pipeline_executor import PipelineExecutor

        req_id = int(request.get("request_id", 0))
        graph = request.get("graph", {}) or {}
        initial_inputs = request.get("initial_inputs_by_node", {}) or {}

        try:
            executor = PipelineExecutor()
            summary = executor.execute(graph, initial_inputs_by_node=initial_inputs, fail_fast=False)
            node_results = {}
            for node_id, node_result in (summary.node_results or {}).items():
                node_results[str(node_id)] = {
                    "success": bool(getattr(node_result, "success", False)),
                    "node_title": str(getattr(node_result, "node_title", "") or ""),
                    "primary_output": getattr(node_result, "primary_output", None),
                    "error_message": getattr(node_result, "error_message", None),
                }

            self.pipeline_finished.emit(
                {
                    "request_id": req_id,
                    "success": bool(summary.success),
                    "errors": list(summary.errors or []),
                    "warnings": list(summary.warnings or []),
                    "execution_order": list(summary.execution_order or []),
                    "node_results": node_results,
                }
            )
        except Exception as e:
            self.pipeline_finished.emit(
                {
                    "request_id": req_id,
                    "success": False,
                    "errors": [str(e)],
                    "warnings": [],
                    "execution_order": [],
                    "node_results": {},
                }
            )


class NodeEditorWindow(QWidget):
    columns_selected = Signal(list)
    dataset_loaded = Signal(object)
    node_selected = Signal(str, str, dict)  # (node_title, node_type, extra_params)
    node_output_changed = Signal(str, object)  # (node_title, dataframe)
    reader_changed = Signal(str)  # reader_name
    nodes_changed = Signal(list)  # List of node usage data for resource monitor

    def __init__(self) -> None:
        super().__init__()
        from ui.app_settings import AppSettings
        self._settings = AppSettings()
        self._current_selected_node = None  # Track currently selected node
        self._last_graph_snapshot: dict | None = None
        self._pending_selected_node = None
        self._last_snapshot_at = 0.0
        self._snapshot_min_interval_sec = 12.0
        self._selection_preview_timer = QTimer(self)
        self._selection_preview_timer.setSingleShot(True)
        self._selection_preview_timer.timeout.connect(self._flush_selection_preview)
        self._last_selection_emit_key: tuple[str, int] | None = None
        self.setStyleSheet("""
            QWidget#NodeEditorRoot {
                background: rgba(10, 14, 22, 255);
            }
        """)
        self.setObjectName("NodeEditorRoot")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addStretch(1)
        self._refresh_nodes_btn = QPushButton("↻ Refresh Nodes")
        self._refresh_nodes_btn.setToolTip("Rebuild the node canvas and restore recently disappeared nodes")
        self._refresh_nodes_btn.setStyleSheet("""
            QPushButton {
                background: rgba(28, 42, 64, 220);
                color: rgba(200, 225, 255, 235);
                border: 1px solid rgba(60, 120, 200, 55);
                border-radius: 8px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(40, 80, 140, 230);
                border-color: rgba(70, 150, 255, 110);
            }
            QPushButton:pressed {
                background: rgba(25, 55, 110, 240);
            }
        """)
        self._refresh_nodes_btn.clicked.connect(self._refresh_nodes_view)
        controls.addWidget(self._refresh_nodes_btn)
        layout.addLayout(controls)

        self._scene = NodeGraphScene()
        self._view = NodeGraphView(self._scene)
        self._view.setRenderHints(
            QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        )
        self._view.setDragMode(QGraphicsView.RubberBandDrag)
        self._view.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self._view.setBackgroundBrush(Qt.transparent)
        self._view.set_columns_selected_callback(self.columns_selected.emit)
        
        # Set callback for when nodes change (add/delete)
        self._view.set_nodes_changed_callback(self._emit_nodes_changed)
        
        # Timer to periodically update node status
        self._node_update_timer = QTimer(self)
        self._node_update_timer.timeout.connect(lambda: self._emit_nodes_changed(include_snapshot=False))
        self._node_update_timer.start(4000)  # Keep resource panel fresh without constant heavy work

        # Set global callbacks
        global _DATASET_PREVIEW_CALLBACK, _NODE_SELECTED_CALLBACK, _READER_CHANGED_CALLBACK
        _DATASET_PREVIEW_CALLBACK = self.dataset_loaded.emit
        _NODE_SELECTED_CALLBACK = self.node_selected.emit
        _READER_CHANGED_CALLBACK = self.reader_changed.emit
        
        # Set output callback for nodes
        self._view.set_output_callback(self._show_node_output)
        
        # Connect scene selection changed
        self._scene.selectionChanged.connect(self._on_selection_changed)

        # ──────────────────────────────────────────────────────────────
        # Main area: canvas
        # ──────────────────────────────────────────────────────────────
        canvas = QFrame(self)
        canvas.setObjectName("nodeCanvasFrame")
        canvas.setStyleSheet("""
            QFrame#nodeCanvasFrame {
                background: rgba(12, 18, 28, 180);
                border: 1px solid rgba(55, 110, 200, 35);
                border-radius: 12px;
            }
        """)
        cl = QVBoxLayout(canvas)
        cl.setContentsMargins(2, 2, 2, 2)
        cl.addWidget(self._view, 1)
        layout.addWidget(canvas, 1)

        self._seed_demo_nodes()

        # Apply persisted settings (grid + pipeline execution behavior)
        QTimer.singleShot(0, self.apply_settings)

    def _on_selection_changed(self) -> None:
        """Handle node selection changes - show selected node's data in preview."""
        # Check if scene is still valid
        try:
            if not shiboken6.isValid(self._scene):
                return
        except Exception:
            return
        try:
            selected = self._scene.selectedItems()  # Get selected items
        except RuntimeError:
            return

        for item in selected:
            if not isinstance(item, NodeItem):
                continue
            try:
                if not shiboken6.isValid(item):
                    continue
            except Exception:
                continue
            self._pending_selected_node = item
            # Debounce expensive panel updates during drag-select or quick click bursts.
            self._selection_preview_timer.start(120)
            return
        # No node selected
        self._current_selected_node = None
        global _EXTRA_READER_PARAMS
        _EXTRA_READER_PARAMS = {}
        self._pending_selected_node = None
        self._last_selection_emit_key = None

    def _flush_selection_preview(self) -> None:
        """Apply selection side effects after a short debounce window."""
        global _EXTRA_READER_PARAMS, _DATASET_PREVIEW_CALLBACK
        try:
            if hasattr(self, "_view") and self._view and self._view.is_selection_drag_active():
                # Keep deferring while drag-selection is in progress.
                self._selection_preview_timer.start(120)
                return
        except Exception:
            pass
        item = self._pending_selected_node
        if item is None:
            return
        try:
            if not shiboken6.isValid(item):
                return
        except Exception:
            return

        try:
            self._current_selected_node = item
            # Selection preview should show full node output by default, not prior link-filtered columns.
            try:
                self.columns_selected.emit([])
            except Exception:
                pass
            node_type = "Dataset Loader" if "Dataset" in item.title else "Other"
            extra_params = item.get_extra_params()
            _EXTRA_READER_PARAMS = extra_params.copy()
            self.node_selected.emit(item.title, node_type, extra_params)

            df = item.get_dataframe()
            if df is None:
                df = item.get_input_dataframe()

            outputs = getattr(item, "_runtime_outputs", None)
            if df is None and isinstance(outputs, dict):
                df = self._get_primary_output(outputs, item.title)

            preview_payload = self._build_preview_payload(item, df)

            if preview_payload is not None and _DATASET_PREVIEW_CALLBACK:
                _DATASET_PREVIEW_CALLBACK(preview_payload)

            if df is not None:
                emit_key = (str(getattr(item, "node_id", "")), id(df))
                if emit_key != self._last_selection_emit_key:
                    self.node_output_changed.emit(item.title, df)
                    self._last_selection_emit_key = emit_key
        except RuntimeError:
            return

    def _build_preview_payload(self, node: NodeItem, fallback_df):
        """Build a data preview payload, preserving separate split datasets."""
        outputs = getattr(node, "_runtime_outputs", None)
        if isinstance(outputs, dict) and "Split" in str(getattr(node, "title", "") or ""):
            split_keys = [
                "X_train", "X_val", "X_test",
                "y_train", "y_val", "y_test",
            ]
            split_payload = {k: outputs.get(k) for k in split_keys if outputs.get(k) is not None}
            if split_payload:
                return split_payload
        return fallback_df
    
    def _show_node_output(self, node_title: str, dataframe) -> None:
        """Show node output in the output panel (called by node's output/input button)."""
        self.node_output_changed.emit(node_title, dataframe)
        if dataframe is not None:
            self.dataset_loaded.emit(dataframe)
    
    def apply_extra_params(self, params: dict) -> None:
        """Apply extra parameters from Node Properties window to current Dataset Loader."""
        global _EXTRA_READER_PARAMS
        
        # Save params to the current selected node
        if self._current_selected_node:
            self._current_selected_node.set_extra_params(params)
        
        _EXTRA_READER_PARAMS = params.copy()
        # Trigger reload of current dataset with new params
        self._view.reload_current_dataset()

    def apply_column_dtype(self, column: str, dtype_key: str) -> None:
        """
        Directly change a dataframe column datatype and propagate downstream.
        This is triggered from the Data Preview panel.
        """
        if not column or not dtype_key:
            return

        # Pick best node to modify: selected node with data, else Dataset Loader, else any node with data
        target = None
        try:
            if self._current_selected_node and self._current_selected_node.get_dataframe() is not None:
                target = self._current_selected_node
        except Exception:
            target = None

        if target is None:
            try:
                for it in self._scene.items():
                    if isinstance(it, NodeItem) and it.title == "Dataset Loader" and it.get_dataframe() is not None:
                        target = it
                        break
            except Exception:
                target = None

        if target is None:
            try:
                for it in self._scene.items():
                    if isinstance(it, NodeItem) and it.get_dataframe() is not None:
                        target = it
                        break
            except Exception:
                target = None

        if target is None:
            return

        df = target.get_dataframe()
        if df is None or column not in df.columns:
            return

        try:
            import pandas as pd
        except Exception:
            return

        new_df = df.copy()
        s = new_df[column]

        try:
            if dtype_key == "float":
                new_df[column] = pd.to_numeric(s, errors="coerce")
            elif dtype_key == "int":
                new_df[column] = pd.to_numeric(s, errors="coerce").astype("Int64")
            elif dtype_key == "str":
                new_df[column] = s.astype(str)
            elif dtype_key == "datetime":
                new_df[column] = pd.to_datetime(s, errors="coerce")
            elif dtype_key == "category":
                new_df[column] = s.astype("category")
            elif dtype_key == "bool":
                # Robust bool conversion with common string/numeric handling
                if str(s.dtype) == "bool":
                    new_df[column] = s
                else:
                    try:
                        # numeric -> !=0
                        if pd.api.types.is_numeric_dtype(s):
                            new_df[column] = (s.fillna(0) != 0).astype("boolean")
                        else:
                            txt = s.astype(str).str.strip().str.lower()
                            true_set = {"true", "1", "yes", "y", "t", "on"}
                            false_set = {"false", "0", "no", "n", "f", "off"}
                            out = pd.Series(pd.NA, index=s.index, dtype="boolean")
                            out[txt.isin(true_set)] = True
                            out[txt.isin(false_set)] = False
                            new_df[column] = out
                    except Exception:
                        new_df[column] = s.astype("boolean")
            else:
                # Unknown type key
                return
        except Exception:
            return

        target.set_dataframe(new_df)

        # Update preview + connected panels
        self.dataset_loaded.emit(new_df)
        self.node_output_changed.emit(target.title, new_df)

        # Re-run downstream nodes so pipeline stays consistent (if enabled)
        try:
            if self._settings.get_bool(self._settings.PIPELINE_PROPAGATE):
                self._view._propagate_downstream(target)
        except Exception:
            pass

    def apply_settings(self) -> None:
        """Apply AppSettings to Node editor behavior."""
        try:
            grid_enabled = self._settings.get_bool(self._settings.GRID_NODE_ENABLED)
            grid_size = self._settings.get_int(self._settings.GRID_NODE_SIZE)
            if hasattr(self._view, "set_grid"):
                self._view.set_grid(grid_enabled, grid_size)
        except Exception:
            pass

    def shutdown_background_threads(self) -> None:
        """Stop background worker threads owned by this editor."""
        try:
            if hasattr(self, "_view") and self._view is not None and hasattr(self._view, "shutdown_engine_thread"):
                self._view.shutdown_engine_thread()
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.shutdown_background_threads()
        except Exception:
            pass
        super().closeEvent(event)

    def _seed_demo_nodes(self) -> None:
        # Create Dataset Loader node at the center as starting point
        self._view._last_context_pos = self._view.mapFromScene(0, 0)
        self._view._add_node("Dataset Loader")
        # Emit initial node list
        QTimer.singleShot(100, self._emit_nodes_changed)
    
    def _emit_nodes_changed(self, include_snapshot: bool = True) -> None:
        """Collect all nodes from the scene and emit nodes_changed signal."""
        try:
            if not shiboken6.isValid(self._scene):
                return
        except Exception:
            return
        
        node_list = []
        try:
            items = list(self._scene.items())
        except RuntimeError:
            return
        for item in items:
            if not isinstance(item, NodeItem):
                continue
            try:
                if not shiboken6.isValid(item):
                    continue
                has_data = item._loaded_dataframe is not None
                has_input = item._input_dataframe is not None
                
                if has_data:
                    status = "Active"
                elif has_input:
                    status = "Waiting"
                else:
                    status = "Idle"
                
                node_data = {
                    "node_id": item.node_id,
                    "node_name": item.title,
                    "cpu_percent": 0.0,
                    "gpu_percent": 0.0,
                    "ram_mb": 0.0,
                    "status": status,
                }
                node_list.append(node_data)
            except RuntimeError:
                continue
        
        self.nodes_changed.emit(node_list)

        # Keep recovery snapshots, but throttle because serialization is expensive.
        if include_snapshot:
            now = time.monotonic()
            if (now - self._last_snapshot_at) >= self._snapshot_min_interval_sec:
                try:
                    snap = self.to_dict(include_dataframes=False)
                    snap_nodes = len((snap or {}).get("nodes", []) or [])
                    prev_nodes = len((self._last_graph_snapshot or {}).get("nodes", []) or [])
                    if snap_nodes >= prev_nodes:
                        self._last_graph_snapshot = snap
                        self._last_snapshot_at = now
                except Exception:
                    pass

    def _refresh_nodes_view(self) -> None:
        """Rebuild the scene from current graph state and recover lost nodes from snapshot."""
        try:
            current = self.to_dict(include_dataframes=False)
        except Exception:
            current = {"nodes": [], "edges": []}

        current_count = len((current or {}).get("nodes", []) or [])
        backup = self._last_graph_snapshot or {"nodes": [], "edges": []}
        backup_count = len((backup or {}).get("nodes", []) or [])

        restore = current
        if backup_count > current_count:
            restore = backup

        if len((restore or {}).get("nodes", []) or []) == 0:
            return

        try:
            self.from_dict(restore)
            self._emit_nodes_changed()
        except Exception:
            pass
    
    def get_node_count(self) -> int:
        """Get the number of nodes in the scene."""
        count = 0
        for item in self._scene.items():
            if isinstance(item, NodeItem):
                count += 1
        return count

    # ═══════════════════════════════════════════════════════════════
    # Project serialization (Save/Open .typhyproj)
    # ═══════════════════════════════════════════════════════════════

    def to_dict(self, include_dataframes: bool = True) -> dict:
        """Serialize node graph (nodes + edges + per-node params/options)."""
        nodes: list[dict] = []
        edges: list[dict] = []

        # Nodes
        try:
            all_items = list(self._scene.items())
        except RuntimeError:
            return {"nodes": [], "edges": []}
        for item in all_items:
            if not isinstance(item, NodeItem):
                continue
            try:
                if not shiboken6.isValid(item):
                    continue
                pos = item.scenePos()
                df_blob = None
                if include_dataframes:
                    try:
                        df = item.get_dataframe()
                        if df is not None:
                            must_store = item.title in ("Dataset Loader", "Final Output")
                            df_blob = self._serialize_dataframe(df, force=must_store)
                    except Exception:
                        df_blob = None
                nodes.append(
                    {
                        "node_id": item.node_id,
                        "title": item.title,
                        "pos": [float(pos.x()), float(pos.y())],
                        "size": [int(getattr(item, "width", 300)), int(getattr(item, "height", 180))],
                        "extra_params": item.get_extra_params(),
                        "options": self._view._extract_node_options(item),
                        "dataframe": df_blob,
                    }
                )
            except RuntimeError:
                continue

        # Edges
        for item in all_items:
            if isinstance(item, EdgeItem):
                # Edges can hold stale Qt pointers if nodes/ports were removed.
                # Skip (and prune) invalid edges so project save never crashes.
                try:
                    if not shiboken6.isValid(item):
                        continue
                except Exception:
                    continue
                try:
                    src = item.get_source_port()
                    tgt = item.get_target_port()
                except Exception:
                    continue
                if not src or not tgt:
                    continue
                try:
                    if not shiboken6.isValid(src) or not shiboken6.isValid(tgt):
                        continue
                except Exception:
                    continue
                try:
                    src_node = src.parentItem()
                    tgt_node = tgt.parentItem()
                except Exception:
                    continue
                try:
                    if not shiboken6.isValid(src_node) or not shiboken6.isValid(tgt_node):
                        continue
                except Exception:
                    continue
                if not isinstance(src_node, NodeItem) or not isinstance(tgt_node, NodeItem):
                    continue

                columns = []
                data_type = getattr(src, "data_type", "any")
                if item.link:
                    try:
                        columns = list(item.link.columns_passed or [])
                        data_type = item.link.data_type or data_type
                    except Exception:
                        pass
                elif getattr(item, "columns", None):
                    columns = list(item.columns)

                edges.append(
                    {
                        "source": {
                            "node_id": src_node.node_id,
                            "port_name": src.name,
                        },
                        "target": {
                            "node_id": tgt_node.node_id,
                            "port_name": tgt.name,
                        },
                        "data_type": data_type,
                        "columns": columns,
                    }
                )

        return {"nodes": nodes, "edges": edges}

    def from_dict(self, data: dict) -> None:
        """Restore node graph from serialized data."""
        data = data or {}
        nodes_data = list(data.get("nodes", []) or [])
        edges_data = list(data.get("edges", []) or [])

        # Clear existing graph through scene API that safely permits node removal.
        try:
            if hasattr(self._scene, "clear_graph"):
                self._scene.clear_graph()
            else:
                self._scene.clear()
        except Exception:
            pass
        self._current_selected_node = None
        primary_df = None

        # Rebuild nodes
        id_to_node: dict[str, NodeItem] = {}
        for nd in nodes_data:
            title = nd.get("title", "Node")
            pos = nd.get("pos", [0, 0])
            x, y = float(pos[0]), float(pos[1])
            node = self._scene.add_node(title, x, y)

            # Preserve node_id for edge mapping
            saved_id = nd.get("node_id")
            if saved_id:
                node.node_id = str(saved_id)

            # Reapply callbacks & build ports/widgets
            node.set_on_select_callback(self._view._on_node_selected)
            if self._view._output_callback:
                node.set_output_callback(self._view._output_callback)
                node.set_input_callback(self._view._output_callback)

            # Apply node catalog UI (ports + controls)
            try:
                self._view._node_menu.apply_ports(node)
            except Exception:
                pass

            # Restore size
            try:
                size = nd.get("size", [300, 180])
                node.resize_to(int(size[0]), int(size[1]))
            except Exception:
                pass

            # Restore extra params
            try:
                node.set_extra_params(nd.get("extra_params", {}) or {})
            except Exception:
                pass

            # Restore option widget values
            try:
                self._apply_node_options(node, nd.get("options", {}) or {})
            except Exception:
                pass

            # Restore embedded dataframe snapshot (if present)
            try:
                df_blob = nd.get("dataframe")
                if df_blob:
                    df = self._deserialize_dataframe(df_blob)
                    if df is not None:
                        node.set_dataframe(df)
                        if title == "Dataset Loader" and primary_df is None:
                            primary_df = df
            except Exception:
                pass

            id_to_node[node.node_id] = node

        # Rebuild edges
        for ed in edges_data:
            src_info = ed.get("source", {}) or {}
            tgt_info = ed.get("target", {}) or {}
            src_node = id_to_node.get(src_info.get("node_id", ""))
            tgt_node = id_to_node.get(tgt_info.get("node_id", ""))
            if not src_node or not tgt_node:
                continue

            src_port = self._find_port_by_name(src_node, src_info.get("port_name", ""), is_output=True)
            tgt_port = self._find_port_by_name(tgt_node, tgt_info.get("port_name", ""), is_output=False)
            if not src_port or not tgt_port:
                continue

            edge = EdgeItem()
            try:
                edge.set_temporary(False)
            except Exception:
                pass
            try:
                edge.set_data_type(ed.get("data_type", getattr(src_port, "data_type", "any")))
            except Exception:
                pass

            self._scene.addItem(edge)
            edge.connect_ports(src_port, tgt_port)

            cols = list(ed.get("columns", []) or [])
            try:
                edge.set_columns(cols)
            except Exception:
                pass

            # Restore link model for edge styling/inspector
            try:
                from nodes.base.link_model import ColumnConfig, ColumnRole

                link = LinkModel()
                link.source_node_id = src_node.node_id
                link.source_node_name = src_node.title
                link.source_port_name = src_port.name
                link.source_port_id = getattr(src_port, "port_id", "")
                link.target_node_id = tgt_node.node_id
                link.target_node_name = tgt_node.title
                link.target_port_name = tgt_port.name
                link.target_port_id = getattr(tgt_port, "port_id", "")
                link.data_type = ed.get("data_type", getattr(src_port, "data_type", "any"))
                for c in cols:
                    link.columns.append(ColumnConfig(name=c, data_type=link.data_type, role=ColumnRole.FEATURE, enabled=True))
                link._update_counts()
                link.validate()
                edge.set_link_model(link)
            except Exception:
                pass

        # Emit node list for resource monitor
        QTimer.singleShot(50, self._emit_nodes_changed)

        # Repopulate downstream UI panels (Data Preview / Graph / Stats) immediately.
        # Prefer restored Dataset Loader output if present, otherwise fall back to any node with data.
        try:
            if primary_df is None:
                for it in self._scene.items():
                    if isinstance(it, NodeItem):
                        df = it.get_dataframe()
                        if df is not None:
                            primary_df = df
                            break
            if primary_df is not None:
                self.dataset_loaded.emit(primary_df)
        except Exception:
            pass

    def get_primary_dataframe(self):
        """Best-effort: return Dataset Loader dataframe if present, else any node output."""
        try:
            for it in self._scene.items():
                if isinstance(it, NodeItem) and it.title == "Dataset Loader":
                    df = it.get_dataframe()
                    if df is not None:
                        return df
            for it in self._scene.items():
                if isinstance(it, NodeItem):
                    df = it.get_dataframe()
                    if df is not None:
                        return df
        except Exception:
            return None
        return None

    def _serialize_dataframe(self, df, force: bool = False) -> dict | None:
        """
        Serialize a pandas DataFrame into a compact blob suitable for JSON project files.
        Uses JSON(split) -> gzip -> base64.
        If the dataframe is too large, stores only a head() preview unless force=True.
        """
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return None

        if not isinstance(df, pd.DataFrame):
            return None

        # Settings-driven embedding policy
        mode = "preview"
        max_mb = 25
        max_rows = 200_000
        preview_rows = 10_000
        compress_level = 6
        try:
            mode = str(self._settings.get_str(self._settings.EMBED_DATASET_MODE) or "preview").lower()
            max_mb = int(self._settings.get_int(self._settings.EMBED_MAX_MB))
            max_rows = int(self._settings.get_int(self._settings.EMBED_MAX_ROWS))
            preview_rows = int(self._settings.get_int(self._settings.EMBED_PREVIEW_ROWS))
            compress_level = int(self._settings.get_int(self._settings.EMBED_COMPRESSION_LEVEL))
        except Exception:
            pass

        if mode in ("none", "path"):
            return None

        MAX_EST_BYTES = max(1, max_mb) * 1024 * 1024
        MAX_ROWS = max(1000, max_rows)
        PREVIEW_ROWS = max(100, preview_rows)

        truncated = False
        try:
            est = int(df.memory_usage(deep=True).sum())
        except Exception:
            est = 0

        use_df = df
        if mode == "preview":
            # Always store a bounded snapshot unless force=True
            if not force:
                use_df = df.head(PREVIEW_ROWS).copy()
                truncated = True
        else:
            # full: store full unless too big and not forced
            if not force and (est > MAX_EST_BYTES or len(df) > MAX_ROWS):
                use_df = df.head(PREVIEW_ROWS).copy()
                truncated = True

        try:
            import json as _json
            import gzip as _gzip
            import base64 as _base64

            payload = {
                "orient": "split",
                "dtypes": {c: str(t) for c, t in use_df.dtypes.items()},
                "rows": int(use_df.shape[0]),
                "cols": int(use_df.shape[1]),
                "truncated": bool(truncated),
            }
            js = use_df.to_json(orient="split", date_format="iso")
            raw = _json.dumps({"meta": payload, "data": js}).encode("utf-8")
            gz = _gzip.compress(raw, compresslevel=max(1, min(9, compress_level)))
            b64 = _base64.b64encode(gz).decode("utf-8")
            return {"format": "pandas-json-split+gzip+base64", "b64": b64}
        except Exception:
            return None

    def _deserialize_dataframe(self, blob: dict):
        """Inverse of _serialize_dataframe()."""
        if not isinstance(blob, dict):
            return None
        if blob.get("format") != "pandas-json-split+gzip+base64":
            return None
        b64 = blob.get("b64", "")
        if not b64:
            return None

        try:
            import pandas as pd  # type: ignore
            import json as _json
            import gzip as _gzip
            import base64 as _base64
            from io import StringIO

            raw = _gzip.decompress(_base64.b64decode(b64))
            obj = _json.loads(raw.decode("utf-8"))
            js = obj.get("data", "")
            if not js:
                return None
            # Pandas deprecates passing literal JSON strings directly.
            df = pd.read_json(StringIO(js), orient="split")

            # Best-effort dtype restoration
            meta = obj.get("meta", {}) or {}
            dtypes = meta.get("dtypes", {}) or {}
            for col, dtype_str in dtypes.items():
                if col in df.columns:
                    try:
                        # Don't aggressively coerce datetimes here; pandas may already infer them.
                        df[col] = df[col].astype(dtype_str)
                    except Exception:
                        pass
            return df
        except Exception:
            return None

    def _find_port_by_name(self, node: NodeItem, port_name: str, is_output: bool) -> PortItem | None:
        ports = getattr(node, "_outputs" if is_output else "_inputs", []) or []
        for p in ports:
            try:
                if getattr(p, "name", "") == port_name and bool(getattr(p, "is_output", False)) == bool(is_output):
                    return p
            except Exception:
                continue
        # fallback: first port
        return ports[0] if ports else None

    def _apply_node_options(self, node: NodeItem, options: dict) -> None:
        """Apply extracted options back into node control widgets by objectName."""
        if not options:
            return
        root = getattr(node, "_controls_widget", None)
        if not root:
            return

        from PySide6.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QSlider

        def walk(w):
            # Apply value if objectName matches
            key = w.objectName() if hasattr(w, "objectName") else ""
            if key and key in options:
                val = options[key]
                try:
                    if isinstance(w, QLineEdit):
                        w.setText(str(val))
                    elif isinstance(w, QCheckBox):
                        w.setChecked(bool(val))
                    elif isinstance(w, QComboBox):
                        # set by text if exists
                        idx = w.findText(str(val))
                        if idx >= 0:
                            w.setCurrentIndex(idx)
                    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                        w.setValue(float(val))
                    elif isinstance(w, QSlider):
                        w.setValue(int(val))
                    elif hasattr(w, "setCurrentText"):
                        # ComboButton-like
                        w.setCurrentText(str(val))
                except Exception:
                    pass

            # Recurse into children widgets
            for ch in w.children():
                if hasattr(ch, "isWidgetType") and ch.isWidgetType():
                    walk(ch)

        walk(root)


class NodeGraphView(QGraphicsView):
    zoom_changed = Signal(float)
    grid_changed = Signal(bool, int)
    engine_execute_requested = Signal(dict)
    pipeline_execute_requested = Signal(dict)

    def __init__(self, scene) -> None:
        super().__init__(scene)
        from ui.app_settings import AppSettings
        _s = AppSettings()
        self._zoom = 1.0
        self._zoom_min = 0.3
        self._zoom_max = 2.2  # hard max zoom limit
        self._panning = False
        self._last_pan_point = None
        self._inner_grid_zoom_1 = 1.25
        self._inner_grid_zoom_2 = 1.85
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self._last_context_pos = None
        self._node_menu = NodeMenuDialog(self)
        self._drag_edge = None
        self._drag_start_port = None
        self._drag_start_pos = None
        self._link_inspector = LinkInspectorDialog(self)
        self.setDragMode(QGraphicsView.NoDrag)
        self._columns_selected_callback = None
        self._output_callback = None
        self._nodes_changed_callback = None
        self._max_propagation_depth = 128
        self._last_input_highlight_at = 0.0
        self._selection_drag_active = False
        self._engine_request_seq = 0
        self._latest_request_by_node: dict[str, int] = {}
        self._pipeline_request_seq = 0
        self._latest_pipeline_request_id = 0
        self._pipeline_inflight = False
        self._pending_pipeline_payload: dict | None = None
        self._pipeline_debounce_timer = QTimer(self)
        self._pipeline_debounce_timer.setSingleShot(True)
        self._pipeline_debounce_timer.timeout.connect(self._flush_pipeline_request)
        self._engine_thread = QThread(self)
        self._engine_worker = _EngineWorker()
        self._engine_worker.moveToThread(self._engine_thread)
        self.engine_execute_requested.connect(self._engine_worker.execute, Qt.QueuedConnection)
        self.pipeline_execute_requested.connect(self._engine_worker.execute_pipeline, Qt.QueuedConnection)
        self._engine_worker.execution_finished.connect(self._on_engine_result, Qt.QueuedConnection)
        self._engine_worker.pipeline_finished.connect(self._on_pipeline_result, Qt.QueuedConnection)
        self._engine_thread.start()
        self.destroyed.connect(self._shutdown_engine_thread)
        # Grid behavior comes from global app settings
        try:
            self._grid_enabled = _s.get_bool(_s.GRID_NODE_ENABLED)
        except Exception:
            self._grid_enabled = True
        try:
            self._grid_size = max(10, int(_s.get_int(_s.GRID_NODE_SIZE)))
        except Exception:
            self._grid_size = 40
        # Smooth rendering
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        # Enable proper interaction with embedded widgets
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setInteractive(True)
        # Allow scene items to receive focus
        scene.setFocusOnTouch(True)

    @Slot()
    def _shutdown_engine_thread(self, *_args) -> None:
        try:
            if self._engine_thread and self._engine_thread.isRunning():
                self._engine_thread.quit()
                self._engine_thread.wait(1000)
        except Exception:
            pass

    def shutdown_engine_thread(self) -> None:
        """Public shutdown hook for owner widgets/windows."""
        self._shutdown_engine_thread()

    # ──────────────────────────────────────────────────────────────────
    # Public UI actions (used by redesigned toolbar/palette)
    # ──────────────────────────────────────────────────────────────────
    def open_node_menu_at_cursor(self) -> None:
        self._last_context_pos = self.mapFromGlobal(self.cursor().pos())
        self._open_node_menu()

    def add_node_by_title(self, title: str) -> None:
        self._ensure_context_pos()
        node = self._add_node(title)
        try:
            if node is not None:
                self.centerOn(node)
        except Exception:
            pass

    def clear_graph_confirmed(self) -> None:
        try:
            from PySide6.QtWidgets import QMessageBox
            ans = QMessageBox.question(
                self,
                "Clear Graph",
                "Clear the entire graph (all nodes and links)?\n\nThis cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        except Exception:
            return

        try:
            sc = self.scene()
            if hasattr(sc, "clear_graph"):
                sc.clear_graph()
            else:
                sc.clear()
        except Exception:
            pass

        try:
            self._zoom = 1.0
            self.resetTransform()
            self.zoom_changed.emit(float(self._zoom))
        except Exception:
            pass

        if self._nodes_changed_callback:
            try:
                self._nodes_changed_callback()
            except Exception:
                pass

    def zoom_in(self) -> None:
        self._apply_zoom_factor(1.15)

    def zoom_out(self) -> None:
        self._apply_zoom_factor(1 / 1.15)

    def zoom_reset(self) -> None:
        try:
            self.resetTransform()
            self._zoom = 1.0
            self.viewport().update()
            self.zoom_changed.emit(float(self._zoom))
        except Exception:
            pass

    def fit_to_scene(self) -> None:
        try:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self._zoom = float(self.transform().m11()) if self.transform() is not None else 1.0
            self.viewport().update()
            self.zoom_changed.emit(float(self._zoom))
        except Exception:
            pass

    def center_scene(self) -> None:
        try:
            self.centerOn(0, 0)
        except Exception:
            try:
                self.centerOn(self.sceneRect().center())
            except Exception:
                pass

    def _ensure_context_pos(self) -> None:
        if self._last_context_pos is None:
            try:
                self._last_context_pos = self.viewport().rect().center()
            except Exception:
                self._last_context_pos = QPoint(10, 10)

    def set_columns_selected_callback(self, callback) -> None:
        self._columns_selected_callback = callback
    
    def set_output_callback(self, callback) -> None:
        """Set callback for when node output button is clicked."""
        self._output_callback = callback
    
    def set_nodes_changed_callback(self, callback) -> None:
        """Set callback for when nodes are added/removed."""
        self._nodes_changed_callback = callback
    
    def reload_current_dataset(self) -> None:
        """Reload the current dataset with updated parameters."""
        # Find all Dataset Loader nodes and trigger reload
        for item in self.scene().items():
            if isinstance(item, NodeItem) and item.title == "Dataset Loader":
                # Find the config widget in the node's controls
                if item._controls_widget:
                    for i in range(item._controls_layout.count()):
                        widget = item._controls_layout.itemAt(i).widget()
                        if isinstance(widget, DatasetLoaderConfigWidget):
                            widget._load_preview()
                            break

    def wheelEvent(self, event) -> None:
        # Let embedded widgets (dropdowns, scrolls) handle wheel events
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        item = self.itemAt(pos)
        if self._is_on_proxy_widget(item):
            super().wheelEvent(event)
            return
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 1 / 1.15
        self._apply_zoom_factor(factor)
        event.accept()

    def _apply_zoom_factor(self, factor: float) -> None:
        try:
            target_zoom = float(self._zoom) * float(factor)
            clamped_zoom = max(self._zoom_min, min(self._zoom_max, target_zoom))
            if abs(clamped_zoom - float(self._zoom)) < 1e-9:
                return
            applied_factor = clamped_zoom / float(self._zoom)
            self._zoom = clamped_zoom
            self.scale(applied_factor, applied_factor)
            self.viewport().update()
            self.zoom_changed.emit(float(self._zoom))
        except Exception:
            pass
    
    def _is_on_proxy_widget(self, item) -> bool:
        """Check if item is a proxy widget or child of a proxy widget."""
        return self._find_proxy_widget(item) is not None
    
    def _find_proxy_widget(self, item) -> QGraphicsProxyWidget | None:
        """Find the proxy widget that contains this item."""
        if item is None:
            return None
        if isinstance(item, QGraphicsProxyWidget):
            return item
        # Check parent chain for proxy widget
        parent = item.parentItem() if hasattr(item, 'parentItem') else None
        while parent:
            if isinstance(parent, QGraphicsProxyWidget):
                return parent
            parent = parent.parentItem() if hasattr(parent, 'parentItem') else None
        return None

    def mousePressEvent(self, event) -> None:
        try:
            # Check if click is on any embedded widget (proxy or its children)
            item = self.itemAt(event.pos())
            proxy = self._find_proxy_widget(item)
            if proxy:
                try:
                    if shiboken6.isValid(proxy):
                        proxy.setFocus(Qt.MouseFocusReason)
                        self.scene().setFocusItem(proxy)
                        widget = proxy.widget()
                        if widget and shiboken6.isValid(widget):
                            scene_pos = self.mapToScene(event.pos())
                            proxy_pos = proxy.mapFromScene(scene_pos)
                            child = widget.childAt(int(proxy_pos.x()), int(proxy_pos.y()))
                            if child and shiboken6.isValid(child):
                                child.setFocus(Qt.MouseFocusReason)
                                if isinstance(child, QLineEdit):
                                    child.setCursorPosition(len(child.text()))
                                child.activateWindow()
                except RuntimeError:
                    pass
                super().mousePressEvent(event)
                return
            if event.button() == Qt.MiddleButton:
                self._panning = True
                self._last_pan_point = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                self._selection_drag_active = item is None
                if isinstance(item, EdgeItem) and shiboken6.isValid(item):
                    self._link_inspector.set_link(item)
                    self._link_inspector.show()
                    return
                if isinstance(item, PortItem) and item.is_output and shiboken6.isValid(item):
                    self._selection_drag_active = False
                    self._start_connection(item, event.pos())
                    return
        except RuntimeError:
            pass
        super().mousePressEvent(event)

    def _show_context_menu(self, pos: QPoint) -> None:
        self._last_context_pos = pos
        item = self.itemAt(pos)
        if item is None:
            self._open_node_menu()
            return

        menu = QMenu(self)
        node_item = self._find_node_item(item)
        if node_item is not None:
            menu.addAction("Delete Node", lambda: self._confirm_delete_node(node_item))
            menu.addSeparator()

        menu.addAction("Add Node...", self._open_node_menu)
        menu.addAction("Cancel", lambda: None)
        menu.exec(self.mapToGlobal(pos))

    def _confirm_delete_node(self, node: NodeItem) -> None:
        """Delete only after explicit user confirmation to prevent accidental loss."""
        if node is None or node.scene() is None:
            return
        name = str(getattr(node, "title", "this node") or "this node")
        choice = QMessageBox.question(
            self,
            "Delete Node",
            f"Delete '{name}'?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice == QMessageBox.Yes:
            self._delete_node(node, reason="context_menu_confirmed")

    def keyPressEvent(self, event) -> None:
        # Check for Shift+A to open node menu
        if event.key() == Qt.Key_A and event.modifiers() == Qt.ShiftModifier:
            self._last_context_pos = self.mapFromGlobal(self.cursor().pos())
            self._open_node_menu()
            return
        # Hard safety lock: disable keyboard node deletion entirely.
        # Node deletion is only allowed from the context menu with confirmation.
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            event.accept()
            return
        super().keyPressEvent(event)

    def _open_node_menu(self) -> None:
        if self._last_context_pos is None:
            self._last_context_pos = self.mapFromGlobal(self.cursor().pos())
        self._node_menu.open_at_cursor()

    def _add_node(self, title: str) -> NodeItem:
        if self._last_context_pos is None:
            self._ensure_context_pos()
        scene_pos = self.mapToScene(self._last_context_pos)
        node = self.scene().add_node(title, scene_pos.x(), scene_pos.y())
        # Set callback for when node is selected to show data in preview
        node.set_on_select_callback(self._on_node_selected)
        # Set callback for output and input button clicks
        if self._output_callback:
            node.set_output_callback(self._output_callback)
            node.set_input_callback(self._output_callback)
        self._node_menu.apply_ports(node)
        # Notify that nodes have changed
        if self._nodes_changed_callback:
            self._nodes_changed_callback()
        return node
    
    def _on_node_selected(self, dataframe) -> None:
        """Called when a node with data is selected."""
        global _DATASET_PREVIEW_CALLBACK
        if _DATASET_PREVIEW_CALLBACK and dataframe is not None:
            _DATASET_PREVIEW_CALLBACK(dataframe)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._last_pan_point is not None:
            delta = event.pos() - self._last_pan_point
            self._last_pan_point = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        if self._drag_edge:
            try:
                if not shiboken6.isValid(self._drag_edge):
                    self._cleanup_drag()
                    return
                end = self.mapToScene(event.pos())
                self._drag_edge.set_points(self._drag_start_pos, end)
                now = time.monotonic()
                if (now - self._last_input_highlight_at) >= 0.05:
                    self._highlight_inputs(True)
                    self._last_input_highlight_at = now
            except RuntimeError:
                self._cleanup_drag()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._selection_drag_active = False
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._last_pan_point = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        if self._drag_edge and self._drag_start_port:
            try:
                item = self.itemAt(event.pos())
                # Validate the drag_start_port is still alive
                if not shiboken6.isValid(self._drag_start_port):
                    self._safe_remove_drag_edge()
                    self._cleanup_drag()
                    return
                if isinstance(item, PortItem) and not item.is_output:
                    if not shiboken6.isValid(item):
                        self._safe_remove_drag_edge()
                        self._cleanup_drag()
                        return
                    if not self._is_compatible(self._drag_start_port, item):
                        self._safe_remove_drag_edge()
                        self._cleanup_drag()
                        return
                    
                    # Get source node for column data
                    source_node = self._drag_start_port.parentItem()
                    if not source_node or not shiboken6.isValid(source_node):
                        self._safe_remove_drag_edge()
                        self._cleanup_drag()
                        return
                    
                    # Get source DataFrame
                    source_df = source_node.get_dataframe()
                    
                    # If source has no data yet, allow connection with all columns selected by default
                    if source_df is None or (hasattr(source_df, 'empty') and source_df.empty):
                        column_configs = [{"name": "*", "data_type": "numeric", "role": "feature", "enabled": True}]
                        column_names = ["*"]
                    else:
                        # Show enhanced column selection popup
                        column_configs = self._column_popup(source_node)
                        if not column_configs:
                            self._safe_remove_drag_edge()
                            self._cleanup_drag()
                            return
                        column_names = [c["name"] for c in column_configs]
                    
                    # Build link model with full config
                    link_model = self._build_link_model(
                        self._drag_start_port,
                        item,
                        column_configs,
                    )
                    
                    # Update edge with link data
                    self._drag_edge.set_temporary(False)
                    self._drag_edge.set_link_model(link_model)
                    
                    # Connect ports for dynamic position updates
                    self._drag_edge.connect_ports(self._drag_start_port, item)
                    
                    if self._columns_selected_callback:
                        self._columns_selected_callback(column_names)

                    # Immediately preview the exact selected columns from source.
                    # This avoids momentary/full-data previews overriding link selection.
                    global _DATASET_PREVIEW_CALLBACK
                    if _DATASET_PREVIEW_CALLBACK and source_df is not None:
                        try:
                            preview_df = self._build_link_preview_dataframe(source_df, column_names)
                            _DATASET_PREVIEW_CALLBACK(preview_df)
                        except Exception:
                            pass
                    
                    # Propagate data from source to target node
                    target_node = item.parentItem()
                    if target_node and shiboken6.isValid(target_node):
                        if self._would_create_cycle(source_node, target_node):
                            QMessageBox.warning(
                                self,
                                "Invalid Link",
                                "This connection would create a cycle in the pipeline graph.\n"
                                "Cyclic execution can freeze the UI, so this link was blocked.",
                            )
                            self._safe_remove_drag_edge()
                            self._cleanup_drag()
                            return
                        self._propagate_data(
                            source_node,
                            target_node,
                            column_names,
                            target_port=item,
                            source_port=self._drag_start_port,
                        )
                else:
                    self._safe_remove_drag_edge()
            except RuntimeError:
                # Qt object deleted during drag — just clean up safely
                pass
            except Exception:
                pass
            self._cleanup_drag()
            return
        super().mouseReleaseEvent(event)

    def is_selection_drag_active(self) -> bool:
        return bool(self._selection_drag_active)

    def _safe_remove_drag_edge(self) -> None:
        """Safely remove the temporary drag edge from the scene."""
        try:
            if self._drag_edge and shiboken6.isValid(self._drag_edge):
                sc = self.scene()
                if sc and shiboken6.isValid(sc):
                    sc.removeItem(self._drag_edge)
        except (RuntimeError, Exception):
            pass

    def _build_link_preview_dataframe(self, source_df, column_names: list[str]):
        """Return a preview dataframe matching link-selected columns."""
        if source_df is None:
            return None
        if isinstance(source_df, pd.Series):
            source_df = source_df.to_frame()
        if not isinstance(source_df, pd.DataFrame):
            return source_df

        if not column_names or "*" in column_names:
            return source_df

        selected = [c for c in column_names if c in source_df.columns]
        if not selected:
            return source_df
        return source_df[selected].copy()

    def drawBackground(self, painter: QPainter, rect) -> None:
        # Gradient background
        from PySide6.QtGui import QLinearGradient, QBrush
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QColor(14, 20, 30))
        grad.setColorAt(0.5, QColor(18, 26, 38))
        grad.setColorAt(1.0, QColor(12, 18, 28))
        painter.fillRect(rect, QBrush(grad))

        if not getattr(self, "_grid_enabled", True):
            return
        painter.save()
        current_zoom = float(self.transform().m11()) if self.transform() is not None else 1.0
        grid = max(10, int(getattr(self, "_grid_size", 40)))
        left = int(rect.left()) - (int(rect.left()) % grid)
        top = int(rect.top()) - (int(rect.top()) % grid)
        # Point-grid background (major + minor + zoom-based inner-grid)
        major_step = grid * 4

        # Draw inner sub-grid only when zoomed in
        if current_zoom >= self._inner_grid_zoom_1:
            inner_grid = max(6, grid // 2)
            if current_zoom >= self._inner_grid_zoom_2:
                inner_grid = max(4, grid // 3)
            il = int(rect.left()) - (int(rect.left()) % inner_grid)
            it = int(rect.top()) - (int(rect.top()) % inner_grid)
            painter.setPen(QPen(QColor(80, 150, 235, 30), 1))
            for x in range(il, int(rect.right()), inner_grid):
                for y in range(it, int(rect.bottom()), inner_grid):
                    painter.drawPoint(x, y)

        for x in range(left, int(rect.right()), grid):
            for y in range(top, int(rect.bottom()), grid):
                is_major = (x % major_step == 0) and (y % major_step == 0)
                if is_major:
                    painter.setPen(QPen(QColor(110, 185, 255, 105), 3))
                else:
                    painter.setPen(QPen(QColor(85, 155, 235, 62), 2))
                painter.drawPoint(x, y)
        painter.restore()

    def set_grid(self, enabled: bool, grid_size: int) -> None:
        self._grid_enabled = bool(enabled)
        try:
            self._grid_size = max(10, int(grid_size))
        except Exception:
            self._grid_size = 40
        self.viewport().update()
        try:
            self.grid_changed.emit(bool(self._grid_enabled), int(self._grid_size))
        except Exception:
            pass

    def _start_connection(self, port: PortItem, pos: QPoint) -> None:
        self._drag_start_port = port
        start = port.scenePos()
        self._drag_start_pos = start
        edge = EdgeItem()
        edge.set_temporary(True)  # Mark as temporary during drag
        edge.set_data_type(port.data_type)  # Use port's data type for color
        edge.set_points(start, start)
        self.scene().addItem(edge)
        self._drag_edge = edge

    def _find_node_item(self, item):
        current = item
        while current is not None:
            if isinstance(current, NodeItem):
                return current
            current = current.parentItem()
        return None

    def _delete_node(self, node: NodeItem, reason: str = "unknown") -> None:
        """Delete a node and its connected edges.

        Safety policy: deletion is allowed only from explicit, confirmed context menu action.
        """
        if reason != "context_menu_confirmed":
            return
        if node is None:
            return
        # Never delete while a connection drag is active.
        if getattr(self, "_drag_edge", None) is not None or getattr(self, "_drag_start_port", None) is not None:
            return
        # Safety check - ensure Qt object and scene are still valid.
        try:
            if not shiboken6.isValid(node):
                return
        except Exception:
            return
        if node.scene() is None:
            return

        all_ports = list(getattr(node, "_inputs", [])) + list(getattr(node, "_outputs", []))

        # Remove connected edges first (check by link model AND by port reference)
        try:
            scene_items = list(self.scene().items())
        except Exception:
            scene_items = []
        for item in scene_items:
            if not isinstance(item, EdgeItem):
                continue
            try:
                if not shiboken6.isValid(item):
                    continue
                remove = False
                # Check via link model
                if item.link:
                    if item.link.source_node_id == node.node_id or item.link.target_node_id == node.node_id:
                        remove = True
                # Check via port references
                if not remove:
                    src = item.get_source_port()
                    tgt = item.get_target_port()
                    if src in all_ports or tgt in all_ports:
                        remove = True
                if remove:
                    try:
                        if hasattr(item, "disconnect_ports"):
                            item.disconnect_ports()
                    except Exception:
                        pass
                    self.scene().removeItem(item)
            except (RuntimeError, Exception):
                continue
        # Remove the node
        try:
            sc = self.scene()
            if hasattr(sc, "remove_node"):
                sc.remove_node(node)
            else:
                sc.removeItem(node)
        except Exception:
            return
        # Notify that nodes have changed
        if self._nodes_changed_callback:
            self._nodes_changed_callback()

    def _highlight_inputs(self, enabled: bool) -> None:
        for item in self.scene().items():
            if isinstance(item, PortItem) and not item.is_output:
                valid = True
                if enabled and self._drag_start_port:
                    valid = self._is_compatible(self._drag_start_port, item)
                item.set_highlight(enabled, valid=valid)

    def _cleanup_drag(self) -> None:
        self._drag_edge = None
        self._drag_start_port = None
        self._drag_start_pos = None
        self._highlight_inputs(False)
    
    def _propagate_data(self, source_node, target_node, columns: list[str], target_port=None, source_port=None) -> None:
        """Propagate data from source node to target node and execute target's logic."""
        import pandas as pd
        from nodes.registry import get_node_runtime

        # Safety: ensure both nodes are still alive
        try:
            if not shiboken6.isValid(source_node) or not shiboken6.isValid(target_node):
                return
        except Exception:
            return

        # Resolve payload from the specific source port when possible.
        payload = self._resolve_source_payload(source_node, source_port)
        if payload is None:
            return

        # Keep a DataFrame copy for nodes that expect table-like input.
        input_df = None
        if isinstance(payload, pd.DataFrame):
            source_df = payload
            # Filter to selected columns if specified
            if columns and columns != ["*"]:
                available_cols = [c for c in columns if c in source_df.columns]
                if available_cols:
                    input_df = source_df[available_cols].copy()
                else:
                    input_df = source_df.copy()
            else:
                input_df = source_df.copy()
        elif isinstance(payload, pd.Series):
            input_df = payload.to_frame().copy()

        node_title = str(getattr(target_node, "title", "") or "")
        is_column_joiner = node_title in ("Column Joiner", "DataFrame Joiner")
        execution_input_df = input_df
        runtime_inputs = None

        if is_column_joiner:
            if input_df is None:
                return
            # Collect per-port inputs so this node can join 2+ incoming DataFrames.
            port_name = ""
            try:
                port_name = str(getattr(target_port, "name", "") or "")
            except Exception:
                port_name = ""
            if not port_name:
                port_name = "Input"

            try:
                if not hasattr(target_node, "_column_joiner_inputs") or not isinstance(target_node._column_joiner_inputs, dict):
                    target_node._column_joiner_inputs = {}
                target_node._column_joiner_inputs[port_name] = input_df.copy()
                input_store = target_node._column_joiner_inputs
            except Exception:
                input_store = {port_name: input_df.copy()}

            # Require at least the first 2 declared input ports before executing.
            required_ports: list[str] = []
            try:
                required_ports = [p.name for p in getattr(target_node, "_inputs", []) if getattr(p, "name", "")]
            except Exception:
                required_ports = []
            required_ports = required_ports[:2] if required_ports else ["Left DataFrame", "Right DataFrame"]
            if not all(rp in input_store for rp in required_ports):
                return

            # Build ordered input frames (required ports first, then any extras).
            options = self._extract_node_options(target_node)
            reset_index = bool(options.get("Reset Index", True))
            frames: list[pd.DataFrame] = []
            ordered_ports = required_ports + [k for k in input_store.keys() if k not in required_ports]
            for pname in ordered_ports:
                frame = input_store.get(pname)
                if frame is None:
                    continue
                try:
                    f = frame.copy()
                    if reset_index:
                        f = f.reset_index(drop=True)
                except Exception:
                    continue
                frames.append(f)

            if not frames:
                return
            joined_input_df = pd.concat(frames, axis=1)
            target_node.set_input_dataframe(joined_input_df)
            execution_input_df = joined_input_df
            runtime_inputs = self._prepare_runtime_inputs(node_title, execution_input_df, target_node)
        else:
            # Store input dataframe in target node (for input view button) when available.
            if input_df is not None:
                target_node.set_input_dataframe(input_df)
            runtime_inputs = self._prepare_runtime_inputs(node_title, execution_input_df, target_node)

        if runtime_inputs is None:
            runtime_inputs = self._prepare_runtime_inputs(node_title, execution_input_df, target_node)

        # Use dedicated engine thread for runtime-backed nodes.
        if get_node_runtime(node_title) is not None:
            options = self._extract_node_options(target_node)
            self._submit_engine_request(
                target_node=target_node,
                title=node_title,
                input_df=execution_input_df,
                inputs=runtime_inputs,
                options=options,
            )
            return

        # Fallback for inline/basic nodes remains synchronous.
        output_df = self._execute_node_logic(target_node, execution_input_df, explicit_inputs=runtime_inputs)
        self._apply_node_result(target_node, output_df, getattr(target_node, "_runtime_outputs", None), trigger_downstream=True)
    
    def _propagate_downstream(self, source_node, ancestry: set[str] | None = None, depth: int = 0) -> None:
        """Propagate from a source node using engine PipelineExecutor in background."""
        try:
            if not shiboken6.isValid(source_node):
                return
        except Exception:
            return

        source_df = source_node.get_dataframe()
        if source_df is None:
            return
        graph_payload, initial_inputs = self._build_runtime_subgraph_payload(source_node, source_df)
        if not graph_payload.get("nodes"):
            return
        self._submit_pipeline_request(graph_payload, initial_inputs)

    def _build_runtime_subgraph_payload(self, source_node, source_df):
        from nodes.base.edge import EdgeItem

        source_id = str(getattr(source_node, "node_id", "") or "")
        if not source_id:
            return {"nodes": [], "edges": []}, {}

        node_map: dict[str, NodeItem] = {}
        adjacency: dict[str, set[str]] = {}
        edges_payload: list[dict] = []

        try:
            scene_items = list(self.scene().items())
        except Exception:
            return {"nodes": [], "edges": []}, {}

        for item in scene_items:
            if isinstance(item, NodeItem):
                try:
                    if shiboken6.isValid(item):
                        nid = str(getattr(item, "node_id", "") or "")
                        if nid:
                            node_map[nid] = item
                            adjacency.setdefault(nid, set())
                except Exception:
                    continue

        for item in scene_items:
            if not isinstance(item, EdgeItem):
                continue
            try:
                if not shiboken6.isValid(item):
                    continue
                src_port = item.get_source_port()
                tgt_port = item.get_target_port()
                if not src_port or not tgt_port:
                    continue
                if not shiboken6.isValid(src_port) or not shiboken6.isValid(tgt_port):
                    continue
                src_node = src_port.parentItem()
                tgt_node = tgt_port.parentItem()
                if not isinstance(src_node, NodeItem) or not isinstance(tgt_node, NodeItem):
                    continue
                src_id = str(getattr(src_node, "node_id", "") or "")
                tgt_id = str(getattr(tgt_node, "node_id", "") or "")
                if not src_id or not tgt_id:
                    continue
                adjacency.setdefault(src_id, set()).add(tgt_id)
                adjacency.setdefault(tgt_id, set())
                cols = []
                try:
                    cols = list(getattr(item.link, "columns_passed", []) or [])
                except Exception:
                    cols = list(getattr(item, "columns", []) or [])
                edges_payload.append(
                    {
                        "source": {"node_id": src_id, "port_name": str(getattr(src_port, "name", "") or "")},
                        "target": {"node_id": tgt_id, "port_name": str(getattr(tgt_port, "name", "") or "")},
                        "columns": cols,
                        "data_type": str(getattr(item.link, "data_type", getattr(src_port, "data_type", "any")) or "any"),
                    }
                )
            except Exception:
                continue

        reachable: set[str] = set()
        stack = [source_id]
        while stack:
            cur = stack.pop()
            if cur in reachable:
                continue
            reachable.add(cur)
            for nxt in adjacency.get(cur, set()):
                if nxt not in reachable:
                    stack.append(nxt)

        nodes_payload: list[dict] = []
        for nid in sorted(reachable):
            node = node_map.get(nid)
            if node is None:
                continue
            nodes_payload.append(
                {
                    "node_id": nid,
                    "title": str(getattr(node, "title", "") or ""),
                    "options": self._extract_node_options(node),
                }
            )

        reachable_edges = [
            e for e in edges_payload
            if str((e.get("source") or {}).get("node_id", "")) in reachable
            and str((e.get("target") or {}).get("node_id", "")) in reachable
        ]

        initial_inputs = {
            source_id: {
                "Data": source_df,
                "Chunk": source_df,
            }
        }
        return {"nodes": nodes_payload, "edges": reachable_edges}, initial_inputs

    def _submit_pipeline_request(self, graph_payload: dict, initial_inputs_by_node: dict) -> None:
        self._pending_pipeline_payload = {
            "graph": graph_payload,
            "initial_inputs_by_node": initial_inputs_by_node,
        }
        if self._pipeline_inflight:
            return
        self._pipeline_debounce_timer.start(90)

    def _flush_pipeline_request(self) -> None:
        if self._pipeline_inflight:
            return
        pending = self._pending_pipeline_payload
        if not pending:
            return
        self._pending_pipeline_payload = None
        self._pipeline_request_seq += 1
        request_id = int(self._pipeline_request_seq)
        self._latest_pipeline_request_id = request_id
        self._pipeline_inflight = True
        self.pipeline_execute_requested.emit(
            {
                "request_id": request_id,
                "graph": pending.get("graph", {}) or {},
                "initial_inputs_by_node": pending.get("initial_inputs_by_node", {}) or {},
            }
        )

    @Slot(dict)
    def _on_pipeline_result(self, result: dict) -> None:
        request_id = int(result.get("request_id", 0))
        self._pipeline_inflight = False
        if request_id != self._latest_pipeline_request_id:
            if self._pending_pipeline_payload:
                self._pipeline_debounce_timer.start(30)
            return
        node_results = result.get("node_results", {}) or {}
        if not isinstance(node_results, dict):
            if self._pending_pipeline_payload:
                self._pipeline_debounce_timer.start(30)
            return

        for node_id, res in node_results.items():
            if not isinstance(res, dict):
                continue
            target_node = self._find_node_by_id(str(node_id))
            if target_node is None:
                continue
            output_df = res.get("primary_output")
            self._apply_node_result(target_node, output_df, None, trigger_downstream=False)

        if self._pending_pipeline_payload:
            self._pipeline_debounce_timer.start(30)

    def _submit_engine_request(self, target_node, title: str, input_df, inputs: dict, options: dict) -> None:
        try:
            if not shiboken6.isValid(target_node):
                return
        except Exception:
            return
        node_id = str(getattr(target_node, "node_id", "") or "")
        if not node_id:
            return
        self._engine_request_seq += 1
        request_id = int(self._engine_request_seq)
        self._latest_request_by_node[node_id] = request_id
        self.engine_execute_requested.emit(
            {
                "request_id": request_id,
                "node_id": node_id,
                "title": title,
                "input_df": input_df,
                "inputs": inputs,
                "options": options or {},
            }
        )

    @Slot(dict)
    def _on_engine_result(self, result: dict) -> None:
        node_id = str(result.get("node_id", "") or "")
        request_id = int(result.get("request_id", 0))
        if not node_id:
            return
        if self._latest_request_by_node.get(node_id) != request_id:
            return

        target_node = self._find_node_by_id(node_id)
        if target_node is None:
            return

        outputs = result.get("outputs", {}) or {}
        output_df = result.get("primary_output")
        self._apply_node_result(target_node, output_df, outputs, trigger_downstream=True)

    def _find_node_by_id(self, node_id: str):
        if not node_id:
            return None
        try:
            for item in self.scene().items():
                if isinstance(item, NodeItem) and str(getattr(item, "node_id", "") or "") == node_id:
                    try:
                        if shiboken6.isValid(item):
                            return item
                    except Exception:
                        continue
        except Exception:
            return None
        return None

    def _apply_node_result(self, target_node, output_df, runtime_outputs=None, trigger_downstream: bool = False) -> None:
        try:
            if not shiboken6.isValid(target_node):
                return
        except Exception:
            return

        if isinstance(runtime_outputs, dict):
            target_node._runtime_outputs = runtime_outputs

        if output_df is not None:
            target_node.set_dataframe(output_df)

        if self._output_callback:
            target_node.set_output_callback(self._output_callback)
            target_node.set_input_callback(self._output_callback)

        if target_node.title == "Final Output" and output_df is not None:
            global _DATASET_PREVIEW_CALLBACK
            if _DATASET_PREVIEW_CALLBACK:
                _DATASET_PREVIEW_CALLBACK(output_df)

        if trigger_downstream:
            self._propagate_downstream(target_node)

    def _resolve_source_payload(self, source_node, source_port=None):
        """Get the payload produced by the exact output port when possible."""
        if source_node is None:
            return None
        try:
            outputs = getattr(source_node, "_runtime_outputs", None)
            if isinstance(outputs, dict) and outputs:
                port_name = str(getattr(source_port, "name", "") or "")
                if port_name and port_name in outputs:
                    return outputs.get(port_name)
        except Exception:
            pass
        try:
            return source_node.get_dataframe()
        except Exception:
            return None

    def _collect_connected_inputs(self, node) -> dict:
        """Collect target inputs from connected upstream source ports."""
        from nodes.base.edge import EdgeItem

        connected: dict = {}
        try:
            scene_items = list(self.scene().items())
        except (RuntimeError, AttributeError):
            return connected
        for item in scene_items:
            try:
                if not isinstance(item, EdgeItem):
                    continue
                if not shiboken6.isValid(item):
                    continue
                src_port = item.get_source_port()
                tgt_port = item.get_target_port()
                if not src_port or not tgt_port:
                    continue
                if not shiboken6.isValid(src_port) or not shiboken6.isValid(tgt_port):
                    continue
                tgt_node = tgt_port.parentItem()
                if tgt_node is not node:
                    continue
                src_node = src_port.parentItem()
                if not src_node or not shiboken6.isValid(src_node):
                    continue
                payload = self._resolve_source_payload(src_node, src_port)
                if payload is None:
                    continue
                connected[str(getattr(tgt_port, "name", "") or "")] = payload
            except RuntimeError:
                continue
        return connected

    def _execute_node_logic(self, node, input_df, explicit_inputs: dict | None = None):
        """Execute the node's processing logic using node runtime classes."""
        import pandas as pd
        import numpy as np
        from nodes.registry import get_node_runtime
        from nodes.base.node_runtime import NodeResult
        
        title = node.title
        
        try:
            # Get node options from controls
            options = self._extract_node_options(node)
            
            # Try to use the registered node runtime
            runtime_class = get_node_runtime(title)
            
            if runtime_class:
                # Create node data structure for runtime
                node_data = {"options": self._options_to_list(options)}
                runtime = runtime_class(node.node_id, node_data)
                
                # Prepare inputs based on node type
                inputs = explicit_inputs if explicit_inputs is not None else self._prepare_runtime_inputs(title, input_df, node)
                
                # Execute the node
                result = runtime.execute(inputs)
                
                if isinstance(result, NodeResult):
                    # Store all outputs on the node
                    node._runtime_outputs = result.outputs
                    
                    # Return the primary output DataFrame
                    primary_output = self._get_primary_output(result.outputs, title)
                    return primary_output
                else:
                    return input_df
            else:
                # Fallback to inline execution for basic nodes
                return self._execute_inline(title, input_df, options)
                
        except Exception as e:
            print(f"Error executing {title}: {e}")
            import traceback
            traceback.print_exc()
            return input_df
    
    def _options_to_list(self, options: dict) -> list[dict]:
        """Convert options dict to list format for node runtime."""
        return [{"label": k, "value": v} for k, v in options.items()]
    
    def _prepare_runtime_inputs(self, title: str, input_df, node) -> dict:
        """Prepare the inputs dict for node runtime based on node type."""
        # Default input mapping
        inputs = {"Data": input_df, "Chunk": input_df}

        # Merge inputs connected to explicit target ports.
        connected_inputs = self._collect_connected_inputs(node)
        if connected_inputs:
            inputs.update(connected_inputs)

        # Handle specific node types that need fallback mapping
        if "Split" in title:
            if "Features" not in inputs:
                inputs["Features"] = input_df
            inputs.setdefault("Target", None)
        
        elif title in ("Classification Model", "Regression Model"):
            inputs.setdefault("X_train", input_df)
            inputs.setdefault("y_train", None)
            inputs.setdefault("X_test", None)
        
        elif title == "Neural Network":
            inputs.setdefault("X_train", input_df)
            inputs.setdefault("y_train", None)
        
        elif title in ("Clustering Model",):
            inputs.setdefault("Features", input_df)
        
        elif title in ("Anomaly Detector", "Anomaly Model"):
            inputs.setdefault("Features", input_df)
        
        elif title == "Feature Scaler":
            inputs.setdefault("Features", input_df)
        
        elif title == "Feature Selector":
            inputs.setdefault("Features", input_df)
            inputs.setdefault("Target", None)
        
        elif title in ("Metrics Evaluator", "Metrics"):
            inputs.setdefault("Trained Model", None)
            inputs.setdefault("X_test", input_df)
            inputs.setdefault("y_test", None)
        
        elif title in ("Model Explainer",):
            inputs.setdefault("Trained Model", None)
            inputs.setdefault("X_test", input_df)
        
        elif title in ("Inference Node",):
            inputs.setdefault("Model", None)
            inputs.setdefault("New Data", input_df)
        
        elif title == "Dataset Merger":
            if connected_inputs:
                inputs.setdefault("Dataset A", None)
                inputs.setdefault("Dataset B", None)
            else:
                inputs.setdefault("Dataset A", input_df)
                inputs.setdefault("Dataset B", None)
        
        elif title == "Final Output":
            inputs.setdefault("Data", input_df)
        
        elif title == "Data Preview":
            inputs.setdefault("Data", input_df)
        
        return inputs
    
    def _get_primary_output(self, outputs: dict, title: str):
        """Get the primary output DataFrame from node outputs."""
        import pandas as pd
        import numpy as np
        
        # Priority order for output keys
        if "Split" in title:
            # For split nodes, return training features
            if "X_train" in outputs:
                val = outputs["X_train"]
                if isinstance(val, pd.DataFrame):
                    return val
                if isinstance(val, pd.Series):
                    return val.to_frame()
        
        # Common output keys (order matters - first match wins)
        priority_keys = [
            "Filtered Chunk", "Filtered Data", "Clean Chunk", "Converted Chunk",
            "Scaled Features", "Encoded Features", "Data", "Raw Data",
            "Merged Data", "Features", "X_train", "Predictions",
            "Scaled Data", "Encoded Data", "Preview", "Processed Data",
            "Joined DataFrame", "Selected DataFrame", "Cluster Labels",
            "Anomaly Scores", "Anomaly Labels",
        ]
        
        for key in priority_keys:
            if key in outputs and outputs[key] is not None:
                val = outputs[key]
                if isinstance(val, pd.DataFrame):
                    return val
                if isinstance(val, pd.Series):
                    return val.to_frame()
                if isinstance(val, np.ndarray) and val.ndim <= 2:
                    return pd.DataFrame(val)
        
        # Return first DataFrame or array found
        for v in outputs.values():
            if isinstance(v, pd.DataFrame):
                return v
            if isinstance(v, pd.Series):
                return v.to_frame()
        
        return None
    
    def _execute_inline(self, title: str, input_df, options) -> "pd.DataFrame":
        """Fallback inline execution for nodes without runtime classes."""
        import pandas as pd
        import numpy as np
        
        if title == "Filter Node":
            return self._execute_filter(input_df, options)
        elif title == "Missing Value Handler":
            return self._execute_missing_handler(input_df, options)
        elif title in ("Feature Scaler", "Scaling"):
            return self._execute_scaler(input_df, options)
        elif title in ("Categorical Encoder", "Encoding"):
            return self._execute_encoder(input_df, options)
        elif title == "Column Selector":
            return self._execute_column_selector(input_df, options)
        elif title == "Outlier Handler":
            return self._execute_outlier_handler(input_df, options)
        elif title == "Data Type Converter":
            return self._execute_type_converter(input_df, options)
        elif title in ("DataFrame Column Selector", "DF Column Selector"):
            return self._execute_dataframe_column_selector(input_df, options)
        elif title in ("Column Joiner", "DataFrame Joiner"):
            return self._execute_column_joiner(input_df, options)
        else:
            return input_df
    
    def _execute_type_converter(self, df, options) -> "pd.DataFrame":
        """Execute data type converter logic."""
        df = df.copy()
        columns = options.get("Columns", "")
        target_type = options.get("Target Type", "float")
        
        if columns:
            cols = [c.strip() for c in columns.split(",") if c.strip()]
        else:
            cols = df.columns.tolist()
        
        for col in cols:
            if col in df.columns:
                try:
                    if target_type == "float":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif target_type == "int":
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    elif target_type == "str":
                        df[col] = df[col].astype(str)
                    elif target_type == "datetime":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif target_type == "category":
                        df[col] = df[col].astype("category")
                except Exception:
                    pass
        return df
    
    def _extract_node_options(self, node) -> dict:
        """Extract option values from node's control widgets."""
        options = {}
        try:
            if not shiboken6.isValid(node):
                return options
            cw = getattr(node, '_controls_widget', None)
            if cw and shiboken6.isValid(cw):
                self._extract_from_widget(cw, options)
        except RuntimeError:
            pass
        return options
    
    def _extract_from_widget(self, widget, options: dict, parent_label: str = "") -> None:
        """Recursively extract values from widget tree."""
        from PySide6.QtWidgets import (
            QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
            QComboBox, QSlider, QLabel, QFormLayout, QHBoxLayout, QVBoxLayout
        )

        try:
            if not shiboken6.isValid(widget):
                return
        except Exception:
            return

        try:
            # Check if this widget has a value
            widget_name = widget.objectName() or parent_label
            
            if isinstance(widget, QLineEdit):
                if widget_name:
                    options[widget_name] = widget.text()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                if widget_name:
                    options[widget_name] = widget.value()
            elif isinstance(widget, QCheckBox):
                if widget_name:
                    options[widget_name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                if widget_name:
                    options[widget_name] = widget.currentText()
            elif hasattr(widget, 'currentText'):  # ComboButton
                if widget_name:
                    options[widget_name] = widget.currentText()
            elif isinstance(widget, QSlider):
                if widget_name:
                    options[widget_name] = widget.value()
            
            # Check layout for child widgets
            layout = widget.layout()
            if layout and shiboken6.isValid(layout):
                current_label = ""
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item:
                        child_widget = item.widget()
                        if child_widget and shiboken6.isValid(child_widget):
                            if isinstance(child_widget, QLabel):
                                current_label = child_widget.text().replace(":", "").strip()
                            else:
                                self._extract_from_widget(child_widget, options, current_label)
                                current_label = ""
            
            # Check children directly
            for child in widget.children():
                if hasattr(child, 'isWidgetType') and child.isWidgetType():
                    if shiboken6.isValid(child):
                        self._extract_from_widget(child, options, "")
        except RuntimeError:
            pass
    
    def _execute_filter(self, df, options) -> pd.DataFrame:
        """Execute filter node logic."""
        import pandas as pd
        import numpy as np
        
        column = options.get("Column", options.get("column", ""))
        condition = options.get("Condition", options.get("condition", ">"))
        value = options.get("Value", options.get("value", "0"))
        enabled = options.get("Enabled", options.get("enabled", True))
        
        # Try to find column from the dataframe if not in options
        if not column and len(df.columns) > 0:
            column = df.columns[0]
        
        if not enabled or not column or column not in df.columns:
            return df
        
        try:
            # Convert value type
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
                mask = pd.Series([True] * len(df), index=df.index)
            
            return df[mask].reset_index(drop=True)
        except Exception as e:
            print(f"Filter error: {e}")
            return df
    
    def _execute_missing_handler(self, df, options) -> pd.DataFrame:
        """Execute missing value handler logic."""
        import numpy as np
        
        strategy = options.get("Strategy", "Mean")
        fill_value = options.get("Fill Value", "0")
        
        df = df.copy()
        
        try:
            if strategy == "Mean":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif strategy == "Median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif strategy == "Mode":
                for col in df.columns:
                    if df[col].isnull().any():
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val[0])
            elif strategy == "Constant":
                df = df.fillna(fill_value)
            elif strategy == "Drop Rows":
                df = df.dropna()
            
            return df
        except Exception:
            return df
    
    def _execute_scaler(self, df, options) -> pd.DataFrame:
        """Execute feature scaler logic."""
        import numpy as np
        
        method = options.get("Method", "StandardScaler")
        df = df.copy()
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if method == "StandardScaler":
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
            elif method == "MinMaxScaler":
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
            
            return df
        except Exception:
            return df
    
    def _execute_encoder(self, df, options) -> pd.DataFrame:
        """Execute categorical encoder logic."""
        import pandas as pd
        
        method = options.get("Method", "One-Hot")
        df = df.copy()
        
        try:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            
            if method == "One-Hot":
                df = pd.get_dummies(df, columns=cat_cols)
            elif method == "Label":
                for col in cat_cols:
                    df[col] = df[col].astype("category").cat.codes
            
            return df
        except Exception:
            return df
    
    def _execute_column_selector(self, df, options) -> pd.DataFrame:
        """Execute column selector logic."""
        features_str = options.get("Features", "")
        
        if features_str:
            cols = [c.strip() for c in features_str.split(",") if c.strip()]
            existing = [c for c in cols if c in df.columns]
            if existing:
                return df[existing]
        
        return df
    
    def _execute_outlier_handler(self, df, options) -> pd.DataFrame:
        """Execute outlier handler logic."""
        import numpy as np
        
        method = options.get("Method", "IQR")
        threshold = float(options.get("Threshold", 1.5))
        action = options.get("Action", "Remove")
        
        df = df.copy()
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if method == "IQR":
                for col in numeric_cols:
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
                    
                    if action == "Remove":
                        df = df[(df[col] >= lower) & (df[col] <= upper)]
                    elif action == "Cap":
                        df[col] = df[col].clip(lower, upper)
            
            return df.reset_index(drop=True)
        except Exception:
            return df

    def _execute_dataframe_column_selector(self, df, options) -> pd.DataFrame:
        """
        Execute simple DataFrame column subset selection.
        Equivalent to: df1 = df[['col1', 'col2']]
        """
        columns_raw = str(options.get("Columns", "") or "").strip()
        if not columns_raw:
            return df

        cols = [c.strip() for c in columns_raw.split(",") if c.strip()]
        if not cols:
            return df

        existing = [c for c in cols if c in df.columns]
        if not existing:
            return df
        return df[existing].copy()

    def _execute_column_joiner(self, df, options) -> pd.DataFrame:
        """
        Join 2+ DataFrames/series side-by-side.
        Preserves original column names and only de-duplicates true collisions.
        """
        if df is None:
            return df
        out = df.copy()
        try:
            if not out.columns.duplicated().any():
                return out
            seen: dict[str, int] = {}
            new_cols: list[str] = []
            for col in out.columns:
                name = str(col)
                if name not in seen:
                    seen[name] = 0
                    new_cols.append(name)
                else:
                    seen[name] += 1
                    new_cols.append(f"{name}_{seen[name]}")
            out.columns = new_cols
            return out
        except Exception:
            return out

    def _is_compatible(self, output_port: PortItem, input_port: PortItem) -> bool:
        try:
            if output_port.parentItem() is input_port.parentItem():
                return False
        except Exception:
            return False
        if output_port.data_type == input_port.data_type:
            return True
        return input_port.data_type == "any"

    def _would_create_cycle(self, source_node: NodeItem, target_node: NodeItem) -> bool:
        """Return True if adding source->target would introduce a cycle."""
        try:
            if source_node is target_node:
                return True
        except Exception:
            return True

        source_id = str(getattr(source_node, "node_id", ""))
        target_id = str(getattr(target_node, "node_id", ""))
        if not source_id or not target_id:
            return False

        adjacency: dict[str, set[str]] = {}
        try:
            scene_items = list(self.scene().items())
        except Exception:
            return False

        for item in scene_items:
            if not isinstance(item, EdgeItem):
                continue
            try:
                if not shiboken6.isValid(item):
                    continue
                src_port = item.get_source_port()
                tgt_port = item.get_target_port()
                if not src_port or not tgt_port:
                    continue
                if not shiboken6.isValid(src_port) or not shiboken6.isValid(tgt_port):
                    continue
                src_node = src_port.parentItem()
                tgt_node = tgt_port.parentItem()
                if not isinstance(src_node, NodeItem) or not isinstance(tgt_node, NodeItem):
                    continue
                a = str(getattr(src_node, "node_id", ""))
                b = str(getattr(tgt_node, "node_id", ""))
                if a and b:
                    adjacency.setdefault(a, set()).add(b)
            except Exception:
                continue

        # Simulate adding source->target and check if target can reach source.
        adjacency.setdefault(source_id, set()).add(target_id)
        stack = [target_id]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == source_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            for nxt in adjacency.get(current, set()):
                if nxt not in seen:
                    stack.append(nxt)
        return False

    def _build_link_model(
        self,
        output_port: PortItem,
        input_port: PortItem,
        column_configs: list[dict],
    ) -> LinkModel:
        from nodes.base.link_model import ColumnConfig, ColumnRole
        
        link = LinkModel()
        src_node = output_port.parentItem()
        tgt_node = input_port.parentItem()
        
        # Identity
        link.source_node_id = getattr(src_node, "node_id", "")
        link.source_node_name = getattr(src_node, "title", "Unknown")
        link.source_port_id = output_port.port_id
        link.source_port_name = output_port.name
        link.target_node_id = getattr(tgt_node, "node_id", "")
        link.target_node_name = getattr(tgt_node, "title", "Unknown")
        link.target_port_id = input_port.port_id
        link.target_port_name = input_port.name
        
        # Columns
        for cfg in column_configs:
            if cfg.get("enabled", True):
                role = ColumnRole(cfg.get("role", "feature"))
                col = ColumnConfig(
                    name=cfg["name"],
                    data_type=cfg.get("type", output_port.data_type),
                    role=role,
                    enabled=cfg.get("enabled", True),
                    missing_pct=cfg.get("missing_pct", 0.0),
                )
                link.columns.append(col)
                link.columns_passed.append(cfg["name"])
                link.column_types[cfg["name"]] = cfg.get("type", output_port.data_type)
        
        link.data_type = output_port.data_type
        link._update_counts()
        link.validate()
        link.estimate_memory()
        
        return link

    def _column_popup(self, source_node=None) -> list[dict]:
        """
        Enhanced column selection popup with role assignment.
        Returns list of column configs: [{"name": str, "role": str, "enabled": bool, "type": str}, ...]
        """
        dialog = ColumnSelectionDialog(self, source_node)
        if dialog.exec() != QDialog.Accepted:
            return []
        return dialog.get_selected_columns()


class NodeMenuDialog(QDialog):
    def __init__(self, view: NodeGraphView) -> None:
        super().__init__(view)
        self._view = view
        self._recent: list[str] = []
        self._favorites: set[str] = set()
        self._catalog = _build_node_catalog()
        self.setWindowTitle("Add Node")
        self.setModal(False)
        self.setWindowFlag(Qt.Tool, True)
        self.resize(720, 420)

        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(
                    x1:0, y1:0, x2:0.3, y2:1,
                    stop:0 rgba(18, 28, 42, 245),
                    stop:1 rgba(12, 18, 30, 250)
                );
            }
            QLabel {
                color: rgba(160, 205, 255, 220);
                font-weight: 600;
            }
            QLineEdit {
                background: rgba(22, 32, 50, 220);
                color: rgba(200, 225, 255, 230);
                border: 1px solid rgba(55, 110, 200, 60);
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }
            QLineEdit:focus { border-color: rgba(70, 150, 255, 140); }
            QListWidget {
                background: rgba(16, 24, 38, 200);
                color: rgba(180, 215, 255, 220);
                border: 1px solid rgba(55, 110, 200, 40);
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 5px 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background: rgba(40, 80, 160, 140);
            }
            QListWidget::item:hover:!selected {
                background: rgba(30, 55, 90, 100);
            }
            QPushButton {
                background: rgba(28, 42, 64, 220);
                color: rgba(180, 215, 255, 230);
                border: 1px solid rgba(60, 120, 200, 50);
                border-radius: 6px;
                padding: 7px 18px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(40, 80, 140, 220);
                border-color: rgba(70, 150, 255, 100);
            }
        """)

        # Fade-in animation
        self._opacity_fx = QGraphicsOpacityEffect(self)
        self._opacity_fx.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_fx)

        root = QVBoxLayout(self)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search nodes...")
        self._search.textChanged.connect(self._refresh_nodes)
        root.addWidget(self._search)

        content = QHBoxLayout()
        root.addLayout(content)

        self._categories = QListWidget()
        self._categories.setSelectionMode(QAbstractItemView.SingleSelection)
        # All categories matching the node catalog
        for name in ["All", "Data", "Preprocessing", "Split", "Model", "Training", "Evaluation", "Output", "AI", "System", "Utility"]:
            self._categories.addItem(name)
        self._categories.setCurrentRow(0)
        self._categories.currentTextChanged.connect(self._refresh_nodes)
        content.addWidget(self._categories, 1)

        self._nodes = QListWidget()
        self._nodes.itemDoubleClicked.connect(self._add_selected)
        self._nodes.setContextMenuPolicy(Qt.CustomContextMenu)
        self._nodes.customContextMenuRequested.connect(self._node_context_menu)
        content.addWidget(self._nodes, 2)

        sidebar = QVBoxLayout()
        content.addLayout(sidebar, 1)

        self._recent_list = QListWidget()
        self._recent_list.itemDoubleClicked.connect(self._add_from_sidebar)
        self._favorite_list = QListWidget()
        self._favorite_list.itemDoubleClicked.connect(self._add_from_sidebar)
        self._ai_list = QListWidget()
        self._ai_list.itemDoubleClicked.connect(self._add_from_sidebar)

        sidebar.addWidget(QLabel("📋 Recently Used"))
        sidebar.addWidget(self._recent_list)
        sidebar.addWidget(QLabel("⭐ Favorites"))
        sidebar.addWidget(self._favorite_list)
        sidebar.addWidget(QLabel("🧠 AI Suggested"))
        sidebar.addWidget(self._ai_list)

        buttons = QHBoxLayout()
        root.addLayout(buttons)
        self._add_button = QPushButton("Add Node")
        self._add_button.clicked.connect(self._add_selected)
        self._close_button = QPushButton("Close")
        self._close_button.clicked.connect(self.close)
        buttons.addStretch(1)
        buttons.addWidget(self._add_button)
        buttons.addWidget(self._close_button)

        self._refresh_nodes()
        self._refresh_sidebars()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._show_anim = QPropertyAnimation(self._opacity_fx, b"opacity", self)
        self._show_anim.setDuration(250)
        self._show_anim.setStartValue(0.0)
        self._show_anim.setEndValue(1.0)
        self._show_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._show_anim.start()

    def open_at_cursor(self) -> None:
        self._refresh_nodes()
        self._refresh_sidebars()
        self.move(self._view.mapToGlobal(self._view.viewport().rect().center()))
        self.show()
        self.raise_()
        self.activateWindow()

    def _refresh_nodes(self) -> None:
        self._nodes.clear()
        category = self._categories.currentItem().text()
        term = self._search.text().strip().lower()
        for node in self._catalog:
            if category != "All" and node["category"] != category:
                continue
            if term and term not in node["name"].lower() and term not in node["desc"].lower():
                continue
            item = QListWidgetItem(f"{node['name']} — {node['desc']}")
            item.setData(Qt.UserRole, node["name"])
            self._nodes.addItem(item)

    def _refresh_sidebars(self) -> None:
        self._recent_list.clear()
        for name in self._recent[:8]:
            self._recent_list.addItem(name)
        self._favorite_list.clear()
        for name in sorted(self._favorites):
            self._favorite_list.addItem(name)
        self._ai_list.clear()
        # AI suggested nodes based on common workflows
        ai_suggestions = [
            "Dataset Loader",
            "Missing Value Handler", 
            "Feature Scaler",
            "Train/Test Split",
            "Model Selector",
            "Training Controller",
            "Metrics Evaluator",
        ]
        for name in ai_suggestions:
            self._ai_list.addItem(name)

    def _node_context_menu(self, pos: QPoint) -> None:
        item = self._nodes.itemAt(pos)
        if not item:
            return
        name = item.data(Qt.UserRole)
        menu = QMenu(self)
        if name in self._favorites:
            menu.addAction("Remove Favorite", lambda: self._toggle_favorite(name))
        else:
            menu.addAction("Add to Favorites", lambda: self._toggle_favorite(name))
        menu.exec(self._nodes.mapToGlobal(pos))

    def _toggle_favorite(self, name: str) -> None:
        if name in self._favorites:
            self._favorites.remove(name)
        else:
            self._favorites.add(name)
        self._refresh_sidebars()

    def _add_from_sidebar(self, item) -> None:
        """Add node from sidebar lists (recent, favorites, AI suggested)."""
        name = item.text()
        self._recent = [name] + [n for n in self._recent if n != name]
        self._recent = self._recent[:10]
        self._view._add_node(name)
        self._refresh_sidebars()

    def _add_selected(self) -> None:
        item = self._nodes.currentItem()
        if not item:
            return
        name = item.data(Qt.UserRole)
        self._recent = [name] + [n for n in self._recent if n != name]
        self._recent = self._recent[:10]
        self._view._add_node(name)
        self._refresh_sidebars()

    def apply_ports(self, node) -> None:
        spec = next((n for n in self._catalog if n["name"] == node.title), None)
        if not spec:
            return
        for inp in spec["inputs"]:
            node.add_input(inp, _infer_port_type(inp))
        for out in spec["outputs"]:
            node.add_output(out, _infer_port_type(out))
        for option in spec.get("options", []):
            control = _build_option_widget(option, node)
            if control is not None:
                node.add_control(control)


def _build_node_catalog() -> list[dict]:
    return [
        # ═══════════════════════════════════════════════════════════════
        # 1️⃣ DATA INGESTION NODES
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "Dataset Loader",
            "category": "Data",
            "desc": "Load dataset chunks from files",
            "inputs": [],
            "outputs": ["Raw Data", "Feature Candidates", "Target Candidates", "Schema", "Stats"],
            "options": [
                {"type": "dataset_loader"},
            ],
        },
        {
            "name": "Dataset Merger",
            "category": "Data",
            "desc": "Combine multiple datasets",
            "inputs": ["Dataset A", "Dataset B"],
            "outputs": ["Merged Data", "Schema"],
            "options": [
                {"type": "combo", "label": "Merge Type", "items": ["Join (Columns)", "Concat (Rows)", "Left Join", "Right Join", "Inner Join", "Outer Join"]},
                {"type": "text", "label": "Join Key", "value": "id"},
                {"type": "combo", "label": "Handle Mismatch", "items": ["Fill NaN", "Drop Rows", "Error"]},
                {"type": "check", "label": "Reset Index", "value": True},
            ],
        },
        {
            "name": "Column Selector",
            "category": "Data",
            "desc": "Select features/target columns",
            "inputs": ["Chunk"],
            "outputs": ["Features", "Target"],
            "options": [
                {"type": "text", "label": "Features", "value": "col1,col2"},
                {"type": "text", "label": "Target", "value": "target"},
                {"type": "check", "label": "Drop Selected", "value": False},
            ],
        },
        {
            "name": "DataFrame Column Selector",
            "category": "Data",
            "desc": "Select DataFrame columns like df[['col1','col2']]",
            "inputs": ["Chunk"],
            "outputs": ["Selected DataFrame"],
            "options": [
                {"type": "text", "label": "Columns", "value": "col1,col2"},
            ],
        },
        {
            "name": "Data Preview",
            "category": "Data",
            "desc": "Preview and inspect data",
            "inputs": ["Data"],
            "outputs": ["Preview"],
            "options": [
                {"type": "spin", "label": "rows", "min": 1, "max": 10000, "value": 100},
            ],
        },
        {
            "name": "Filter Node",
            "category": "Data",
            "desc": "Filter rows by condition",
            "inputs": ["Chunk"],
            "outputs": ["Filtered Chunk", "Rejected Rows"],
            "options": [
                {"type": "text", "label": "Column", "value": "col1"},
                {"type": "combo", "label": "Condition", "items": [">", "<", ">=", "<=", "==", "!=", "contains", "startswith", "isnull", "notnull"]},
                {"type": "text", "label": "Value", "value": "0"},
                {"type": "check", "label": "Enabled", "value": True},
            ],
        },
        # ═══════════════════════════════════════════════════════════════
        # 2️⃣ DATA PREPROCESSING NODES
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "Missing Value Handler",
            "category": "Preprocessing",
            "desc": "Handle null/missing values",
            "inputs": ["Chunk"],
            "outputs": ["Clean Chunk", "Missing Report"],
            "options": [
                {"type": "combo", "label": "Strategy", "items": ["Mean", "Median", "Mode", "Constant", "Forward Fill", "Backward Fill", "Interpolate", "Drop Rows", "Drop Columns"]},
                {"type": "text", "label": "Fill Value", "value": "0"},
                {"type": "double", "label": "Drop Threshold", "min": 0.0, "max": 1.0, "value": 0.5, "step": 0.05},
                {"type": "check", "label": "Selected Columns Only", "value": False},
            ],
        },
        {
            "name": "Data Type Converter",
            "category": "Preprocessing",
            "desc": "Convert column data types",
            "inputs": ["Chunk"],
            "outputs": ["Converted Chunk"],
            "options": [
                {"type": "text", "label": "Columns", "value": "col1,col2"},
                {"type": "combo", "label": "Target Type", "items": ["int", "float", "str", "bool", "datetime", "category"]},
                {"type": "combo", "label": "Date Format", "items": ["Auto", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"]},
                {"type": "check", "label": "Coerce Errors", "value": True},
            ],
        },
        {
            "name": "Column Joiner",
            "category": "Preprocessing",
            "desc": "Join 2+ DataFrames side-by-side to add extra columns",
            "inputs": ["Left DataFrame", "Right DataFrame"],
            "outputs": ["Joined DataFrame"],
            "options": [
                {"type": "check", "label": "Reset Index", "value": True},
            ],
        },
        {
            "name": "Categorical Encoder",
            "category": "Preprocessing",
            "desc": "Encode categorical columns",
            "inputs": ["Chunk"],
            "outputs": ["Encoded Features", "Encoder State"],
            "options": [
                {"type": "combo", "label": "Method", "items": ["One-Hot", "Label", "Ordinal", "Target", "Binary", "Frequency"]},
                {"type": "combo", "label": "Handle Unknown", "items": ["Ignore", "Error", "Infrequent"]},
                {"type": "spin", "label": "Max Categories", "min": 2, "max": 100, "value": 10},
                {"type": "check", "label": "Drop First", "value": False},
            ],
        },
        {
            "name": "Feature Scaler",
            "category": "Preprocessing",
            "desc": "Scale/normalize numeric features",
            "inputs": ["Features"],
            "outputs": ["Scaled Features", "Scaler State"],
            "options": [
                {"type": "combo", "label": "Method", "items": ["StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer", "PowerTransformer", "QuantileTransformer"]},
                {"type": "check", "label": "With Mean", "value": True},
                {"type": "check", "label": "With Std", "value": True},
                {"type": "check", "label": "Clip Outliers", "value": False},
            ],
        },
        {
            "name": "Feature Selector",
            "category": "Preprocessing",
            "desc": "Select best features",
            "inputs": ["Features", "Target"],
            "outputs": ["Selected Features", "Feature Scores"],
            "options": [
                {"type": "combo", "label": "Method", "items": ["Variance Threshold", "Correlation", "SelectKBest", "RFE", "L1 Regularization", "Tree Importance", "Mutual Information"]},
                {"type": "spin", "label": "K Features", "min": 1, "max": 100, "value": 10},
                {"type": "double", "label": "Threshold", "min": 0.0, "max": 1.0, "value": 0.1, "step": 0.01},
                {"type": "check", "label": "AI Suggestion", "value": True},
            ],
        },
        {
            "name": "Outlier Handler",
            "category": "Preprocessing",
            "desc": "Detect and handle outliers",
            "inputs": ["Chunk"],
            "outputs": ["Clean Chunk", "Outlier Mask"],
            "options": [
                {"type": "combo", "label": "Method", "items": ["IQR", "Z-Score", "Isolation Forest", "LOF", "DBSCAN"]},
                {"type": "double", "label": "Threshold", "min": 1.0, "max": 5.0, "value": 1.5, "step": 0.1},
                {"type": "combo", "label": "Action", "items": ["Remove", "Cap", "Replace Mean", "Replace Median", "Flag Only"]},
            ],
        },
        {
            "name": "Text Preprocessor",
            "category": "Preprocessing",
            "desc": "Process text columns",
            "inputs": ["Chunk"],
            "outputs": ["Processed Text"],
            "options": [
                {"type": "text", "label": "Column", "value": "text"},
                {"type": "check", "label": "Lowercase", "value": True},
                {"type": "check", "label": "Remove Punctuation", "value": True},
                {"type": "check", "label": "Remove Stopwords", "value": True},
                {"type": "check", "label": "Stemming", "value": False},
                {"type": "check", "label": "Lemmatization", "value": True},
            ],
        },
        # ═══════════════════════════════════════════════════════════════
        # 3️⃣ DATA SPLITTING & FLOW CONTROL
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "Train/Test Split",
            "category": "Split",
            "desc": "Split into train/test sets",
            "inputs": ["Features", "Target"],
            "outputs": ["X_train", "X_test", "y_train", "y_test"],
            "options": [
                {"type": "double", "label": "Test Size", "min": 0.05, "max": 0.5, "value": 0.2, "step": 0.05},
                {"type": "spin", "label": "Seed", "min": 0, "max": 9999, "value": 42},
                {"type": "check", "label": "Shuffle", "value": True},
                {"type": "check", "label": "Stratify", "value": False},
            ],
        },
        {
            "name": "Train/Val/Test Split",
            "category": "Split",
            "desc": "Three-way data split",
            "inputs": ["Features", "Target"],
            "outputs": ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
            "options": [
                {"type": "double", "label": "Train Size", "min": 0.5, "max": 0.9, "value": 0.7, "step": 0.05},
                {"type": "double", "label": "Val Size", "min": 0.05, "max": 0.3, "value": 0.15, "step": 0.05},
                {"type": "spin", "label": "Seed", "min": 0, "max": 9999, "value": 42},
                {"type": "check", "label": "Stratify", "value": True},
            ],
        },
        {
            "name": "Cross Validation Split",
            "category": "Split",
            "desc": "K-Fold cross validation",
            "inputs": ["Features", "Target"],
            "outputs": ["Fold Iterator", "Fold Info"],
            "options": [
                {"type": "spin", "label": "K Folds", "min": 2, "max": 20, "value": 5},
                {"type": "combo", "label": "Strategy", "items": ["KFold", "StratifiedKFold", "GroupKFold", "TimeSeriesSplit", "RepeatedKFold"]},
                {"type": "check", "label": "Shuffle", "value": True},
                {"type": "spin", "label": "Seed", "min": 0, "max": 9999, "value": 42},
            ],
        },
        {
            "name": "Time Series Split",
            "category": "Split",
            "desc": "Time-order-safe split for forecasting/time series",
            "inputs": ["Features", "Target"],
            "outputs": ["Split Iterator", "Split Info"],
            "options": [
                {"type": "spin", "label": "N Splits", "min": 2, "max": 20, "value": 5},
                {"type": "spin", "label": "Max Train Size", "min": 0, "max": 1000000, "value": 0},
                {"type": "spin", "label": "Test Size", "min": 0, "max": 1000000, "value": 0},
                {"type": "spin", "label": "Gap", "min": 0, "max": 10000, "value": 0},
            ],
        },
        {
            "name": "Batch Controller",
            "category": "Split",
            "desc": "Control data streaming",
            "inputs": ["Data Stream"],
            "outputs": ["Batched Data", "Batch Info"],
            "options": [
                {"type": "spin", "label": "Batch Size", "min": 8, "max": 10000, "value": 32},
                {"type": "check", "label": "Shuffle", "value": True},
                {"type": "check", "label": "Drop Last", "value": False},
                {"type": "spin", "label": "Buffer Size", "min": 100, "max": 100000, "value": 1000},
                {"type": "spin", "label": "Prefetch", "min": 1, "max": 10, "value": 2},
            ],
        },
        {
            "name": "Conditional Router",
            "category": "Split",
            "desc": "Route data by condition",
            "inputs": ["Data"],
            "outputs": ["True Branch", "False Branch"],
            "options": [
                {"type": "combo", "label": "Condition Type", "items": ["Column Value", "Row Count", "Data Shape", "Custom Expression"]},
                {"type": "text", "label": "Column", "value": "status"},
                {"type": "combo", "label": "Operator", "items": ["==", "!=", ">", "<", ">=", "<=", "in", "not in"]},
                {"type": "text", "label": "Value", "value": "active"},
            ],
        },
        # ═══════════════════════════════════════════════════════════════
        # 4️⃣ MODEL NODES (CORE ML)
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "Model Selector",
            "category": "Model",
            "desc": "Choose ML algorithm",
            "inputs": [],
            "outputs": ["Model Config"],
            "options": [
                {"type": "combo", "label": "Task", "items": ["Classification", "Regression", "Clustering", "Anomaly Detection"]},
                {
                    "type": "combo",
                    "label": "Algorithm",
                    "items": [
                        "Logistic Regression",
                        "Random Forest",
                        "Gradient Boosting",
                        "XGBoost",
                        "LightGBM",
                        "CatBoost",
                        "SVM",
                        "KNN",
                        "Decision Tree",
                        "Naive Bayes",
                        "Linear Regression",
                        "Ridge",
                        "Lasso",
                        "ElasticNet",
                        "AdaBoost",
                    ],
                },
                {"type": "check", "label": "AI Recommend", "value": True},
            ],
        },
        {
            "name": "Classification Model",
            "category": "Model",
            "desc": "Train a classification model directly",
            "inputs": ["X_train", "y_train", "X_test"],
            "outputs": ["Trained Model", "Predictions", "Probabilities", "Feature Importance"],
            "options": [
                {"type": "combo", "label": "Algorithm", "items": ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "SVM", "KNN", "Decision Tree", "Naive Bayes", "AdaBoost"]},
            ],
        },
        {
            "name": "Regression Model",
            "category": "Model",
            "desc": "Train a regression model directly",
            "inputs": ["X_train", "y_train", "X_test"],
            "outputs": ["Trained Model", "Predictions", "Feature Importance"],
            "options": [
                {"type": "combo", "label": "Algorithm", "items": ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "SVR", "KNN", "Decision Tree", "AdaBoost"]},
            ],
        },
        {
            "name": "Neural Network",
            "category": "Model",
            "desc": "Deep learning model",
            "inputs": ["X_train", "y_train"],
            "outputs": ["NN Model", "Architecture"],
            "options": [
                {"type": "combo", "label": "Architecture", "items": ["MLP", "CNN", "RNN", "LSTM", "GRU", "Transformer", "AutoEncoder"]},
                {"type": "text", "label": "Hidden Layers", "value": "128,64,32"},
                {"type": "combo", "label": "Activation", "items": ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "GELU", "Swish"]},
                {"type": "double", "label": "Dropout", "min": 0.0, "max": 0.8, "value": 0.2, "step": 0.05},
                {"type": "combo", "label": "Optimizer", "items": ["Adam", "SGD", "AdamW", "RMSprop", "Adagrad"]},
                {"type": "check", "label": "Batch Norm", "value": True},
            ],
        },
        {
            "name": "Clustering Model",
            "category": "Model",
            "desc": "Unsupervised clustering",
            "inputs": ["Features"],
            "outputs": ["Cluster Labels", "Cluster Centers"],
            "options": [
                {"type": "combo", "label": "Algorithm", "items": ["K-Means", "DBSCAN", "Hierarchical", "GMM", "OPTICS", "Spectral", "Birch"]},
                {"type": "spin", "label": "N Clusters", "min": 2, "max": 50, "value": 5},
                {"type": "combo", "label": "Linkage", "items": ["ward", "complete", "average", "single"]},
                {"type": "double", "label": "Eps", "min": 0.01, "max": 10.0, "value": 0.5, "step": 0.1},
            ],
        },
        {
            "name": "Anomaly Detector",
            "category": "Model",
            "desc": "Detect anomalies",
            "inputs": ["Features"],
            "outputs": ["Anomaly Scores", "Anomaly Labels"],
            "options": [
                {"type": "combo", "label": "Method", "items": ["Isolation Forest", "One-Class SVM", "LOF", "Elliptic Envelope", "Autoencoder"]},
                {"type": "double", "label": "Contamination", "min": 0.01, "max": 0.5, "value": 0.1, "step": 0.01},
                {"type": "spin", "label": "N Estimators", "min": 50, "max": 500, "value": 100},
            ],
        },
        # ═══════════════════════════════════════════════════════════════
        # 5️⃣ TRAINING & OPTIMIZATION
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "Training Controller",
            "category": "Training",
            "desc": "Train model incrementally",
            "inputs": ["Model Config", "X_train", "y_train", "X_val", "y_val"],
            "outputs": ["Trained Model", "Training History"],
            "options": [
                {"type": "spin", "label": "Epochs", "min": 1, "max": 1000, "value": 50},
                {"type": "double", "label": "Learning Rate", "min": 0.00001, "max": 1.0, "value": 0.001, "step": 0.0001},
                {"type": "spin", "label": "Batch Size", "min": 8, "max": 2048, "value": 32},
                {"type": "check", "label": "Use GPU", "value": False},
                {"type": "check", "label": "Early Stopping", "value": True},
                {"type": "spin", "label": "Patience", "min": 1, "max": 50, "value": 10},
                {"type": "combo", "label": "LR Scheduler", "items": ["None", "StepLR", "CosineAnnealing", "ReduceOnPlateau", "OneCycle"]},
            ],
        },
        {
            "name": "Hyperparameter Tuner",
            "category": "Training",
            "desc": "Optimize hyperparameters",
            "inputs": ["Model Config", "X_train", "y_train"],
            "outputs": ["Best Params", "Tuning Results"],
            "options": [
                {"type": "combo", "label": "Search Method", "items": ["Grid Search", "Random Search", "Bayesian", "Optuna", "Hyperband"]},
                {"type": "spin", "label": "N Iterations", "min": 5, "max": 200, "value": 20},
                {"type": "spin", "label": "CV Folds", "min": 2, "max": 10, "value": 5},
                {"type": "combo", "label": "Metric", "items": ["accuracy", "f1", "roc_auc", "precision", "recall", "mse", "r2"]},
                {"type": "check", "label": "Parallel", "value": True},
                {"type": "spin", "label": "N Jobs", "min": -1, "max": 16, "value": -1},
            ],
        },
        {
            "name": "Ensemble Builder",
            "category": "Training",
            "desc": "Combine multiple models",
            "inputs": ["Model A", "Model B", "Model C"],
            "outputs": ["Ensemble Model"],
            "options": [
                {"type": "combo", "label": "Method", "items": ["Voting", "Bagging", "Boosting", "Stacking", "Blending"]},
                {"type": "combo", "label": "Voting", "items": ["hard", "soft"]},
                {"type": "check", "label": "Use Weights", "value": False},
                {"type": "text", "label": "Weights", "value": "1,1,1"},
            ],
        },
        # ═══════════════════════════════════════════════════════════════
        # 6️⃣ EVALUATION & ANALYSIS
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "Metrics Evaluator",
            "category": "Evaluation",
            "desc": "Compute performance metrics",
            "inputs": ["Trained Model", "X_test", "y_test"],
            "outputs": ["Metrics", "Predictions"],
            "options": [
                {"type": "check", "label": "Accuracy", "value": True},
                {"type": "check", "label": "Precision", "value": True},
                {"type": "check", "label": "Recall", "value": True},
                {"type": "check", "label": "F1 Score", "value": True},
                {"type": "check", "label": "ROC AUC", "value": True},
                {"type": "check", "label": "Confusion Matrix", "value": True},
                {"type": "check", "label": "MSE/RMSE", "value": True},
                {"type": "check", "label": "MAE", "value": True},
                {"type": "check", "label": "R² Score", "value": True},
            ],
        },
        {
            "name": "Visualization Node",
            "category": "Evaluation",
            "desc": "Generate visual reports",
            "inputs": ["Metrics", "Training History"],
            "outputs": ["Graph Data"],
            "options": [
                {"type": "combo", "label": "Chart Type", "items": ["Loss Curve", "Accuracy Curve", "Confusion Matrix", "ROC Curve", "PR Curve", "Feature Importance", "Learning Curve", "Residual Plot"]},
                {"type": "check", "label": "Live Update", "value": True},
                {"type": "check", "label": "Save Figure", "value": False},
                {"type": "combo", "label": "Theme", "items": ["Dark", "Light", "Seaborn", "Minimal"]},
            ],
        },
        {
            "name": "Model Explainer",
            "category": "Evaluation",
            "desc": "Explain model predictions",
            "inputs": ["Trained Model", "X_test"],
            "outputs": ["Explanations", "Feature Importance"],
            "options": [
                {"type": "combo", "label": "Method", "items": ["SHAP", "LIME", "Permutation", "Tree Explainer", "Partial Dependence"]},
                {"type": "spin", "label": "N Samples", "min": 10, "max": 1000, "value": 100},
                {"type": "check", "label": "Global Importance", "value": True},
                {"type": "check", "label": "Local Explanations", "value": False},
            ],
        },
        # ═══════════════════════════════════════════════════════════════
        # 7️⃣ DEPLOYMENT & OUTPUT
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "Model Export",
            "category": "Output",
            "desc": "Save trained model",
            "inputs": ["Trained Model", "Preprocessors"],
            "outputs": ["Export Path"],
            "options": [
                {"type": "combo", "label": "Format", "items": ["pickle", "joblib", "onnx", "pmml", "tensorflow", "pytorch"]},
                {"type": "text", "label": "Model Name", "value": "my_model"},
                {"type": "check", "label": "Include Pipeline", "value": True},
                {"type": "check", "label": "Compress", "value": False},
                {"type": "text", "label": "Version", "value": "1.0.0"},
            ],
        },
        {
            "name": "Inference Node",
            "category": "Output",
            "desc": "Run predictions",
            "inputs": ["Model", "New Data"],
            "outputs": ["Predictions", "Probabilities"],
            "options": [
                {"type": "combo", "label": "Output Type", "items": ["Labels", "Probabilities", "Both"]},
                {"type": "check", "label": "Apply Threshold", "value": True},
                {"type": "double", "label": "Threshold", "min": 0.0, "max": 1.0, "value": 0.5, "step": 0.05},
                {"type": "check", "label": "Batch Mode", "value": True},
            ],
        },
        {
            "name": "Report Generator",
            "category": "Output",
            "desc": "Generate analysis report",
            "inputs": ["Metrics", "Visualizations", "Explanations"],
            "outputs": ["Report File"],
            "options": [
                {"type": "combo", "label": "Format", "items": ["HTML", "PDF", "Markdown", "JSON"]},
                {"type": "check", "label": "Include Graphs", "value": True},
                {"type": "check", "label": "Include Data Summary", "value": True},
                {"type": "check", "label": "Include Code", "value": False},
            ],
        },
        {
            "name": "Final Output",
            "category": "Output",
            "desc": "Display final pipeline result in Data Preview",
            "inputs": ["Data"],
            "outputs": [],
            "options": [
                {"type": "text", "label": "Output Name", "value": "Pipeline Result"},
                {"type": "check", "label": "Auto Preview", "value": True},
            ],
        },
        # ═══════════════════════════════════════════════════════════════
        # 8️⃣ AI ADVISORY & SYSTEM NODES
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "AI Advisor",
            "category": "AI",
            "desc": "Get intelligent suggestions",
            "inputs": ["Data", "Pipeline State"],
            "outputs": ["Suggestions", "Warnings"],
            "options": [
                {"type": "check", "label": "Algorithm Suggestions", "value": True},
                {"type": "check", "label": "Preprocessing Tips", "value": True},
                {"type": "check", "label": "Hyperparameter Hints", "value": True},
                {"type": "check", "label": "Data Quality Warnings", "value": True},
                {"type": "combo", "label": "Verbosity", "items": ["Minimal", "Normal", "Detailed"]},
            ],
        },
        {
            "name": "Resource Manager",
            "category": "System",
            "desc": "Control hardware usage",
            "inputs": [],
            "outputs": ["Resource Limits"],
            "options": [
                {"type": "spin", "label": "Max CPU %", "min": 10, "max": 100, "value": 80},
                {"type": "spin", "label": "Max GPU %", "min": 10, "max": 100, "value": 70},
                {"type": "spin", "label": "Memory Cap (MB)", "min": 256, "max": 65536, "value": 4096},
                {"type": "combo", "label": "GPU Device", "items": ["Auto", "GPU 0", "GPU 1", "CPU Only"]},
                {"type": "combo", "label": "Priority", "items": ["Low", "Normal", "High"]},
                {"type": "check", "label": "Memory Mapping", "value": True},
            ],
        },
        # ═══════════════════════════════════════════════════════════════
        # 9️⃣ UTILITY & CONTROL NODES
        # ═══════════════════════════════════════════════════════════════
        {
            "name": "Debug Inspector",
            "category": "Utility",
            "desc": "Inspect data and debug",
            "inputs": ["Any Data"],
            "outputs": ["Passthrough"],
            "options": [
                {"type": "check", "label": "Show Shape", "value": True},
                {"type": "check", "label": "Show Types", "value": True},
                {"type": "check", "label": "Show Stats", "value": True},
                {"type": "check", "label": "Show Sample", "value": True},
                {"type": "spin", "label": "Sample Rows", "min": 1, "max": 100, "value": 5},
                {"type": "check", "label": "Breakpoint", "value": False},
            ],
        },
        {
            "name": "Data Logger",
            "category": "Utility",
            "desc": "Log data to file/console",
            "inputs": ["Data"],
            "outputs": ["Passthrough"],
            "options": [
                {"type": "combo", "label": "Output", "items": ["Console", "File", "Both"]},
                {"type": "text", "label": "Log File", "value": "pipeline.log"},
                {"type": "combo", "label": "Level", "items": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                {"type": "check", "label": "Timestamp", "value": True},
            ],
        },
        {
            "name": "Checkpoint Node",
            "category": "Utility",
            "desc": "Save/load pipeline state",
            "inputs": ["State"],
            "outputs": ["State"],
            "options": [
                {"type": "combo", "label": "Action", "items": ["Save", "Load", "Auto"]},
                {"type": "text", "label": "Checkpoint Name", "value": "checkpoint"},
                {"type": "check", "label": "Include Data", "value": False},
                {"type": "check", "label": "Include Model", "value": True},
            ],
        },
        {
            "name": "Timer Node",
            "category": "Utility",
            "desc": "Measure execution time",
            "inputs": ["Start Signal"],
            "outputs": ["Elapsed Time", "End Signal"],
            "options": [
                {"type": "combo", "label": "Unit", "items": ["Seconds", "Milliseconds", "Minutes"]},
                {"type": "check", "label": "Log Time", "value": True},
                {"type": "check", "label": "Cumulative", "value": False},
            ],
        },
        {
            "name": "Loop Controller",
            "category": "Utility",
            "desc": "Iterate over data/params",
            "inputs": ["Iterator"],
            "outputs": ["Current Item", "Index", "Done Signal"],
            "options": [
                {"type": "spin", "label": "Max Iterations", "min": 1, "max": 10000, "value": 100},
                {"type": "check", "label": "Break on Error", "value": True},
                {"type": "check", "label": "Progress Bar", "value": True},
            ],
        },
        {
            "name": "Note/Comment",
            "category": "Utility",
            "desc": "Add documentation",
            "inputs": [],
            "outputs": [],
            "options": [
                {"type": "text", "label": "Title", "value": "Note"},
                {"type": "text", "label": "Description", "value": "Add your notes here..."},
                {"type": "combo", "label": "Color", "items": ["Yellow", "Blue", "Green", "Red", "Purple"]},
            ],
        },
    ]


class ComboButton(QPushButton):
    """Custom combo box using popup - works properly inside QGraphicsProxyWidget."""
    
    valueChanged = Signal(str)
    
    def __init__(self, items: list[str], parent=None) -> None:
        super().__init__(parent)
        self._items = items
        self._current_index = 0
        self._current_text = items[0] if items else ""
        self._active_popup = None
        self._update_text()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(40, 50, 65, 220);
                border: 1px solid rgba(100, 120, 150, 150);
                border-radius: 4px;
                padding: 4px 8px;
                text-align: left;
                color: rgba(220, 230, 240, 230);
            }
            QPushButton:hover {
                background-color: rgba(55, 70, 90, 230);
            }
            QPushButton:pressed {
                background-color: rgba(35, 45, 60, 230);
            }
        """)
        self.clicked.connect(self._show_popup)
    
    def _update_text(self) -> None:
        self.setText(f"{self._current_text} ▼")
    
    def currentText(self) -> str:
        return self._current_text
    
    def currentIndex(self) -> int:
        return self._current_index
    
    def setCurrentIndex(self, index: int) -> None:
        if 0 <= index < len(self._items):
            self._current_index = index
            self._current_text = self._items[index]
            self._update_text()
    
    def setCurrentText(self, text: str) -> None:
        if text in self._items:
            self._current_index = self._items.index(text)
            self._current_text = text
            self._update_text()
    
    def _show_popup(self) -> None:
        # Close existing popup if open
        if self._active_popup:
            try:
                from shiboken6 import isValid
                if isValid(self._active_popup):
                    self._active_popup.close()
            except (RuntimeError, ImportError):
                pass
            self._active_popup = None
            return
        
        from PySide6.QtWidgets import QListWidget, QListWidgetItem, QApplication
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QCursor
        
        # Create popup list
        popup = QListWidget(QApplication.activeWindow())
        popup.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        popup.setAttribute(Qt.WA_DeleteOnClose)
        popup.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        popup.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        popup.setStyleSheet("""
            QListWidget {
                background-color: rgba(35, 45, 60, 250);
                border: 1px solid rgba(100, 120, 150, 180);
                border-radius: 6px;
                padding: 4px;
                outline: none;
            }
            QListWidget::item {
                color: rgba(220, 230, 240, 230);
                padding: 6px 12px;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: rgba(60, 80, 110, 200);
            }
            QListWidget::item:selected {
                background-color: rgba(70, 100, 140, 220);
            }
            QScrollBar:vertical {
                background-color: rgba(30, 40, 55, 150);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(100, 120, 150, 180);
                border-radius: 4px;
                min-height: 20px;
            }
        """)
        
        for item_text in self._items:
            item = QListWidgetItem(item_text)
            item.setSizeHint(QSize(150, 28))
            popup.addItem(item)
            if item_text == self._current_text:
                item.setSelected(True)
                popup.setCurrentItem(item)
        
        self._active_popup = popup
        
        def on_select(item):
            self._current_text = item.text()
            self._current_index = self._items.index(item.text())
            self._update_text()
            self.valueChanged.emit(self._current_text)
            popup.close()
            self._active_popup = None
        
        def on_focus_out(event):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._close_popup_if_not_focused(popup))
            QListWidget.focusOutEvent(popup, event)
        
        popup.itemClicked.connect(on_select)
        popup.focusOutEvent = on_focus_out
        
        # Calculate popup height based on items
        popup_height = min(280, len(self._items) * 32 + 16)
        popup.setFixedSize(180, popup_height)
        
        # Position below the button
        btn_global_pos = self.mapToGlobal(self.rect().bottomLeft())
        popup.move(btn_global_pos)
        popup.show()
        popup.setFocus()
        popup.activateWindow()
    
    def _close_popup_if_not_focused(self, popup) -> None:
        try:
            from shiboken6 import isValid
            if popup and isValid(popup) and not popup.hasFocus():
                popup.close()
                self._active_popup = None
        except (RuntimeError, ImportError):
            self._active_popup = None


def _build_option_widget(option: dict, parent_node=None) -> QWidget | None:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    label = QLabel(option.get("label", ""))
    label.setStyleSheet("color: rgba(255, 255, 255, 190);")
    layout.addWidget(label)

    widget: QWidget | None = None
    opt_type = option.get("type")
    label_name = option.get("label", "")
    
    if opt_type == "combo":
        # Use custom ComboButton instead of QComboBox (works in QGraphicsProxyWidget)
        items = option.get("items", [])
        widget = ComboButton(items)
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.setObjectName(label_name)
    elif opt_type == "spin":
        widget = QSpinBox()
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.setMinimum(option.get("min", 0))
        widget.setMaximum(option.get("max", 100))
        widget.setValue(option.get("value", 0))
        widget.setObjectName(label_name)
    elif opt_type == "double":
        widget = QDoubleSpinBox()
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.setMinimum(option.get("min", 0.0))
        widget.setMaximum(option.get("max", 1.0))
        widget.setSingleStep(option.get("step", 0.1))
        widget.setValue(option.get("value", 0.0))
        widget.setObjectName(label_name)
    elif opt_type == "check":
        widget = QCheckBox()
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.setChecked(option.get("value", False))
        widget.setObjectName(label_name)
    elif opt_type == "slider":
        widget = QSlider(Qt.Horizontal)
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.setMinimum(option.get("min", 0))
        widget.setMaximum(option.get("max", 100))
        widget.setValue(option.get("value", 0))
        widget.setObjectName(label_name)
    elif opt_type == "text":
        widget = QLineEdit()
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.setAttribute(Qt.WA_InputMethodEnabled, True)
        widget.setText(option.get("value", ""))
        widget.setObjectName(label_name)
    elif opt_type == "file":
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        path_edit = QLineEdit()
        path_edit.setFocusPolicy(Qt.StrongFocus)
        path_edit.setAttribute(Qt.WA_InputMethodEnabled, True)
        path_edit.setText(option.get("value", ""))
        browse = QPushButton("Browse")
        browse.setFocusPolicy(Qt.StrongFocus)

        def pick_file() -> None:
            from PySide6.QtWidgets import QApplication
            parent = None
            for w in QApplication.topLevelWidgets():
                if w.isVisible() and w.windowTitle():
                    parent = w
                    break
            path, _ = QFileDialog.getOpenFileName(
                parent,
                "Select Dataset",
                "",
                "Data Files (*.csv *.parquet *.json);;All Files (*)",
            )
            if path:
                path_edit.setText(path)

        browse.clicked.connect(pick_file)
        layout.addWidget(path_edit, 1)
        layout.addWidget(browse)
        widget = container
    elif opt_type == "dataset_loader":
        widget = DatasetLoaderConfigWidget()
        if parent_node:
            widget.set_parent_node(parent_node)

    if widget is None:
        return None

    widget.setStyleSheet("color: rgba(255, 255, 255, 220);")
    layout.addWidget(widget, 1)
    return row


class DatasetLoaderConfigWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._parent_node = None  # Reference to parent NodeItem
        self._last_path = ""  # Track path changes for auto-load
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

        # Use a button with popup menu instead of QComboBox (works better in QGraphicsProxyWidget)
        self._reader_select = QPushButton("read_csv ▼")
        self._reader_select.setFocusPolicy(Qt.StrongFocus)
        self._reader_select.setStyleSheet("""
            QPushButton {
                background-color: rgba(40, 50, 65, 220);
                border: 1px solid rgba(100, 120, 150, 150);
                border-radius: 4px;
                padding: 6px 12px;
                text-align: left;
                color: rgba(220, 230, 240, 230);
            }
            QPushButton:hover {
                background-color: rgba(55, 70, 90, 230);
            }
            QPushButton:pressed {
                background-color: rgba(35, 45, 60, 230);
            }
        """)
        self._current_reader = "read_csv"
        self._reader_select.clicked.connect(self._show_reader_menu)
        self._readers_list = _all_pandas_readers()

        self._layout.addWidget(QLabel("Reader Function"))
        self._layout.addWidget(self._reader_select)

        self._form_container = QWidget()
        self._form_layout = QFormLayout(self._form_container)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._form_layout.setSpacing(6)
        self._layout.addWidget(self._form_container)

        self._field_widgets: dict[str, QWidget] = {}

        self._status_label = QLabel("Select a file to load preview")
        self._status_label.setStyleSheet("color: rgba(180, 200, 255, 200); font-size: 11px;")
        self._layout.addWidget(self._status_label)

        self._rebuild_form(self._current_reader)
    
    def set_parent_node(self, node) -> None:
        """Set reference to parent NodeItem to store loaded data."""
        self._parent_node = node
    
    def _show_reader_menu(self) -> None:
        """Show scrollable popup to select reader function."""
        from PySide6.QtWidgets import QListWidget, QListWidgetItem, QApplication
        from PySide6.QtCore import QSize, QTimer
        from PySide6.QtGui import QCursor
        
        # Store reference to close later
        if hasattr(self, '_active_popup') and self._active_popup:
            self._active_popup.close()
            self._active_popup = None
            return
        
        # Get main window as parent
        main_window = None
        for widget in QApplication.topLevelWidgets():
            if widget.isVisible() and widget.windowTitle():
                main_window = widget
                break
        
        # Create popup
        popup = QListWidget(main_window)
        popup.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        popup.setAttribute(Qt.WA_DeleteOnClose)
        popup.setAttribute(Qt.WA_ShowWithoutActivating, False)
        popup.setFocusPolicy(Qt.StrongFocus)
        popup.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        popup.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        popup.setStyleSheet("""
            QListWidget {
                background-color: rgb(35, 45, 60);
                border: 1px solid rgb(80, 100, 130);
                border-radius: 6px;
                outline: none;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px 16px;
                color: rgb(220, 230, 240);
                border-radius: 4px;
                margin: 1px 2px;
            }
            QListWidget::item:hover {
                background-color: rgb(60, 80, 110);
            }
            QListWidget::item:selected {
                background-color: rgb(70, 100, 140);
            }
            QScrollBar:vertical {
                background-color: rgb(30, 40, 55);
                width: 8px;
                border-radius: 4px;
                margin: 4px 2px;
            }
            QScrollBar::handle:vertical {
                background-color: rgb(100, 120, 150);
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)
        
        for reader in self._readers_list:
            item = QListWidgetItem(reader)
            item.setSizeHint(QSize(180, 30))
            popup.addItem(item)
            if reader == self._current_reader:
                item.setSelected(True)
                popup.setCurrentItem(item)
        
        self._active_popup = popup
        
        def on_select(item):
            self._select_reader(item.text())
            popup.close()
            self._active_popup = None
        
        def on_focus_out(event):
            # Close popup when it loses focus
            QTimer.singleShot(100, lambda: self._close_popup_if_not_focused(popup))
            QListWidget.focusOutEvent(popup, event)
        
        popup.itemClicked.connect(on_select)
        popup.focusOutEvent = on_focus_out
        
        # Size and position at cursor
        popup.setFixedSize(200, 280)
        popup.move(QCursor.pos())
        popup.show()
        popup.setFocus()
        popup.activateWindow()
    
    def _close_popup_if_not_focused(self, popup) -> None:
        """Close popup if it doesn't have focus."""
        try:
            # Check if popup is still a valid Qt object
            from shiboken6 import isValid
            if popup and isValid(popup) and not popup.hasFocus():
                popup.close()
                self._active_popup = None
        except (RuntimeError, ImportError):
            # Object already deleted or shiboken not available
            self._active_popup = None
    
    def _select_reader(self, reader_name: str) -> None:
        """Handle reader selection from menu."""
        global _READER_CHANGED_CALLBACK
        self._current_reader = reader_name
        self._reader_select.setText(f"{reader_name} ▼")
        self._rebuild_form(reader_name)
        self._on_settings_changed()
        # Notify properties window of reader change
        if _READER_CHANGED_CALLBACK:
            _READER_CHANGED_CALLBACK(reader_name)
    
    def _on_settings_changed(self) -> None:
        """Called when any setting changes - auto-load if path exists."""
        from PySide6.QtCore import QTimer
        QTimer.singleShot(500, self._check_and_load)
    
    def _check_and_load(self) -> None:
        """Check if path changed and auto-load."""
        kwargs = _collect_reader_kwargs(self._field_widgets)
        path = kwargs.get("path", "")
        if path and path != self._last_path:
            self._last_path = path
            self._load_preview()

    def _rebuild_form(self, reader_name: str) -> None:
        while self._form_layout.rowCount():
            self._form_layout.removeRow(0)
        self._field_widgets.clear()

        for field in _reader_fields(reader_name):
            widget = _build_field_widget(field, auto_load_callback=self._on_settings_changed)
            self._field_widgets[field["name"]] = widget
            self._form_layout.addRow(field["label"], widget)

    def _load_preview(self) -> None:
        global _DATASET_PREVIEW_CALLBACK, _EXTRA_READER_PARAMS
        reader_name = self._current_reader
        kwargs = _collect_reader_kwargs(self._field_widgets)

        path = kwargs.pop("path", None)
        if not path:
            self._status_label.setText("⚠ Please enter a path or URL first.")
            return

        self._status_label.setText("Loading...")

        try:
            func = getattr(pd, reader_name, None)
            if func is None:
                self._status_label.setText(f"❌ Unknown reader: {reader_name}")
                return

            # Build kwargs for the reader
            read_kwargs = {}
            for k, v in kwargs.items():
                if v not in (None, "", 0) or k in ("lines",):
                    read_kwargs[k] = v
            
            # Merge extra parameters from Node Properties window
            if _EXTRA_READER_PARAMS:
                read_kwargs.update(_EXTRA_READER_PARAMS)

            # Load full dataset
            if reader_name in ("read_csv", "read_table"):
                read_kwargs.pop("chunksize", None)
                df = func(path, **read_kwargs)
            elif reader_name == "read_json":
                read_kwargs.pop("chunksize", None)
                df = func(path, **read_kwargs)
            elif reader_name == "read_excel":
                df = func(path, **read_kwargs)
            elif reader_name == "read_html":
                tables = func(path, **read_kwargs)
                df = tables[0] if tables else pd.DataFrame()
            elif reader_name in ("read_sql", "read_sql_query"):
                query = kwargs.get("query", "")
                con = kwargs.get("con", "")
                if not query or not con:
                    self._status_label.setText("⚠ Query and connection required.")
                    return
                df = func(query, con)
            elif reader_name == "read_sql_table":
                table = kwargs.get("table", "")
                con = kwargs.get("con", "")
                if not table or not con:
                    self._status_label.setText("⚠ Table name and connection required.")
                    return
                df = func(table, con)
            else:
                df = func(path, **read_kwargs)

            self._status_label.setText(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

            # Store dataframe on parent node for selection callback
            if self._parent_node:
                self._parent_node.set_dataframe(df)

            if _DATASET_PREVIEW_CALLBACK:
                _DATASET_PREVIEW_CALLBACK(df)

        except FileNotFoundError:
            self._status_label.setText(f"❌ File not found: {path}")
        except Exception as e:
            error_msg = str(e)[:80]
            self._status_label.setText(f"❌ Error: {error_msg}")


def _all_pandas_readers() -> list[str]:
    return [
        "read_csv",
        "read_table",
        "read_parquet",
        "read_json",
        "read_excel",
        "read_feather",
        "read_pickle",
        "read_hdf",
        "read_stata",
        "read_sas",
        "read_spss",
        "read_orc",
        "read_xml",
        "read_html",
        "read_fwf",
        "read_clipboard",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "read_gbq",
    ]


def _reader_signature(reader_name: str) -> str:
    func = getattr(pd, reader_name, None)
    if func is None:
        return "Unknown reader."
    try:
        return str(inspect.signature(func))
    except Exception:
        return "Signature unavailable."


def _reader_fields(reader_name: str) -> list[dict]:
    base = [{"name": "path", "label": "Path / URL", "type": "path"}]
    if reader_name in ("read_csv", "read_table"):
        return base + [
            {"name": "sep", "label": "Delimiter", "type": "text", "value": ","},
            {"name": "encoding", "label": "Encoding", "type": "text", "value": ""},
            {"name": "chunksize", "label": "Chunk Size", "type": "spin", "min": 1000, "max": 100000, "value": 5000},
        ]
    if reader_name == "read_json":
        return base + [
            {"name": "lines", "label": "Lines (jsonl)", "type": "check", "value": True},
            {"name": "chunksize", "label": "Chunk Size", "type": "spin", "min": 1000, "max": 100000, "value": 5000},
        ]
    if reader_name == "read_excel":
        return base + [{"name": "sheet_name", "label": "Sheet", "type": "text", "value": "0"}]
    if reader_name in ("read_parquet", "read_feather", "read_pickle", "read_orc"):
        return base
    if reader_name == "read_xml":
        return base + [{"name": "xpath", "label": "XPath", "type": "text", "value": ""}]
    if reader_name == "read_html":
        return base + [{"name": "match", "label": "Match", "type": "text", "value": ""}]
    if reader_name == "read_fwf":
        return base + [{"name": "widths", "label": "Widths", "type": "text", "value": ""}]
    if reader_name in ("read_sql", "read_sql_query"):
        return [
            {"name": "query", "label": "SQL Query", "type": "multiline", "value": ""},
            {"name": "con", "label": "Connection String", "type": "text", "value": ""},
        ]
    if reader_name == "read_sql_table":
        return [
            {"name": "table", "label": "Table Name", "type": "text", "value": ""},
            {"name": "con", "label": "Connection String", "type": "text", "value": ""},
        ]
    if reader_name == "read_gbq":
        return [
            {"name": "query", "label": "GBQ Query", "type": "multiline", "value": ""},
            {"name": "project_id", "label": "Project ID", "type": "text", "value": ""},
        ]
    if reader_name in ("read_sas", "read_stata", "read_spss", "read_hdf"):
        return base
    if reader_name == "read_clipboard":
        return [{"name": "sep", "label": "Delimiter", "type": "text", "value": "\t"}]
    return base


def _build_field_widget(field: dict, auto_load_callback=None) -> QWidget:
    ftype = field.get("type")
    if ftype == "path":
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        path_edit = QLineEdit()
        path_edit.setFocusPolicy(Qt.StrongFocus)
        path_edit.setAttribute(Qt.WA_InputMethodEnabled, True)
        browse = QPushButton("Browse")
        browse.setFocusPolicy(Qt.StrongFocus)
        
        def on_browse():
            _browse_file(path_edit)
            # Trigger auto-load after file is selected
            if auto_load_callback:
                auto_load_callback()
        
        browse.clicked.connect(on_browse)
        # Also trigger on text edit finished
        if auto_load_callback:
            path_edit.editingFinished.connect(auto_load_callback)
        
        layout.addWidget(path_edit, 1)
        layout.addWidget(browse)
        return container
    if ftype == "spin":
        widget = QSpinBox()
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.setMinimum(field.get("min", 0))
        widget.setMaximum(field.get("max", 100))
        widget.setValue(field.get("value", 0))
        return widget
    if ftype == "check":
        widget = QCheckBox()
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.setChecked(field.get("value", False))
        return widget
    if ftype == "multiline":
        widget = QTextEdit()
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.setAttribute(Qt.WA_InputMethodEnabled, True)
        widget.setFixedHeight(60)
        widget.setPlainText(field.get("value", ""))
        return widget
    widget = QLineEdit()
    widget.setFocusPolicy(Qt.StrongFocus)
    widget.setAttribute(Qt.WA_InputMethodEnabled, True)
    widget.setText(field.get("value", ""))
    return widget


def _browse_file(target: QLineEdit) -> None:
    # Get the top-level window as parent to avoid covering issues
    from PySide6.QtWidgets import QApplication
    parent = None
    for widget in QApplication.topLevelWidgets():
        if widget.isVisible() and widget.windowTitle():
            parent = widget
            break
    
    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Select Dataset",
        "",
        "Data Files (*.csv *.xlsx *.xls *.json *.parquet *.feather *.pkl *.hdf *.stata *.sas *.spss *.orc *.xml);;All Files (*)",
    )
    if path:
        target.setText(path)


def _collect_reader_kwargs(field_widgets: dict) -> dict:
    """Collect values from form field widgets into kwargs dict."""
    kwargs: dict = {}
    for name, widget in field_widgets.items():
        if isinstance(widget, QLineEdit):
            value = widget.text().strip()
            if value:
                kwargs[name] = value
        elif isinstance(widget, QSpinBox):
            kwargs[name] = widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            kwargs[name] = widget.value()
        elif isinstance(widget, QCheckBox):
            kwargs[name] = widget.isChecked()
        elif isinstance(widget, QTextEdit):
            value = widget.toPlainText().strip()
            if value:
                kwargs[name] = value
        elif isinstance(widget, QWidget):
            # Path container with QLineEdit inside
            line_edit = widget.findChild(QLineEdit)
            if line_edit and line_edit.text().strip():
                kwargs[name] = line_edit.text().strip()
    return kwargs


def _infer_port_type(name: str) -> str:
    lower = name.lower()
    if "target" in lower or lower.startswith("y_"):
        return "target"
    if "schema" in lower or "categor" in lower:
        return "categorical"
    if "tensor" in lower:
        return "tensor"
    if "metric" in lower or "graph" in lower:
        return "metrics"
    return "numeric"


class ColumnSelectionDialog(QDialog):
    """
    Enhanced column selection dialog with role assignment.
    
    Features:
    - Column list with checkboxes
    - Role assignment (Feature/Target/Ignore)
    - Data type display
    - Missing value percentage
    - AI suggestions
    - Select All / None buttons
    """
    
    def __init__(self, parent=None, source_node=None) -> None:
        super().__init__(parent)
        self._source_node = source_node
        self._column_widgets: list[dict] = []
        self._truncated_column_count = 0
        # Guard rails to keep the connection dialog responsive on large datasets.
        self._max_columns_to_render = 200
        self._missing_stats_sample_rows = 5000
        
        self.setWindowTitle("🔗 Select Columns to Pass")
        self.setModal(True)
        self.resize(550, 480)
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(28, 35, 45, 250);
            }
            QLabel {
                color: rgba(220, 230, 240, 220);
            }
            QCheckBox {
                color: rgba(220, 230, 240, 220);
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid rgba(100, 120, 150, 180);
                background-color: rgba(30, 40, 55, 200);
            }
            QCheckBox::indicator:checked {
                background-color: rgba(70, 140, 200, 220);
            }
            QComboBox {
                background-color: rgba(40, 50, 65, 220);
                border: 1px solid rgba(100, 120, 150, 150);
                border-radius: 4px;
                padding: 4px 8px;
                color: rgba(220, 230, 240, 230);
            }
            QPushButton {
                background-color: rgba(50, 65, 85, 220);
                border: 1px solid rgba(100, 120, 150, 150);
                border-radius: 6px;
                padding: 8px 16px;
                color: rgba(220, 230, 240, 220);
            }
            QPushButton:hover {
                background-color: rgba(60, 80, 105, 230);
            }
            QPushButton:pressed {
                background-color: rgba(40, 55, 75, 230);
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("📊 Choose which columns to pass through this link")
        header.setStyleSheet("font-size: 14px; font-weight: 600; padding: 8px 0;")
        layout.addWidget(header)
        
        # Quick actions
        actions = QHBoxLayout()
        select_all = QPushButton("✓ Select All")
        select_all.clicked.connect(self._select_all)
        select_none = QPushButton("✗ Select None")
        select_none.clicked.connect(self._select_none)
        select_features = QPushButton("📊 Features Only")
        select_features.clicked.connect(self._select_features_only)
        actions.addWidget(select_all)
        actions.addWidget(select_none)
        actions.addWidget(select_features)
        actions.addStretch()
        layout.addLayout(actions)
        
        # Column list header
        col_header = QHBoxLayout()
        col_header.addWidget(QLabel("Column"), 2)
        col_header.addWidget(QLabel("Type"), 1)
        col_header.addWidget(QLabel("Role"), 1)
        col_header.addWidget(QLabel("Missing"), 1)
        layout.addLayout(col_header)
        
        # Scrollable column list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(280)
        
        self._columns_container = QWidget()
        self._columns_layout = QVBoxLayout(self._columns_container)
        self._columns_layout.setSpacing(4)
        self._columns_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll.setWidget(self._columns_container)
        layout.addWidget(scroll)
        
        # Populate columns
        self._populate_columns()
        
        # Summary
        self._summary = QLabel("0 columns selected")
        self._summary.setStyleSheet("color: rgba(150, 180, 220, 200); font-size: 12px;")
        layout.addWidget(self._summary)
        
        # AI suggestion (placeholder)
        ai_box = QHBoxLayout()
        ai_label = QLabel("🧠 AI Suggestion:")
        ai_label.setStyleSheet("color: rgba(150, 200, 255, 180);")
        self._ai_suggestion = QLabel("Consider marking 'target' as Target role")
        self._ai_suggestion.setStyleSheet("color: rgba(200, 220, 255, 150); font-style: italic;")
        ai_box.addWidget(ai_label)
        ai_box.addWidget(self._ai_suggestion, 1)
        layout.addLayout(ai_box)
        
        # Buttons
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("✓ Confirm Connection")
        confirm_btn.setStyleSheet("background-color: rgba(60, 140, 80, 220);")
        confirm_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(confirm_btn)
        layout.addLayout(buttons)
        
        self._update_summary()
    
    def _populate_columns(self) -> None:
        """Populate column list from source node data."""
        # Try to get columns from source node's dataframe
        columns_data = []
        
        if self._source_node:
            df = getattr(self._source_node, "_loaded_dataframe", None)
            if df is not None:
                try:
                    total_cols = len(df.columns)
                    visible_cols = list(df.columns[: self._max_columns_to_render])
                    self._truncated_column_count = max(0, total_cols - len(visible_cols))
                except Exception:
                    visible_cols = list(df.columns)
                    self._truncated_column_count = 0

                # Missing-percentage on full columns can freeze for large data.
                # Use a bounded sample to keep connection UX fast.
                try:
                    sampled_df = df.head(self._missing_stats_sample_rows)
                except Exception:
                    sampled_df = df

                sample_len = len(sampled_df) if sampled_df is not None else 0

                for col in visible_cols:
                    dtype = str(sampled_df[col].dtype if sampled_df is not None and col in sampled_df.columns else df[col].dtype)
                    if "int" in dtype or "float" in dtype:
                        col_type = "numeric"
                    elif "object" in dtype or "str" in dtype:
                        col_type = "categorical"
                    elif "datetime" in dtype:
                        col_type = "datetime"
                    elif "bool" in dtype:
                        col_type = "bool"
                    else:
                        col_type = "unknown"

                    if sample_len > 0 and sampled_df is not None and col in sampled_df.columns:
                        try:
                            missing_pct = (sampled_df[col].isna().sum() / sample_len) * 100
                        except Exception:
                            missing_pct = 0.0
                    else:
                        missing_pct = 0.0
                    columns_data.append({
                        "name": col,
                        "type": col_type,
                        "missing_pct": missing_pct,
                    })
        
        # Fallback to demo columns if no data
        if not columns_data:
            columns_data = [
                {"name": "id", "type": "numeric", "missing_pct": 0},
                {"name": "feature_1", "type": "numeric", "missing_pct": 2.5},
                {"name": "feature_2", "type": "numeric", "missing_pct": 0},
                {"name": "category", "type": "categorical", "missing_pct": 5.0},
                {"name": "target", "type": "numeric", "missing_pct": 0},
            ]
        
        for col_data in columns_data:
            self._add_column_row(col_data)
    
    def _add_column_row(self, col_data: dict) -> None:
        """Add a row for a column."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(8)
        
        # Checkbox with column name
        checkbox = QCheckBox(col_data["name"])
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(self._update_summary)
        row_layout.addWidget(checkbox, 2)
        
        # Type label with color
        type_label = QLabel(col_data["type"])
        type_colors = {
            "numeric": "rgba(70, 130, 200, 200)",
            "categorical": "rgba(70, 180, 120, 200)",
            "datetime": "rgba(150, 100, 200, 200)",
            "bool": "rgba(200, 150, 100, 200)",
            "unknown": "rgba(150, 150, 150, 200)",
        }
        type_label.setStyleSheet(f"color: {type_colors.get(col_data['type'], 'white')};")
        row_layout.addWidget(type_label, 1)
        
        # Role dropdown
        role_combo = QComboBox()
        role_combo.addItems(["Feature", "Target", "Ignore", "Index"])
        # Auto-suggest target role
        if "target" in col_data["name"].lower() or "label" in col_data["name"].lower():
            role_combo.setCurrentText("Target")
        elif "id" in col_data["name"].lower() or "index" in col_data["name"].lower():
            role_combo.setCurrentText("Index")
        row_layout.addWidget(role_combo, 1)
        
        # Missing percentage
        missing_pct = col_data.get("missing_pct", 0)
        missing_label = QLabel(f"{missing_pct:.1f}%")
        if missing_pct > 50:
            missing_label.setStyleSheet("color: rgba(220, 80, 80, 200);")
        elif missing_pct > 20:
            missing_label.setStyleSheet("color: rgba(230, 180, 50, 200);")
        else:
            missing_label.setStyleSheet("color: rgba(100, 200, 100, 200);")
        row_layout.addWidget(missing_label, 1)
        
        self._columns_layout.addWidget(row)
        self._column_widgets.append({
            "widget": row,
            "checkbox": checkbox,
            "role": role_combo,
            "data": col_data,
        })
    
    def _select_all(self) -> None:
        for item in self._column_widgets:
            item["checkbox"].setChecked(True)
    
    def _select_none(self) -> None:
        for item in self._column_widgets:
            item["checkbox"].setChecked(False)
    
    def _select_features_only(self) -> None:
        for item in self._column_widgets:
            role = item["role"].currentText()
            item["checkbox"].setChecked(role == "Feature")
    
    def _update_summary(self) -> None:
        selected = sum(1 for item in self._column_widgets if item["checkbox"].isChecked())
        total = len(self._column_widgets)
        features = sum(1 for item in self._column_widgets 
                      if item["checkbox"].isChecked() and item["role"].currentText() == "Feature")
        targets = sum(1 for item in self._column_widgets 
                     if item["checkbox"].isChecked() and item["role"].currentText() == "Target")
        summary = f"📊 {selected}/{total} columns selected • {features} features • {targets} targets"
        if self._truncated_column_count > 0:
            summary += f" • showing first {total} of {total + self._truncated_column_count}"
        self._summary.setText(summary)
    
    def get_selected_columns(self) -> list[dict]:
        """Return list of selected column configurations."""
        result = []
        for item in self._column_widgets:
            if item["checkbox"].isChecked():
                result.append({
                    "name": item["data"]["name"],
                    "type": item["data"]["type"],
                    "role": item["role"].currentText().lower(),
                    "enabled": True,
                    "missing_pct": item["data"].get("missing_pct", 0),
                })
        return result


class LinkInspectorDialog(QDialog):
    """
    Comprehensive link properties inspector.
    
    Features:
    - Source/Target info
    - Column configuration with role editing
    - Data statistics preview
    - Validation status
    - Performance estimation
    - Enable/Disable/Lock controls
    """
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._edge: EdgeItem | None = None
        
        self.setWindowTitle("🔗 Link Properties")
        self.setWindowFlag(Qt.Tool, True)
        self.resize(480, 600)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(
                    x1:0, y1:0, x2:0.3, y2:1,
                    stop:0 rgba(18, 28, 42, 248),
                    stop:1 rgba(12, 18, 30, 252)
                );
            }
            QLabel {
                color: rgba(180, 215, 255, 220);
            }
            QGroupBox {
                color: rgba(150, 195, 250, 210);
                border: 1px solid rgba(55, 110, 200, 50);
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background: rgba(28, 42, 64, 220);
                color: rgba(180, 215, 255, 230);
                border: 1px solid rgba(60, 120, 200, 50);
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background: rgba(40, 80, 140, 220);
                border-color: rgba(70, 150, 255, 100);
            }
        """)

        # Fade-in animation
        self._opacity_fx = QGraphicsOpacityEffect(self)
        self._opacity_fx.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_fx)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Header with state indicator
        header = QHBoxLayout()
        self._title = QLabel("📊 Link Properties")
        self._title.setStyleSheet("font-size: 16px; font-weight: 700; color: rgba(140, 200, 255, 240);")
        self._state_badge = QLabel("● Normal")
        self._state_badge.setStyleSheet("color: rgba(100, 200, 100, 220);")
        header.addWidget(self._title)
        header.addStretch()
        header.addWidget(self._state_badge)
        layout.addLayout(header)
        
        # Connection info
        from PySide6.QtWidgets import QGroupBox
        conn_group = QGroupBox("Connection")
        conn_layout = QVBoxLayout(conn_group)
        self._source_label = QLabel("From: —")
        self._target_label = QLabel("To: —")
        self._type_label = QLabel("Data Type: —")
        conn_layout.addWidget(self._source_label)
        conn_layout.addWidget(self._target_label)
        conn_layout.addWidget(self._type_label)
        layout.addWidget(conn_group)
        
        # Column configuration
        col_group = QGroupBox("Columns")
        col_layout = QVBoxLayout(col_group)
        
        self._columns_list = QListWidget()
        self._columns_list.setMaximumHeight(150)
        self._columns_list.setStyleSheet("""
            QListWidget {
                background-color: rgba(30, 40, 55, 180);
                border: 1px solid rgba(80, 100, 130, 150);
                border-radius: 4px;
            }
            QListWidget::item {
                color: rgba(220, 230, 240, 220);
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: rgba(70, 100, 140, 180);
            }
        """)
        col_layout.addWidget(self._columns_list)
        
        col_summary = QHBoxLayout()
        self._col_count_label = QLabel("0 columns")
        self._memory_label = QLabel("Memory: Unknown")
        col_summary.addWidget(self._col_count_label)
        col_summary.addStretch()
        col_summary.addWidget(self._memory_label)
        col_layout.addLayout(col_summary)
        layout.addWidget(col_group)
        
        # Validation status
        valid_group = QGroupBox("Validation")
        valid_layout = QVBoxLayout(valid_group)
        self._validation_status = QLabel("✓ Valid")
        self._validation_status.setStyleSheet("color: rgba(100, 200, 100, 220);")
        self._validation_details = QLabel("")
        self._validation_details.setWordWrap(True)
        self._validation_details.setStyleSheet("color: rgba(180, 180, 180, 180); font-size: 11px;")
        valid_layout.addWidget(self._validation_status)
        valid_layout.addWidget(self._validation_details)
        layout.addWidget(valid_group)
        
        # Controls
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)
        
        self._enabled_check = QCheckBox("Enabled")
        self._enabled_check.setChecked(True)
        self._enabled_check.setStyleSheet("color: rgba(220, 230, 240, 220);")
        
        self._locked_check = QCheckBox("Locked (read-only)")
        self._locked_check.setStyleSheet("color: rgba(220, 230, 240, 220);")
        
        ctrl_layout.addWidget(self._enabled_check)
        ctrl_layout.addWidget(self._locked_check)
        layout.addWidget(ctrl_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self._reconfigure_btn = QPushButton("🔧 Reconfigure Columns")
        self._reconfigure_btn.clicked.connect(self._reconfigure_columns)
        
        self._remove_btn = QPushButton("🗑️ Remove Link")
        self._remove_btn.setStyleSheet("background-color: rgba(180, 60, 60, 180);")
        self._remove_btn.clicked.connect(self._remove_link)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        
        btn_layout.addWidget(self._reconfigure_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self._remove_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        # Connect signals
        self._enabled_check.stateChanged.connect(self._toggle_enabled)
        self._locked_check.stateChanged.connect(self._toggle_locked)
    
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._show_anim = QPropertyAnimation(self._opacity_fx, b"opacity", self)
        self._show_anim.setDuration(250)
        self._show_anim.setStartValue(0.0)
        self._show_anim.setEndValue(1.0)
        self._show_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._show_anim.start()

    def set_link(self, edge: EdgeItem) -> None:
        """Set the link to display."""
        from nodes.base.link_model import LinkState
        
        self._edge = edge
        link = edge.link
        
        if not link:
            self._source_label.setText("From: Unknown")
            self._target_label.setText("To: Unknown")
            self._type_label.setText(f"Data Type: {edge._data_type}")
            self._col_count_label.setText(f"{len(edge.columns)} columns")
            self._columns_list.clear()
            for col in edge.columns:
                self._columns_list.addItem(col)
            return
        
        # Update connection info
        self._source_label.setText(f"From: {link.source_node_name} → {link.source_port_name}")
        self._target_label.setText(f"To: {link.target_node_name} → {link.target_port_name}")
        self._type_label.setText(f"Data Type: {link.data_type}")
        
        # Update state badge
        state_colors = {
            LinkState.NORMAL: ("● Normal", "rgba(100, 200, 100, 220)"),
            LinkState.WARNING: ("⚠ Warning", "rgba(230, 180, 50, 220)"),
            LinkState.ERROR: ("✗ Error", "rgba(220, 80, 80, 220)"),
            LinkState.DISABLED: ("○ Disabled", "rgba(150, 150, 150, 220)"),
            LinkState.LOCKED: ("🔒 Locked", "rgba(100, 140, 180, 220)"),
            LinkState.EXECUTING: ("⟳ Executing", "rgba(100, 200, 255, 220)"),
            LinkState.COMPLETED: ("✓ Completed", "rgba(100, 200, 100, 220)"),
        }
        text, color = state_colors.get(link.state, ("?", "gray"))
        self._state_badge.setText(text)
        self._state_badge.setStyleSheet(f"color: {color};")
        
        # Update columns
        self._columns_list.clear()
        for col in link.columns:
            role_icon = {"feature": "📊", "target": "🎯", "ignore": "⊘", "index": "#"}.get(col.role.value, "")
            type_str = col.data_type[:3] if col.data_type else "?"
            item_text = f"{role_icon} {col.name} [{type_str}]"
            if col.missing_pct > 0:
                item_text += f" ({col.missing_pct:.1f}% missing)"
            item = QListWidgetItem(item_text)
            if not col.enabled:
                item.setForeground(QColor(150, 150, 150))
            self._columns_list.addItem(item)
        
        # Update summary
        self._col_count_label.setText(f"{link.enabled_columns}/{link.total_columns} columns enabled")
        self._memory_label.setText(f"Memory: {link.estimated_memory_cost}")
        
        # Update validation
        if link.is_valid:
            if link.validation_warnings:
                self._validation_status.setText("⚠ Warnings")
                self._validation_status.setStyleSheet("color: rgba(230, 180, 50, 220);")
                self._validation_details.setText("\n".join(link.validation_warnings))
            else:
                self._validation_status.setText("✓ Valid")
                self._validation_status.setStyleSheet("color: rgba(100, 200, 100, 220);")
                self._validation_details.setText("No issues detected")
        else:
            self._validation_status.setText("✗ Errors")
            self._validation_status.setStyleSheet("color: rgba(220, 80, 80, 220);")
            self._validation_details.setText("\n".join(link.validation_errors))
        
        # Update controls
        self._enabled_check.setChecked(link.enabled)
        self._locked_check.setChecked(link.locked)
    
    def _toggle_enabled(self) -> None:
        from nodes.base.link_model import LinkState
        if self._edge and self._edge.link:
            enabled = self._enabled_check.isChecked()
            self._edge.link.enabled = enabled
            if not enabled:
                self._edge.set_state(LinkState.DISABLED)
            else:
                self._edge.link.validate()
                self._edge.set_state(self._edge.link.state)
    
    def _toggle_locked(self) -> None:
        from nodes.base.link_model import LinkState
        if self._edge and self._edge.link:
            locked = self._locked_check.isChecked()
            self._edge.link.locked = locked
            if locked:
                self._edge.set_state(LinkState.LOCKED)
            else:
                self._edge.link.validate()
                self._edge.set_state(self._edge.link.state)
    
    def _reconfigure_columns(self) -> None:
        """Open column selection dialog to reconfigure."""
        # TODO: Implement reconfiguration
        pass
    
    def _remove_link(self) -> None:
        """Remove the link from the scene."""
        if self._edge:
            scene = self._edge.scene()
            if scene:
                scene.removeItem(self._edge)
        self.close()