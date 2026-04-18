from __future__ import annotations

from pathlib import Path
import sys
import ctypes
import json
import base64
import hashlib

from PySide6.QtGui import (
    QIcon, QPixmap, QPalette, QColor, QPainter, QLinearGradient, QBrush, QAction,
)
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect,
    QParallelAnimationGroup, QSequentialAnimationGroup, Property, Signal, QObject, QThread, Slot,
)
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QFrame,
    QStatusBar,
    QScrollArea,
    QDockWidget,
    QDialog,
    QFileDialog,
    QGraphicsOpacityEffect,
    QMenu,
)

from ui.app_settings import AppSettings
from ui.windows.settings_window import SettingsWindow
from ui.windows.node_editor_window import NodeEditorWindow
from ui.windows.node_properties_window import NodePropertiesWindow
from ui.windows.data_window import DataPreviewWindow
from ui.windows.data_statistics_window import DataStatisticsWindow
from ui.windows.data_profiler_window import DataProfilerWindow
from ui.windows.node_output_window import NodeOutputWindow
from ui.windows.ai_advisor_window import AIAdvisorWindow
from ui.windows.model_output_window import ModelOutputWindow

try:
    import shiboken6  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shiboken6 = None


# ══════════════════════════════════════════════════════════════════════
#  Gradient background widget — fallback when acrylic blur is unavailable
# ══════════════════════════════════════════════════════════════════════

class _GradientBackground(QWidget):
    """Full-window gradient background painted behind everything."""

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QColor(12, 16, 24))
        grad.setColorAt(0.45, QColor(18, 26, 38))
        grad.setColorAt(1.0, QColor(10, 14, 22))
        p.fillRect(self.rect(), QBrush(grad))
        p.end()


# ══════════════════════════════════════════════════════════════════════
#  Animated card frame — fade-in / slide-up entrance + hover lift
# ══════════════════════════════════════════════════════════════════════

class _AnimatedCard(QFrame):
    """A card that fades & slides in on show and gently lifts on hover."""

    _HOVER_LIFT_PX = 6
    _ENTER_DURATION = 420  # ms

    def __init__(self, delay_ms: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._delay_ms = delay_ms
        self._base_y: int | None = None
        self._hovered = False
        self._hover_anim: QPropertyAnimation | None = None

        # Opacity effect for fade animation
        self._opacity_fx = QGraphicsOpacityEffect(self)
        self._opacity_fx.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_fx)

        # Animations (created lazily on first showEvent)
        self._entrance_played = False

    # -- entrance animation ---------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._entrance_played:
            return
        self._entrance_played = True
        QTimer.singleShot(self._delay_ms, self._play_entrance)

    def _play_entrance(self) -> None:
        if not self.isVisible():
            return
        # Fade in
        fade = QPropertyAnimation(self._opacity_fx, b"opacity", self)
        fade.setDuration(self._ENTER_DURATION)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.OutCubic)

        # Slide up
        start_pos = QPoint(self.x(), self.y() + 30)
        end_pos = QPoint(self.x(), self.y())
        slide = QPropertyAnimation(self, b"pos", self)
        slide.setDuration(self._ENTER_DURATION)
        slide.setStartValue(start_pos)
        slide.setEndValue(end_pos)
        slide.setEasingCurve(QEasingCurve.OutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(fade)
        group.addAnimation(slide)
        group.start()

    # -- hover lift -----------------------------------------------------------
    def enterEvent(self, event) -> None:  # noqa: N802
        super().enterEvent(event)
        if self._hovered:
            return
        self._hovered = True
        # Refresh base position from current layout position so leave animation
        # always restores correctly after resizes/reflows.
        self._base_y = self.y()
        if self._hover_anim is not None:
            self._hover_anim.stop()
        self._hover_anim = QPropertyAnimation(self, b"pos", self)
        self._hover_anim.setDuration(180)
        self._hover_anim.setStartValue(QPoint(self.x(), self.y()))
        self._hover_anim.setEndValue(QPoint(self.x(), self._base_y - self._HOVER_LIFT_PX))
        self._hover_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._hover_anim.start()

    def leaveEvent(self, event) -> None:  # noqa: N802
        super().leaveEvent(event)
        if not self._hovered:
            return
        self._hovered = False
        if self._base_y is None:
            return
        if self._hover_anim is not None:
            self._hover_anim.stop()
        self._hover_anim = QPropertyAnimation(self, b"pos", self)
        self._hover_anim.setDuration(180)
        self._hover_anim.setStartValue(QPoint(self.x(), self.y()))
        self._hover_anim.setEndValue(QPoint(self.x(), self._base_y))
        self._hover_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._hover_anim.start()


# ══════════════════════════════════════════════════════════════════════
#  Animated push-button with hover glow
# ══════════════════════════════════════════════════════════════════════

class _GlowButton(QPushButton):
    """QPushButton that smoothly interpolates its background colour on hover."""

    _NORMAL = "rgba(28, 40, 56, 220)"
    _HOVER  = "rgba(40, 90, 160, 240)"
    _PRESSED = "rgba(20, 50, 100, 255)"
    _BORDER_NORMAL = "rgba(80, 140, 220, 50)"
    _BORDER_HOVER  = "rgba(80, 160, 255, 120)"

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._bg = QColor(28, 40, 56, 220)
        self._apply_style()

    # Expose QColor property for animation
    def _get_bg(self) -> QColor:
        return self._bg

    def _set_bg(self, c: QColor) -> None:
        self._bg = c
        self._apply_style()

    bgColor = Property(QColor, _get_bg, _set_bg)

    def _apply_style(self) -> None:
        r, g, b, a = self._bg.red(), self._bg.green(), self._bg.blue(), self._bg.alpha()
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba({r},{g},{b},{a});
                color: rgba(220, 235, 255, 240);
                border: 1px solid {self._BORDER_NORMAL};
                border-radius: 7px;
                padding: 7px 14px;
                font-weight: 500;
            }}
        """)

    def enterEvent(self, event) -> None:  # noqa: N802
        super().enterEvent(event)
        self._color_anim = QPropertyAnimation(self, b"bgColor", self)
        self._color_anim.setDuration(200)
        self._color_anim.setEndValue(QColor(40, 90, 160, 240))
        self._color_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._color_anim.start()


class _ProjectSaveWorker(QObject):
    save_finished = Signal(dict)

    @Slot(dict)
    def write_project(self, request: dict) -> None:
        path = str(request.get("path", "") or "")
        payload = request.get("payload", {})
        autosave = bool(request.get("autosave", False))
        if not path:
            self.save_finished.emit({"ok": False, "path": path, "autosave": autosave, "error": "Missing save path"})
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.save_finished.emit({"ok": True, "path": path, "autosave": autosave, "error": ""})
        except Exception as e:
            self.save_finished.emit({"ok": False, "path": path, "autosave": autosave, "error": str(e)})

    def leaveEvent(self, event) -> None:  # noqa: N802
        super().leaveEvent(event)
        self._color_anim = QPropertyAnimation(self, b"bgColor", self)
        self._color_anim.setDuration(280)
        self._color_anim.setEndValue(QColor(28, 40, 56, 220))
        self._color_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._color_anim.start()


class MainWindow(QMainWindow):
    _save_requested = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Typhydrion")
        self.setMinimumSize(1100, 700)
        self._settings = AppSettings()
        self._current_project_path: str | None = None
        self._project_dirty: bool = False
        self._suppress_dirty_tracking: bool = False
        self._last_clean_fingerprint: str | None = None
        self._autosave_timer: QTimer | None = None
        self._autosave_inflight: bool = False
        self._autosave_pending: bool = False
        self._base_font_point_size: int | None = None
        self._base_font_pixel_size: int | None = None
        self._logo_pixmap: QPixmap | None = None
        self._logo_label: QLabel | None = None
        self._menu_card: QFrame | None = None
        self._recent_card: QFrame | None = None
        self._recent_menu: QMenu | None = None
        self._recent_menu_empty_action: QAction | None = None
        self._recent_body_layout: QVBoxLayout | None = None
        self._todo_card: QFrame | None = None
        self._workspace_built = False
        self._view_menu = None
        self._workspace_menu = None
        self._dock_by_title: dict[str, QDockWidget] = {}
        self._data_preview_widget = None
        self._data_profiler_widget: DataProfilerWindow | None = None
        self._acrylic_ok = False  # will be True if Windows blur succeeded
        self._save_thread = QThread(self)
        self._save_worker = _ProjectSaveWorker()
        self._save_worker.moveToThread(self._save_thread)
        self._save_requested.connect(self._save_worker.write_project, Qt.QueuedConnection)
        self._save_worker.save_finished.connect(self._on_async_save_finished, Qt.QueuedConnection)
        self._save_thread.start()
        try:
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self._shutdown_background_threads)
        except Exception:
            pass
        self._set_app_icon()
        self._enable_translucency()
        self._build_menu()
        self._build_home()

        # Apply settings after UI is built; also handles autosave/open-last.
        QTimer.singleShot(50, self._apply_settings_to_app)

    def _set_app_icon(self) -> None:
        icon_path = self._project_root() / "assets" / "icons" / "app_logo.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _enable_translucency(self) -> None:
        # Windows-only blur effects
        if sys.platform != "win32":
            self._acrylic_ok = False
            return

        # User setting can disable acrylic/blur
        try:
            if not self._settings.get_bool(self._settings.UI_TRANSPARENCY_ENABLED):
                self._acrylic_ok = False
                return
        except Exception:
            pass

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        hwnd = int(self.winId())
        # Allow configurable acrylic gradient
        gradient = None
        try:
            gradient = self._settings.get_int(self._settings.UI_ACRYLIC_GRADIENT_HEX)
        except Exception:
            gradient = None
        if self._enable_acrylic_blur(hwnd, gradient_override=gradient):
            self._acrylic_ok = True
        else:
            self._acrylic_ok = bool(self._enable_aero_blur(hwnd))

    def _enable_aero_blur(self, hwnd: int) -> bool:
        try:
            class DWM_BLURBEHIND(ctypes.Structure):
                _fields_ = [
                    ("dwFlags", ctypes.c_uint),
                    ("fEnable", ctypes.c_int),
                    ("hRgnBlur", ctypes.c_void_p),
                    ("fTransitionOnMaximized", ctypes.c_int),
                ]

            DWM_BB_ENABLE = 0x00000001
            blur_behind = DWM_BLURBEHIND(DWM_BB_ENABLE, 1, None, 0)
            ctypes.windll.dwmapi.DwmEnableBlurBehindWindow(
                ctypes.c_void_p(hwnd),
                ctypes.byref(blur_behind),
            )
            return True
        except Exception:
            return False

    def _enable_acrylic_blur(self, hwnd: int, gradient_override: int | None = None) -> bool:
        class ACCENT_POLICY(ctypes.Structure):
            _fields_ = [
                ("AccentState", ctypes.c_int),
                ("AccentFlags", ctypes.c_int),
                ("GradientColor", ctypes.c_int),
                ("AnimationId", ctypes.c_int),
            ]

        class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
            _fields_ = [
                ("Attribute", ctypes.c_int),
                ("Data", ctypes.c_void_p),
                ("SizeOfData", ctypes.c_size_t),
            ]

        WCA_ACCENT_POLICY = 19
        ACCENT_ENABLE_ACRYLICBLURBEHIND = 4
        ACCENT_ENABLE_BLURBEHIND = 3

        # ARGB: 0xAABBGGRR (lower alpha = lighter blur)
        gradient = int(gradient_override) if gradient_override is not None else 0x518e7400
        policy = ACCENT_POLICY(
            ACCENT_ENABLE_ACRYLICBLURBEHIND,
            0,
            gradient,
            0,
        )
        data = WINDOWCOMPOSITIONATTRIBDATA(
            WCA_ACCENT_POLICY,
            ctypes.cast(ctypes.pointer(policy), ctypes.c_void_p),
            ctypes.sizeof(policy),
        )

        set_wca = getattr(ctypes.windll.user32, "SetWindowCompositionAttribute", None)
        if not set_wca:
            return False

        result = set_wca(ctypes.c_void_p(hwnd), ctypes.byref(data))
        if result == 0:
            # Fallback to basic blur if acrylic fails
            policy.AccentState = ACCENT_ENABLE_BLURBEHIND
            data.Data = ctypes.cast(ctypes.pointer(policy), ctypes.c_void_p)
            return bool(set_wca(ctypes.c_void_p(hwnd), ctypes.byref(data)))

        return True

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet("""
            QMenuBar {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(14, 22, 34, 240),
                    stop:0.5 rgba(18, 28, 42, 240),
                    stop:1 rgba(14, 22, 34, 240)
                );
                color: rgba(170, 210, 255, 230);
                border-bottom: 1px solid rgba(50, 110, 200, 35);
                padding: 2px 4px;
                font-weight: 500;
            }
            QMenuBar::item {
                padding: 5px 12px;
                border-radius: 4px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background: rgba(40, 80, 160, 120);
            }
            QMenu {
                background-color: rgba(18, 26, 40, 240);
                border: 1px solid rgba(60, 120, 200, 40);
                border-radius: 6px;
                padding: 6px 0;
                color: rgba(180, 215, 255, 230);
            }
            QMenu::item {
                padding: 6px 28px;
            }
            QMenu::item:selected {
                background: rgba(40, 80, 160, 130);
                border-radius: 3px;
            }
            QMenu::separator {
                height: 1px;
                background: rgba(60, 120, 200, 30);
                margin: 4px 12px;
            }
        """)

        file_menu = menu_bar.addMenu("File")
        action_home = file_menu.addAction("Home")
        action_new = file_menu.addAction("New Project")
        action_open = file_menu.addAction("Open Project...")
        action_templates = file_menu.addAction("Templates...")
        action_settings = file_menu.addAction("Settings")
        action_load_data = file_menu.addAction("Load Dataset...")
        recent_menu = file_menu.addMenu("Recent")
        self._recent_menu = recent_menu
        self._recent_menu_empty_action = recent_menu.addAction("No recent projects")
        self._recent_menu_empty_action.setEnabled(False)
        action_save = file_menu.addAction("Save Project")
        action_save_latest = file_menu.addAction("Save Latest Project")
        action_save_as = file_menu.addAction("Save Project As...")
        file_menu.addSeparator()
        action_exit = file_menu.addAction("Exit")

        view_menu = menu_bar.addMenu("View")
        action_reset_layout = view_menu.addAction("Reset Layout")

        workspace_menu = menu_bar.addMenu("Workspace")
        action_data_view = workspace_menu.addAction("Data View")
        action_data_stats = workspace_menu.addAction("Data Statistics")
        action_ai = workspace_menu.addAction("AI / Model Control")

        analysis_menu = menu_bar.addMenu("Data Analysis")
        action_open_profiler = analysis_menu.addAction("Open Data Profiler")
        analysis_menu.addSeparator()
        action_focus_preview = analysis_menu.addAction("Open Data Preview")
        action_focus_stats = analysis_menu.addAction("Open Data Statistics")
        analysis_menu.addSeparator()
        action_add_corr = analysis_menu.addAction("Add Correlation Matrix (Heatmap)")
        action_add_missing = analysis_menu.addAction("Add Missing Values Chart")
        action_add_target = analysis_menu.addAction("Add Target Distribution Chart")

        help_menu = menu_bar.addMenu("Help")
        action_about = help_menu.addAction("About")

        action_home.triggered.connect(self._go_home)
        action_new.triggered.connect(self._start_new_project)
        action_open.triggered.connect(self._open_project_file)
        action_templates.triggered.connect(self._start_template_project)
        action_settings.triggered.connect(self._open_settings)
        action_load_data.triggered.connect(self._placeholder_action)
        action_save.triggered.connect(self._save_project_file)
        action_save_latest.triggered.connect(self._save_latest_project)
        action_save_as.triggered.connect(lambda: self._save_project_file(save_as=True))
        action_reset_layout.triggered.connect(self._reset_layout)
        action_data_view.triggered.connect(lambda: self._focus_dock("Data Preview"))
        action_data_stats.triggered.connect(lambda: self._focus_dock("Data Statistics"))
        action_ai.triggered.connect(lambda: self._focus_dock("AI Advisor"))
        action_exit.triggered.connect(self.close)
        action_about.triggered.connect(self._show_about)

        def ensure_workspace_then(fn):
            def _wrapped():
                if not self._workspace_built:
                    self._build_workspace()
                fn()
            return _wrapped

        action_open_profiler.triggered.connect(self._open_data_analysis_profiler)
        action_focus_preview.triggered.connect(ensure_workspace_then(lambda: self._focus_dock("Data Preview")))
        action_focus_stats.triggered.connect(ensure_workspace_then(lambda: self._focus_dock("Data Statistics")))

        # Graph/visualization window removed.
        action_add_corr.setEnabled(False)
        action_add_missing.setEnabled(False)
        action_add_target.setEnabled(False)

        self._action_new = action_new
        self._action_open = action_open
        self._action_load_data = action_load_data
        self._view_menu = view_menu
        self._workspace_menu = workspace_menu
        self._refresh_recent_projects_ui()

    def _build_home(self) -> None:
        # Use gradient background as the central widget so it paints behind everything
        central = _GradientBackground()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)

        # Logo with fade-in animation
        self._logo_label = QLabel()
        self._logo_label.setAlignment(Qt.AlignCenter)
        self._logo_label.setStyleSheet("background: transparent;")
        logo_path = self._project_root() / "assets" / "icons" / "app_logo.png"
        if logo_path.exists():
            self._logo_pixmap = QPixmap(str(logo_path))
            self._update_logo_size()
        else:
            self._logo_label.setText("Typhydrion")
            self._logo_label.setStyleSheet(
                "font-size: 36px; font-weight: 700; color: rgba(120, 180, 255, 220); "
                "background: transparent;"
            )

        layout.addWidget(self._logo_label)

        # Animate logo fade
        logo_opacity = QGraphicsOpacityEffect(self._logo_label)
        logo_opacity.setOpacity(0.0)
        self._logo_label.setGraphicsEffect(logo_opacity)
        self._logo_fade = QPropertyAnimation(logo_opacity, b"opacity", self)
        self._logo_fade.setDuration(700)
        self._logo_fade.setStartValue(0.0)
        self._logo_fade.setEndValue(1.0)
        self._logo_fade.setEasingCurve(QEasingCurve.OutCubic)
        QTimer.singleShot(100, self._logo_fade.start)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(28)
        cards_row.setContentsMargins(40, 28, 40, 0)
        cards_row.addStretch(1)

        self._menu_card = self._build_menu_card(delay_ms=200)
        self._recent_card = self._build_recent_card(delay_ms=340)
        self._todo_card = self._build_card(
            title="Todos",
            body="No tasks yet",
            delay_ms=480,
        )

        cards_row.addWidget(self._menu_card)
        cards_row.addWidget(self._recent_card)
        cards_row.addWidget(self._todo_card)
        cards_row.addStretch(1)
        layout.addLayout(cards_row)
        layout.addStretch(2)
        self.setCentralWidget(central)
        self._build_footer()
        self._refresh_recent_projects_ui()

    def _start_new_project(self) -> None:
        if not self._confirm_continue_with_unsaved_changes():
            return
        if not self._workspace_built:
            self._build_workspace()
        self._current_project_path = None
        self._set_project_dirty(False)
        self._last_clean_fingerprint = self._project_state_fingerprint()
        self.statusBar().showMessage("New project ready.")

    def _project_file_filter(self) -> str:
        return "Typhydrion Project (*.typhyproj)"

    def _build_project_payload(self, include_dataframes: bool = True) -> dict:
        node_editor = self.centralWidget()
        if hasattr(node_editor, "to_dict"):
            try:
                graph_payload = node_editor.to_dict(include_dataframes=include_dataframes)
            except TypeError:
                graph_payload = node_editor.to_dict()
        else:
            graph_payload = {}
        return {
            "version": 1,
            "app": "Typhydrion",
            "qt": {
                "geometry_b64": base64.b64encode(bytes(self.saveGeometry())).decode("utf-8"),
                "state_b64": base64.b64encode(bytes(self.saveState())).decode("utf-8"),
            },
            "node_editor": graph_payload,
        }

    def _queue_async_save(self, path: str, payload: dict, autosave: bool) -> None:
        if not path:
            return
        self._save_requested.emit({"path": path, "payload": payload, "autosave": bool(autosave)})

    @Slot(dict)
    def _on_async_save_finished(self, result: dict) -> None:
        autosave = bool(result.get("autosave", False))
        ok = bool(result.get("ok", False))
        path = str(result.get("path", "") or "")
        error = str(result.get("error", "") or "")

        if autosave:
            self._autosave_inflight = False

        if ok:
            try:
                if path:
                    self._settings.add_recent_project(path)
                    self._refresh_recent_projects_ui()
            except Exception:
                pass
            self._set_project_dirty(False)
            self._last_clean_fingerprint = None
            name = Path(path).name if path else "project"
            if autosave:
                self.statusBar().showMessage(f"Autosaved: {name}")
            else:
                self.statusBar().showMessage(f"Saved: {name}")
        else:
            if autosave:
                self.statusBar().showMessage(f"Autosave failed: {error}")
            else:
                QMessageBox.warning(self, "Save Failed", f"Could not save project:\n{error}")

        if autosave and self._autosave_pending:
            self._autosave_pending = False
            QTimer.singleShot(100, self._autosave_tick)

    def _save_project_file(self, save_as: bool = False) -> bool:
        """Save current workspace to a .typhyproj file (nodes + graph cards + dock layout)."""
        if not self._workspace_built:
            self._build_workspace()

        if save_as or not self._current_project_path:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Project",
                self._current_project_path or "",
                self._project_file_filter(),
            )
            if not path:
                return False
            if not path.lower().endswith(".typhyproj"):
                path = path + ".typhyproj"
            self._current_project_path = path

        payload = self._build_project_payload()

        try:
            with open(self._current_project_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            try:
                self._settings.add_recent_project(self._current_project_path)
                self._refresh_recent_projects_ui()
            except Exception:
                pass
            self._set_project_dirty(False)
            self._last_clean_fingerprint = self._project_state_fingerprint()
            self.statusBar().showMessage(f"Saved: {Path(self._current_project_path).name}")
            return True
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Could not save project:\n{e}")
            return False

    def _save_latest_project(self) -> None:
        """Save directly to the latest/current project path."""
        if self._current_project_path:
            self._save_project_file(save_as=False)
            return
        # No known latest path yet: fall back to Save As once.
        self._save_project_file(save_as=True)

    def _open_project_file(self) -> None:
        """Open a .typhyproj file and restore workspace state."""
        if not self._confirm_continue_with_unsaved_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            self._current_project_path or "",
            self._project_file_filter(),
        )
        if not path:
            return

        # Use unified loader so we always update recents/last path and apply layout.
        self._open_project_file_path(path)
        return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Open Failed", f"Could not open project:\n{e}")
            return

        if not self._workspace_built:
            self._build_workspace()

        self._current_project_path = path

        node_editor = self.centralWidget()

        try:
            if hasattr(node_editor, "from_dict"):
                node_editor.from_dict(payload.get("node_editor", {}))
        except Exception as e:
            QMessageBox.warning(self, "Load Warning", f"Node graph restore failed:\n{e}")

        # Restore dock layout last (so docks exist)
        qt = payload.get("qt", {})
        try:
            geom_b64 = qt.get("geometry_b64", "")
            state_b64 = qt.get("state_b64", "")
            if geom_b64:
                self.restoreGeometry(base64.b64decode(geom_b64))
            if state_b64:
                self.restoreState(base64.b64decode(state_b64))
        except Exception:
            # Non-fatal
            pass

        self.statusBar().showMessage(f"Opened: {Path(path).name}")

    def _start_template_project(self) -> None:
        if not self._workspace_built:
            self._build_workspace()
        self.statusBar().showMessage("Template project loaded.")

    def _open_settings(self) -> None:
        dlg = SettingsWindow(self)
        dlg.settings_applied.connect(self._apply_settings_to_app)
        dlg.exec()

    def _apply_settings_to_app(self) -> None:
        """Apply persisted settings to running components."""
        # Theme (best-effort; many widgets have their own stylesheets)
        try:
            app = QApplication.instance()
            if app is not None:
                theme = self._settings.get_str(self._settings.UI_THEME).lower()
                if theme in {"dark", "light"}:
                    app.setStyle("Fusion")
                    pal = QPalette()
                    if theme == "dark":
                        pal.setColor(QPalette.Window, QColor(22, 28, 36))
                        pal.setColor(QPalette.WindowText, QColor(235, 240, 245))
                        pal.setColor(QPalette.Base, QColor(18, 22, 28))
                        pal.setColor(QPalette.AlternateBase, QColor(28, 34, 42))
                        pal.setColor(QPalette.ToolTipBase, QColor(235, 240, 245))
                        pal.setColor(QPalette.ToolTipText, QColor(235, 240, 245))
                        pal.setColor(QPalette.Text, QColor(235, 240, 245))
                        pal.setColor(QPalette.Button, QColor(30, 38, 48))
                        pal.setColor(QPalette.ButtonText, QColor(235, 240, 245))
                        pal.setColor(QPalette.BrightText, QColor(255, 80, 80))
                        pal.setColor(QPalette.Highlight, QColor(17, 141, 255))
                        pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
                    else:
                        pal = app.style().standardPalette()
                    app.setPalette(pal)

                    # Global stylesheet for consistent colours & subtle transitions
                    if theme == "dark":
                        app.setStyleSheet("""
                            QToolTip {
                                background: rgba(18, 28, 44, 240);
                                color: rgba(180, 215, 255, 230);
                                border: 1px solid rgba(55, 110, 200, 60);
                                border-radius: 4px;
                                padding: 4px 8px;
                            }
                            QScrollBar:vertical {
                                background: rgba(14, 20, 30, 180);
                                width: 8px;
                                border-radius: 4px;
                            }
                            QScrollBar::handle:vertical {
                                background: rgba(55, 100, 180, 120);
                                border-radius: 4px;
                                min-height: 24px;
                            }
                            QScrollBar::handle:vertical:hover {
                                background: rgba(70, 130, 220, 160);
                            }
                            QScrollBar:horizontal {
                                background: rgba(14, 20, 30, 180);
                                height: 8px;
                                border-radius: 4px;
                            }
                            QScrollBar::handle:horizontal {
                                background: rgba(55, 100, 180, 120);
                                border-radius: 4px;
                                min-width: 24px;
                            }
                            QScrollBar::handle:horizontal:hover {
                                background: rgba(70, 130, 220, 160);
                            }
                            QScrollBar::add-line, QScrollBar::sub-line {
                                width: 0; height: 0;
                            }
                        """)
                    else:
                        app.setStyleSheet("")
        except Exception:
            pass

        # Global font scaling
        try:
            app = QApplication.instance()
            if app is not None:
                f = app.font()
                scale = max(70, min(160, int(self._settings.get_int(self._settings.UI_FONT_SCALE_PERCENT))))
                # Some Qt fonts are pixel-sized (pointSize() == -1). Scale both paths safely.
                cur_pt = int(f.pointSize()) if f.pointSize() and f.pointSize() > 0 else 0
                if cur_pt > 0:
                    if self._base_font_point_size is None:
                        self._base_font_point_size = max(6, cur_pt)
                    new_pt = max(7, int(round(self._base_font_point_size * (scale / 100.0))))
                    if new_pt != cur_pt:
                        f.setPointSize(new_pt)
                        app.setFont(f)
                else:
                    cur_px = int(f.pixelSize()) if f.pixelSize() and f.pixelSize() > 0 else 0
                    if cur_px > 0:
                        if self._base_font_pixel_size is None:
                            self._base_font_pixel_size = max(9, cur_px)
                        new_px = max(9, int(round(self._base_font_pixel_size * (scale / 100.0))))
                        if new_px != cur_px:
                            f.setPixelSize(new_px)
                            app.setFont(f)
        except Exception:
            pass

        # Refresh transparency on Windows (best-effort)
        try:
            if sys.platform == "win32" and self._settings.get_bool(self._settings.UI_TRANSPARENCY_ENABLED):
                hwnd = int(self.winId())
                gradient = self._settings.get_int(self._settings.UI_ACRYLIC_GRADIENT_HEX)
                self._enable_acrylic_blur(hwnd, gradient_override=gradient)
        except Exception:
            pass

        # Autosave timer
        try:
            enabled = self._settings.get_bool(self._settings.AUTOSAVE_ENABLED)
            interval_ms = max(10_000, self._settings.get_int(self._settings.AUTOSAVE_INTERVAL_SEC) * 1000)
            if enabled:
                if self._autosave_timer is None:
                    self._autosave_timer = QTimer(self)
                    self._autosave_timer.timeout.connect(self._autosave_tick)
                self._autosave_timer.start(interval_ms)
            else:
                if self._autosave_timer is not None:
                    self._autosave_timer.stop()
        except Exception:
            pass

        # Apply settings to workspace widgets if they exist
        try:
            if self._workspace_built and self.centralWidget() is not None:
                node_editor = self.centralWidget()
                if hasattr(node_editor, "apply_settings"):
                    node_editor.apply_settings()
        except Exception:
            pass

        # Open last project (once, right after startup)
        try:
            startup = self._settings.get_str(self._settings.UI_STARTUP_MODE).lower()
            if not self._workspace_built and startup == "last_project" and self._settings.get_bool(self._settings.PROJECT_OPEN_LAST):
                last_path = self._settings.get_str(self._settings.PROJECT_LAST_PATH)
                if last_path:
                    self._open_project_file_path(last_path)
        except Exception:
            pass

    def _autosave_tick(self) -> None:
        if not self._workspace_built:
            return
        # Autosave only if we have some path to write to without prompting.
        try:
            if not self._current_project_path:
                folder = self._settings.get_str(self._settings.PROJECT_DEFAULT_FOLDER) or str(self._project_root())
                self._current_project_path = str(Path(folder) / "autosave.typhyproj")
            if self._autosave_inflight:
                self._autosave_pending = True
                return
            payload = self._build_project_payload(include_dataframes=False)
            self._autosave_inflight = True
            self._autosave_pending = False
            self._queue_async_save(self._current_project_path, payload, autosave=True)
        except Exception:
            self._autosave_inflight = False
            pass

    def _open_project_file_path(self, path: str) -> None:
        """Open project from a known path (no dialogs)."""
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Open Failed", f"Could not open project:\n{e}")
            return

        if not self._workspace_built:
            self._build_workspace()

        self._suppress_dirty_tracking = True
        self._current_project_path = path
        self._settings.add_recent_project(path)
        self._refresh_recent_projects_ui()

        node_editor = self.centralWidget()

        try:
            if hasattr(node_editor, "from_dict"):
                node_editor.from_dict(payload.get("node_editor", {}))
        except Exception as e:
            QMessageBox.warning(self, "Load Warning", f"Node graph restore failed:\n{e}")

        qt = payload.get("qt", {})
        try:
            geom_b64 = qt.get("geometry_b64", "")
            state_b64 = qt.get("state_b64", "")
            if geom_b64:
                self.restoreGeometry(base64.b64decode(geom_b64))
            if state_b64:
                self.restoreState(base64.b64decode(state_b64))
        except Exception:
            pass
        finally:
            self._suppress_dirty_tracking = False
            self._set_project_dirty(False)
            self._last_clean_fingerprint = self._project_state_fingerprint()

        self.statusBar().showMessage(f"Opened: {Path(path).name}")

    def _go_home(self) -> None:
        if not self._confirm_continue_with_unsaved_changes():
            return
        self._shutdown_background_threads()
        self._workspace_built = False
        self._logo_label = None
        self._menu_card = None
        self._recent_card = None
        self._recent_body_layout = None
        self._todo_card = None
        self._dock_by_title.clear()
        self.setCentralWidget(QWidget())
        for dock in self.findChildren(QDockWidget):
            self.removeDockWidget(dock)
            dock.deleteLater()
        self._build_home()

    def _build_workspace(self) -> None:
        self._shutdown_background_threads(stop_save_thread=False)
        self._workspace_built = True
        self._logo_label = None
        self._menu_card = None
        self._recent_card = None
        self._recent_body_layout = None
        self._todo_card = None
        node_editor = NodeEditorWindow()
        self.setCentralWidget(node_editor)
        
        # ═══════════════════════════════════════════════════════════════
        # LEFT PANEL - AI Advisor
        # ═══════════════════════════════════════════════════════════════
        self._add_dock(
            "AI Advisor",
            AIAdvisorWindow(),
            Qt.LeftDockWidgetArea,
            "rgba(48, 54, 42, 200)",
        )
        
        # ═══════════════════════════════════════════════════════════════
        # RIGHT PANELS - Data Preview, Data Statistics, Node Properties, Model Output
        # ═══════════════════════════════════════════════════════════════
        
        # Data Preview panel (first on right)
        data_preview = DataPreviewWindow()
        self._data_preview_widget = data_preview
        self._add_dock(
            "Data Preview",
            data_preview,
            Qt.RightDockWidgetArea,
            "rgba(32, 56, 64, 200)",
        )
        
        # Data Statistics panel
        data_statistics = DataStatisticsWindow()
        self._data_statistics_widget = data_statistics
        self._add_dock(
            "Data Statistics",
            data_statistics,
            Qt.RightDockWidgetArea,
            "rgba(36, 48, 72, 200)",
        )

        # Node Output panel
        node_output = NodeOutputWindow()
        self._node_output_widget = node_output
        self._add_dock(
            "Node Output",
            node_output,
            Qt.RightDockWidgetArea,
            "rgba(42, 52, 68, 200)",
        )
        
        # Node Properties panel
        node_properties = NodePropertiesWindow()
        self._node_properties_widget = node_properties
        self._add_dock(
            "Node Properties",
            node_properties,
            Qt.RightDockWidgetArea,
            "rgba(34, 46, 66, 200)",
        )
        
        # Model Output panel
        self._add_dock(
            "Model Output",
            ModelOutputWindow(),
            Qt.RightDockWidgetArea,
            "rgba(40, 44, 58, 200)",
        )
        
        # Tabify all right-side docks together
        data_preview_dock = self._dock_by_title.get("Data Preview")
        data_stats_dock = self._dock_by_title.get("Data Statistics")
        node_output_dock = self._dock_by_title.get("Node Output")
        node_props_dock = self._dock_by_title.get("Node Properties")
        model_output_dock = self._dock_by_title.get("Model Output")
        
        if data_preview_dock and data_stats_dock:
            self.tabifyDockWidget(data_preview_dock, data_stats_dock)
        if data_stats_dock and node_output_dock:
            self.tabifyDockWidget(data_stats_dock, node_output_dock)
        if node_output_dock and node_props_dock:
            self.tabifyDockWidget(node_output_dock, node_props_dock)
        if node_props_dock and model_output_dock:
            self.tabifyDockWidget(node_props_dock, model_output_dock)
        
        # Set Data Preview as the active tab
        if data_preview_dock:
            data_preview_dock.raise_()
        
        # ═══════════════════════════════════════════════════════════════
        # BOTTOM PANELS removed (Graph window removed)
        graph_dock = None
        
        # Set default bottom tab
        try:
            prefer_graph = bool(self._settings.get_bool(self._settings.DEFAULT_GRAPH_DOCK_ACTIVE))
        except Exception:
            prefer_graph = True
        if prefer_graph and graph_dock:
            graph_dock.raise_()
        
        # ═══════════════════════════════════════════════════════════════
        # CONNECT SIGNALS
        # ═══════════════════════════════════════════════════════════════
        node_editor.columns_selected.connect(data_preview.set_columns)
        node_editor.dataset_loaded.connect(data_preview.set_dataframe)
        node_editor.dataset_loaded.connect(data_statistics.set_dataframe)
        node_editor.dataset_loaded.connect(node_output.set_dataframe)
        data_preview.dtype_change_requested.connect(node_editor.apply_column_dtype)
        node_editor.node_output_changed.connect(node_output.set_node_output)
        node_editor.node_selected.connect(node_properties.set_node)
        node_editor.reader_changed.connect(node_properties.set_reader)
        
        # Connect Node Properties "Apply" button to update reader parameters
        node_properties.properties_changed.connect(node_editor.apply_extra_params)
        
        # Track project dirty state for save prompts.
        node_editor.nodes_changed.connect(lambda _nodes: self._mark_project_dirty())
        node_editor.node_output_changed.connect(lambda *_: self._mark_project_dirty())
        node_properties.properties_changed.connect(lambda *_: self._mark_project_dirty())
        self._graph_window = None

    def _add_dock(
        self,
        title: str,
        widget: QWidget,
        area: Qt.DockWidgetArea,
        color: str,
    ) -> None:
        wrapped = self._wrap_card(widget, color)
        dock = QDockWidget(title, self)
        dock.setObjectName(title)
        dock.setWidget(wrapped)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setStyleSheet("""
            QDockWidget {
                color: rgba(160, 210, 255, 220);
                font-weight: 600;
                titlebar-close-icon: url(none);
            }
            QDockWidget::title {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(22, 36, 56, 220),
                    stop:1 rgba(16, 26, 40, 220)
                );
                border: 1px solid rgba(60, 120, 200, 30);
                border-radius: 4px;
                padding: 5px 8px;
                text-align: left;
            }
        """)
        self.addDockWidget(area, dock)

        # Animate dock fade-in (store ref on dock so GC doesn't kill it)
        fx = QGraphicsOpacityEffect(dock)
        fx.setOpacity(0.0)
        dock.setGraphicsEffect(fx)
        anim = QPropertyAnimation(fx, b"opacity", dock)
        anim.setDuration(400)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        dock._fade_anim = anim          # prevent garbage-collection
        anim.finished.connect(lambda: dock.setGraphicsEffect(None))  # remove effect after done
        anim.start()

        action = dock.toggleViewAction()
        if self._view_menu:
            self._view_menu.addAction(action)
        self._dock_by_title[title] = dock
    
    def _focus_dock(self, title: str) -> None:
        dock = self._dock_by_title.get(title)
        if not dock:
            return
        dock.show()
        dock.raise_()

    def _wrap_card(self, widget: QWidget, color: str) -> QWidget:
        card = QFrame()
        card.setObjectName("dockCard")
        card.setStyleSheet(
            f"""
            QFrame#dockCard {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0.4, y2:1,
                    stop:0 {color},
                    stop:1 rgba(12, 18, 28, 240)
                );
                border: 1px solid rgba(60, 120, 200, 35);
                border-radius: 10px;
            }}
            """
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(widget)
        return card

    def _reset_layout(self) -> None:
        if not self._workspace_built:
            return
        for dock in self.findChildren(QDockWidget):
            self.removeDockWidget(dock)
            dock.deleteLater()
        self._build_workspace()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_logo_size()
        self._update_card_sizes()
        self._update_status_sizes()

    def _update_logo_size(self) -> None:
        if not self._logo_label:
            return
        if not self._is_widget_valid(self._logo_label):
            return
        if not self._logo_pixmap:
            return
        target = min(self.width(), self.height()) * 0.5
        size = max(64, int(target))
        self._logo_label.setPixmap(
            self._logo_pixmap.scaled(
                size,
                size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    @staticmethod
    def _is_widget_valid(widget: QWidget) -> bool:
        if shiboken6 is None:
            return True
        try:
            return shiboken6.isValid(widget)
        except Exception:
            return False

    def _update_card_sizes(self) -> None:
        if not self._menu_card or not self._recent_card or not self._todo_card:
            return
        if not self._is_widget_valid(self._menu_card):
            return
        if not self._is_widget_valid(self._recent_card):
            return
        if not self._is_widget_valid(self._todo_card):
            return
        target_width = int(self.width() * 0.32)
        target_height = int(self.height() * 0.24)
        width = max(240, target_width)
        height = max(140, target_height)
        self._menu_card.setFixedSize(width, height)
        self._recent_card.setFixedSize(width, height)
        self._todo_card.setFixedSize(width, height)

    def _update_status_sizes(self) -> None:
        # Resource monitor removed; footer now only shows status text.
        return

    def _build_card(self, title: str, body: str, delay_ms: int = 0) -> _AnimatedCard:
        card = _AnimatedCard(delay_ms=delay_ms)
        card.setObjectName("card")
        card.setMinimumWidth(260)
        card.setStyleSheet(self._card_stylesheet())

        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = self._build_card_header(title)

        body_label = QLabel(body)
        body_label.setStyleSheet(
            "color: rgba(180, 200, 225, 200); font-size: 13px; background: transparent;"
        )
        body_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        layout.addWidget(header)
        layout.addWidget(body_label)
        layout.addStretch(1)
        return card

    def _build_recent_card(self, delay_ms: int = 0) -> _AnimatedCard:
        card = _AnimatedCard(delay_ms=delay_ms)
        card.setObjectName("card")
        card.setMinimumWidth(260)
        card.setStyleSheet(self._card_stylesheet())

        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = self._build_card_header("Recent Projects")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget { background: transparent; }"
            "QScrollBar:vertical { width: 5px; }"
        )

        scroll_body = QWidget()
        scroll_body.setStyleSheet("background: transparent;")
        self._recent_body_layout = QVBoxLayout(scroll_body)
        self._recent_body_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_body_layout.setSpacing(6)
        scroll.setWidget(scroll_body)

        layout.addWidget(header)
        layout.addWidget(scroll)
        layout.addStretch(1)
        return card

    def _recent_project_paths(self) -> list[str]:
        recents = self._settings.get_list(self._settings.PROJECT_RECENTS)
        seen: set[str] = set()
        out: list[str] = []
        for raw in recents:
            p = str(raw or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _open_recent_project(self, path: str) -> None:
        if not path:
            return
        if not self._confirm_continue_with_unsaved_changes():
            return
        p = Path(path)
        if not p.exists():
            recents = [r for r in self._recent_project_paths() if r != path]
            self._settings.set_value(self._settings.PROJECT_RECENTS, recents)
            self._refresh_recent_projects_ui()
            QMessageBox.warning(self, "Recent Project Missing", f"File not found:\n{path}")
            return
        self._open_project_file_path(path)

    def _set_project_dirty(self, dirty: bool) -> None:
        self._project_dirty = bool(dirty)
        base = "Typhydrion"
        if self._current_project_path:
            base = f"{Path(self._current_project_path).name} - Typhydrion"
        if self._project_dirty:
            base = f"* {base}"
        self.setWindowTitle(base)

    def _mark_project_dirty(self) -> None:
        if self._suppress_dirty_tracking:
            return
        # Keep per-action dirty marking O(1). Full fingerprinting serializes the
        # entire graph and can block the UI when triggered by frequent signals.
        self._set_project_dirty(True)

    def _project_state_fingerprint(self) -> str | None:
        """Stable digest of project content used to suppress false dirty prompts."""
        if not self._workspace_built:
            return None
        node_editor = self.centralWidget()
        try:
            payload = {
                "node_editor": (
                    node_editor.to_dict(include_dataframes=False)
                    if hasattr(node_editor, "to_dict")
                    else {}
                ),
            }
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            return hashlib.sha1(raw.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _has_unsaved_changes(self) -> bool:
        if not self._workspace_built:
            return False
        if not self._project_dirty:
            return False
        fp = self._project_state_fingerprint()
        if fp is not None and self._last_clean_fingerprint is not None and fp == self._last_clean_fingerprint:
            self._set_project_dirty(False)
            return False
        return True

    def _confirm_continue_with_unsaved_changes(self) -> bool:
        """Prompt user to save unsaved changes before destructive navigation."""
        if not self._has_unsaved_changes():
            return True
        ans = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved project changes.\nDo you want to save them?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if ans == QMessageBox.Save:
            return bool(self._save_project_file(save_as=not bool(self._current_project_path)))
        if ans == QMessageBox.Discard:
            return True
        return False

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._confirm_continue_with_unsaved_changes():
            event.ignore()
            return
        self._shutdown_background_threads()
        super().closeEvent(event)

    def _shutdown_background_threads(self, stop_save_thread: bool = True) -> None:
        """Stop worker threads to avoid 'QThread destroyed while running'."""
        try:
            cw = self.centralWidget()
            if cw is not None and hasattr(cw, "shutdown_background_threads"):
                cw.shutdown_background_threads()
        except Exception:
            pass
        if not stop_save_thread:
            return
        try:
            if self._save_thread is not None and self._save_thread.isRunning():
                self._save_thread.quit()
                self._save_thread.wait(1200)
        except Exception:
            pass

    def _refresh_recent_projects_ui(self) -> None:
        recents = self._recent_project_paths()

        # Refresh File > Recent menu
        try:
            if self._recent_menu is not None:
                self._recent_menu.clear()
                if not recents:
                    empty = self._recent_menu.addAction("No recent projects")
                    empty.setEnabled(False)
                    self._recent_menu_empty_action = empty
                else:
                    for path in recents:
                        name = Path(path).name or path
                        act = self._recent_menu.addAction(name)
                        act.setToolTip(path)
                        act.triggered.connect(lambda checked=False, p=path: self._open_recent_project(p))
        except Exception:
            pass

        # Refresh Home recent card list
        try:
            if self._recent_body_layout is None:
                return

            while self._recent_body_layout.count():
                item = self._recent_body_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

            if not recents:
                lbl = QLabel("No recent projects")
                lbl.setStyleSheet(
                    "color: rgba(180, 200, 225, 200); font-size: 13px; background: transparent;"
                )
                self._recent_body_layout.addWidget(lbl)
                self._recent_body_layout.addStretch(1)
                return

            for path in recents[:10]:
                name = Path(path).stem or Path(path).name or "Project"
                btn = _GlowButton(name)
                btn.setToolTip(path)
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(lambda checked=False, p=path: self._open_recent_project(p))
                self._recent_body_layout.addWidget(btn)
            self._recent_body_layout.addStretch(1)
        except Exception:
            pass

    def _build_menu_card(self, delay_ms: int = 0) -> _AnimatedCard:
        card = _AnimatedCard(delay_ms=delay_ms)
        card.setObjectName("card")
        card.setMinimumWidth(260)
        card.setStyleSheet(self._card_stylesheet())

        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = self._build_card_header("Menu")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }"
                             "QScrollArea > QWidget { background: transparent; }"
                             "QScrollBar:vertical { width: 5px; }")

        scroll_body = QWidget()
        scroll_body.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_body)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(6)

        buttons_info = [
            ("New Project",                  self._action_new.trigger),
            ("Open Project",                 self._action_open.trigger),
            ("Templates",                    self._start_template_project),
            ("Settings",                     self._open_settings),
            ("Load Dataset",                 self._action_load_data.trigger),
            ("Data Analysis",                self._open_data_analysis_from_home),
            ("Data Visualization Dashboard", self._open_visualization_dashboard_from_home),
        ]

        for text, slot in buttons_info:
            btn = _GlowButton(text)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(slot)
            scroll_layout.addWidget(btn)

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_body)

        layout.addWidget(header)
        layout.addWidget(scroll)
        layout.addStretch(1)
        return card

    def _open_data_analysis_from_home(self) -> None:
        """Home menu shortcut: open the data profiler workflow."""
        self._open_data_analysis_profiler()

    def _open_data_analysis_profiler(self) -> None:
        # Dedicated Data Analysis layout in the same main window (no other cards/docks).
        self._workspace_built = False
        self._logo_label = None
        self._menu_card = None
        self._recent_card = None
        self._recent_body_layout = None
        self._todo_card = None
        self._dock_by_title.clear()
        self.setCentralWidget(QWidget())
        for dock in self.findChildren(QDockWidget):
            self.removeDockWidget(dock)
            dock.deleteLater()

        central = _GradientBackground()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = self._build_card_header("Data Analysis")
        layout.addWidget(header)

        self._data_profiler_widget = DataProfilerWindow(central)
        layout.addWidget(self._data_profiler_widget, 1)
        self.setCentralWidget(central)

    def _open_visualization_dashboard_from_home(self) -> None:
        """Graph window removed."""
        QMessageBox.information(
            self,
            "Graph Window Removed",
            "Graph / Visualization has been removed from this build.",
        )

    def _build_footer(self) -> None:
        status = QStatusBar()
        status.setObjectName("footer")
        status.setContentsMargins(6, 0, 6, 0)
        status.setStyleSheet(
            """
            QStatusBar#footer {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(14, 20, 30, 220),
                    stop:0.5 rgba(18, 28, 42, 230),
                    stop:1 rgba(14, 20, 30, 220)
                );
                color: rgba(180, 210, 245, 220);
                border-top: 1px solid rgba(60, 120, 200, 40);
            }
            QLabel {
                color: rgba(160, 200, 240, 210);
                font-size: 12px;
            }
            """
        )
        status.showMessage("")
        self.setStatusBar(status)
        self._update_status_sizes()

    @staticmethod
    def _card_stylesheet() -> str:
        """Shared stylesheet for all home-screen cards."""
        return """
            _AnimatedCard#card {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(22, 34, 52, 210),
                    stop:1 rgba(16, 24, 38, 230)
                );
                border: 1px solid rgba(70, 130, 220, 55);
                border-radius: 14px;
            }
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget { background: transparent; }
            QScrollArea QAbstractScrollArea { background: transparent; }
            QScrollArea::viewport { background: transparent; }
        """

    def _build_card_header(self, title: str) -> QFrame:
        header = QFrame()
        header.setObjectName("cardHeader")
        header.setStyleSheet(
            """
            QFrame#cardHeader {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(50, 100, 180, 50),
                    stop:1 rgba(30, 60, 120, 20)
                );
                border-radius: 8px;
                border-bottom: 1px solid rgba(80, 160, 255, 40);
            }
            """
        )

        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 7, 12, 7)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            "font-size: 15px; font-weight: 700; "
            "color: rgba(160, 210, 255, 240); background: transparent;"
        )
        header_layout.addWidget(title_label)
        return header
    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About Typhydrion",
            "Typhydrion\n"
            "A node-based, incremental ML pipeline builder.",
        )

    def _placeholder_action(self) -> None:
        QMessageBox.information(
            self,
            "Not Implemented",
            "This action will be implemented in the next step.",
        )

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]