from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QBrush,
    QPen,
    QColor,
    QPainterPath,
    QLinearGradient,
    QPainter,
)
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsPathItem,
    QGraphicsTextItem,
    QGraphicsEllipseItem,
    QGraphicsProxyWidget,
    QWidget,
    QVBoxLayout,
    QGraphicsDropShadowEffect,
)
import uuid
import weakref
import shiboken6

from nodes.base.port import PortItem


# ── Rounded-rect helper ────────────────────────────────────────────────────
def _rounded_rect_path(w: float, h: float, r: float = 10.0) -> QPainterPath:
    """Return a QPainterPath representing a rounded rectangle."""
    p = QPainterPath()
    p.addRoundedRect(QRectF(0, 0, w, h), r, r)
    return p


def _top_rounded_path(w: float, h: float, r: float = 10.0) -> QPainterPath:
    """Rounded top corners, flat bottom."""
    p = QPainterPath()
    p.moveTo(r, 0)
    p.lineTo(w - r, 0)
    p.arcTo(QRectF(w - 2 * r, 0, 2 * r, 2 * r), 90, -90)
    p.lineTo(w, h)
    p.lineTo(0, h)
    p.lineTo(0, r)
    p.arcTo(QRectF(0, 0, 2 * r, 2 * r), 180, -90)
    p.closeSubpath()
    return p


def _bottom_rounded_path(w: float, h: float, r: float = 10.0) -> QPainterPath:
    """Flat top, rounded bottom corners."""
    p = QPainterPath()
    p.moveTo(0, 0)
    p.lineTo(w, 0)
    p.lineTo(w, h - r)
    p.arcTo(QRectF(w - 2 * r, h - 2 * r, 2 * r, 2 * r), 0, -90)
    p.lineTo(r, h)
    p.arcTo(QRectF(0, h - 2 * r, 2 * r, 2 * r), 270, -90)
    p.closeSubpath()
    return p


_CONTROLS_STYLE = """
QWidget#nodeControls { background: transparent; }
QLabel { background: transparent; color: rgba(215, 228, 245, 215); }
QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
    background-color: rgba(28, 40, 58, 230);
    border: 1px solid rgba(90, 130, 190, 110);
    border-radius: 4px;
    padding: 3px 6px;
    color: rgba(225, 235, 250, 235);
    selection-background-color: rgba(70, 130, 220, 200);
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
    border-color: rgba(90, 160, 255, 210);
}
QSpinBox, QDoubleSpinBox { padding-right: 18px; }
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    width: 16px;
    background: rgba(45, 65, 95, 200);
    border-left: 1px solid rgba(90, 130, 190, 90);
}
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 4px; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 4px; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: rgba(65, 100, 150, 230); }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(__UP_ARROW__); width: 8px; height: 5px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(__DOWN_ARROW__); width: 8px; height: 5px; }
QPushButton {
    background-color: rgba(40, 60, 90, 225);
    border: 1px solid rgba(90, 140, 210, 120);
    border-radius: 4px;
    padding: 4px 10px;
    color: rgba(220, 232, 250, 235);
}
QPushButton:hover { background-color: rgba(55, 85, 130, 235); }
QPushButton:pressed { background-color: rgba(30, 48, 75, 235); }
QCheckBox { background: transparent; color: rgba(215, 228, 245, 215); spacing: 6px; }
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid rgba(100, 140, 200, 170);
    border-radius: 3px;
    background-color: rgba(28, 40, 58, 230);
}
QCheckBox::indicator:checked {
    background-color: rgba(70, 140, 230, 235);
    border-color: rgba(120, 175, 255, 230);
}
QSlider::groove:horizontal { height: 4px; background: rgba(60, 90, 130, 160); border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 6px; background: rgba(90, 160, 255, 230); }
"""

_controls_style_cache: str | None = None


def _controls_style() -> str:
    """Controls stylesheet with spin-box arrow images rendered on first use (needs a QApplication)."""
    global _controls_style_cache
    if _controls_style_cache is not None:
        return _controls_style_cache

    import tempfile
    from pathlib import Path
    from PySide6.QtGui import QPixmap, QPolygonF

    out_dir = Path(tempfile.gettempdir()) / "typhydrion_ui"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, points in (
        ("up", [(0, 10), (16, 10), (8, 0)]),
        ("down", [(0, 0), (16, 0), (8, 10)]),
    ):
        pix = QPixmap(16, 10)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(200, 220, 250, 230))
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        painter.end()
        path = out_dir / f"spin_{name}.png"
        pix.save(str(path))
        paths[name] = path.as_posix()

    _controls_style_cache = (
        _CONTROLS_STYLE.replace("__UP_ARROW__", paths["up"]).replace("__DOWN_ARROW__", paths["down"])
    )
    return _controls_style_cache


class NodeItem(QGraphicsItemGroup):
    _CORNER_R = 10.0  # corner radius for the card
    _PORT_TOP = 46
    _PORT_STEP = 24
    _PORT_LABEL_INSET = 12
    _PORT_LABEL_GAP = 24
    _CONTROLS_INSET = 10
    _FOOTER_H = 14
    _MIN_HEIGHT = 120
    _MAX_AUTO_WIDTH = 460
    _MAX_AUTO_HEIGHT = 620

    def __init__(self, title: str, width: int = 300, height: int = 180) -> None:
        super().__init__()
        self.node_id = uuid.uuid4().hex
        self.title = title
        self.width = width
        self.height = height
        self._inputs: list[PortItem] = []
        self._outputs: list[PortItem] = []
        self._background: QGraphicsPathItem | None = None
        self._header: QGraphicsPathItem | None = None
        self._body: QGraphicsRectItem | None = None
        self._footer: QGraphicsPathItem | None = None
        self._title_item: QGraphicsTextItem | None = None
        self._status_item: QGraphicsEllipseItem | None = None
        self._controls_widget: QWidget | None = None
        self._controls_layout: QVBoxLayout | None = None
        self._controls_proxy: QGraphicsProxyWidget | None = None
        self._fixed_width = width
        self._manual_width: int | None = None
        self._manual_height: int | None = None
        self._resize_handle: ResizeHandle | None = None
        self._input_btn: QGraphicsProxyWidget | None = None
        self._output_btn: QGraphicsProxyWidget | None = None

        # Store loaded data for this node
        self._loaded_dataframe = None  # Output data
        self._input_dataframe = None   # Input data received from connection
        self._on_select_callback = None
        self._output_callback = None
        self._input_callback = None    # Callback for input view button
        self._extra_params = {}        # Extra parameters from Node Properties window

        self._build_card()
        self.setHandlesChildEvents(False)
        self.setFlags(
            QGraphicsItemGroup.ItemIsMovable
            | QGraphicsItemGroup.ItemIsSelectable
            | QGraphicsItemGroup.ItemSendsScenePositionChanges
        )
        self.setCursor(Qt.ArrowCursor)

    # ────────────────────────────────────────────────────────────────────
    # Card construction
    # ────────────────────────────────────────────────────────────────────

    def _build_card(self) -> None:
        R = self._CORNER_R

        # ── background (full rounded rect with gradient) ──
        self._background = QGraphicsPathItem(_rounded_rect_path(self.width, self.height, R))
        grad = QLinearGradient(0, 0, 0, self.height)
        grad.setColorAt(0, QColor(24, 36, 54, 210))
        grad.setColorAt(1, QColor(16, 24, 38, 230))
        self._background.setBrush(QBrush(grad))
        self._border_pen = QPen(QColor(60, 120, 200, 55), 1.2)
        self._background.setPen(self._border_pen)
        self._background.setFlag(QGraphicsPathItem.ItemIsSelectable, False)
        self.addToGroup(self._background)

        # Soft drop shadow instead of blur
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(18)
        shadow.setColor(QColor(10, 20, 40, 120))
        shadow.setOffset(0, 4)
        self._background.setGraphicsEffect(shadow)
        self._shadow = shadow

        # ── header (rounded top) ──
        self._header = QGraphicsPathItem(_top_rounded_path(self.width, 30, R))
        header_grad = QLinearGradient(0, 0, self.width, 0)
        header_grad.setColorAt(0, QColor(32, 52, 82, 150))
        header_grad.setColorAt(1, QColor(26, 42, 68, 130))
        self._header.setBrush(QBrush(header_grad))
        self._header.setPen(QPen(Qt.transparent))
        self._header.setFlag(QGraphicsPathItem.ItemIsSelectable, False)
        self.addToGroup(self._header)

        # ── body (flat rect) ──
        self._body = QGraphicsRectItem(0, 30, self.width, self.height - 30 - self._FOOTER_H)
        self._body.setBrush(QBrush(QColor(20, 32, 50, 60)))
        self._body.setPen(QPen(Qt.transparent))
        self._body.setFlag(QGraphicsRectItem.ItemIsSelectable, False)
        self.addToGroup(self._body)

        # ── footer (rounded bottom) ──
        self._footer = QGraphicsPathItem(_bottom_rounded_path(self.width, self._FOOTER_H, R))
        self._footer.setPos(0, self.height - self._FOOTER_H)
        self._footer.setBrush(QBrush(QColor(24, 40, 64, 90)))
        self._footer.setPen(QPen(Qt.transparent))
        self._footer.setFlag(QGraphicsPathItem.ItemIsSelectable, False)
        self.addToGroup(self._footer)

        # ── title text ──
        self._title_item = QGraphicsTextItem(self.title)
        self._title_item.setDefaultTextColor(QColor(170, 210, 255, 240))
        self._title_item.setPos(10, 5)
        self._title_item.setFlag(QGraphicsTextItem.ItemIsSelectable, False)
        self.addToGroup(self._title_item)

        # ── status dot ──
        self._status_item = QGraphicsEllipseItem(0, 0, 10, 10)
        self._status_item.setBrush(QBrush(QColor(70, 200, 90)))
        self._status_item.setPen(QPen(QColor(0, 0, 0, 80), 0.8))
        self._status_item.setPos(self.width - 18, 10)
        self._status_item.setFlag(QGraphicsEllipseItem.ItemIsSelectable, False)
        self.addToGroup(self._status_item)

        # Input/Output header buttons
        self._input_btn = self._create_header_button("📥", self.width - 64, 5, is_input=True)
        self._output_btn = self._create_header_button("📤", self.width - 40, 5, is_input=False)

        # ── controls proxy ──
        self._controls_widget = QWidget()
        self._controls_widget.setObjectName("nodeControls")
        self._controls_widget.setStyleSheet(_controls_style())
        self._controls_widget.setAttribute(Qt.WA_TranslucentBackground)
        self._controls_layout = QVBoxLayout(self._controls_widget)
        self._controls_layout.setContentsMargins(6, 6, 6, 6)
        self._controls_layout.setSpacing(6)

        self._controls_proxy = QGraphicsProxyWidget()
        self._controls_proxy.setWidget(self._controls_widget)
        self._controls_proxy.setPos(self._CONTROLS_INSET, self._controls_top())
        self._controls_proxy.setAcceptedMouseButtons(Qt.AllButtons)
        self._controls_proxy.setFlag(QGraphicsProxyWidget.ItemIsFocusable, True)
        self._controls_proxy.setFlag(QGraphicsProxyWidget.ItemAcceptsInputMethod, True)
        self._controls_proxy.setFlag(QGraphicsProxyWidget.ItemIsSelectable, False)
        self._controls_proxy.setActive(True)
        self.addToGroup(self._controls_proxy)

        # ── resize handle ──
        self._resize_handle = ResizeHandle(self)
        self._resize_handle.setPos(self.width - 12, self.height - 12)
        self.addToGroup(self._resize_handle)

    # ────────────────────────────────────────────────────────────────────
    # Header buttons
    # ────────────────────────────────────────────────────────────────────

    def _create_header_button(self, text: str, x: float, y: float, is_input: bool = False) -> QGraphicsProxyWidget:
        from PySide6.QtWidgets import QPushButton

        btn = QPushButton(text)
        btn.setFixedSize(22, 20)
        btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(40, 65, 100, 160);
                border: 1px solid rgba(80, 140, 220, 60);
                border-radius: 4px;
                color: white;
                font-size: 11px;
                padding: 0;
            }
            QPushButton:hover {
                background-color: rgba(55, 90, 140, 200);
                border-color: rgba(100, 170, 255, 100);
            }
            QPushButton:pressed {
                background-color: rgba(30, 50, 80, 200);
            }
        """)

        if is_input:
            btn.clicked.connect(self._on_input_btn_clicked)
        else:
            btn.clicked.connect(self._on_output_btn_clicked)

        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(btn)
        proxy.setPos(x, y)
        proxy.setFlag(QGraphicsProxyWidget.ItemIsSelectable, False)
        proxy.setAcceptedMouseButtons(Qt.LeftButton)
        self.addToGroup(proxy)
        return proxy

    def _on_output_btn_clicked(self) -> None:
        if self._output_callback:
            self._output_callback(self.title + " (Output)", self._loaded_dataframe)

    def _on_input_btn_clicked(self) -> None:
        if self._input_callback:
            self._input_callback(self.title + " (Input)", self._input_dataframe)

    def set_output_callback(self, callback) -> None:
        self._output_callback = callback

    def set_input_callback(self, callback) -> None:
        self._input_callback = callback

    def set_input_dataframe(self, df) -> None:
        self._input_dataframe = df

    def get_input_dataframe(self):
        return self._input_dataframe

    def set_extra_params(self, params: dict) -> None:
        self._extra_params = params.copy() if params else {}

    def get_extra_params(self) -> dict:
        return self._extra_params.copy()

    # ────────────────────────────────────────────────────────────────────
    # Controls
    # ────────────────────────────────────────────────────────────────────

    def add_control(self, widget: QWidget) -> None:
        if (
            not self._controls_layout
            or not shiboken6.isValid(self)
            or (self._controls_widget is not None and not shiboken6.isValid(self._controls_widget))
        ):
            return
        try:
            self._controls_layout.addWidget(widget)
            # Children added to an already-visible container stay hidden until the next event
            # loop pass, and hidden widgets don't count toward the layout's size hint.
            widget.show()
        except RuntimeError:
            return
        self._auto_resize()
        from PySide6.QtCore import QTimer
        weak_self = weakref.ref(self)

        def _safe_resize() -> None:
            node = weak_self()
            if node is None:
                return
            try:
                if not shiboken6.isValid(node):
                    return
            except Exception:
                return
            node._auto_resize()

        QTimer.singleShot(100, _safe_resize)

    # ────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ────────────────────────────────────────────────────────────────────

    def _rebuild_paths(self) -> None:
        """Rebuild all rounded-rect QPainterPaths after a size change."""
        try:
            if not shiboken6.isValid(self):
                return
        except Exception:
            return
        R = self._CORNER_R
        if self._background and shiboken6.isValid(self._background):
            self._background.setPath(_rounded_rect_path(self.width, self.height, R))
            grad = QLinearGradient(0, 0, 0, self.height)
            grad.setColorAt(0, QColor(24, 36, 54, 210))
            grad.setColorAt(1, QColor(16, 24, 38, 230))
            self._background.setBrush(QBrush(grad))
        if self._header and shiboken6.isValid(self._header):
            self._header.setPath(_top_rounded_path(self.width, 30, R))
        if self._body and shiboken6.isValid(self._body):
            self._body.setRect(0, 30, self.width, self.height - 30 - self._FOOTER_H)
        if self._footer and shiboken6.isValid(self._footer):
            self._footer.setPath(_bottom_rounded_path(self.width, self._FOOTER_H, R))
            self._footer.setPos(0, self.height - self._FOOTER_H)

    def _auto_resize(self) -> None:
        if not self._controls_widget:
            return
        try:
            if not shiboken6.isValid(self):
                return
            if not shiboken6.isValid(self._controls_widget):
                self._controls_widget = None
                return
        except Exception:
            return

        fit = self._fit_size()
        if fit is None:
            return
        desired_w, desired_h = fit

        if self._manual_width is not None:
            desired_w = max(desired_w, self._manual_width)
        if self._manual_height is not None:
            desired_h = max(desired_h, self._manual_height)

        if desired_h != self.height or desired_w != self.width:
            self.prepareGeometryChange()
            self.height = int(desired_h)
            self.width = int(desired_w)
        self._layout_contents()

    def _fit_size(self) -> tuple[int, int] | None:
        """Smallest card size that shows all ports and controls."""
        has_controls = False
        hint_w = hint_h = 0
        try:
            if self._controls_widget is not None and shiboken6.isValid(self._controls_widget):
                self._controls_widget.setMinimumSize(0, 0)
                self._controls_widget.setMaximumSize(16777215, 16777215)
                if self._controls_layout is not None:
                    self._controls_layout.activate()
                    has_controls = self._controls_layout.count() > 0
                hint = self._controls_widget.sizeHint()
                hint_w, hint_h = hint.width(), hint.height()
        except RuntimeError:
            self._controls_widget = None
            return None

        max_left = max((p.label_width() for p in self._inputs), default=0)
        max_right = max((p.label_width() for p in self._outputs), default=0)
        ports_w = int(max_left + max_right) + self._PORT_LABEL_GAP + 2 * self._PORT_LABEL_INSET

        width = max(self._fixed_width, ports_w)
        if has_controls:
            width = max(width, hint_w + 2 * self._CONTROLS_INSET)
        width = min(width, self._MAX_AUTO_WIDTH)

        height = self._controls_top() + (hint_h if has_controls else 0) + self._FOOTER_H + 2
        height = min(max(height, self._MIN_HEIGHT), self._MAX_AUTO_HEIGHT)
        return int(width), int(height)

    def has_manual_size(self) -> bool:
        return self._manual_width is not None or self._manual_height is not None

    def fit_to_content(self) -> None:
        """Drop any manual size and shrink/grow the card to fit its content."""
        self._manual_width = None
        self._manual_height = None
        self._auto_resize()

    def _controls_top(self) -> int:
        rows = max(len(self._inputs), len(self._outputs))
        return self._PORT_TOP - 6 + rows * self._PORT_STEP + 4

    def _layout_contents(self) -> None:
        """Position header items, ports and the controls panel for the current size."""
        self._rebuild_paths()

        if self._status_item:
            self._status_item.setPos(self.width - 18, 10)
        if self._output_btn:
            self._output_btn.setPos(self.width - 40, 5)
        if self._input_btn:
            self._input_btn.setPos(self.width - 64, 5)
        if self._title_item:
            self._title_item.setPos(10, 5)
        if self._resize_handle:
            self._resize_handle.setPos(self.width - 12, self.height - 12)

        for index, port in enumerate(self._inputs):
            y = self._PORT_TOP + index * self._PORT_STEP
            port.setPos(0, y)
            port.set_label_pos(self._PORT_LABEL_INSET, y - 11)
        for index, port in enumerate(self._outputs):
            y = self._PORT_TOP + index * self._PORT_STEP
            port.setPos(self.width, y)
            port.set_label_pos(self.width - self._PORT_LABEL_INSET - port.label_width(), y - 11)

        if self._controls_proxy and self._controls_widget:
            try:
                if shiboken6.isValid(self._controls_proxy) and shiboken6.isValid(self._controls_widget):
                    top = self._controls_top()
                    self._controls_proxy.setPos(self._CONTROLS_INSET, top)
                    avail_w = max(80, int(self.width - 2 * self._CONTROLS_INSET))
                    avail_h = max(0, int(self.height - top - self._FOOTER_H - 2))
                    hint_h = self._controls_widget.sizeHint().height()
                    self._controls_widget.setFixedSize(avail_w, min(hint_h, avail_h))
                    self._controls_proxy.setVisible(avail_h > 0)
                else:
                    self._controls_proxy = None
                    self._controls_widget = None
            except RuntimeError:
                self._controls_proxy = None
                self._controls_widget = None

        self._update_connected_edges()

    # ────────────────────────────────────────────────────────────────────
    # Ports
    # ────────────────────────────────────────────────────────────────────

    def add_input(self, name: str, data_type: str = "numeric") -> PortItem:
        port = PortItem(name, is_output=False, data_type=data_type, parent=self)
        self._inputs.append(port)
        self._layout_contents()
        self._auto_resize()
        return port

    def add_output(self, name: str, data_type: str = "numeric") -> PortItem:
        port = PortItem(name, is_output=True, data_type=data_type, parent=self)
        self._outputs.append(port)
        self._layout_contents()
        self._auto_resize()
        return port

    # ────────────────────────────────────────────────────────────────────
    # Selection / movement
    # ────────────────────────────────────────────────────────────────────

    def itemChange(self, change, value):
        try:
            if not shiboken6.isValid(self):
                return super().itemChange(change, value)
        except Exception:
            return super().itemChange(change, value)

        if change == QGraphicsItemGroup.ItemSelectedChange and self._background:
            try:
                if value:
                    glow = QPen(QColor(60, 150, 255, 170), 2)
                    self._background.setPen(glow)
                    if hasattr(self, "_shadow"):
                        self._shadow.setBlurRadius(28)
                        self._shadow.setColor(QColor(40, 100, 220, 140))
                    if self._on_select_callback and self._loaded_dataframe is not None:
                        self._on_select_callback(self._loaded_dataframe)
                else:
                    self._background.setPen(self._border_pen)
                    if hasattr(self, "_shadow"):
                        self._shadow.setBlurRadius(18)
                        self._shadow.setColor(QColor(10, 20, 40, 120))
            except RuntimeError:
                pass

        if change == QGraphicsItemGroup.ItemPositionChange:
            try:
                from ui.app_settings import AppSettings
                s = AppSettings()
                if s.get_bool(s.NODE_SNAP_TO_GRID):
                    grid = max(10, int(s.get_int(s.GRID_NODE_SIZE)))
                    p = value if isinstance(value, QPointF) else QPointF(value)
                    x = round(p.x() / grid) * grid
                    y = round(p.y() / grid) * grid
                    return QPointF(x, y)
            except Exception:
                pass

        if change == QGraphicsItemGroup.ItemScenePositionHasChanged:
            self._update_connected_edges()
            try:
                sc = self.scene()
                if sc and shiboken6.isValid(sc) and hasattr(sc, "ensure_item_in_scene"):
                    sc.ensure_item_in_scene(self)
            except Exception:
                pass

        return super().itemChange(change, value)

    def _update_connected_edges(self) -> None:
        if not self.scene() or not shiboken6.isValid(self.scene()):
            return

        edges = []
        seen = set()
        for port in (self._inputs + self._outputs):
            try:
                if hasattr(port, "connected_edges"):
                    for edge in port.connected_edges():
                        key = id(edge)
                        if key in seen:
                            continue
                        seen.add(key)
                        edges.append(edge)
            except Exception:
                continue

        for edge in edges:
            try:
                if shiboken6.isValid(edge):
                    edge.update_position()
            except RuntimeError:
                continue

    # ────────────────────────────────────────────────────────────────────
    # Data helpers
    # ────────────────────────────────────────────────────────────────────

    def set_dataframe(self, df) -> None:
        self._loaded_dataframe = df

    def get_dataframe(self):
        return self._loaded_dataframe

    def set_on_select_callback(self, callback) -> None:
        self._on_select_callback = callback

    # ────────────────────────────────────────────────────────────────────
    # Geometry overrides
    # ────────────────────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(-2, -2, self.width + 4, self.height + 4)

    def shape(self):
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width, self.height), self._CORNER_R, self._CORNER_R)
        return path

    def paint(self, painter, option, widget=None):
        # Enable anti-aliasing for smooth edges on the whole group
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        # Don't call super().paint() — prevents default selection box

    # ────────────────────────────────────────────────────────────────────
    # Manual resize
    # ────────────────────────────────────────────────────────────────────

    def resize_to(self, width: int, height: int) -> None:
        try:
            if not shiboken6.isValid(self):
                return
        except Exception:
            return

        fit = self._fit_size()
        min_w, min_h = fit if fit is not None else (220, 160)
        new_width = max(min_w, int(width))
        new_height = max(min_h, int(height))
        if new_width == self.width and new_height == self.height:
            return

        self.prepareGeometryChange()

        self._manual_width = new_width
        self._manual_height = new_height
        self.width = new_width
        self.height = new_height

        self._layout_contents()


class ResizeHandle(QGraphicsRectItem):
    def __init__(self, node: NodeItem) -> None:
        super().__init__(0, 0, 12, 12, node)
        self._node = node
        self.setBrush(QBrush(QColor(80, 140, 220, 100)))
        self.setPen(QPen(QColor(60, 120, 200, 60), 0.8))
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, False)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event) -> None:
        self.setCursor(Qt.SizeFDiagCursor)
        self.setBrush(QBrush(QColor(80, 160, 255, 160)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.unsetCursor()
        self.setBrush(QBrush(QColor(80, 140, 220, 100)))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        try:
            if shiboken6.isValid(self._node):
                self._node.fit_to_content()
        except RuntimeError:
            pass
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        try:
            if not shiboken6.isValid(self._node):
                event.accept()
                return
            pos = event.scenePos()
            node_pos = self._node.scenePos()
            width = int(pos.x() - node_pos.x())
            height = int(pos.y() - node_pos.y())
            self._node.resize_to(width, height)
        except RuntimeError:
            pass
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        try:
            if shiboken6.isValid(self._node) and self._node.scene():
                self._node.scene().update()
        except RuntimeError:
            pass
        event.accept()
