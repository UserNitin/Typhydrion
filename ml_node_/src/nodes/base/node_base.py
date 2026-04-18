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


class NodeItem(QGraphicsItemGroup):
    _CORNER_R = 10.0  # corner radius for the card

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
        self._left_margin = 60
        self._right_margin = 80
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
        self._body = QGraphicsRectItem(0, 30, self.width, self.height - 54)
        self._body.setBrush(QBrush(QColor(20, 32, 50, 60)))
        self._body.setPen(QPen(Qt.transparent))
        self._body.setFlag(QGraphicsRectItem.ItemIsSelectable, False)
        self.addToGroup(self._body)

        # ── footer (rounded bottom) ──
        self._footer = QGraphicsPathItem(_bottom_rounded_path(self.width, 24, R))
        self._footer.setPos(0, self.height - 24)
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
        self._controls_widget.setStyleSheet("background: transparent;")
        self._controls_widget.setAttribute(Qt.WA_TranslucentBackground)
        self._controls_layout = QVBoxLayout(self._controls_widget)
        self._controls_layout.setContentsMargins(6, 6, 6, 6)
        self._controls_layout.setSpacing(6)

        self._controls_proxy = QGraphicsProxyWidget()
        self._controls_proxy.setWidget(self._controls_widget)
        self._controls_proxy.setPos(self._left_margin, 36)
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
            self._body.setRect(0, 30, self.width, self.height - 54)
        if self._footer and shiboken6.isValid(self._footer):
            self._footer.setPath(_bottom_rounded_path(self.width, 24, R))
            self._footer.setPos(0, self.height - 24)

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

        try:
            self._controls_widget.adjustSize()
            hint = self._controls_widget.sizeHint()
        except RuntimeError:
            self._controls_widget = None
            return

        max_left = max((p.label_width() for p in self._inputs), default=0)
        max_right = max((p.label_width() for p in self._outputs), default=0)
        self._left_margin = max(60, int(max_left) + 20)
        self._right_margin = max(80, int(max_right) + 20)

        ports_h = 42 + max(len(self._inputs), len(self._outputs)) * 24 + 30
        content_h = hint.height() + 62
        desired_h = max(180, min(content_h, 300), ports_h)
        max_height = 350
        desired_h = min(desired_h, max_height)

        desired_w = self._fixed_width
        if self._manual_width is not None:
            desired_w = max(desired_w, self._manual_width)
        if self._manual_height is not None:
            desired_h = max(desired_h, self._manual_height)

        if desired_h == self.height and desired_w == self.width:
            return

        self.prepareGeometryChange()

        self.height = desired_h
        self.width = desired_w

        self._rebuild_paths()

        if self._status_item:
            self._status_item.setPos(self.width - 18, 10)
        if self._output_btn:
            self._output_btn.setPos(self.width - 40, 5)
        if self._input_btn:
            self._input_btn.setPos(self.width - 64, 5)
        if self._title_item:
            self._title_item.setPos(10, 5)
        if self._controls_proxy and self._controls_widget:
            try:
                if shiboken6.isValid(self._controls_proxy) and shiboken6.isValid(self._controls_widget):
                    self._controls_proxy.setPos(self._left_margin, 36)
                    available_width = max(120, int(self.width - self._left_margin - self._right_margin))
                    self._controls_widget.setFixedWidth(available_width)
                    max_content_h = self.height - 82
                    if hint.height() > max_content_h:
                        self._controls_widget.setFixedHeight(max_content_h)
                    else:
                        self._controls_widget.setMinimumHeight(hint.height())
                else:
                    self._controls_proxy = None
                    self._controls_widget = None
            except RuntimeError:
                self._controls_proxy = None
                self._controls_widget = None
        if self._resize_handle:
            self._resize_handle.setPos(self.width - 12, self.height - 12)
        for index, port in enumerate(self._inputs):
            y = 42 + index * 24
            port.setPos(0, y)
            port.set_label_pos(8, y - 8)
        for index, port in enumerate(self._outputs):
            y = 42 + index * 24
            port.setPos(self.width, y)
            port.set_label_pos(self.width - self._right_margin + 6, y - 8)

    # ────────────────────────────────────────────────────────────────────
    # Ports
    # ────────────────────────────────────────────────────────────────────

    def add_input(self, name: str, data_type: str = "numeric") -> PortItem:
        port = PortItem(name, is_output=False, data_type=data_type, parent=self)
        y = 42 + len(self._inputs) * 24
        port.setPos(0, y)
        port.set_label_pos(8, y - 8)
        self._inputs.append(port)
        self._auto_resize()
        return port

    def add_output(self, name: str, data_type: str = "numeric") -> PortItem:
        port = PortItem(name, is_output=True, data_type=data_type, parent=self)
        y = 42 + len(self._outputs) * 24
        port.setPos(self.width, y)
        port.set_label_pos(self.width - self._right_margin + 6, y - 8)
        self._outputs.append(port)
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

        new_width = max(220, width)
        new_height = max(160, height)

        self.prepareGeometryChange()

        self._manual_width = new_width
        self._manual_height = new_height
        self.width = new_width
        self.height = new_height

        self._rebuild_paths()

        if self._status_item:
            self._status_item.setPos(self.width - 18, 10)
        if self._output_btn:
            self._output_btn.setPos(self.width - 40, 5)
        if self._input_btn:
            self._input_btn.setPos(self.width - 64, 5)
        if self._controls_proxy and self._controls_widget:
            try:
                if shiboken6.isValid(self._controls_proxy) and shiboken6.isValid(self._controls_widget):
                    available_width = max(120, int(self.width - self._left_margin - self._right_margin))
                    self._controls_widget.setFixedWidth(available_width)
                    max_content_h = self.height - 82
                    self._controls_widget.setFixedHeight(max_content_h)
                else:
                    self._controls_proxy = None
                    self._controls_widget = None
            except RuntimeError:
                self._controls_proxy = None
                self._controls_widget = None
        if self._resize_handle:
            self._resize_handle.setPos(self.width - 12, self.height - 12)
        for index, port in enumerate(self._outputs):
            y = 42 + index * 24
            port.setPos(self.width, y)
            port.set_label_pos(self.width - self._right_margin + 6, y - 8)


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
