from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QGraphicsDropShadowEffect


def _tool_btn(text: str, tooltip: str = "", checkable: bool = False) -> QPushButton:
    btn = QPushButton(text)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setCheckable(checkable)
    if tooltip:
        btn.setToolTip(tooltip)
    btn.setStyleSheet("""
        QPushButton {
            background: rgba(28, 42, 64, 220);
            color: rgba(200, 225, 255, 235);
            border: 1px solid rgba(60, 120, 200, 55);
            border-radius: 8px;
            padding: 6px 10px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: rgba(40, 80, 140, 230);
            border-color: rgba(70, 150, 255, 110);
        }
        QPushButton:pressed {
            background: rgba(25, 55, 110, 240);
        }
        QPushButton:checked {
            background: rgba(20, 90, 160, 210);
            border-color: rgba(100, 180, 255, 140);
        }
        QPushButton:disabled {
            background: rgba(28, 42, 64, 120);
            color: rgba(200, 225, 255, 120);
            border-color: rgba(60, 120, 200, 30);
        }
    """)
    return btn


class NodeEditorToolbar(QFrame):
    add_node_clicked = Signal()
    zoom_in_clicked = Signal()
    zoom_out_clicked = Signal()
    zoom_reset_clicked = Signal()
    fit_clicked = Signal()
    center_clicked = Signal()
    clear_clicked = Signal()
    grid_toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setObjectName("nodeEditorToolbar")
        self.setStyleSheet("""
            QFrame#nodeEditorToolbar {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(18, 28, 42, 245),
                    stop:1 rgba(14, 20, 32, 245)
                );
                border: 1px solid rgba(55, 110, 200, 45);
                border-radius: 12px;
            }
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setYOffset(3)
        shadow.setColor(QColor(20, 60, 140, 50))
        self.setGraphicsEffect(shadow)

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(10)

        title = QLabel("🧩 Node Editor")
        title.setStyleSheet("color: rgba(215, 235, 255, 245); font-size: 15px; font-weight: 800;")
        row.addWidget(title)

        hint = QLabel("Shift+A: add node   •   Middle-mouse: pan   •   Wheel: zoom")
        hint.setStyleSheet("color: rgba(140, 175, 215, 200); font-size: 11px; font-weight: 500;")
        row.addWidget(hint)

        row.addStretch(1)

        self._grid_btn = _tool_btn("☰ Grid", "Toggle canvas grid", checkable=True)
        self._grid_btn.setChecked(True)
        self._grid_btn.toggled.connect(self.grid_toggled.emit)
        row.addWidget(self._grid_btn)

        self._add_btn = _tool_btn("＋ Add", "Add a node (Shift+A)")
        self._add_btn.clicked.connect(self.add_node_clicked.emit)
        row.addWidget(self._add_btn)

        self._zoom_out = _tool_btn("－", "Zoom out")
        self._zoom_out.clicked.connect(self.zoom_out_clicked.emit)
        row.addWidget(self._zoom_out)

        self._zoom_in = _tool_btn("＋", "Zoom in")
        self._zoom_in.clicked.connect(self.zoom_in_clicked.emit)
        row.addWidget(self._zoom_in)

        self._zoom_reset = _tool_btn("100%", "Reset zoom")
        self._zoom_reset.clicked.connect(self.zoom_reset_clicked.emit)
        row.addWidget(self._zoom_reset)

        self._fit = _tool_btn("Fit", "Fit graph to view")
        self._fit.clicked.connect(self.fit_clicked.emit)
        row.addWidget(self._fit)

        self._center = _tool_btn("Center", "Center view")
        self._center.clicked.connect(self.center_clicked.emit)
        row.addWidget(self._center)

        self._clear = _tool_btn("Clear", "Clear all nodes and links")
        self._clear.clicked.connect(self.clear_clicked.emit)
        row.addWidget(self._clear)

    def set_grid_checked(self, checked: bool) -> None:
        try:
            self._grid_btn.blockSignals(True)
            self._grid_btn.setChecked(bool(checked))
        finally:
            self._grid_btn.blockSignals(False)

