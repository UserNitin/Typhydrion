from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel


class NodeEditorStatusBar(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("nodeEditorStatusBar")
        self.setStyleSheet("""
            QFrame#nodeEditorStatusBar {
                background: rgba(14, 20, 32, 210);
                border: 1px solid rgba(55, 110, 200, 30);
                border-radius: 10px;
            }
            QLabel {
                color: rgba(170, 205, 245, 210);
                font-size: 11px;
                background: transparent;
            }
        """)

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(10)

        self._left = QLabel("Ready")
        self._left.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self._left, 1)

        self._zoom = QLabel("Zoom: 100%")
        row.addWidget(self._zoom, 0)

        self._grid = QLabel("Grid: On")
        row.addWidget(self._grid, 0)

        self._selection = QLabel("Selected: 0")
        row.addWidget(self._selection, 0)

    def set_message(self, text: str) -> None:
        self._left.setText(text or "")

    def set_zoom(self, zoom: float) -> None:
        try:
            pct = int(round(float(zoom) * 100))
        except Exception:
            pct = 100
        self._zoom.setText(f"Zoom: {pct}%")

    def set_grid(self, enabled: bool, size: int | None = None) -> None:
        if enabled:
            if size is not None:
                self._grid.setText(f"Grid: On ({int(size)})")
            else:
                self._grid.setText("Grid: On")
        else:
            self._grid.setText("Grid: Off")

    def set_selected_count(self, n: int) -> None:
        try:
            n = int(n)
        except Exception:
            n = 0
        self._selection.setText(f"Selected: {n}")

