from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QPen, QColor
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem
import uuid
import shiboken6


class PortItem(QGraphicsEllipseItem):
    def __init__(self, name: str, is_output: bool, data_type: str = "numeric", parent=None) -> None:
        super().__init__(-6, -6, 12, 12, parent)
        self.port_id = uuid.uuid4().hex
        self.name = name
        self.is_output = is_output
        self.data_type = data_type
        self._default_brush = QBrush(self._color_for_type(data_type))
        self.setBrush(self._default_brush)
        self.setPen(QPen(Qt.black, 1))
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, False)

        self._label = QGraphicsTextItem(name, parent)
        self._label.setDefaultTextColor(Qt.white)
        self._label.setFlag(QGraphicsTextItem.ItemIsSelectable, False)
        self._connected_edges: list = []

    def set_label_pos(self, x: float, y: float) -> None:
        self._label.setPos(x, y)

    def set_highlight(self, enabled: bool, valid: bool = True) -> None:
        if enabled:
            self.setBrush(QBrush(Qt.green if valid else Qt.red))
        else:
            self.setBrush(self._default_brush)

    def label_width(self) -> float:
        return self._label.boundingRect().width()

    def register_edge(self, edge) -> None:
        """Register an edge connected to this port."""
        if edge is None:
            return
        try:
            if edge not in self._connected_edges:
                self._connected_edges.append(edge)
        except Exception:
            pass

    def unregister_edge(self, edge) -> None:
        """Unregister an edge from this port."""
        if edge is None:
            return
        try:
            self._connected_edges = [e for e in self._connected_edges if e is not edge]
        except Exception:
            pass

    def connected_edges(self) -> list:
        """Return currently valid connected edges and prune stale references."""
        valid = []
        for edge in list(self._connected_edges):
            try:
                if edge is not None and shiboken6.isValid(edge):
                    valid.append(edge)
            except Exception:
                continue
        self._connected_edges = valid
        return valid

    @staticmethod
    def _color_for_type(data_type: str) -> QColor:
        mapping = {
            "numeric": QColor(80, 150, 255),
            "categorical": QColor(80, 200, 120),
            "target": QColor(200, 80, 200),
            "tensor": QColor(220, 160, 60),
            "metrics": QColor(80, 200, 200),
            "any": QColor(200, 200, 200),
        }
        return mapping.get(data_type, QColor(200, 200, 200))