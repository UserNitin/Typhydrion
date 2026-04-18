from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QGraphicsScene
import shiboken6

from nodes.base.node_base import NodeItem


class NodeGraphScene(QGraphicsScene):
    def __init__(self) -> None:
        super().__init__()
        self.setSceneRect(-2000, -2000, 4000, 4000)
        # Safety lock: prevent accidental node removal from unknown code paths.
        self._allow_node_removal = False

    def removeItem(self, item) -> None:
        """Block NodeItem removal unless explicitly authorized."""
        if isinstance(item, NodeItem) and not self._allow_node_removal:
            return
        try:
            if not shiboken6.isValid(item):
                return
            super().removeItem(item)
        except RuntimeError:
            pass

    def remove_node(self, node: NodeItem) -> None:
        """Explicitly remove a node from the scene."""
        try:
            if not shiboken6.isValid(node):
                return
        except Exception:
            return
        self._allow_node_removal = True
        try:
            super().removeItem(node)
        except RuntimeError:
            pass
        finally:
            self._allow_node_removal = False

    def clear_graph(self) -> None:
        """Explicitly clear the full graph, including nodes."""
        self._allow_node_removal = True
        try:
            super().clear()
        except RuntimeError:
            pass
        finally:
            self._allow_node_removal = False

    def add_node(self, title: str, x: float, y: float) -> NodeItem:
        try:
            from ui.app_settings import AppSettings
            s = AppSettings()
            w = int(s.get_int(s.NODE_DEFAULT_W))
            h = int(s.get_int(s.NODE_DEFAULT_H))
            node = NodeItem(title, width=w, height=h)
        except Exception:
            node = NodeItem(title)
        node.setPos(x, y)
        self.addItem(node)
        self.ensure_item_in_scene(node)
        return node

    def ensure_item_in_scene(self, item, margin: float = 450.0) -> None:
        """Expand scene rect so the given item remains reachable/visible."""
        try:
            if item is None or not shiboken6.isValid(item):
                return
            item_rect = item.sceneBoundingRect()
            if item_rect.isNull():
                return
            padded = item_rect.adjusted(-margin, -margin, margin, margin)
            current = self.sceneRect()
            if current.isNull():
                self.setSceneRect(padded)
                return
            if not current.contains(padded):
                self.setSceneRect(current.united(padded))
        except Exception:
            return

    def ensure_rect_in_scene(self, rect: QRectF, margin: float = 250.0) -> None:
        """Expand scene rect to include an arbitrary scene-space rectangle."""
        try:
            if rect.isNull():
                return
            padded = rect.adjusted(-margin, -margin, margin, margin)
            current = self.sceneRect()
            if current.isNull():
                self.setSceneRect(padded)
                return
            if not current.contains(padded):
                self.setSceneRect(current.united(padded))
        except Exception:
            return