from __future__ import annotations

from collections import defaultdict
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QPushButton,
    QFrame,
)


class NodePaletteWidget(QWidget):
    """Lightweight node palette (search + category + list)."""

    node_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._catalog: list[dict[str, Any]] = []

        self.setObjectName("nodePalette")
        self.setStyleSheet("""
            QWidget#nodePalette {
                background: rgba(12, 18, 30, 235);
                border: 1px solid rgba(55, 110, 200, 35);
                border-radius: 12px;
            }
            QLabel {
                color: rgba(170, 210, 255, 235);
                font-weight: 700;
            }
            QLineEdit {
                background: rgba(22, 32, 50, 220);
                color: rgba(200, 225, 255, 230);
                border: 1px solid rgba(55, 110, 200, 55);
                border-radius: 8px;
                padding: 7px 10px;
                font-size: 12px;
            }
            QComboBox {
                background: rgba(22, 32, 50, 220);
                color: rgba(200, 225, 255, 220);
                border: 1px solid rgba(55, 110, 200, 45);
                border-radius: 8px;
                padding: 6px 10px;
                font-size: 12px;
            }
            QListWidget {
                background: rgba(16, 24, 38, 210);
                color: rgba(190, 220, 255, 220);
                border: 1px solid rgba(55, 110, 200, 30);
                border-radius: 10px;
                padding: 4px;
            }
            QListWidget::item { padding: 6px 8px; border-radius: 6px; }
            QListWidget::item:hover:!selected { background: rgba(30, 55, 90, 110); }
            QListWidget::item:selected { background: rgba(40, 80, 160, 150); }
            QPushButton {
                background: rgba(28, 42, 64, 220);
                color: rgba(200, 225, 255, 230);
                border: 1px solid rgba(60, 120, 200, 45);
                border-radius: 8px;
                padding: 7px 10px;
                font-weight: 650;
            }
            QPushButton:hover {
                background: rgba(40, 80, 140, 230);
                border-color: rgba(70, 150, 255, 110);
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("Nodes")
        title.setStyleSheet("font-size: 13px;")
        header.addWidget(title)
        header.addStretch(1)
        self._add_btn = QPushButton("Add")
        self._add_btn.setToolTip("Add selected node")
        self._add_btn.clicked.connect(self._emit_selected)
        header.addWidget(self._add_btn)
        root.addLayout(header)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search nodes…")
        self._search.textChanged.connect(self._refresh)
        root.addWidget(self._search)

        self._category = QComboBox()
        self._category.currentTextChanged.connect(self._refresh)
        root.addWidget(self._category)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _it: self._emit_selected())
        root.addWidget(self._list, 1)

        footer = QFrame()
        footer.setStyleSheet("background: transparent;")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(8)
        hint = QLabel("Tip: Double-click to add")
        hint.setStyleSheet("color: rgba(140, 175, 215, 190); font-weight: 500; font-size: 11px;")
        fl.addWidget(hint)
        fl.addStretch(1)
        root.addWidget(footer)

    def set_catalog(self, catalog: list[dict[str, Any]]) -> None:
        self._catalog = list(catalog or [])
        cats = sorted({str(n.get("category", "")).strip() for n in self._catalog if n.get("category")})
        self._category.blockSignals(True)
        try:
            self._category.clear()
            self._category.addItem("All")
            for c in cats:
                self._category.addItem(c)
        finally:
            self._category.blockSignals(False)
        self._refresh()

    def _emit_selected(self) -> None:
        it = self._list.currentItem()
        if not it:
            return
        name = it.data(Qt.UserRole)
        if name:
            self.node_requested.emit(str(name))

    def _refresh(self) -> None:
        q = (self._search.text() or "").strip().lower()
        cat = (self._category.currentText() or "All").strip()

        # group items by category order, but show as a flat list
        out: list[dict[str, Any]] = []
        for n in self._catalog:
            name = str(n.get("name", "")).strip()
            if not name:
                continue
            ncat = str(n.get("category", "")).strip() or "Other"
            desc = str(n.get("desc", "")).strip()
            if cat != "All" and ncat != cat:
                continue
            if q and (q not in name.lower()) and (q not in desc.lower()):
                continue
            out.append(n)

        # Stable ordering: category then name
        out.sort(key=lambda n: (str(n.get("category", "")), str(n.get("name", ""))))

        self._list.clear()
        for n in out[:500]:
            name = str(n.get("name", ""))
            desc = str(n.get("desc", ""))
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            if desc:
                item.setToolTip(desc)
            self._list.addItem(item)

        if self._list.count() > 0 and self._list.currentRow() < 0:
            self._list.setCurrentRow(0)

