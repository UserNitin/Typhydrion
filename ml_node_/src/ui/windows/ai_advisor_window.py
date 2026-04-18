from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class AIAdvisorWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("AI Advisor")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        subtitle = QLabel("Suggestions and explanations will appear here.")
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 170);")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch(1)