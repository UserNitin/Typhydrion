from __future__ import annotations

from datetime import datetime
from pathlib import Path
import webbrowser

import pandas as pd
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from ydata_profiling import ProfileReport  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ProfileReport = None

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    QWebEngineView = None


class DataProfilerWindow(QWidget):
    """Interactive data profiling window powered by ydata-profiling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(560, 420)
        self.setObjectName("dataProfilerRoot")
        self.setStyleSheet(
            """
            QWidget#dataProfilerRoot {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(18, 28, 44, 220),
                    stop:1 rgba(14, 20, 32, 230)
                );
                border: 1px solid rgba(70, 130, 220, 40);
                border-radius: 12px;
            }
            QLabel#profilerTitle {
                color: rgba(228, 240, 255, 245);
                font-size: 30px;
                font-weight: 700;
                background: transparent;
            }
            QLabel#profilerSubtitle {
                color: rgba(165, 205, 245, 220);
                font-size: 13px;
                background: transparent;
            }
            QLineEdit {
                background-color: rgba(15, 24, 38, 235);
                color: rgba(225, 240, 255, 240);
                border: 1px solid rgba(80, 145, 230, 65);
                border-radius: 8px;
                padding: 7px 10px;
                selection-background-color: rgba(65, 135, 230, 160);
            }
            QLineEdit:focus {
                border: 1px solid rgba(110, 180, 255, 140);
            }
            QPushButton {
                background-color: rgba(28, 44, 66, 235);
                color: rgba(214, 232, 252, 240);
                border: 1px solid rgba(90, 155, 235, 80);
                border-radius: 8px;
                padding: 7px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(38, 62, 94, 245);
                border-color: rgba(120, 190, 255, 140);
            }
            QPushButton:pressed {
                background-color: rgba(20, 40, 64, 245);
            }
            QPushButton:disabled {
                color: rgba(160, 180, 205, 140);
                background-color: rgba(24, 32, 44, 180);
                border-color: rgba(90, 120, 150, 50);
            }
            QFrame#viewerFrame {
                background-color: rgba(10, 16, 24, 230);
                border: 1px solid rgba(70, 130, 220, 55);
                border-radius: 10px;
            }
            """
        )

        self._dataframe: pd.DataFrame | None = None
        self._report_path: Path | None = None
        self._data_label = "Loaded Dataset"

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title = QLabel("Data Profiler")
        title.setObjectName("profilerTitle")
        root.addWidget(title)

        subtitle = QLabel("Select a dataset and generate a detailed ydata-profiling report.")
        subtitle.setObjectName("profilerSubtitle")
        root.addWidget(subtitle)

        controls = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Choose dataset file (.csv, .tsv, .xlsx, .json, .parquet)")
        controls.addWidget(self._path_edit, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._pick_file)
        controls.addWidget(browse_btn)

        self._generate_btn = QPushButton("Generate Profile")
        self._generate_btn.clicked.connect(self._generate_profile)
        controls.addWidget(self._generate_btn)

        self._open_browser_btn = QPushButton("Open in Browser")
        self._open_browser_btn.setEnabled(False)
        self._open_browser_btn.clicked.connect(self._open_in_browser)
        controls.addWidget(self._open_browser_btn)

        root.addLayout(controls)

        self._status = QLabel("Ready")
        self._status.setStyleSheet(
            "color: rgba(160, 210, 255, 220); background: transparent; font-size: 13px;"
        )
        root.addWidget(self._status)

        self._viewer = None
        if QWebEngineView is not None:
            viewer_frame = QFrame()
            viewer_frame.setObjectName("viewerFrame")
            vf_layout = QVBoxLayout(viewer_frame)
            vf_layout.setContentsMargins(1, 1, 1, 1)
            vf_layout.setSpacing(0)

            self._viewer = QWebEngineView()
            self._viewer.setHtml(
                """
                <html><body style="
                    margin:0;
                    background:#0a1220;
                    color:#a9cfff;
                    font-family:Segoe UI, Arial, sans-serif;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    height:100vh;
                ">
                    Generate a profile to view report.
                </body></html>
                """
            )
            vf_layout.addWidget(self._viewer)
            root.addWidget(viewer_frame, 1)
        else:
            note = QLabel(
                "Embedded report view is unavailable (QtWebEngine not installed).\n"
                "The generated report can still be opened in your default browser."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: rgba(240, 190, 120, 220);")
            root.addWidget(note, 1)

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "Datasets (*.csv *.tsv *.xlsx *.xls *.json *.parquet);;All Files (*.*)",
        )
        if path:
            self._path_edit.setText(path)

    @staticmethod
    def _read_dataset(path: Path) -> pd.DataFrame:
        ext = path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path)
        if ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if ext == ".json":
            return pd.read_json(path)
        if ext == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file extension: {ext}")

    def _set_busy(self, busy: bool) -> None:
        self._generate_btn.setEnabled(not busy)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _generate_profile(self) -> None:
        if ProfileReport is None:
            QMessageBox.warning(
                self,
                "Missing Dependency",
                "ydata-profiling is not installed.\nRun: pip install ydata-profiling",
            )
            return

        try:
            self._set_busy(True)
            raw_path = self._path_edit.text().strip()
            if raw_path:
                dataset_path = Path(raw_path)
                if not dataset_path.exists():
                    QMessageBox.warning(self, "Invalid File", "Selected dataset file does not exist.")
                    return
                self._status.setText("Loading dataset...")
                QApplication.processEvents()
                df = self._read_dataset(dataset_path)
                self._dataframe = df
                self._data_label = dataset_path.name
            elif self._dataframe is not None and not self._dataframe.empty:
                df = self._dataframe
            else:
                QMessageBox.information(
                    self,
                    "Dataset Required",
                    "Choose a dataset file, or load a dataset in workspace first.",
                )
                return

            rows, cols = df.shape
            use_minimal = rows > 100_000 or cols > 120
            self._status.setText(f"Generating profile for {rows:,} rows x {cols} columns...")
            QApplication.processEvents()

            profile = ProfileReport(
                df,
                title=f"Data Profile - {self._data_label}",
                explorative=True,
                minimal=use_minimal,
            )

            reports_dir = Path.cwd() / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            out_path = reports_dir / f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            profile.to_file(str(out_path))
            self._report_path = out_path

            self._status.setText(f"Profile generated: {out_path.name}")
            self._open_browser_btn.setEnabled(True)

            if self._viewer is not None:
                self._viewer.setUrl(QUrl.fromLocalFile(str(out_path.resolve())))
            else:
                webbrowser.open(out_path.resolve().as_uri())

        except Exception as exc:
            QMessageBox.critical(self, "Profile Failed", f"Could not generate profile report:\n{exc}")
            self._status.setText("Profile generation failed.")
        finally:
            self._set_busy(False)

    def _open_in_browser(self) -> None:
        if self._report_path is None or not self._report_path.exists():
            QMessageBox.information(self, "No Report", "Generate a profile report first.")
            return
        webbrowser.open(self._report_path.resolve().as_uri())

    def set_dataframe(self, df: pd.DataFrame | None) -> None:
        """Receive active dataset from workspace and allow profiling without file picker."""
        if df is None or df.empty:
            self._dataframe = None
            self._data_label = "Loaded Dataset"
            return
        self._dataframe = df
        self._data_label = "Workspace Dataset"
        self._status.setText(f"Workspace dataset ready: {df.shape[0]:,} rows x {df.shape[1]} columns")
