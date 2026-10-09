"""
Run a small preprocessing path and capture Typhydrion output windows,
then insert the screenshots into the report (Figures 4–7).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

ROOT = Path(r"d:\collage\major project")
SRC = ROOT / "ml_node_" / "src"
SHOTS = ROOT / "diagrams" / "screenshots"
SAMPLE_CSV = SHOTS / "sample_preprocess.csv"
DOC_DOWNLOADS = (
    Path(r"C:\Users\NITIN\Downloads")
    / "Typhydrion_Visual_ML_Pipeline_Builder_Report_Through_Preprocessing.docx"
)
DOC_PROJECT = ROOT / "Typhydrion_Visual_ML_Pipeline_Builder_Report_Through_Preprocessing.docx"

sys.path.insert(0, str(SRC))


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [22, 31, None, 45, 28, 36, 19, None, 52, 41, 27, 33],
            "income": [28000, 54000, 41000, None, 36000, 72000, 19000, 48000, None, 61000, 33000, 45000],
            "city": ["Bengaluru", "Delhi", "Mumbai", "Bengaluru", None, "Chennai", "Delhi", "Mumbai", "Chennai", "Bengaluru", "Delhi", None],
            "plan": ["Basic", "Plus", "Basic", "Pro", "Plus", "Pro", "Basic", None, "Plus", "Pro", "Basic", "Plus"],
            "score": [0.42, 0.81, None, 0.67, 0.55, 0.91, 0.33, 0.74, 0.62, None, 0.48, 0.70],
            "visits": [3, 8, 5, 12, 4, None, 2, 7, 9, 6, 3, 5],
        }
    )


def _run_pipeline(csv_path: Path) -> dict:
    from nodes.io.dataset_loader_node import DatasetLoaderNode
    from nodes.preprocess.encoding_node import EncodingNode
    from nodes.preprocess.missing_value_node import MissingValueNode
    from nodes.preprocess.scaling_node import ScalingNode

    loader = DatasetLoaderNode("loader", {"options": {"path": str(csv_path), "reader": "read_csv"}})
    loaded = loader.run({})
    if not loaded.success:
        raise RuntimeError(loaded.error_message)

    raw = loaded.outputs["Raw Data"]
    missing = MissingValueNode("missing", {"options": {"Strategy": "Median"}})
    cleaned = missing.run({"Chunk": raw})
    if not cleaned.success:
        raise RuntimeError(cleaned.error_message)

    encoder = EncodingNode("encode", {"options": {"Method": "One-Hot", "Max Categories": 10}})
    encoded = encoder.run({"Chunk": cleaned.outputs["Clean Chunk"]})
    if not encoded.success:
        raise RuntimeError(encoded.error_message)

    scaler = ScalingNode("scale", {"options": {"Method": "StandardScaler"}})
    scaled = scaler.run({"Features": encoded.outputs["Encoded Features"]})
    if not scaled.success:
        raise RuntimeError(scaled.error_message)

    return {
        "raw": raw,
        "clean": cleaned.outputs["Clean Chunk"],
        "missing_report": cleaned.outputs["Missing Report"],
        "encoded": encoded.outputs["Encoded Features"],
        "scaled": scaled.outputs["Scaled Features"],
        "stats": loaded.outputs["Stats"],
    }


def _numpy_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Cast pandas StringDtype columns to object so the statistics window can inspect them."""
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.StringDtype):
            out[col] = out[col].astype(object)
    return out


def _grab(app, widget, path: Path, size=(1180, 700), prepare=None) -> None:
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

    host = QWidget()
    host.setWindowTitle("Typhydrion")
    host.setStyleSheet("background-color: #121c2c;")
    layout = QVBoxLayout(host)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.addWidget(widget)
    host.resize(*size)
    host.show()
    for _ in range(8):
        QApplication.processEvents()
    if prepare is not None:
        prepare()
        for _ in range(4):
            QApplication.processEvents()
    widget.repaint()
    QApplication.processEvents()
    host.grab().save(str(path), "PNG")
    host.hide()
    host.close()
    QApplication.processEvents()


def capture() -> dict[str, Path]:
    from PySide6.QtWidgets import QApplication
    from ui.windows.data_statistics_window import DataStatisticsWindow
    from ui.windows.data_window import DataPreviewWindow
    from ui.windows.node_output_window import NodeOutputWindow

    SHOTS.mkdir(parents=True, exist_ok=True)
    frame = _sample_frame()
    frame.to_csv(SAMPLE_CSV, index=False)
    outputs = _run_pipeline(SAMPLE_CSV)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(
        """
        QWidget { color: rgba(230, 238, 248, 230); }
        """
    )

    paths = {
        "preview": SHOTS / "output_data_preview.png",
        "stats": SHOTS / "output_data_statistics.png",
        "missing": SHOTS / "output_missing_report.png",
        "scaled": SHOTS / "output_scaled_features.png",
    }

    preview = DataPreviewWindow()
    # Preview cells are shown as text so nullable integers/NaN do not overflow Qt items.
    preview_df = outputs["raw"].copy()
    preview_df = preview_df.apply(
        lambda col: col.map(lambda v: "" if pd.isna(v) else v)
    )
    preview.set_dataframe(preview_df)
    _grab(app, preview, paths["preview"])

    stats = DataStatisticsWindow()
    stats.set_dataframe(_numpy_friendly(outputs["raw"]))
    _grab(app, stats, paths["stats"])

    missing_win = NodeOutputWindow()
    missing_win.set_node(
        "Missing Value Handler",
        {
            "Clean Chunk": outputs["clean"],
            "Missing Report": outputs["missing_report"],
        },
    )
    missing_win._port_combo.setCurrentText("Missing Report")
    QApplication = __import__("PySide6.QtWidgets", fromlist=["QApplication"]).QApplication
    QApplication.processEvents()
    _grab(app, missing_win, paths["missing"])

    scaled_win = NodeOutputWindow()
    scaled_win.set_node(
        "Feature Scaler",
        {"Scaled Features": outputs["scaled"]},
    )
    scaled_win._tabs.setCurrentIndex(2)  # Summary tab
    _grab(app, scaled_win, paths["scaled"], size=(1480, 820))

    print("Captured screenshots:")
    for key, path in paths.items():
        print(f"  {key}: {path} ({path.stat().st_size} bytes)")
    print(
        "Pipeline shapes:",
        "raw", outputs["raw"].shape,
        "clean", outputs["clean"].shape,
        "scaled", outputs["scaled"].shape,
    )
    paths.update(capture_node_windows())
    return paths


def capture_node_windows() -> dict[str, Path]:
    """Screenshot the node editor graph and the node properties panel."""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QApplication
    from nodes.base.edge import EdgeItem
    from ui.windows.node_editor_window import NodeEditorWindow
    from ui.windows.node_properties_window import NodePropertiesWindow

    app = QApplication.instance() or QApplication(sys.argv)
    editor = NodeEditorWindow()
    scene = editor._scene
    view = editor._view
    scene.clear_graph()

    titles = [
        "Dataset Loader",
        "Missing Value Handler",
        "Categorical Encoder",
        "Feature Scaler",
    ]
    nodes = []
    for title in titles:
        node = scene.add_node(title, 0, 0)
        view._node_menu.apply_ports(node)
        nodes.append(node)

    # Left-to-right so the connecting curves stay readable.
    gap_x = 110
    x = 0.0
    for node in nodes:
        node.setPos(x, 0)
        x += node.sceneBoundingRect().width() + gap_x

    links = [
        (0, "Raw Data", 1, "Chunk"),
        (1, "Clean Chunk", 2, "Chunk"),
        (2, "Encoded Features", 3, "Features"),
    ]
    for src_i, src_name, dst_i, dst_name in links:
        source = next(p for p in nodes[src_i]._outputs if p.name == src_name)
        target = next(p for p in nodes[dst_i]._inputs if p.name == dst_name)
        edge = EdgeItem()
        scene.addItem(edge)
        edge.set_temporary(False)
        edge.connect_ports(source, target)

    editor.resize(1600, 640)
    editor.show()
    for _ in range(8):
        QApplication.processEvents()

    bounds = nodes[0].sceneBoundingRect()
    for node in nodes[1:]:
        bounds = bounds.united(node.sceneBoundingRect())
    bounds = bounds.adjusted(-36, -28, 48, 28)
    for node in nodes:
        for port in node._inputs + node._outputs:
            for edge in port.connected_edges():
                edge.update_position()

    paths = {
        "editor": SHOTS / "node_editor_window.png",
        "props_loader": SHOTS / "node_properties_loader.png",
        "props_missing": SHOTS / "node_properties_missing.png",
    }

    # Render the scene directly. Grabbing the graphics view crashes on this Qt build.
    img = QImage(1680, 560, QImage.Format_ARGB32)
    img.fill(QColor("#0e1624"))
    painter = QPainter(img)
    scene.render(painter, QRectF(0, 0, 1680, 560), bounds, Qt.KeepAspectRatio)
    painter.end()
    img.save(str(paths["editor"]))

    loader_props = NodePropertiesWindow()
    loader_props.set_node("Dataset Loader", "Dataset Loader")
    _grab(app, loader_props, paths["props_loader"], size=(980, 780))

    missing_props = NodePropertiesWindow()
    missing_props.set_node(
        "Missing Value Handler",
        "Missing Value Handler",
        {"Columns": "age, income, city, score", "Inplace": True, "Limit": 0},
    )
    _grab(app, missing_props, paths["props_missing"], size=(980, 720))

    try:
        editor._node_update_timer.stop()
        view.shutdown_engine_thread()
        editor.hide()
    except Exception:
        pass

    print("Captured node windows:")
    for key, path in paths.items():
        print(f"  {key}: {path} ({path.stat().st_size} bytes)")
    return paths


def _set_run_font(run, size_pt=11, bold=False, name="Times New Roman"):
    run.bold = bold
    run.font.size = Pt(size_pt)
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:cs"), name)


def _clear_runs(paragraph):
    for child in list(paragraph._element):
        if child.tag.endswith("}r") or child.tag.endswith("}hyperlink") or child.tag.endswith("}drawing"):
            paragraph._element.remove(child)


def _para_after(paragraph) -> Paragraph:
    new_el = OxmlElement("w:p")
    paragraph._element.addnext(new_el)
    return Paragraph(new_el, paragraph._parent)


def _caption(paragraph, text: str) -> None:
    _clear_runs(paragraph)
    run = paragraph.add_run(text)
    _set_run_font(run, size_pt=11, bold=True)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(10)


def _picture(paragraph, image_path: Path) -> None:
    _clear_runs(paragraph)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(6.1))


def _find_para(doc: Document, startswith: str):
    for p in doc.paragraphs:
        if p.text.strip().startswith(startswith):
            return p
    return None


def _last_before(doc: Document, start_prefix: str, stop_prefix: str | None):
    """Last non-empty paragraph in a section, so new figures sit at the end of it."""
    paras = doc.paragraphs
    start = None
    for i, p in enumerate(paras):
        if p.text.strip().startswith(start_prefix):
            start = i
            break
    if start is None:
        return None
    last = paras[start]
    for j in range(start + 1, len(paras)):
        text = paras[j].text.strip()
        if stop_prefix and text.startswith(stop_prefix):
            break
        if text:
            last = paras[j]
    return last


def _body_after_heading(doc: Document, heading_prefix: str):
    paras = doc.paragraphs
    for i, p in enumerate(paras):
        if p.text.strip().startswith(heading_prefix):
            for j in range(i + 1, min(i + 6, len(paras))):
                text = paras[j].text.strip()
                if text and not text.startswith("Figure") and not text.startswith("Table"):
                    return paras[j]
            return paras[i]
    return None


def insert_screenshots(doc_path: Path, shots: dict[str, Path]) -> None:
    doc = Document(str(doc_path))
    refresh = {
        "Figure 4:": shots["preview"],
        "Figure 5:": shots["stats"],
        "Figure 6:": shots["missing"],
        "Figure 7:": shots["scaled"],
        "Figure 8:": shots["editor"],
        "Figure 9:": shots["props_loader"],
        "Figure 10:": shots["props_missing"],
    }
    if any(p.text.strip().startswith("Figure 4:") for p in doc.paragraphs):
        print("Refreshing existing output screenshots in", doc_path.name)
        paras = doc.paragraphs
        for i, p in enumerate(paras):
            text = p.text.strip()
            for prefix, image in refresh.items():
                if text.startswith(prefix) and i > 0:
                    _picture(paras[i - 1], image)
                    break

    blocks = [
        (
            "9.1",
            [
                (shots["preview"], "Figure 4: Data Preview window — raw dataset returned by the Dataset Loader (nulls visible before imputation)."),
                (shots["stats"], "Figure 5: Data Statistics window — row/column counts, memory, missing percentage, and column quality for the loaded dataset."),
            ],
        ),
        (
            "9.2",
            [
                (shots["missing"], "Figure 6: Node Output window — Missing Report from the Missing Value Handler (Median strategy, mode fallback for categories)."),
                (shots["scaled"], "Figure 7: Node Output window — Scaled Features summary after one-hot encoding and StandardScaler."),
            ],
        ),
        (
            "5.5",
            [
                (shots["editor"], "Figure 8: Node editor window — preprocessing graph with Dataset Loader, Missing Value Handler, Categorical Encoder, and Feature Scaler."),
                (shots["props_loader"], "Figure 9: Node properties window for the Dataset Loader (pandas reader parameters)."),
                (shots["props_missing"], "Figure 10: Node properties window for the Missing Value Handler."),
            ],
        ),
    ]

    existing = {p.text.strip() for p in doc.paragraphs}
    for heading, items in blocks:
        pending = []
        for image, caption in items:
            prefix = caption.split(":", 1)[0] + ":"
            if any(text.startswith(prefix) for text in existing):
                continue
            pending.append((image, caption))
        if not pending:
            continue
        stop = {"5.5": "5.6", "9.1": "9.2", "9.2": "9.3"}.get(heading)
        anchor = _last_before(doc, heading, stop) if stop else _body_after_heading(doc, heading)
        if anchor is None:
            print("Heading not found:", heading)
            continue
        for image, caption in pending:
            img_p = _para_after(anchor)
            cap_p = _para_after(img_p)
            _picture(img_p, image)
            _caption(cap_p, caption)
            anchor = cap_p
        print(f"Inserted screenshots under {heading} in {doc_path.name}")

    doc.save(str(doc_path))
    print("Saved:", doc_path)


def main() -> None:
    shots = capture()
    # Project copy is the writable source of truth; Downloads may be open in Word.
    source = DOC_PROJECT if DOC_PROJECT.exists() else DOC_DOWNLOADS
    insert_screenshots(source, shots)
    try:
        shutil.copy2(source, DOC_DOWNLOADS)
        print("Updated:", DOC_DOWNLOADS)
    except PermissionError:
        alt = DOC_DOWNLOADS.with_name(
            DOC_DOWNLOADS.stem + "_With_Output_Screenshots" + DOC_DOWNLOADS.suffix
        )
        shutil.copy2(source, alt)
        print("Downloads file is open. Wrote:", alt)


if __name__ == "__main__":
    main()
