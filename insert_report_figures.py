"""
Generate Typhydrion report figures and insert them into the Word report
at Figure 1 (architecture), Figure 2 (DFD Level 0), and Figure 3 (DFD Level 1).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

BASE = Path(r"d:\collage\major project")
DIAGRAMS = BASE / "diagrams"
DOC_DOWNLOADS = (
    Path(r"C:\Users\NITIN\Downloads")
    / "Typhydrion_Visual_ML_Pipeline_Builder_Report_Through_Preprocessing.docx"
)
DOC_PROJECT = BASE / "Typhydrion_Visual_ML_Pipeline_Builder_Report_Through_Preprocessing.docx"
PROJECT_NAME = "VisualML"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rounded_rect(draw, box, fill, outline, width=2, radius=12):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _center_text(draw, box, text, font, fill="#1a1a1a"):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2), text, font=font, fill=fill)


def _multiline_center(draw, box, lines, font, fill="#1a1a1a", gap=4):
    x0, y0, x1, y1 = box
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + gap * (len(lines) - 1)
    y = (y0 + y1 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text(((x0 + x1 - w) / 2, y), line, font=font, fill=fill)
        y += h + gap


def _arrow(draw, start, end, fill="#334155", width=3):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = end
    # simple arrow head pointing right/down depending on direction
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) >= abs(dy):  # horizontal
        draw.polygon([(x1, y1), (x1 - 10, y1 - 6), (x1 - 10, y1 + 6)], fill=fill)
    else:  # vertical
        draw.polygon([(x1, y1), (x1 - 6, y1 - 10), (x1 + 6, y1 - 10)], fill=fill)


def make_architecture() -> Path:
    """Figure 1 — layered system architecture for Typhydrion."""
    w, h = 1600, 1100
    img = Image.new("RGB", (w, h), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_f = _font(28, bold=True)
    layer_f = _font(18, bold=True)
    box_f = _font(16)
    small_f = _font(14)

    draw.text((40, 24), f"{PROJECT_NAME} — System Architecture (Through Preprocessing)", font=title_f, fill="#0f172a")

    layers = [
        ("User Layer", "#e0f2fe", "#0369a1", ["Desktop Operator / Analyst"]),
        (
            "Presentation Layer (PySide6)",
            "#dbeafe",
            "#1d4ed8",
            ["Main Window", "Node Editor / Scene", "Properties Panel", "Data Preview / Profiler"],
        ),
        (
            "Application Layer",
            "#ede9fe",
            "#6d28d9",
            ["Project Helpers", "Settings", "Async Run Workers"],
        ),
        (
            "Execution Layer",
            "#fef3c7",
            "#b45309",
            ["Graph Validator", "Scheduler", "Pipeline Executor", "Node Runner / Registry"],
        ),
        (
            "Node Runtimes (this phase)",
            "#dcfce7",
            "#15803d",
            ["Dataset Loader", "Missing / Types / Outliers", "Encode / Scale / Feature Select"],
        ),
        (
            "Local Persistence",
            "#f1f5f9",
            "#334155",
            ["Source files (D1)", "In-memory port data (D2)", "cache / logs / reports (D3)"],
        ),
    ]

    y = 80
    layer_h = 140
    margin = 50
    for name, fill, outline, boxes in layers:
        _rounded_rect(draw, (margin, y, w - margin, y + layer_h), fill, outline, width=2, radius=14)
        draw.text((margin + 18, y + 12), name, font=layer_f, fill=outline)
        n = len(boxes)
        gap = 16
        inner_left = margin + 20
        inner_right = w - margin - 20
        usable = inner_right - inner_left
        box_w = (usable - gap * (n - 1)) / n
        box_top = y + 48
        box_bot = y + layer_h - 16
        for i, label in enumerate(boxes):
            bx0 = inner_left + i * (box_w + gap)
            bx1 = bx0 + box_w
            _rounded_rect(draw, (bx0, box_top, bx1, box_bot), "#ffffff", outline, width=2, radius=10)
            lines = label.split(" / ")
            if len(lines) == 1 and len(label) > 28:
                # wrap long labels roughly
                words = label.split()
                mid = len(words) // 2
                lines = [" ".join(words[:mid]), " ".join(words[mid:])]
            _multiline_center(draw, (bx0, box_top, bx1, box_bot), lines, box_f if len(lines) == 1 else small_f)
        # downward arrow between layers
        if name != layers[-1][0]:
            _arrow(draw, (w // 2, y + layer_h + 2), (w // 2, y + layer_h + 22), fill=outline, width=3)
        y += layer_h + 28

    draw.text(
        (40, h - 36),
        "Figure 1: Layered architecture — UI configures nodes; engine validates/schedules; preprocessing runtimes write local artifacts.",
        font=small_f,
        fill="#475569",
    )
    out = DIAGRAMS / "typhydrion_architecture.png"
    img.save(out, "PNG")
    return out


def make_dfd_level0() -> Path:
    """Figure 2 — Level 0 context DFD."""
    w, h = 1500, 780
    img = Image.new("RGB", (w, h), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_f = _font(28, bold=True)
    ent_f = _font(18, bold=True)
    proc_f = _font(20, bold=True)
    flow_f = _font(14)
    small_f = _font(14)

    draw.text((40, 24), f"{PROJECT_NAME} — Data Flow Diagram (Level 0)", font=title_f, fill="#0f172a")

    # External entity left
    left = (60, 280, 320, 420)
    _rounded_rect(draw, left, "#fee2e2", "#b91c1c", width=3, radius=8)
    _multiline_center(draw, left, ["External", "Tabular File", "(CSV / Excel / …)"], ent_f, fill="#7f1d1d")

    # Process bubble
    cx, cy, r = 750, 350, 160
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#dbeafe", outline="#1d4ed8", width=4)
    _multiline_center(
        draw,
        (cx - r, cy - r, cx + r, cy + r),
        ["0.0", PROJECT_NAME, "Preprocessing", "Pipeline"],
        proc_f,
        fill="#1e3a8a",
    )

    # External entity right (outputs as sink)
    right = (1180, 220, 1440, 340)
    _rounded_rect(draw, right, "#dcfce7", "#15803d", width=3, radius=8)
    _multiline_center(draw, right, ["Prepared Feature", "Table"], ent_f, fill="#14532d")

    right2 = (1180, 400, 1440, 520)
    _rounded_rect(draw, right2, "#fef9c3", "#a16207", width=3, radius=8)
    _multiline_center(draw, right2, ["Quality Metadata", "& Reports"], ent_f, fill="#713f12")

    # Data store bottom
    store = (560, 580, 940, 680)
    draw.rectangle(store, fill="#f1f5f9", outline="#334155", width=3)
    draw.line((560, 600, 940, 600), fill="#334155", width=2)
    draw.line((560, 660, 940, 660), fill="#334155", width=2)
    _center_text(draw, (560, 600, 940, 660), "D1/D3 Local files, cache, logs, reports", ent_f)

    # Flows
    _arrow(draw, (320, 350), (cx - r - 4, 350), fill="#b91c1c", width=3)
    draw.text((340, 310), "Raw tabular data", font=flow_f, fill="#b91c1c")

    _arrow(draw, (cx + r + 4, 280), (1180, 280), fill="#15803d", width=3)
    draw.text((960, 248), "Prepared frame", font=flow_f, fill="#15803d")

    _arrow(draw, (cx + r + 4, 420), (1180, 460), fill="#a16207", width=3)
    draw.text((960, 430), "Schema / missing / state", font=flow_f, fill="#a16207")

    _arrow(draw, (750, cy + r + 2), (750, 580), fill="#334155", width=3)
    draw.text((770, 530), "Caches & profiler outputs", font=flow_f, fill="#334155")

    draw.text(
        (40, h - 40),
        f"Figure 2: Level-0 context — external file enters {PROJECT_NAME}; prepared table and quality metadata are produced.",
        font=small_f,
        fill="#475569",
    )
    out = DIAGRAMS / "typhydrion_dfd_level0.png"
    img.save(out, "PNG")
    return out


def make_dfd_level1() -> Path:
    """Figure 3 — Level 1 process flow through preprocessing."""
    w, h = 1700, 980
    img = Image.new("RGB", (w, h), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_f = _font(26, bold=True)
    ent_f = _font(15, bold=True)
    proc_f = _font(14, bold=True)
    flow_f = _font(12)
    small_f = _font(13)

    draw.text((40, 20), f"{PROJECT_NAME} — Data Flow Diagram (Level 1, Through Preprocessing)", font=title_f, fill="#0f172a")

    # External entities
    user_box = (40, 80, 200, 170)
    file_box = (40, 420, 200, 510)
    _rounded_rect(draw, user_box, "#fee2e2", "#b91c1c", width=2, radius=6)
    _multiline_center(draw, user_box, ["User", "(Analyst)"], ent_f, fill="#7f1d1d")
    _rounded_rect(draw, file_box, "#fee2e2", "#b91c1c", width=2, radius=6)
    _multiline_center(draw, file_box, ["Tabular", "File"], ent_f, fill="#7f1d1d")

    def process(box, num, title, fill="#dbeafe", outline="#1d4ed8"):
        draw.ellipse(box, fill=fill, outline=outline, width=3)
        _multiline_center(draw, box, [num, title], proc_f, fill="#1e3a8a", gap=2)

    def store(box, label):
        draw.rectangle(box, fill="#f1f5f9", outline="#334155", width=2)
        y0 = box[1]
        draw.line((box[0], y0 + 14, box[2], y0 + 14), fill="#334155", width=2)
        draw.line((box[0], box[3] - 14, box[2], box[3] - 14), fill="#334155", width=2)
        _center_text(draw, (box[0], y0 + 14, box[2], box[3] - 14), label, ent_f)

    # Processes row
    p1 = (280, 70, 460, 210)
    p2 = (540, 70, 720, 210)
    p3 = (800, 70, 980, 210)
    p4 = (1060, 70, 1240, 210)
    process(p1, "1.0", "Configure &\nValidate Graph")
    process(p2, "2.0", "Dataset\nLoader")
    process(p3, "3.0", "Preview /\nProfile")
    process(p4, "4.0", "Clean:\nMissing/Types\nOutliers", fill="#fef3c7", outline="#b45309")

    p5 = (540, 300, 720, 440)
    p6 = (800, 300, 980, 440)
    p7 = (1060, 300, 1240, 440)
    process(p5, "5.0", "Encode\nCategories", fill="#ede9fe", outline="#6d28d9")
    process(p6, "6.0", "Scale\nFeatures", fill="#ede9fe", outline="#6d28d9")
    process(p7, "7.0", "Feature\nSelect", fill="#ede9fe", outline="#6d28d9")

    p8 = (1320, 300, 1500, 440)
    process(p8, "8.0", "Emit\nPrepared\nOutputs", fill="#dcfce7", outline="#15803d")

    # Stores
    d1 = (260, 560, 480, 650)
    d2 = (700, 560, 980, 650)
    d3 = (1200, 560, 1480, 650)
    store(d1, "D1 Source File")
    store(d2, "D2 Graph Port Data")
    store(d3, "D3 Cache / Logs / Reports")

    # Output sinks
    out1 = (1520, 80, 1680, 170)
    out2 = (1520, 200, 1680, 290)
    _rounded_rect(draw, out1, "#dcfce7", "#15803d", width=2, radius=6)
    _multiline_center(draw, out1, ["Prepared", "Frame"], ent_f, fill="#14532d")
    _rounded_rect(draw, out2, "#fef9c3", "#a16207", width=2, radius=6)
    _multiline_center(draw, out2, ["Metadata", "Reports"], ent_f, fill="#713f12")

    # Arrows — user to configure
    _arrow(draw, (200, 125), (280, 140), fill="#b91c1c", width=2)
    draw.text((205, 95), "config", font=flow_f, fill="#b91c1c")

    # 1 -> 2
    _arrow(draw, (460, 140), (540, 140), fill="#334155", width=2)
    draw.text((475, 112), "run order", font=flow_f, fill="#334155")

    # file -> 2
    _arrow(draw, (200, 465), (600, 210), fill="#b91c1c", width=2)
    draw.text((220, 330), "raw file path", font=flow_f, fill="#b91c1c")

    # 2 -> D1 / D2
    draw.line([(630, 210), (630, 560)], fill="#334155", width=2)
    draw.polygon([(630, 560), (624, 550), (636, 550)], fill="#334155")
    draw.text((640, 380), "read", font=flow_f, fill="#334155")

    # 2 -> 3
    _arrow(draw, (720, 140), (800, 140), fill="#334155", width=2)
    draw.text((735, 112), "raw frame", font=flow_f, fill="#334155")

    # 3 -> 4
    _arrow(draw, (980, 140), (1060, 140), fill="#334155", width=2)
    draw.text((990, 112), "inspect", font=flow_f, fill="#334155")

    # 4 -> 5
    _arrow(draw, (1150, 210), (650, 300), fill="#334155", width=2)
    draw.text((880, 240), "cleaned chunk", font=flow_f, fill="#334155")

    # 5 -> 6 -> 7 -> 8
    _arrow(draw, (720, 370), (800, 370), fill="#6d28d9", width=2)
    _arrow(draw, (980, 370), (1060, 370), fill="#6d28d9", width=2)
    _arrow(draw, (1240, 370), (1320, 370), fill="#15803d", width=2)
    draw.text((740, 345), "encoded", font=flow_f, fill="#6d28d9")
    draw.text((1000, 345), "scaled", font=flow_f, fill="#6d28d9")
    draw.text((1255, 345), "selected", font=flow_f, fill="#15803d")

    # 8 -> outputs
    _arrow(draw, (1500, 340), (1520, 140), fill="#15803d", width=2)
    _arrow(draw, (1500, 380), (1520, 245), fill="#a16207", width=2)

    # 3/8 -> D3
    draw.line([(890, 210), (890, 280), (1340, 280), (1340, 560)], fill="#64748b", width=2)
    draw.polygon([(1340, 560), (1334, 550), (1346, 550)], fill="#64748b")
    draw.text((1350, 480), "profiler HTML", font=flow_f, fill="#64748b")

    # ports to D2
    draw.line([(890, 440), (890, 560)], fill="#334155", width=2)
    draw.polygon([(890, 560), (884, 550), (896, 550)], fill="#334155")
    draw.text((900, 490), "node outputs", font=flow_f, fill="#334155")

    draw.text(
        (40, h - 50),
        "Figure 3: Level-1 flow — load → inspect → clean → encode → scale → select → prepared outputs (+ local stores).",
        font=small_f,
        fill="#475569",
    )
    draw.text(
        (40, h - 28),
        "Note: Split / train / evaluate nodes exist in the codebase but are outside this report phase.",
        font=small_f,
        fill="#64748b",
    )
    out = DIAGRAMS / "typhydrion_dfd_level1.png"
    img.save(out, "PNG")
    return out


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
        if child.tag.endswith("}r") or child.tag.endswith("}hyperlink"):
            paragraph._element.remove(child)


def _insert_picture_in_paragraph(paragraph, image_path: Path, width_inches: float = 6.0):
    """Replace paragraph content with a centered picture."""
    _clear_runs(paragraph)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    return paragraph


def insert_into_doc(doc_path: Path, figures: dict[str, Path]):
    if not doc_path.exists():
        print("Skip missing:", doc_path)
        return

    doc = Document(str(doc_path))
    paras = doc.paragraphs

    # Map caption starts -> (image key, updated caption)
    targets = [
        (
            ("Figure 1",),
            "arch",
            "Figure 1: System architecture — User → Node Editor UI → Graph Validator / "
            "Scheduler → Node Runtimes (Loader → Preprocess…) → Local data/cache, logs, and reports.",
        ),
        (
            ("Figure 2",),
            "dfd0",
            "Figure 2: Level-0 data flow — External tabular file → Typhydrion preprocessing "
            "pipeline → Prepared feature table + quality metadata.",
        ),
        (
            ("Figure 3",),
            "dfd1",
            "Figure 3: Level-1 process flow — Loader → Preview/Profile → Missing/Types/Outliers → "
            "Encode → Scale → Feature Select → Prepared outputs.",
        ),
    ]

    for i, p in enumerate(paras):
        text = p.text.strip()
        for prefixes, key, new_caption in targets:
            if any(text.startswith(prefix) for prefix in prefixes):
                img_para = None
                for j in range(i + 1, min(i + 5, len(paras))):
                    t = paras[j].text.strip()
                    style = paras[j].style.name if paras[j].style else ""
                    if t.startswith("Figure") or t.startswith("5.") or t.startswith("8.") or t.startswith("9."):
                        break
                    if style.startswith("Heading"):
                        break
                    has_drawing = bool(paras[j]._element.findall(".//" + qn("w:drawing")))
                    if not t or has_drawing:
                        img_para = paras[j]
                        break
                if img_para is None:
                    from docx.oxml import OxmlElement
                    from docx.text.paragraph import Paragraph

                    new_el = OxmlElement("w:p")
                    p._element.addnext(new_el)
                    img_para = Paragraph(new_el, p._parent)

                _insert_picture_in_paragraph(img_para, figures[key], width_inches=6.1)
                _clear_runs(p)
                run = p.add_run(new_caption)
                _set_run_font(run, size_pt=11, bold=True)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                print(f"  Updated {key} after caption at para {i} in {doc_path.name}")
                break

    doc.save(str(doc_path))
    print("Saved:", doc_path)


def main():
    DIAGRAMS.mkdir(parents=True, exist_ok=True)
    arch = make_architecture()
    dfd0 = make_dfd_level0()
    dfd1 = make_dfd_level1()
    print("Generated:", arch)
    print("Generated:", dfd0)
    print("Generated:", dfd1)

    figures = {"arch": arch, "dfd0": dfd0, "dfd1": dfd1}
    insert_into_doc(DOC_DOWNLOADS, figures)
    # Keep project copy identical to Downloads (avoids leftover unused media)
    import shutil

    shutil.copy2(DOC_DOWNLOADS, DOC_PROJECT)
    print("Copied to:", DOC_PROJECT)


if __name__ == "__main__":
    main()
