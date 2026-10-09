"""Set the whole report to 13 pt and finish filling the Section 7 page."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph

DOWNLOADS = Path(r"C:\Users\NITIN\Downloads")
SOURCE = DOWNLOADS / "Typhydrion_Report_Sections_3_7_Expanded_13pt_Grayscale_DFD.docx"
TARGET = DOWNLOADS / "Typhydrion_Visual_ML_Pipeline_Builder_Report_Through_Preprocessing.docx"
BACKUP = DOWNLOADS / "Typhydrion_Report_Before_13pt.docx"

SECTION_7_EXTRA = (
    "pandas carries the DataFrame between nodes, NumPy supports numeric arrays, "
    "scikit-learn supplies the preprocessing algorithms, and ydata-profiling produces "
    "HTML quality reports, so the whole stack runs locally without a cloud account."
)
SECTION_7_MARKER = "pandas carries the DataFrame between nodes"


def set_run_font(run, size: float) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    for key in ("ascii", "hAnsi", "cs"):
        fonts.set(qn(f"w:{key}"), "Times New Roman")


def size_for(paragraph: Paragraph) -> float:
    style = paragraph.style.name if paragraph.style else ""
    return 14 if style == "Heading 1" else 13


def add_section_7_paragraph(doc: Document) -> None:
    if any(p.text.strip().startswith(SECTION_7_MARKER) for p in doc.paragraphs):
        return
    anchor = next(p for p in doc.paragraphs if p.text.strip() == "8. DATA FLOW DIAGRAM")
    element = OxmlElement("w:p")
    anchor._element.addprevious(element)
    paragraph = Paragraph(element, anchor._parent)
    paragraph.style = "Normal"
    set_run_font(paragraph.add_run(SECTION_7_EXTRA), 13)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = paragraph.paragraph_format
    fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(6)


def apply_13pt(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(13)
    for paragraph in doc.paragraphs:
        size = size_for(paragraph)
        for run in paragraph.runs:
            set_run_font(run, size)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        set_run_font(run, 13)


def main() -> None:
    if not BACKUP.exists() and TARGET.exists():
        shutil.copy2(TARGET, BACKUP)
    doc = Document(str(SOURCE))
    add_section_7_paragraph(doc)
    if "--whole" in sys.argv:
        apply_13pt(doc)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0]) if args else TARGET
    doc.save(str(out))
    print("Saved:", out)


if __name__ == "__main__":
    main()
