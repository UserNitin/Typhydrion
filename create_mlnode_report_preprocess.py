"""
Rewrite the Alliance MCA report for Typhydrion (ml_node_)
through data preprocessing only. Later model/output sections
are replaced with a short deferred-work note.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, Twips
from docx.oxml import OxmlElement

SRC = Path(
    r"C:\Users\NITIN\Downloads"
    r"\Predictive_Analytics_for_Strategic_Intelligence_in_eSports_Report_Final_Reviewed.docx"
)
OUT = Path(
    r"C:\Users\NITIN\Downloads"
    r"\Typhydrion_Visual_ML_Pipeline_Builder_Report_Through_Preprocessing.docx"
)
OUT_PROJECT = Path(
    r"d:\collage\major project"
    r"\Typhydrion_Visual_ML_Pipeline_Builder_Report_Through_Preprocessing.docx"
)

PROJECT_TITLE = (
    "Typhydrion: A Visual Node-Based Machine Learning "
    "Pipeline Builder (Through Preprocessing)"
)
SHORT_TITLE = "Typhydrion — Visual Node-Based ML Pipeline Builder"


def set_run_font(run, *, bold=None, size_pt=None, name="Times New Roman"):
    if bold is not None:
        run.bold = bold
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    run.font.name = name
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:cs"), name)


def clear_paragraph(paragraph):
    p = paragraph._element
    for child in list(p):
        if child.tag.endswith("}r") or child.tag.endswith("}hyperlink"):
            p.remove(child)


def set_paragraph_text(paragraph, text, *, bold=False, size_pt=12, align=None, justify=False):
    clear_paragraph(paragraph)
    run = paragraph.add_run(text)
    set_run_font(run, bold=bold, size_pt=size_pt)
    if align is not None:
        paragraph.alignment = align
    elif justify:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = paragraph.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE


def add_para_after(paragraph, text, *, style="Normal", bold=False, size_pt=12, justify=True):
    new_p = OxmlElement("w:p")
    paragraph._element.addnext(new_p)
    # Bind as paragraph
    from docx.text.paragraph import Paragraph

    p = Paragraph(new_p, paragraph._parent)
    if style:
        try:
            p.style = style
        except KeyError:
            pass
    set_paragraph_text(p, text, bold=bold, size_pt=size_pt, justify=justify)
    return p


def delete_element(element):
    parent = element.getparent()
    if parent is not None:
        parent.remove(element)


def clear_table(table):
    tbl = table._tbl
    delete_element(tbl)


def fill_table(table, rows):
    """Replace table contents with rows (list of lists). Resize as needed."""
    # Clear existing rows except keep structure by rewriting cells
    # Easier: delete table and insert new one after a marker paragraph.
    raise NotImplementedError


def create_table_after(paragraph, data, col_widths=None):
    """Create a table immediately after paragraph and return it."""
    doc = paragraph.part.document
    # Insert a temporary empty paragraph marker, add table at end, then move
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    try:
        table.style = "Table Grid"
    except KeyError:
        pass  # some templates lack built-in table styles
    for i, row in enumerate(data):
        for j, cell_text in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(cell_text))
            set_run_font(run, bold=(i == 0), size_pt=10)
            p.paragraph_format.space_after = Pt(2)
    # Move table XML after paragraph
    paragraph._element.addnext(table._tbl)
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)
    return table


def remove_inline_shapes(doc):
    body = doc.element.body
    # Remove drawings from paragraphs
    for p in body.iter(qn("w:drawing")):
        parent = p.getparent()
        if parent is not None:
            parent.remove(p)
    for p in body.iter(qn("w:pict")):
        parent = p.getparent()
        if parent is not None:
            parent.remove(p)


def rebuild_document():
    doc = Document(str(SRC))
    remove_inline_shapes(doc)

    # --- Cover page text updates ---
    replacements = {
        "Project Name: Predictive Analytics for Strategic Intelligence in eSports":
            f"Project Name: {SHORT_TITLE}",
        "Project Title: Predictive Analytics for Strategic Intelligence in eSports":
            f"Project Title: {SHORT_TITLE}",
    }
    for p in doc.paragraphs:
        if p.text in replacements:
            align = p.alignment
            bold = any(r.bold for r in p.runs) if p.runs else True
            set_paragraph_text(p, replacements[p.text], bold=bold, size_pt=12, align=align)

    # Delete all tables (will recreate needed ones)
    for table in list(doc.tables):
        clear_table(table)

    # Build paragraph index map by finding headings
    paras = doc.paragraphs

    def find_exact(text):
        for i, p in enumerate(paras):
            if p.text.strip() == text:
                return i
        return None

    def find_startswith(prefix):
        for i, p in enumerate(paras):
            if p.text.strip().startswith(prefix):
                return i
        return None

    # Content map: from Abstract through end — rewrite in place where possible,
    # then truncate leftover body after References.

    # ACK stays mostly same — lightly retarget one sentence if needed
    # Keep acknowledgement as-is (guide/student names).

    # === 1. ABSTRACT ===
    i = find_exact("1. ABSTRACT")
    abstract_paras = [
        (
            "This project develops Typhydrion (codebase ml_node_), a desktop machine-learning "
            "workflow builder with a visual node-based editor. Instead of writing every "
            "pipeline step as a separate script, the user connects reusable nodes for data "
            "loading, inspection, and preprocessing, then executes the graph through a "
            "validated scheduling engine. The application is built with PySide6 (Qt for Python) "
            "and organizes local artifacts in working folders for data cache, logs, and reports."
        ),
        (
            "The present report covers the system through the preprocessing stage. At this "
            "stage the implemented path includes dataset loading from common tabular formats "
            "(CSV, Excel, JSON, Parquet, and other pandas readers), data preview and profiling "
            "support, missing-value handling, data-type conversion, categorical encoding, "
            "numeric scaling, outlier handling, feature selection, and optional text "
            "preprocessing. Graph validation, topological scheduling, and pipeline execution "
            "orchestrate these nodes so that cleaned feature tables can be produced repeatably "
            "from a visual workflow."
        ),
        (
            "Later stages such as train/test splitting, model training, evaluation dashboards, "
            "and deployment packaging are present in the codebase as node categories but are "
            "outside the detailed scope of this report phase. The contribution documented here "
            "is a coherent data-to-preprocessing pipeline that reduces manual spreadsheet and "
            "ad-hoc script work while keeping each transformation inspectable as a graph node."
        ),
    ]
    # Replace next normal paragraphs after abstract heading
    _rewrite_section_body(paras, i, abstract_paras, stop_heading="2. INTRODUCTION")

    # === 2. INTRODUCTION ===
    i = find_exact("2. INTRODUCTION")
    # 2.1
    i21 = find_exact("2.1 eSports data and decision making")
    if i21 is not None:
        set_paragraph_text(
            paras[i21],
            "2.1 Machine-learning workflows and decision support",
            bold=True,
            size_pt=12,
        )
        intro_21 = [
            (
                "Modern analytical work typically proceeds as a pipeline: load tabular data, "
                "inspect quality, repair missing or inconsistent values, encode categories, "
                "scale numeric fields, select useful features, and only then train models. "
                "When these steps live in disconnected notebooks or spreadsheets, definitions "
                "drift, joins are repeated by hand, and it becomes difficult to replay the same "
                "preparation for a new file. A visual node graph makes each transformation "
                "explicit: ports carry data between nodes, options are stored on the node, and "
                "the execution engine runs the graph in dependency order."
            ),
            (
                "Predictive analytics still depends on sound preprocessing. Scaling choices, "
                "imputation strategy, and encoding method change the feature space that any "
                "later model will see. Strategic value at this stage is therefore operational: "
                "a reusable, inspectable preparation path that documents what was done to the "
                "data before modeling begins. Typhydrion focuses on that path by exposing "
                "preprocessing as first-class nodes inside a desktop editor rather than as "
                "hidden helper code."
            ),
        ]
        _rewrite_section_body(paras, i21, intro_21, stop_heading="2.2")

    # 2.2 - original had broken heading; find "2.2 Project Context" or following heading text
    i22 = find_startswith("2.2")
    if i22 is None:
        # The original had "2.2 Project Context" as Normal, then a Heading 2 with long text
        i22 = find_exact("2.2 Project Context")
    if i22 is not None:
        set_paragraph_text(paras[i22], "2.2 Project context", bold=True, size_pt=12)
        # Also fix the next heading-like long paragraph if it exists
        intro_22 = [
            (
                "Typhydrion is engineered as a local desktop application. The main window "
                "hosts a node editor, node properties panel, data preview/statistics/profiler "
                "windows, output views, and settings. Users assemble a directed acyclic graph "
                "(DAG) from a palette of nodes. The registry maps visual titles such as "
                "Dataset Loader, Missing Value Handler, Categorical Encoder, Feature Scaler, "
                "Outlier Handler, Feature Selector, and Text Preprocessor to runtime classes."
            ),
            (
                "In this report phase, strategic capability is bounded to data ingestion and "
                "preprocessing. The system does not claim live telemetry ingestion, cloud "
                "model serving, or completed end-to-end competition forecasting. Those "
                "capabilities belong to later modules (split, model, training, evaluation) "
                "and remain outside the detailed narrative until they are documented with "
                "verified runs."
            ),
        ]
        _rewrite_section_body(paras, i22, intro_22, stop_heading="2.3")

    # Fix the orphan long Heading 2 from original if still present
    for p in paras:
        t = p.text.strip()
        if t.startswith("In this specific implementation, strategic intelligence"):
            clear_paragraph(p)
            p.style = "Normal"

    i23 = find_startswith("2.3")
    if i23 is not None:
        set_paragraph_text(
            paras[i23],
            "2.3 Aim, objectives, and scope",
            bold=True,
            size_pt=12,
        )
        intro_23 = [
            (
                "The aim is to provide a desktop environment where tabular datasets can be "
                "loaded, inspected, and prepared for modeling through connected preprocessing "
                "nodes. The implementation objectives for this phase are to: (1) load datasets "
                "through configurable pandas readers; (2) expose preview, schema, and "
                "statistics; (3) handle missing values with explicit strategies; (4) convert "
                "data types safely; (5) encode categorical columns; (6) scale numeric "
                "features; (7) detect and treat outliers; (8) select informative features; "
                "and (9) execute the resulting graph with validation and scheduling."
            ),
            "Objectives stated as checklist items:",
            "• Provide a visual node editor with ports, edges, and property configuration.",
            "• Validate the graph for structural errors and produce a topological run order.",
            "• Implement preprocessing nodes that emit cleaned frames plus metadata reports.",
            "• Keep intermediate artifacts and profiling reports in local project folders.",
            (
                "Scope for this report stops after preprocessing. Train/validation splits, "
                "classifier/regressor training, metrics dashboards, and export/inference nodes "
                "exist in the repository but are not evaluated in depth here."
            ),
        ]
        _rewrite_section_body(paras, i23, intro_23, stop_heading="3. PROBLEM STATEMENT")

    # === 3. PROBLEM STATEMENT ===
    i = find_exact("3. PROBLEM STATEMENT")
    problem = [
        (
            "The project addresses the difficulty of turning raw tabular files into a "
            "consistent, reusable prepared dataset. Manual cleaning in spreadsheets or "
            "one-off scripts makes it hard to compare strategies (for example mean versus "
            "median imputation), to preserve the same encoding mapping across runs, and to "
            "see missingness or outlier decisions before modeling. When preparation steps "
            "are not represented as an explicit workflow, leakage, silent type errors, and "
            "irreproducible feature sets become likely."
        ),
        (
            "The specific problem implemented in this phase is a visual preprocessing "
            "pipeline: load a dataset, inspect it, apply a chosen sequence of cleaning and "
            "transformation nodes, and obtain a prepared feature table with accompanying "
            "reports (missingness summary, encoder state, scaler state, outlier mask, "
            "feature scores). The user remains responsible for choosing strategies; the "
            "tool makes those choices visible and repeatable."
        ),
        (
            "The system does not claim to solve model selection or production monitoring in "
            "this report phase. Its practical objective is to organize evidence about the "
            "data and produce prepared inputs that later modeling stages can consume."
        ),
    ]
    _rewrite_section_body(paras, i, problem, stop_heading="3.1")

    i31 = find_startswith("3.1")
    if i31 is not None:
        set_paragraph_text(
            paras[i31],
            "3.1 Operational impact of the problem",
            bold=True,
            size_pt=12,
        )
        body_31 = [
            (
                "Fragmented preparation has a direct operational cost. An analyst repeating "
                "the same cleaning for each new CSV must reapply filters, retype conversions, "
                "and recheck null rates. Small differences in column names, date parsing, or "
                "category handling produce incompatible feature matrices. A node graph with "
                "saved options reduces these avoidable inconsistencies and makes the "
                "preparation path reviewable by a supervisor or teammate."
            ),
        ]
        _rewrite_section_body(paras, i31, body_31, stop_heading="3.2")

    i32 = find_startswith("3.2")
    if i32 is not None:
        set_paragraph_text(
            paras[i32],
            "3.2 Functional requirements and boundaries",
            bold=True,
            size_pt=12,
        )
        body_32 = [
            (
                "The application needs to accept a file path and reader configuration; "
                "preview rows and schema; allow the user to connect preprocessing nodes; "
                "execute the graph; and surface node outputs and metadata. It should fail "
                "clearly when required inputs are missing (for example no file path on the "
                "loader, or no target for a supervised feature-selection method). These "
                "requirements describe a decision-support preparation tool: modeling choices "
                "remain with the analyst, and no cleaned table is presented as a finished "
                "prediction."
            ),
        ]
        _rewrite_section_body(paras, i32, body_32, stop_heading="4. EXISTING SYSTEM")

    # === 4. EXISTING SYSTEM ===
    i = find_exact("4. EXISTING SYSTEM")
    # just heading

    i41 = find_startswith("4.1")
    if i41 is not None:
        set_paragraph_text(paras[i41], "4.1 Traditional analysis", bold=True, size_pt=12)
        body_41 = [
            (
                "A conventional workflow for preparing ML data may involve opening CSV files "
                "in a spreadsheet, deleting incomplete rows by hand, typing formulas for "
                "averages, and exporting a cleaned sheet to a notebook. This can work for a "
                "small file, but it requires repeated filtering and arithmetic when formats "
                "change or when several preparation strategies must be compared. Definitions "
                "for missing-value rules and encoding often remain undocumented."
            ),
        ]
        _rewrite_section_body(paras, i41, body_41, stop_heading="4.2")

    i42 = find_startswith("4.2")
    if i42 is not None:
        set_paragraph_text(
            paras[i42],
            "4.2 Limitations addressed by this project",
            bold=True,
            size_pt=12,
        )
        body_42 = [
            (
                "Relevant limitations are the effort required to chain repeated cleaning "
                "steps, the absence of a single visual surface connecting loaders to "
                "transformers, and the difficulty of inspecting missingness or outliers "
                "before interpreting downstream results. A manual reviewer may also apply "
                "inconsistent encodings between training and later reuse. The project "
                "responds with a shared node runtime, fitted-state storage on encoding and "
                "scaling nodes, missing-value reports, and graph-level execution."
            ),
            "Table 1: Comparison of manual preparation and the implemented system",
        ]
        _rewrite_section_body(paras, i42, body_42, stop_heading="4.3")
        # Insert comparison table after the caption paragraph
        caption_idx = None
        for j, p in enumerate(paras):
            if p.text.strip().startswith("Table 1:"):
                caption_idx = j
                break
        if caption_idx is not None:
            create_table_after(
                paras[caption_idx],
                [
                    ["Aspect", "Manual preparation", "Typhydrion (this phase)"],
                    [
                        "Record organization",
                        "Files and sheets reviewed separately.",
                        "Dataset Loader produces a frame plus schema/stats outputs.",
                    ],
                    [
                        "Missing values",
                        "Ad-hoc delete/fill without a saved strategy.",
                        "Missing Value Handler with mean/median/mode/constant/ffill/bfill/interpolate/drop.",
                    ],
                    [
                        "Encoding / scaling",
                        "Often rewritten per notebook.",
                        "Categorical Encoder and Feature Scaler nodes with stored state.",
                    ],
                    [
                        "Outliers / features",
                        "Informal visual checks.",
                        "Outlier Handler and Feature Selector with scores metadata.",
                    ],
                    [
                        "Reproducibility",
                        "Hard to replay exact steps.",
                        "Graph of nodes executed in topological order by the pipeline engine.",
                    ],
                ],
                col_widths=[1.4, 2.2, 2.6],
            )
            note_idx = caption_idx + 1
            # Find a following empty/normal para to put note
            for j in range(caption_idx + 1, min(caption_idx + 5, len(paras))):
                if not paras[j].text.strip() or paras[j].text.strip().startswith("This comparison"):
                    set_paragraph_text(
                        paras[j],
                        "This comparison describes design intent for the preprocessing phase "
                        "rather than a timed user study.",
                        justify=True,
                    )
                    break

    i43 = find_startswith("4.3")
    if i43 is not None:
        set_paragraph_text(
            paras[i43],
            "4.3 Difference from a full AutoML or cloud platform",
            bold=True,
            size_pt=12,
        )
        body_43 = [
            (
                "The supplied implementation should be understood as a local visual pipeline "
                "builder rather than a managed cloud AutoML service. It does not automatically "
                "choose the best model for every problem, host APIs on the internet, or ingest "
                "streaming telemetry. The current report focuses on completed tabular "
                "preparation steps that the user configures explicitly."
            ),
        ]
        _rewrite_section_body(paras, i43, body_43, stop_heading="5. PROPOSED SYSTEM")

    # === 5. PROPOSED SYSTEM ===
    i51 = find_startswith("5.1")
    if i51 is not None:
        set_paragraph_text(paras[i51], "5.1 System overview", bold=True, size_pt=12)
        body_51 = [
            (
                "The proposed system is a PySide6 desktop application. The home shell "
                "(MainWindow) opens the node editor and supporting windows. The user places "
                "nodes from the palette, configures options (file path, imputation strategy, "
                "scaler method, and similar), connects ports, and runs either a single node "
                "or the full pipeline. The engine validates the graph, schedules levels, and "
                "executes node runtimes through the registry."
            ),
            (
                "Figure 1 (conceptual): User → Node Editor UI → Graph Validator / Scheduler → "
                "Node Runtimes (Loader → Preprocess…) → Local data/cache, logs, and reports."
            ),
        ]
        _rewrite_section_body(paras, i51, body_51, stop_heading="5.2")

    i52 = find_startswith("5.2")
    if i52 is not None:
        set_paragraph_text(paras[i52], "5.2 System modules", bold=True, size_pt=12)
        body_52 = [
            "Table 2: Modules covered through preprocessing",
        ]
        _rewrite_section_body(paras, i52, body_52, stop_heading="5.3")
        for j, p in enumerate(paras):
            if p.text.strip().startswith("Table 2:"):
                create_table_after(
                    p,
                    [
                        ["Module", "Role in this phase"],
                        [
                            "UI shell and node editor",
                            "Visual graph editing, property panels, async run workers, data preview hooks.",
                        ],
                        [
                            "Node registry",
                            "Maps titles such as Dataset Loader and Missing Value Handler to runtime classes.",
                        ],
                        [
                            "Graph validator / scheduler",
                            "Checks structure, detects cycles, builds topological execution levels.",
                        ],
                        [
                            "Pipeline executor / node runner",
                            "Feeds upstream outputs into nodes, records timing, errors, and terminal outputs.",
                        ],
                        [
                            "Data IO nodes",
                            "Load, preview, merge, select columns, filter rows.",
                        ],
                        [
                            "Preprocessing nodes",
                            "Missing values, type conversion, encoding, scaling, outliers, feature selection, text cleanup.",
                        ],
                        [
                            "Local artifact folders",
                            "data/, logs/, reports/ hold caches, diagnostics, and profiler HTML outputs.",
                        ],
                    ],
                    col_widths=[2.0, 4.2],
                )
                break

    i53 = find_startswith("5.3")
    if i53 is not None:
        set_paragraph_text(
            paras[i53],
            "5.3 Data preprocessing (detailed)",
            bold=True,
            size_pt=12,
        )
        body_53 = [
            (
                "Preprocessing is the focus of this report. After the Dataset Loader emits a "
                "raw frame with schema and statistics, the user chains transformation nodes. "
                "Each node accepts a chunk or feature frame on its input port, applies a "
                "configured strategy, and emits a cleaned frame plus optional metadata "
                "(reports or fitted state)."
            ),
            (
                "Missing Value Handler supports mean, median, mode, constant fill, forward "
                "and backward fill, linear interpolation for numeric columns, and dropping "
                "rows or columns using a missingness threshold. It can restrict processing to "
                "selected columns and always produces a missingness report (counts and "
                "percentages by column) before mutation."
            ),
            (
                "Data Type Converter coerces columns to int, float, string, bool, datetime, "
                "or category, with optional date formats and error coercion. Categorical "
                "Encoder supports one-hot (with max-category capping), label, ordinal, "
                "target, binary, and frequency encoding, and stores an encoder state for "
                "reuse. Feature Scaler applies StandardScaler, MinMaxScaler, RobustScaler, "
                "MaxAbsScaler, Normalizer, PowerTransformer, or QuantileTransformer on "
                "numeric columns, with optional outlier clipping after scaling."
            ),
            (
                "Outlier Handler detects anomalies with IQR, Z-score, Isolation Forest, or "
                "LOF and can remove, cap, replace with mean/median, or flag rows. Feature "
                "Selector offers variance threshold, correlation filtering, SelectKBest, "
                "RFE, L1 regularization, tree importance, and mutual information, returning "
                "selected columns and score dictionaries. Text Preprocessor lowercases text, "
                "strips punctuation, and removes a basic stopword list on a chosen column."
            ),
            (
                "Together these nodes form a configurable preparation path. Typical order is "
                "load → (optional filter/column select) → missing-value handling → type "
                "conversion → outlier handling → encoding → scaling → feature selection. "
                "The graph may differ by dataset; the engine only requires a valid DAG."
            ),
        ]
        # Stop before 5.4 — we will replace remaining proposed-system subsections
        _rewrite_section_body(paras, i53, body_53, stop_heading="5.4")

    # Replace 5.4–5.7 with deferred-scope notes (still before models deep-dive)
    i54 = find_startswith("5.4")
    if i54 is not None:
        set_paragraph_text(
            paras[i54],
            "5.4 Subsequent modules (deferred in this report)",
            bold=True,
            size_pt=12,
        )
        body_54 = [
            (
                "The codebase also registers split nodes (train/test, train/val/test, cross "
                "validation, time-series split), model nodes (classification, regression, "
                "clustering, anomaly, neural network), training utilities, and evaluation "
                "nodes (metrics, visualization, report, explainer). Those modules are "
                "acknowledged for architecture completeness but are not specified, trained, "
                "or evaluated in this report phase."
            ),
        ]
        _rewrite_section_body(paras, i54, body_54, stop_heading="5.5")

    i55 = find_startswith("5.5")
    if i55 is not None:
        set_paragraph_text(
            paras[i55],
            "5.5 Preprocessing workflow",
            bold=True,
            size_pt=12,
        )
        body_55 = [
            "• The user opens Typhydrion and creates or loads a project graph.",
            "• A Dataset Loader node is configured with path and pandas reader options.",
            "• Preview/statistics/profiler windows are used to inspect nulls, dtypes, and distributions.",
            "• Preprocessing nodes are connected according to the chosen cleaning strategy.",
            "• The pipeline is validated and executed; each node writes outputs and metadata.",
            "• The prepared frame is available for later split/model nodes in a future phase.",
        ]
        _rewrite_section_body(paras, i55, body_55, stop_heading="5.6")

    i56 = find_startswith("5.6")
    if i56 is not None:
        set_paragraph_text(
            paras[i56],
            "5.6 System architecture",
            bold=True,
            size_pt=12,
        )
        body_56 = [
            (
                "The user layer is the desktop operator. The presentation layer is PySide6 "
                "(main window, node editor view/scene, palette, toolbars, property and data "
                "windows). The application layer includes settings and project helpers. The "
                "execution layer comprises graph validation, scheduling, node running, and "
                "pipeline orchestration. The node layer implements IO and preprocessing "
                "runtimes under a shared NodeRuntime contract (NodeResult, NodeContext). "
                "Persistence for this phase is file-based (datasets on disk; caches, logs, "
                "and HTML profile reports under project folders). There is no separate cloud "
                "database or remote model service in the inspected local design."
            ),
        ]
        _rewrite_section_body(paras, i56, body_56, stop_heading="5.7")

    i57 = find_startswith("5.7")
    if i57 is not None:
        set_paragraph_text(
            paras[i57],
            "5.7 Limitations and safeguards",
            bold=True,
            size_pt=12,
        )
        body_57 = [
            (
                "Preprocessing quality depends on user choices and on the source file. "
                "Imputation and encoding cannot invent ground truth; they only apply "
                "declared rules. Some advanced options (for example full NLP stemming "
                "libraries) are only partially implemented. Large files require attention "
                "to chunking settings and available memory. Fitted scaler/encoder state "
                "must be reused consistently when a later modeling phase is added."
            ),
            (
                "The application is a local academic prototype. Dependency pinning, "
                "automated engine tests, and formal packaging metadata are recommended "
                "hardening steps. Users should only load trusted datasets and review "
                "node metadata before treating a cleaned table as analysis-ready."
            ),
        ]
        _rewrite_section_body(paras, i57, body_57, stop_heading="6. ABOUT DATASET")

    # === 6. ABOUT DATASET ===
    i61 = find_startswith("6.1")
    if i61 is not None:
        set_paragraph_text(
            paras[i61],
            "6.1 Dataset handling and scope",
            bold=True,
            size_pt=12,
        )
        body_61 = [
            (
                "Typhydrion is dataset-agnostic in this phase: the Dataset Loader accepts "
                "paths configured by the user and dispatches to pandas readers (for example "
                "read_csv, read_excel, read_json, read_parquet, and related functions). The "
                "project is not locked to a single domain file. Sample or cached frames may "
                "reside under the application's data/ directory for local experiments."
            ),
            (
                "Because this report stops at preprocessing, no competition-specific "
                "tournament register or third-party scorecard feed is required. Any "
                "tabular file with identifiable columns can be used to demonstrate loading, "
                "missing-value reports, encoding, and scaling."
            ),
            "Table 3: Primary data artifacts relevant to preprocessing",
        ]
        _rewrite_section_body(paras, i61, body_61, stop_heading="6.2")
        for j, p in enumerate(paras):
            if p.text.strip().startswith("Table 3:"):
                create_table_after(
                    p,
                    [
                        ["Artifact / output", "Produced by", "Purpose"],
                        [
                            "Raw Data frame",
                            "Dataset Loader",
                            "Source table for all downstream nodes.",
                        ],
                        [
                            "Schema / Stats",
                            "Dataset Loader",
                            "dtypes, nullability, unique counts, missing totals.",
                        ],
                        [
                            "Missing Report",
                            "Missing Value Handler",
                            "Per-column missing counts and percentages.",
                        ],
                        [
                            "Encoder / Scaler State",
                            "Encoding / Scaling nodes",
                            "Fitted mappings for reproducible transforms.",
                        ],
                        [
                            "Outlier Mask",
                            "Outlier Handler",
                            "Boolean flags of detected extreme values.",
                        ],
                        [
                            "Feature Scores",
                            "Feature Selector",
                            "Ranking used to keep informative columns.",
                        ],
                        [
                            "Profiler HTML",
                            "Data profiler window / reports/",
                            "Human-readable quality overview.",
                        ],
                    ],
                    col_widths=[1.8, 1.8, 2.6],
                )
                break

    i62 = find_startswith("6.2")
    if i62 is not None:
        set_paragraph_text(
            paras[i62],
            "6.2 Data structure and types",
            bold=True,
            size_pt=12,
        )
        body_62 = [
            (
                "Loaded tables are pandas DataFrames. Numeric columns are the primary input "
                "to scaling, outlier detection, and many feature-selection methods. Object "
                "and category columns are the primary input to encoding. Datetime conversion "
                "is available through the type-converter node. The loader also lists feature "
                "and target candidates to help the user plan later supervised steps."
            ),
            "Table 4: Column roles during preprocessing",
        ]
        _rewrite_section_body(paras, i62, body_62, stop_heading="6.3")
        for j, p in enumerate(paras):
            if p.text.strip().startswith("Table 4:"):
                create_table_after(
                    p,
                    [
                        ["Role", "Typical dtypes", "Handled by"],
                        [
                            "Numeric features",
                            "int, float",
                            "Imputation, scaling, outliers, feature selection",
                        ],
                        [
                            "Categorical features",
                            "object, category",
                            "Mode/constant fill, encoding",
                        ],
                        [
                            "Text fields",
                            "object (free text)",
                            "Text Preprocessor",
                        ],
                        [
                            "Temporal fields",
                            "datetime (after conversion)",
                            "Data Type Converter; later time-series split (deferred)",
                        ],
                        [
                            "Target (optional now)",
                            "numeric or label",
                            "Required only for supervised selection methods",
                        ],
                    ],
                    col_widths=[1.6, 1.8, 2.8],
                )
                break

    i63 = find_startswith("6.3")
    if i63 is not None:
        set_paragraph_text(paras[i63], "6.3 Data quality", bold=True, size_pt=12)
        body_63 = [
            (
                "Quality checks in this phase are node- and window-driven rather than a "
                "single hard-coded domain validator. The loader reports overall missing "
                "percentage and column groups. The Missing Value Handler emits a structured "
                "missing report. The data statistics and profiler windows support "
                "distribution and type review. Users should still verify source provenance "
                "and business definitions outside the tool."
            ),
        ]
        _rewrite_section_body(paras, i63, body_63, stop_heading="6.4")

    i64 = find_startswith("6.4")
    if i64 is not None:
        set_paragraph_text(
            paras[i64],
            "6.4 Inputs and preprocessing outputs",
            bold=True,
            size_pt=12,
        )
        body_64 = [
            (
                "Pipeline inputs are the file path, reader name/kwargs, and the ordered "
                "graph of preprocessing options. Outputs are cleaned frames on node ports, "
                "plus metadata such as missing reports, encoder/scaler state, outlier masks, "
                "and feature scores. No model target metrics are claimed in this phase."
            ),
            "Table 5: Preprocessing node summary",
        ]
        _rewrite_section_body(paras, i64, body_64, stop_heading="6.5")
        for j, p in enumerate(paras):
            if p.text.strip().startswith("Table 5:"):
                create_table_after(
                    p,
                    [
                        ["Node", "Key options", "Main outputs"],
                        [
                            "Dataset Loader",
                            "path, reader, chunksize, kwargs",
                            "Raw Data, Schema, Stats, candidates",
                        ],
                        [
                            "Missing Value Handler",
                            "strategy, fill value, drop threshold, columns",
                            "Clean Chunk, Missing Report",
                        ],
                        [
                            "Data Type Converter",
                            "columns, target type, date format",
                            "Converted Chunk",
                        ],
                        [
                            "Categorical Encoder",
                            "method, max categories, drop first",
                            "Encoded Features, Encoder State",
                        ],
                        [
                            "Feature Scaler",
                            "method, with mean/std, clip outliers",
                            "Scaled Features, Scaler State",
                        ],
                        [
                            "Outlier Handler",
                            "method, threshold, action",
                            "Clean Chunk, Outlier Mask",
                        ],
                        [
                            "Feature Selector",
                            "method, k features, threshold",
                            "Selected Features, Feature Scores",
                        ],
                        [
                            "Text Preprocessor",
                            "column, lowercase, punctuation, stopwords",
                            "Processed Text",
                        ],
                    ],
                    col_widths=[1.7, 2.2, 2.3],
                )
                break

    i65 = find_startswith("6.5")
    if i65 is not None:
        set_paragraph_text(
            paras[i65],
            "6.5 Transformations in this phase (no train/test claim)",
            bold=True,
            size_pt=12,
        )
        body_65 = [
            (
                "Chronological or random train/test splitting belongs to later split nodes "
                "and is not evaluated here. Transformations documented in this phase are "
                "imputation, type coercion, categorical encoding, numeric scaling, outlier "
                "treatment, unsupervised or supervised feature selection (when a target "
                "port is supplied), and light text normalization. When modeling is added "
                "in a future report, fitted preprocessing state must be applied "
                "consistently to holdout data to avoid leakage."
            ),
        ]
        _rewrite_section_body(paras, i65, body_65, stop_heading="7. TECHNOLOGIES USED")

    # === 7. TECHNOLOGIES ===
    i7 = find_exact("7. TECHNOLOGIES USED")
    if i7 is not None:
        body_7 = [
            (
                "Technologies are taken from the Typhydrion requirements and source imports "
                "relevant to the UI and preprocessing path."
            ),
            "Table 6: Technologies used through preprocessing",
        ]
        _rewrite_section_body(paras, i7, body_7, stop_heading="7.1")
        for j, p in enumerate(paras):
            if p.text.strip().startswith("Table 6:"):
                create_table_after(
                    p,
                    [
                        ["Technology", "Purpose"],
                        ["Python 3.10+", "Application and node runtime logic."],
                        ["PySide6", "Desktop UI, windows, widgets, settings."],
                        ["pandas / NumPy", "Tabular IO, cleaning, encoding helpers, arrays."],
                        ["scikit-learn", "Scalers, encoders support, feature-selection utilities, outlier detectors."],
                        ["SciPy / joblib", "Scientific routines and serialization helpers."],
                        ["PyArrow", "Efficient Parquet/Feather-oriented loading support."],
                        ["ydata-profiling", "HTML data-profile reports from the profiler window."],
                        ["matplotlib", "Plotting support used by analysis windows/nodes."],
                        ["psutil", "Resource-related utilities."],
                    ],
                    col_widths=[1.8, 4.4],
                )
                break

    i71 = find_startswith("7.1")
    if i71 is not None:
        set_paragraph_text(
            paras[i71],
            "7.1 Application and interface technologies",
            bold=True,
            size_pt=12,
        )
        body_71 = [
            (
                "Python provides the application logic. PySide6 maps user actions to the "
                "node editor and supporting windows. Application settings persist UI and "
                "pipeline preferences. This combination keeps the demonstration "
                "self-contained on a local workstation."
            ),
        ]
        _rewrite_section_body(paras, i71, body_71, stop_heading="7.2")

    i72 = find_startswith("7.2")
    if i72 is not None:
        set_paragraph_text(
            paras[i72],
            "7.2 Data preparation technologies",
            bold=True,
            size_pt=12,
        )
        body_72 = [
            (
                "pandas and NumPy support tabular handling. scikit-learn supplies scalers, "
                "feature selectors, and outlier estimators used by preprocessing nodes. "
                "PyArrow assists columnar file formats. ydata-profiling supports richer "
                "quality reports. TensorFlow is listed for later neural/anomaly nodes and "
                "is not required to explain the preprocessing phase itself."
            ),
        ]
        _rewrite_section_body(paras, i72, body_72, stop_heading="8. DATA FLOW DIAGRAM")

    # === 8. DATA FLOW ===
    i8 = find_exact("8. DATA FLOW DIAGRAM")
    if i8 is not None:
        body_8 = [
            (
                "The data-flow view for this phase follows the path visible in the node "
                "editor and engine: the user configures a loader; data moves through "
                "preprocessing nodes; the executor records outputs. Model inference paths "
                "are omitted from the detailed flow because they are out of scope here."
            ),
            (
                "Figure 2 (conceptual Level 0): External tabular file → Typhydrion "
                "preprocessing pipeline → Prepared feature table + quality metadata."
            ),
        ]
        _rewrite_section_body(paras, i8, body_8, stop_heading="8.1")

    i81 = find_startswith("8.1")
    if i81 is not None:
        set_paragraph_text(
            paras[i81],
            "8.1 Level 1 process flow (through preprocessing)",
            bold=True,
            size_pt=12,
        )
        body_81 = [
            (
                "At Level 1, the graph is validated and scheduled. The Dataset Loader reads "
                "the file and emits Raw Data with Schema/Stats. Optional Column Selector or "
                "Filter nodes reduce the working set. Missing Value Handler cleans nulls and "
                "emits a Missing Report. Type conversion, outlier handling, encoding, "
                "scaling, and feature selection run in the user-defined order. Terminal "
                "outputs are the prepared frame and associated metadata. If a required "
                "input port is empty, the node returns a failed NodeResult with an error "
                "message rather than a silent partial transform."
            ),
            (
                "Figure 3 (conceptual Level 1): Loader → Preview/Profile → Missing/"
                "Types/Outliers → Encode → Scale → Feature Select → Prepared outputs."
            ),
        ]
        _rewrite_section_body(paras, i81, body_81, stop_heading="8.2")

    i82 = find_startswith("8.2")
    if i82 is not None:
        set_paragraph_text(
            paras[i82],
            "8.2 Data stores and boundaries",
            bold=True,
            size_pt=12,
        )
        body_82 = [
            (
                "D1 is the source file on disk. D2 is the in-memory/graph data flowing "
                "between node ports during a run. D3 covers local folders (data/cache, "
                "logs, reports) for caches and profiler outputs. Fitted encoder/scaler "
                "state is held on node runtimes for the session and is intended to support "
                "consistent reuse when modeling is added later. Player-telemetry or live "
                "game feeds are outside this boundary."
            ),
        ]
        _rewrite_section_body(paras, i82, body_82, stop_heading="9. OUTPUT")

    # === 9. OUTPUT — retitle to preprocessing outputs ===
    i9 = find_exact("9. OUTPUT")
    if i9 is not None:
        set_paragraph_text(paras[i9], "9. PREPROCESSING OUTPUTS", bold=True, size_pt=14)

    i91 = find_startswith("9.1")
    if i91 is not None:
        set_paragraph_text(
            paras[i91],
            "9.1 Loader and inspection outputs",
            bold=True,
            size_pt=12,
        )
        body_91 = [
            (
                "A successful Dataset Loader run returns a Raw Data frame together with "
                "feature/target candidate lists, a per-column schema (dtype, nullable, "
                "unique count), and aggregate statistics (shape, memory, missing total/"
                "percent, numeric vs categorical column lists). Preview nodes and data "
                "windows show head rows and dtype summaries for interactive checks."
            ),
        ]
        _rewrite_section_body(paras, i91, body_91, stop_heading="9.2")

    i92 = find_startswith("9.2")
    if i92 is not None:
        set_paragraph_text(
            paras[i92],
            "9.2 Cleaning and transformation outputs",
            bold=True,
            size_pt=12,
        )
        body_92 = [
            (
                "Missing Value Handler returns Clean Chunk plus Missing Report metadata "
                "(rows before/after and strategy used). Encoding and scaling nodes return "
                "transformed frames and serializable state dictionaries. Outlier Handler "
                "returns a cleaned frame and an outlier mask. Feature Selector returns the "
                "reduced feature matrix and score map. These outputs are the tangible "
                "deliverables of the present report phase."
            ),
        ]
        _rewrite_section_body(paras, i92, body_92, stop_heading="9.3")

    i93 = find_startswith("9.3")
    if i93 is not None:
        set_paragraph_text(
            paras[i93],
            "9.3 Example preprocessing path",
            bold=True,
            size_pt=12,
        )
        body_93 = [
            (
                "A representative academic demonstration path is: load a CSV → inspect "
                "missingness in the profiler → impute numeric columns with median and "
                "categoricals with mode → one-hot encode categories with a max-category "
                "cap → standardize numeric features → drop low-variance columns. Exact "
                "numeric metrics depend on the user-supplied file and are therefore not "
                "copied from the previous eSports evaluation tables."
            ),
            "Table 7: Representative preprocessing run checklist",
        ]
        _rewrite_section_body(paras, i93, body_93, stop_heading="9.4")
        for j, p in enumerate(paras):
            if p.text.strip().startswith("Table 7:"):
                create_table_after(
                    p,
                    [
                        ["Step", "Node", "Expected evidence"],
                        ["1", "Dataset Loader", "Non-empty Raw Data; schema/stats populated"],
                        ["2", "Missing Value Handler", "Missing Report; reduced null counts"],
                        ["3", "Categorical Encoder", "Encoded Features; encoder_state stored"],
                        ["4", "Feature Scaler", "Scaled Features; scaler_state stored"],
                        ["5", "Feature Selector", "Selected Features; feature scores available"],
                    ],
                    col_widths=[0.7, 2.0, 3.5],
                )
                break

    i94 = find_startswith("9.4")
    if i94 is not None:
        set_paragraph_text(
            paras[i94],
            "9.4 Deferred evaluation results",
            bold=True,
            size_pt=12,
        )
        body_94 = [
            (
                "Classifier/regressor holdout metrics from the previous eSports report are "
                "not applicable to Typhydrion’s preprocessing phase and are intentionally "
                "omitted. Model comparison tables will be produced in a later report after "
                "split and training nodes are exercised on a chosen dataset."
            ),
        ]
        _rewrite_section_body(paras, i94, body_94, stop_heading="9.5")

    i95 = find_startswith("9.5")
    if i95 is not None:
        set_paragraph_text(
            paras[i95],
            "9.5 Functional checks for this phase",
            bold=True,
            size_pt=12,
        )
        body_95 = [
            (
                "Functional verification for this phase focuses on whether loader and "
                "preprocessing nodes return successful NodeResult objects for valid "
                "inputs and clear errors for missing inputs. Full pytest coverage for the "
                "engine is recommended as future work."
            ),
            "Table 8: Preprocessing-oriented test cases",
        ]
        _rewrite_section_body(paras, i95, body_95, stop_heading="9.6")
        for j, p in enumerate(paras):
            if p.text.strip().startswith("Table 8:"):
                create_table_after(
                    p,
                    [
                        ["ID", "Input condition", "Expected behavior"],
                        [
                            "TC-01",
                            "Loader with valid CSV path",
                            "Raw Data and schema/stats returned",
                        ],
                        [
                            "TC-02",
                            "Loader with empty path",
                            "Failed result: no file path specified",
                        ],
                        [
                            "TC-03",
                            "Missing-value node with strategy Mean",
                            "Numeric nulls filled; report emitted",
                        ],
                        [
                            "TC-04",
                            "Encoder on frame with categoricals",
                            "Encoded Features and encoder state",
                        ],
                        [
                            "TC-05",
                            "Scaler on numeric features",
                            "Scaled Features and scaler state",
                        ],
                        [
                            "TC-06",
                            "SelectKBest without target",
                            "Failed result requesting target",
                        ],
                    ],
                    col_widths=[0.8, 2.4, 3.0],
                )
                break

    i96 = find_startswith("9.6")
    if i96 is not None:
        set_paragraph_text(
            paras[i96],
            "9.6 Output interpretation",
            bold=True,
            size_pt=12,
        )
        body_96 = [
            (
                "Preprocessing outputs should be read as prepared evidence for later "
                "modeling, not as predictions. Metadata reports explain what changed. If "
                "null rates remain high after imputation choices, or if encoding explodes "
                "dimensionality, the graph should be adjusted before any training phase."
            ),
        ]
        _rewrite_section_body(paras, i96, body_96, stop_heading="10. CONCLUSION")

    # === 10. CONCLUSION ===
    i10 = find_exact("10. CONCLUSION")
    if i10 is not None:
        body_10 = [
            (
                "This report phase presents Typhydrion as a desktop visual ML workflow "
                "builder and documents the path from dataset loading through "
                "preprocessing. The principal achievement is an inspectable node graph "
                "that can load tabular files, report quality signals, and apply explicit "
                "cleaning, encoding, scaling, outlier, feature-selection, and text "
                "preparation steps under a validated execution engine."
            ),
            (
                "Modeling accuracy claims are deferred until split/train/evaluate nodes "
                "are documented with a chosen dataset and recorded metrics. The system at "
                "this stage is best understood as an academic prototype for reproducible "
                "data preparation inside a node-based desktop environment."
            ),
        ]
        _rewrite_section_body(paras, i10, body_10, stop_heading="10.1")

    i101 = find_startswith("10.1")
    if i101 is not None:
        set_paragraph_text(paras[i101], "10.1 Future scope", bold=True, size_pt=12)
        body_101 = [
            (
                "Future report phases should document train/validation splits, model "
                "training and comparison, metrics and visualization nodes, export/"
                "inference, packaging (requirements/pyproject completion), and automated "
                "tests for the validator, scheduler, and executor. Additional work may "
                "include stronger persistence contracts for projects and fitted "
                "preprocessing state across sessions."
            ),
        ]
        _rewrite_section_body(
            paras, i101, body_101, stop_heading="REFERENCES AND PROJECT ARTIFACTS"
        )

    # === REFERENCES ===
    i_ref = find_startswith("REFERENCES")
    if i_ref is not None:
        refs = [
            (
                "[1] Project source tree ml_node_/ (Typhydrion): src/main.py; "
                "src/ui/main_window.py; src/ui/windows/node_editor_window.py; "
                "src/engine/graph_validator.py; src/engine/scheduler.py; "
                "src/engine/node_runner.py; src/engine/pipeline_executor.py; "
                "src/nodes/registry.py; src/nodes/io/dataset_loader_node.py; "
                "src/nodes/preprocess/*.py; README.md; requirements.txt; "
                "reports/project_overview.md."
            ),
            (
                "[2] Qt for Python (PySide6) documentation, "
                "https://doc.qt.io/qtforpython/"
            ),
            (
                "[3] pandas documentation — IO tools and data cleaning, "
                "https://pandas.pydata.org/docs/"
            ),
            (
                "[4] scikit-learn documentation — preprocessing and feature selection, "
                "https://scikit-learn.org/stable/modules/preprocessing.html"
            ),
            (
                "[5] ydata-profiling documentation, "
                "https://docs.profiling.ydata.ai/"
            ),
        ]
        _rewrite_section_body(paras, i_ref, refs, stop_heading=None)

    # Clear any leftover figure captions that still mention BGMI screenshots
    for p in paras:
        t = p.text.strip()
        if t.startswith("Figure 4:") or t.startswith("Figure 5:") or t.startswith("Figure 6:"):
            clear_paragraph(p)
        if t.startswith("Figure 7:") or t.startswith("Figure 8:") or t.startswith("Figure 9:"):
            clear_paragraph(p)
        if "BGMI" in t or "16score" in t or "Flask" in t:
            # Leave if we already rewrote; catch stragglers in empty leftovers
            if t.startswith("Figure") or "screenshot" in t.lower():
                clear_paragraph(p)

    doc.save(str(OUT))
    doc.save(str(OUT_PROJECT))
    print("Saved:", OUT)
    print("Saved:", OUT_PROJECT)


def _rewrite_section_body(paras, heading_idx, texts, stop_heading):
    """Replace Normal paragraphs after heading until stop_heading with texts.

    Extra existing paragraphs are cleared. If fewer exist than texts, content is
    packed into available paragraphs (last one may concatenate) — better to clear
    and overwrite sequentially.
    """
    if heading_idx is None:
        return

    # Collect body paragraph indices
    body_indices = []
    for j in range(heading_idx + 1, len(paras)):
        style = paras[j].style.name if paras[j].style else ""
        text = paras[j].text.strip()
        if stop_heading and (
            text == stop_heading
            or text.startswith(stop_heading)
            or (style.startswith("Heading") and text[:3].split()[0][:1].isdigit() is False and text.startswith(stop_heading[:3]))
        ):
            # Stop when we hit the next section heading marker
            if text.startswith(stop_heading) or (
                style.startswith("Heading") and stop_heading and text.startswith(stop_heading.split()[0][:2] if False else stop_heading)
            ):
                break
        if stop_heading and text.startswith(stop_heading):
            break
        if style.startswith("Heading"):
            # Next official heading ends this section body
            # But allow if this heading is still part of rewriting target — caller stops via stop_heading
            if stop_heading is None:
                break
            # If heading text starts with stop_heading prefix
            if text.startswith(tuple(stop_heading)) if False else text.startswith(stop_heading):
                break
            # Also stop on any Heading that is not empty when stop_heading provided
            # Exception: we only stop when matches stop_heading
            if stop_heading and text.startswith(stop_heading):
                break
            # For Heading 1/2 that begin next section numbers
            if stop_heading and len(stop_heading) >= 2 and text[: len(stop_heading)] == stop_heading:
                break
            # Generic: if style is Heading and text looks like next section
            if stop_heading:
                # stop only on matching prefix
                if text.startswith(stop_heading):
                    break
                # if heading and different section — detect by first token
                continue_heading = False
                # Actually for Heading paragraphs that are NOT the stop, we should stop
                # because section bodies shouldn't contain headings belonging to later parts
                # unless stop_heading is a Heading 2 inside same section.
                # Safer: stop on any Heading whose text starts with stop_heading OR
                # whose leading number is greater.
                if text.startswith(stop_heading):
                    break
                # If this is a heading and stop_heading is like "5.4", and text is "5.4 ...", break
                # If text is "6. ABOUT", and stop is "5.4", we should have stopped earlier via 5.4.
                # For body collection, skip clearing headings that aren't stop — break before them
                break
        body_indices.append(j)

    # Overwrite
    for k, idx in enumerate(body_indices):
        if k < len(texts):
            # Preserve list-looking lines without forced justify issues
            justify = not texts[k].startswith("•")
            is_table_caption = texts[k].startswith("Table ")
            set_paragraph_text(
                paras[idx],
                texts[k],
                bold=is_table_caption,
                size_pt=12,
                justify=justify and not is_table_caption,
            )
            if is_table_caption or texts[k].startswith("Objectives") or texts[k].startswith("•"):
                paras[idx].alignment = WD_ALIGN_PARAGRAPH.LEFT
        else:
            clear_paragraph(paras[idx])

    # If we have more texts than body paragraphs, append after last body para / heading
    if len(texts) > len(body_indices):
        anchor = paras[body_indices[-1]] if body_indices else paras[heading_idx]
        for extra in texts[len(body_indices) :]:
            justify = not extra.startswith("•")
            is_table_caption = extra.startswith("Table ")
            anchor = add_para_after(
                anchor,
                extra,
                bold=is_table_caption,
                size_pt=12,
                justify=justify and not is_table_caption,
            )


if __name__ == "__main__":
    rebuild_document()
