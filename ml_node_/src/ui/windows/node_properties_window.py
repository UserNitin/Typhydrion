from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
    QCheckBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QSlider,
)
from PySide6.QtCore import Qt, Signal
import inspect
import pandas as pd


# Parameters that are shown on the node card itself (don't show in properties)
NODE_CARD_PARAMS = {
    "read_csv": ["filepath_or_buffer", "sep", "encoding", "chunksize"],
    "read_table": ["filepath_or_buffer", "sep", "encoding", "chunksize"],
    "read_json": ["path_or_buf", "lines", "chunksize"],
    "read_excel": ["io", "sheet_name"],
    "read_parquet": ["path"],
    "read_feather": ["path"],
    "read_pickle": ["filepath_or_buffer"],
    "read_hdf": ["path_or_buf"],
    "read_stata": ["filepath_or_buffer"],
    "read_sas": ["filepath_or_buffer"],
    "read_spss": ["path"],
    "read_orc": ["path"],
    "read_xml": ["path_or_buffer", "xpath"],
    "read_html": ["io", "match"],
    "read_fwf": ["filepath_or_buffer", "widths"],
    "read_clipboard": ["sep"],
    "read_sql": ["sql", "con"],
    "read_sql_query": ["sql", "con"],
    "read_sql_table": ["table_name", "con"],
    "read_gbq": ["query", "project_id"],
}

# Additional parameters for each node type (beyond what's shown on the node card)
# These are EXTRA parameters not shown on the node itself
NODE_EXTRA_PARAMS = {
    "Filter Node": [
        {"label": "Case Sensitive", "type": "check", "value": True, "desc": "Case sensitive string matching"},
        {"label": "Regex", "type": "check", "value": False, "desc": "Use regex for contains/startswith"},
        {"label": "Drop NA", "type": "check", "value": False, "desc": "Drop rows with NA in filter column"},
    ],
    "Missing Value Handler": [
        {"label": "Columns", "type": "text", "value": "", "desc": "Specific columns to process (comma-separated, empty=all)"},
        {"label": "Min Non-NA", "type": "spin", "min": 1, "max": 100, "value": 1, "desc": "Minimum non-NA values required per row"},
        {"label": "Limit", "type": "spin", "min": 0, "max": 1000, "value": 0, "desc": "Max consecutive values to fill (0=unlimited)"},
        {"label": "Inplace", "type": "check", "value": True, "desc": "Modify data in place"},
    ],
    "Data Type Converter": [
        {"label": "Downcast", "type": "combo", "items": ["None", "integer", "signed", "unsigned", "float"], "desc": "Downcast numeric types to save memory"},
        {"label": "Copy", "type": "check", "value": True, "desc": "Return a copy instead of modifying"},
    ],
    "Categorical Encoder": [
        {"label": "Columns", "type": "text", "value": "", "desc": "Specific columns to encode (empty=auto-detect)"},
        {"label": "Min Frequency", "type": "double", "min": 0.0, "max": 1.0, "value": 0.0, "step": 0.01, "desc": "Minimum frequency for categories"},
        {"label": "Sparse Output", "type": "check", "value": False, "desc": "Return sparse matrix (for One-Hot)"},
    ],
    "Feature Scaler": [
        {"label": "Columns", "type": "text", "value": "", "desc": "Specific columns to scale (empty=all numeric)"},
        {"label": "Feature Range Min", "type": "double", "min": -10.0, "max": 10.0, "value": 0.0, "step": 0.1, "desc": "Min value for MinMaxScaler"},
        {"label": "Feature Range Max", "type": "double", "min": -10.0, "max": 10.0, "value": 1.0, "step": 0.1, "desc": "Max value for MinMaxScaler"},
        {"label": "Quantile Range Low", "type": "double", "min": 0.0, "max": 50.0, "value": 25.0, "step": 1.0, "desc": "Lower quantile for RobustScaler"},
        {"label": "Quantile Range High", "type": "double", "min": 50.0, "max": 100.0, "value": 75.0, "step": 1.0, "desc": "Upper quantile for RobustScaler"},
    ],
    "Feature Selector": [
        {"label": "Score Function", "type": "combo", "items": ["f_classif", "f_regression", "chi2", "mutual_info_classif", "mutual_info_regression"], "desc": "Scoring function for SelectKBest"},
        {"label": "Estimator", "type": "combo", "items": ["RandomForest", "LogisticRegression", "LinearRegression", "XGBoost"], "desc": "Estimator for RFE and Tree Importance"},
        {"label": "Step", "type": "double", "min": 0.01, "max": 1.0, "value": 0.1, "step": 0.01, "desc": "Features to remove per iteration (RFE)"},
    ],
    "Outlier Handler": [
        {"label": "Columns", "type": "text", "value": "", "desc": "Specific columns to check (empty=all numeric)"},
        {"label": "N Neighbors", "type": "spin", "min": 1, "max": 100, "value": 20, "desc": "Neighbors for LOF"},
        {"label": "Contamination", "type": "double", "min": 0.0, "max": 0.5, "value": 0.1, "step": 0.01, "desc": "Expected proportion of outliers"},
        {"label": "N Estimators", "type": "spin", "min": 10, "max": 500, "value": 100, "desc": "Trees for Isolation Forest"},
    ],
    "Text Preprocessor": [
        {"label": "Language", "type": "combo", "items": ["english", "spanish", "french", "german", "italian", "portuguese"], "desc": "Language for stopwords"},
        {"label": "Min Token Length", "type": "spin", "min": 1, "max": 10, "value": 2, "desc": "Minimum token length"},
        {"label": "Max Token Length", "type": "spin", "min": 10, "max": 100, "value": 50, "desc": "Maximum token length"},
        {"label": "Custom Stopwords", "type": "text", "value": "", "desc": "Additional stopwords (comma-separated)"},
    ],
    "Train/Test Split": [
        {"label": "Group Column", "type": "text", "value": "", "desc": "Column for group-based splitting"},
        {"label": "N Splits", "type": "spin", "min": 1, "max": 10, "value": 1, "desc": "Number of re-shuffling & splitting iterations"},
    ],
    "Train/Val/Test Split": [
        {"label": "Group Column", "type": "text", "value": "", "desc": "Column for group-based splitting"},
    ],
    "Cross Validation Split": [
        {"label": "N Repeats", "type": "spin", "min": 1, "max": 10, "value": 1, "desc": "Repeats for RepeatedKFold"},
        {"label": "Group Column", "type": "text", "value": "", "desc": "Column for GroupKFold"},
        {"label": "Gap", "type": "spin", "min": 0, "max": 100, "value": 0, "desc": "Gap for TimeSeriesSplit"},
    ],
    "Classification Model": [
        {"label": "N Estimators", "type": "spin", "min": 10, "max": 1000, "value": 100, "desc": "Number of trees/estimators"},
        {"label": "Max Depth", "type": "spin", "min": 1, "max": 100, "value": 10, "desc": "Maximum tree depth"},
        {"label": "Learning Rate", "type": "double", "min": 0.001, "max": 1.0, "value": 0.1, "step": 0.01, "desc": "Learning rate (for boosting)"},
        {"label": "Min Samples Split", "type": "spin", "min": 2, "max": 100, "value": 2, "desc": "Min samples to split node"},
        {"label": "Min Samples Leaf", "type": "spin", "min": 1, "max": 100, "value": 1, "desc": "Min samples per leaf"},
        {"label": "Random State", "type": "spin", "min": 0, "max": 9999, "value": 42, "desc": "Random seed"},
        {"label": "Class Weight", "type": "combo", "items": ["None", "balanced", "balanced_subsample"], "desc": "Class weights for imbalanced data"},
        {"label": "C", "type": "double", "min": 0.001, "max": 100.0, "value": 1.0, "step": 0.1, "desc": "Regularization (SVM, LogReg)"},
        {"label": "Kernel", "type": "combo", "items": ["rbf", "linear", "poly", "sigmoid"], "desc": "SVM kernel type"},
        {"label": "K Neighbors", "type": "spin", "min": 1, "max": 100, "value": 5, "desc": "Number of neighbors (KNN)"},
    ],
    "Regression Model": [
        {"label": "N Estimators", "type": "spin", "min": 10, "max": 1000, "value": 100, "desc": "Number of trees/estimators"},
        {"label": "Max Depth", "type": "spin", "min": 1, "max": 100, "value": 10, "desc": "Maximum tree depth"},
        {"label": "Learning Rate", "type": "double", "min": 0.001, "max": 1.0, "value": 0.1, "step": 0.01, "desc": "Learning rate (for boosting)"},
        {"label": "Min Samples Split", "type": "spin", "min": 2, "max": 100, "value": 2, "desc": "Min samples to split node"},
        {"label": "Min Samples Leaf", "type": "spin", "min": 1, "max": 100, "value": 1, "desc": "Min samples per leaf"},
        {"label": "Random State", "type": "spin", "min": 0, "max": 9999, "value": 42, "desc": "Random seed"},
        {"label": "Alpha", "type": "double", "min": 0.0, "max": 100.0, "value": 1.0, "step": 0.1, "desc": "Regularization strength (Ridge, Lasso)"},
        {"label": "L1 Ratio", "type": "double", "min": 0.0, "max": 1.0, "value": 0.5, "step": 0.1, "desc": "ElasticNet L1/L2 ratio"},
        {"label": "K Neighbors", "type": "spin", "min": 1, "max": 100, "value": 5, "desc": "Number of neighbors (KNN)"},
    ],
    "Clustering Model": [
        {"label": "N Clusters", "type": "spin", "min": 2, "max": 100, "value": 3, "desc": "Number of clusters"},
        {"label": "Max Iter", "type": "spin", "min": 100, "max": 10000, "value": 300, "desc": "Maximum iterations"},
        {"label": "N Init", "type": "spin", "min": 1, "max": 50, "value": 10, "desc": "Number of initializations"},
        {"label": "Random State", "type": "spin", "min": 0, "max": 9999, "value": 42, "desc": "Random seed"},
        {"label": "EPS", "type": "double", "min": 0.01, "max": 10.0, "value": 0.5, "step": 0.1, "desc": "DBSCAN epsilon"},
        {"label": "Min Samples", "type": "spin", "min": 1, "max": 100, "value": 5, "desc": "DBSCAN min samples"},
        {"label": "Linkage", "type": "combo", "items": ["ward", "complete", "average", "single"], "desc": "Hierarchical linkage"},
    ],
    "Anomaly Detector": [
        {"label": "N Estimators", "type": "spin", "min": 10, "max": 500, "value": 100, "desc": "Number of estimators"},
        {"label": "Contamination", "type": "double", "min": 0.0, "max": 0.5, "value": 0.1, "step": 0.01, "desc": "Expected outlier fraction"},
        {"label": "N Neighbors", "type": "spin", "min": 1, "max": 100, "value": 20, "desc": "Neighbors for LOF"},
        {"label": "Random State", "type": "spin", "min": 0, "max": 9999, "value": 42, "desc": "Random seed"},
    ],
    "Model Evaluator": [
        {"label": "CV Folds", "type": "spin", "min": 2, "max": 20, "value": 5, "desc": "Cross-validation folds"},
        {"label": "Scoring", "type": "combo", "items": ["accuracy", "f1", "precision", "recall", "roc_auc", "r2", "mse", "mae"], "desc": "Primary scoring metric"},
        {"label": "Return Train Score", "type": "check", "value": True, "desc": "Include training scores"},
    ],
    "Hyperparameter Tuner": [
        {"label": "CV Folds", "type": "spin", "min": 2, "max": 20, "value": 5, "desc": "Cross-validation folds"},
        {"label": "Scoring", "type": "combo", "items": ["accuracy", "f1", "precision", "recall", "roc_auc", "r2", "neg_mse", "neg_mae"], "desc": "Optimization metric"},
        {"label": "N Iter", "type": "spin", "min": 10, "max": 500, "value": 50, "desc": "Iterations for random/bayesian search"},
        {"label": "Random State", "type": "spin", "min": 0, "max": 9999, "value": 42, "desc": "Random seed"},
        {"label": "Verbose", "type": "spin", "min": 0, "max": 3, "value": 1, "desc": "Verbosity level"},
        {"label": "N Jobs", "type": "spin", "min": -1, "max": 32, "value": -1, "desc": "Parallel jobs (-1=all cores)"},
    ],
    "Export Model": [
        {"label": "Compress", "type": "check", "value": True, "desc": "Compress the saved file"},
        {"label": "Protocol", "type": "spin", "min": 0, "max": 5, "value": 4, "desc": "Pickle protocol version"},
    ],
    "Inference": [
        {"label": "Batch Size", "type": "spin", "min": 1, "max": 10000, "value": 1000, "desc": "Prediction batch size"},
        {"label": "Confidence Threshold", "type": "double", "min": 0.0, "max": 1.0, "value": 0.5, "step": 0.05, "desc": "Classification confidence threshold"},
    ],
    "Dataset Merger": [
        {"label": "Left On", "type": "text", "value": "", "desc": "Left join key column"},
        {"label": "Right On", "type": "text", "value": "", "desc": "Right join key column"},
        {"label": "Suffixes", "type": "text", "value": "_x,_y", "desc": "Suffixes for overlapping columns"},
        {"label": "Validate", "type": "combo", "items": ["None", "one_to_one", "one_to_many", "many_to_one", "many_to_many"], "desc": "Check merge key uniqueness"},
    ],
    "Column Selector": [
        {"label": "Regex", "type": "check", "value": False, "desc": "Use regex for column selection"},
        {"label": "Include Types", "type": "text", "value": "", "desc": "Include columns of these types (e.g., number,object)"},
        {"label": "Exclude Types", "type": "text", "value": "", "desc": "Exclude columns of these types"},
    ],
    # ═══════════════════════════════════════════════════════════════
    # FLOW CONTROL & UTILITY NODES
    # ═══════════════════════════════════════════════════════════════
    "AI Advisor": [
        {"label": "Model Provider", "type": "combo", "items": ["OpenAI", "Anthropic", "Local LLM", "Hugging Face"], "desc": "AI model provider"},
        {"label": "Model Name", "type": "text", "value": "gpt-4", "desc": "Specific model name"},
        {"label": "Temperature", "type": "double", "min": 0.0, "max": 2.0, "value": 0.7, "step": 0.1, "desc": "Creativity/randomness"},
        {"label": "Max Tokens", "type": "spin", "min": 100, "max": 8000, "value": 1000, "desc": "Maximum response length"},
        {"label": "API Key Env", "type": "text", "value": "OPENAI_API_KEY", "desc": "Environment variable for API key"},
        {"label": "Timeout", "type": "spin", "min": 10, "max": 300, "value": 60, "desc": "Request timeout in seconds"},
    ],
    "Note/Comment": [
        {"label": "Font Size", "type": "spin", "min": 8, "max": 24, "value": 12, "desc": "Text font size"},
        {"label": "Bold", "type": "check", "value": False, "desc": "Bold text"},
        {"label": "Italic", "type": "check", "value": False, "desc": "Italic text"},
        {"label": "Background Opacity", "type": "double", "min": 0.0, "max": 1.0, "value": 0.8, "step": 0.1, "desc": "Background transparency"},
        {"label": "Border Width", "type": "spin", "min": 0, "max": 5, "value": 1, "desc": "Border width in pixels"},
    ],
    "Loop Controller": [
        {"label": "Loop Type", "type": "combo", "items": ["Count", "While Condition", "For Each", "Until Condition"], "desc": "Type of loop iteration"},
        {"label": "Start Index", "type": "spin", "min": 0, "max": 1000, "value": 0, "desc": "Starting index for iterations"},
        {"label": "Step Size", "type": "spin", "min": 1, "max": 100, "value": 1, "desc": "Increment per iteration"},
        {"label": "Max Runtime", "type": "spin", "min": 0, "max": 3600, "value": 0, "desc": "Max runtime in seconds (0=unlimited)"},
        {"label": "Parallel", "type": "check", "value": False, "desc": "Run iterations in parallel"},
        {"label": "N Workers", "type": "spin", "min": 1, "max": 32, "value": 4, "desc": "Number of parallel workers"},
    ],
    "Timer Node": [
        {"label": "Precision", "type": "combo", "items": ["seconds", "milliseconds", "microseconds"], "desc": "Time measurement precision"},
        {"label": "Include CPU Time", "type": "check", "value": True, "desc": "Measure CPU time"},
        {"label": "Include Memory", "type": "check", "value": False, "desc": "Track memory usage"},
        {"label": "Log Format", "type": "text", "value": "{name}: {elapsed:.3f}s", "desc": "Custom log format"},
    ],
    "Checkpoint Node": [
        {"label": "Auto Save Interval", "type": "spin", "min": 0, "max": 3600, "value": 0, "desc": "Auto-save interval (0=disabled)"},
        {"label": "Max Checkpoints", "type": "spin", "min": 1, "max": 100, "value": 5, "desc": "Maximum checkpoints to keep"},
        {"label": "Compression", "type": "combo", "items": ["None", "gzip", "bz2", "lzma"], "desc": "Compression algorithm"},
        {"label": "Include Metadata", "type": "check", "value": True, "desc": "Save additional metadata"},
        {"label": "Atomic Write", "type": "check", "value": True, "desc": "Use atomic write operations"},
    ],
    "Data Logger": [
        {"label": "Log Format", "type": "combo", "items": ["text", "json", "csv", "html"], "desc": "Output log format"},
        {"label": "Max File Size", "type": "spin", "min": 1, "max": 1000, "value": 10, "desc": "Max file size in MB"},
        {"label": "Backup Count", "type": "spin", "min": 0, "max": 20, "value": 5, "desc": "Number of backup files"},
        {"label": "Include Timestamps", "type": "check", "value": True, "desc": "Add timestamps to log entries"},
        {"label": "Include Stack Trace", "type": "check", "value": False, "desc": "Include stack traces for errors"},
        {"label": "Flush Interval", "type": "spin", "min": 0, "max": 60, "value": 1, "desc": "Flush to disk interval (seconds)"},
    ],
    "Model Explainer": [
        {"label": "Background Samples", "type": "spin", "min": 10, "max": 1000, "value": 100, "desc": "Background samples for SHAP"},
        {"label": "Max Display", "type": "spin", "min": 5, "max": 50, "value": 20, "desc": "Max features to display"},
        {"label": "Plot Type", "type": "combo", "items": ["summary", "bar", "beeswarm", "waterfall", "force"], "desc": "Default plot type"},
        {"label": "Interaction", "type": "check", "value": False, "desc": "Calculate interaction values"},
        {"label": "Check Additivity", "type": "check", "value": True, "desc": "Verify SHAP values sum to prediction"},
        {"label": "Approximate", "type": "check", "value": False, "desc": "Use approximate SHAP (faster)"},
    ],
    "Training Controller": [
        {"label": "Gradient Clipping", "type": "double", "min": 0.0, "max": 10.0, "value": 1.0, "step": 0.1, "desc": "Max gradient norm"},
        {"label": "Warmup Steps", "type": "spin", "min": 0, "max": 10000, "value": 0, "desc": "Learning rate warmup steps"},
        {"label": "Weight Decay", "type": "double", "min": 0.0, "max": 1.0, "value": 0.01, "step": 0.001, "desc": "L2 regularization"},
        {"label": "Accumulation Steps", "type": "spin", "min": 1, "max": 64, "value": 1, "desc": "Gradient accumulation steps"},
        {"label": "Mixed Precision", "type": "check", "value": False, "desc": "Use mixed precision training"},
        {"label": "Deterministic", "type": "check", "value": False, "desc": "Ensure reproducibility"},
        {"label": "Log Steps", "type": "spin", "min": 1, "max": 1000, "value": 10, "desc": "Logging frequency"},
    ],
    "Report Generator": [
        {"label": "Template", "type": "combo", "items": ["Default", "Minimal", "Detailed", "Executive", "Custom"], "desc": "Report template"},
        {"label": "Page Size", "type": "combo", "items": ["A4", "Letter", "Legal", "A3"], "desc": "PDF page size"},
        {"label": "Font Family", "type": "text", "value": "Arial", "desc": "Font family for text"},
        {"label": "Font Size", "type": "spin", "min": 8, "max": 16, "value": 11, "desc": "Base font size"},
        {"label": "Include TOC", "type": "check", "value": True, "desc": "Include table of contents"},
        {"label": "Include Summary", "type": "check", "value": True, "desc": "Include executive summary"},
        {"label": "Max Figures", "type": "spin", "min": 1, "max": 50, "value": 20, "desc": "Maximum figures to include"},
    ],
    "Resource Manager": [
        {"label": "CPU Threshold", "type": "double", "min": 0.0, "max": 100.0, "value": 80.0, "step": 5.0, "desc": "CPU usage threshold (%)"},
        {"label": "Memory Threshold", "type": "double", "min": 0.0, "max": 100.0, "value": 80.0, "step": 5.0, "desc": "Memory usage threshold (%)"},
        {"label": "Check Interval", "type": "spin", "min": 1, "max": 60, "value": 5, "desc": "Resource check interval (seconds)"},
        {"label": "Auto Scale", "type": "check", "value": False, "desc": "Automatically adjust resources"},
        {"label": "Alert Email", "type": "text", "value": "", "desc": "Email for resource alerts"},
        {"label": "Kill On Exceed", "type": "check", "value": False, "desc": "Kill process if limits exceeded"},
    ],
    "Visualization Node": [
        {"label": "Figure Size Width", "type": "double", "min": 4.0, "max": 20.0, "value": 10.0, "step": 0.5, "desc": "Figure width in inches"},
        {"label": "Figure Size Height", "type": "double", "min": 3.0, "max": 15.0, "value": 6.0, "step": 0.5, "desc": "Figure height in inches"},
        {"label": "DPI", "type": "spin", "min": 72, "max": 300, "value": 100, "desc": "Resolution (dots per inch)"},
        {"label": "Style", "type": "combo", "items": ["seaborn", "ggplot", "bmh", "dark_background", "fivethirtyeight", "grayscale"], "desc": "Matplotlib style"},
        {"label": "Color Palette", "type": "combo", "items": ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdYlBu"], "desc": "Color palette"},
        {"label": "Show Grid", "type": "check", "value": True, "desc": "Show grid lines"},
        {"label": "Legend Position", "type": "combo", "items": ["best", "upper right", "upper left", "lower right", "lower left", "center"], "desc": "Legend position"},
    ],
    "Model Selector": [
        {"label": "Selection Metric", "type": "combo", "items": ["accuracy", "f1", "precision", "recall", "roc_auc", "r2", "mse", "mae"], "desc": "Metric for model selection"},
        {"label": "CV Folds", "type": "spin", "min": 2, "max": 20, "value": 5, "desc": "Cross-validation folds"},
        {"label": "Refit", "type": "check", "value": True, "desc": "Refit best model on full data"},
        {"label": "Return All", "type": "check", "value": False, "desc": "Return all models, not just best"},
        {"label": "Verbose", "type": "spin", "min": 0, "max": 3, "value": 1, "desc": "Verbosity level"},
    ],
    "Batch Controller": [
        {"label": "Memory Limit", "type": "spin", "min": 100, "max": 64000, "value": 1000, "desc": "Memory limit per batch (MB)"},
        {"label": "Timeout Per Batch", "type": "spin", "min": 0, "max": 3600, "value": 0, "desc": "Timeout per batch (0=unlimited)"},
        {"label": "Retry Failed", "type": "spin", "min": 0, "max": 5, "value": 0, "desc": "Retry failed batches"},
        {"label": "Save Progress", "type": "check", "value": True, "desc": "Save progress between batches"},
        {"label": "Progress File", "type": "text", "value": "batch_progress.json", "desc": "Progress checkpoint file"},
    ],
    "Debug Inspector": [
        {"label": "Max Rows Display", "type": "spin", "min": 5, "max": 1000, "value": 100, "desc": "Max rows to display"},
        {"label": "Max Columns Display", "type": "spin", "min": 5, "max": 100, "value": 50, "desc": "Max columns to display"},
        {"label": "Precision", "type": "spin", "min": 1, "max": 10, "value": 4, "desc": "Decimal precision"},
        {"label": "Show Memory Usage", "type": "check", "value": True, "desc": "Display memory usage"},
        {"label": "Show Dtypes", "type": "check", "value": True, "desc": "Display column data types"},
        {"label": "Profile Data", "type": "check", "value": False, "desc": "Generate detailed data profile"},
    ],
    "Ensemble Builder": [
        {"label": "Ensemble Type", "type": "combo", "items": ["Voting", "Stacking", "Bagging", "Boosting"], "desc": "Type of ensemble"},
        {"label": "Final Estimator", "type": "combo", "items": ["LogisticRegression", "LinearRegression", "RandomForest", "XGBoost", "None"], "desc": "Meta-learner for stacking"},
        {"label": "CV Folds", "type": "spin", "min": 2, "max": 10, "value": 5, "desc": "CV folds for stacking"},
        {"label": "Passthrough", "type": "check", "value": False, "desc": "Include original features in stacking"},
        {"label": "N Jobs", "type": "spin", "min": -1, "max": 32, "value": -1, "desc": "Parallel jobs (-1=all cores)"},
    ],
    "Neural Network": [
        {"label": "Loss Function", "type": "combo", "items": ["mse", "mae", "binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy", "huber"], "desc": "Loss function"},
        {"label": "Metrics", "type": "text", "value": "accuracy", "desc": "Metrics to track (comma-separated)"},
        {"label": "Early Stop Patience", "type": "spin", "min": 1, "max": 50, "value": 10, "desc": "Early stopping patience"},
        {"label": "Early Stop Min Delta", "type": "double", "min": 0.0, "max": 0.1, "value": 0.001, "step": 0.001, "desc": "Minimum improvement"},
        {"label": "Reduce LR Factor", "type": "double", "min": 0.1, "max": 0.9, "value": 0.5, "step": 0.1, "desc": "LR reduction factor"},
        {"label": "Reduce LR Patience", "type": "spin", "min": 1, "max": 20, "value": 5, "desc": "LR reduction patience"},
        {"label": "Validation Split", "type": "double", "min": 0.0, "max": 0.5, "value": 0.2, "step": 0.05, "desc": "Validation data fraction"},
        {"label": "Shuffle", "type": "check", "value": True, "desc": "Shuffle training data"},
        {"label": "Verbose", "type": "spin", "min": 0, "max": 2, "value": 1, "desc": "Training verbosity"},
    ],
    "Conditional Router": [
        {"label": "Else Branch", "type": "check", "value": True, "desc": "Enable else/false branch"},
        {"label": "Evaluate All", "type": "check", "value": False, "desc": "Evaluate all conditions (not short-circuit)"},
        {"label": "Case Sensitive", "type": "check", "value": True, "desc": "Case sensitive string comparison"},
        {"label": "Null Handling", "type": "combo", "items": ["False", "True", "Error"], "desc": "How to handle null values"},
    ],
    "Final Output": [
        {"label": "Export Format", "type": "combo", "items": ["None", "csv", "parquet", "json", "excel", "pickle"], "desc": "Auto-export format"},
        {"label": "Export Path", "type": "text", "value": "", "desc": "Auto-export file path"},
        {"label": "Include Index", "type": "check", "value": False, "desc": "Include index in export"},
        {"label": "Compression", "type": "combo", "items": ["None", "gzip", "bz2", "zip", "xz"], "desc": "Export compression"},
    ],
    "Metrics Evaluator": [
        {"label": "Bootstrap", "type": "check", "value": False, "desc": "Calculate bootstrap confidence intervals"},
        {"label": "Bootstrap Iterations", "type": "spin", "min": 100, "max": 10000, "value": 1000, "desc": "Number of bootstrap samples"},
        {"label": "Confidence Level", "type": "double", "min": 0.8, "max": 0.99, "value": 0.95, "step": 0.01, "desc": "Confidence interval level"},
        {"label": "Average", "type": "combo", "items": ["binary", "micro", "macro", "weighted", "samples"], "desc": "Averaging method for multiclass"},
        {"label": "Zero Division", "type": "combo", "items": ["warn", "0", "1"], "desc": "Value when division by zero"},
    ],
    "Model Export": [
        {"label": "Export Format", "type": "combo", "items": ["pickle", "joblib", "onnx", "pmml", "h5"], "desc": "Model export format"},
        {"label": "Include Preprocessors", "type": "check", "value": True, "desc": "Include preprocessing pipeline"},
        {"label": "Include Metadata", "type": "check", "value": True, "desc": "Include model metadata"},
        {"label": "Optimize", "type": "check", "value": False, "desc": "Optimize model for inference"},
        {"label": "Quantize", "type": "check", "value": False, "desc": "Quantize model weights"},
    ],
}


class NodePropertiesWindow(QWidget):
    properties_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._current_node = None
        self._current_node_type = "Dataset Loader"
        self._current_reader = "read_csv"
        self._param_widgets: dict[str, tuple[QCheckBox, QWidget, QFrame]] = {}
        self._all_params: list[dict] = []
        self._saved_params: dict = {}  # Saved parameters for current node
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        self._title = QLabel("Node Properties")
        self._title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(self._title)
        
        self._node_name = QLabel("No node selected")
        self._node_name.setStyleSheet("color: rgba(180, 200, 255, 200); font-size: 13px;")
        layout.addWidget(self._node_name)
        
        # Search bar
        self._search_bar = QLineEdit()
        self._search_bar.setPlaceholderText("🔍 Search parameters...")
        self._search_bar.setClearButtonEnabled(True)
        self._search_bar.setStyleSheet("""
            QLineEdit {
                background-color: rgba(30, 40, 55, 180);
                border: 1px solid rgba(80, 100, 130, 150);
                border-radius: 6px;
                padding: 8px 12px;
                color: rgba(220, 230, 240, 220);
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: rgba(100, 140, 200, 200);
                background-color: rgba(35, 48, 65, 200);
            }
            QLineEdit::placeholder {
                color: rgba(150, 170, 200, 120);
            }
        """)
        self._search_bar.textChanged.connect(self._filter_params)
        layout.addWidget(self._search_bar)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                background-color: rgba(30, 40, 55, 150);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(100, 120, 150, 180);
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)
        
        self._params_container = QWidget()
        self._params_layout = QVBoxLayout(self._params_container)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_layout.setSpacing(6)
        self._params_layout.addStretch(1)
        
        scroll.setWidget(self._params_container)
        layout.addWidget(scroll, 1)
        
        self._info_label = QLabel("Select a node to see its parameters")
        self._info_label.setStyleSheet("color: rgba(255, 255, 255, 120); font-size: 11px;")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)
        
        # Apply button
        self._apply_btn = QPushButton("✓ Apply Parameters")
        self._apply_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 130, 90, 200);
                border: 1px solid rgba(80, 160, 110, 180);
                border-radius: 6px;
                padding: 10px 16px;
                color: white;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: rgba(70, 150, 100, 220);
            }
            QPushButton:pressed {
                background-color: rgba(50, 110, 75, 220);
            }
            QPushButton:disabled {
                background-color: rgba(60, 70, 80, 150);
                color: rgba(150, 160, 170, 150);
            }
        """)
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        self._apply_btn.setEnabled(False)
        layout.addWidget(self._apply_btn)

    def _on_apply_clicked(self) -> None:
        """Handle Apply button click - emit properties_changed signal."""
        params = self.get_enabled_params()
        self.properties_changed.emit(params)
        self._info_label.setText(f"✓ Applied {len(params)} parameter(s)")

    def set_node(self, node_title: str, node_type: str = "Dataset Loader", extra_params: dict = None) -> None:
        """Set the currently selected node and show its parameters."""
        self._node_name.setText(f"📦 {node_title}")
        self._current_node = node_title
        self._current_node_type = node_type
        self._saved_params = extra_params or {}
        self._search_bar.clear()
        
        # Clear existing parameters
        self._clear_params()
        
        if node_title == "Dataset Loader":
            self._build_dataset_loader_params()
            self._restore_saved_params()
            self._apply_btn.setEnabled(True)
        elif node_title in NODE_EXTRA_PARAMS:
            self._build_node_params(node_title)
            self._restore_saved_params()
            self._apply_btn.setEnabled(True)
        else:
            # Node without extra parameters
            self._info_label.setText(f"No additional parameters for '{node_title}'.\nAll options are available on the node card.")
            self._apply_btn.setEnabled(False)

    def set_reader(self, reader_name: str) -> None:
        """Update parameters when reader function changes."""
        self._current_reader = reader_name
        self._search_bar.clear()
        if self._current_node:
            self._clear_params()
            self._build_dataset_loader_params()
            self._restore_saved_params()
    
    def _restore_saved_params(self) -> None:
        """Restore saved parameter values from the node."""
        if not self._saved_params:
            return
        
        restored_count = 0
        for name, value in self._saved_params.items():
            if name in self._param_widgets:
                checkbox, widget, frame = self._param_widgets[name]
                # Enable the checkbox
                checkbox.setChecked(True)
                # Set the value
                if isinstance(widget, QComboBox):
                    if isinstance(value, bool):
                        widget.setCurrentText("True" if value else "False")
                    else:
                        widget.setCurrentText(str(value))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                restored_count += 1
        
        if restored_count > 0:
            self._info_label.setText(f"Restored {restored_count} saved parameter(s)")

    def _clear_params(self) -> None:
        """Clear all parameter widgets."""
        self._param_widgets.clear()
        self._all_params.clear()
        while self._params_layout.count() > 1:
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def _connect_checkbox_to_widget(self, checkbox: QCheckBox, widget: QWidget) -> None:
        """Connect checkbox to enable/disable widget - avoids closure issues."""
        def handler(state):
            widget.setEnabled(state != 0)
        checkbox.stateChanged.connect(handler)

    def _filter_params(self, search_text: str) -> None:
        """Filter parameters based on search text."""
        search_lower = search_text.lower().strip()
        visible_count = 0
        
        for name, (checkbox, input_widget, frame) in self._param_widgets.items():
            # Find the param description
            param = next((p for p in self._all_params if p["name"] == name), None)
            desc = param.get("desc", "") if param else ""
            
            # Check if search matches name or description
            matches = (
                not search_lower or 
                search_lower in name.lower() or 
                search_lower in desc.lower()
            )
            frame.setVisible(matches)
            if matches:
                visible_count += 1
        
        if search_lower:
            self._info_label.setText(f"Found {visible_count} parameters matching '{search_text}'")
        else:
            self._info_label.setText(
                f"Showing {len(self._param_widgets)} additional parameters\n"
                "✓ Enable parameters you want to use"
            )

    def _build_node_params(self, node_title: str) -> None:
        """Build parameter widgets for a specific node type."""
        extra_params = NODE_EXTRA_PARAMS.get(node_title, [])
        
        # Convert to the format expected by _create_param_row
        self._all_params = []
        for opt in extra_params:
            param = {
                "name": opt.get("label", ""),
                "type": self._convert_option_type(opt.get("type", "text")),
                "default": opt.get("value"),
                "desc": opt.get("desc", ""),
                "options": opt.get("items", []),
                "min": opt.get("min"),
                "max": opt.get("max"),
                "step": opt.get("step"),
            }
            self._all_params.append(param)
        
        self._info_label.setText(
            f"Showing {len(self._all_params)} additional parameters for '{node_title}'\n"
            "✓ Enable parameters you want to use"
        )
        
        for param in self._all_params:
            row = self._create_param_row(param)
            self._params_layout.insertWidget(self._params_layout.count() - 1, row)
    
    def _convert_option_type(self, opt_type: str) -> str:
        """Convert option type from catalog format to param format."""
        type_map = {
            "text": "str",
            "spin": "int",
            "double": "float",
            "check": "bool",
            "combo": "list",
            "slider": "int",
        }
        return type_map.get(opt_type, "str")

    def _build_dataset_loader_params(self) -> None:
        """Build parameter widgets for the current pandas reader."""
        all_params = _get_reader_params(self._current_reader)
        
        # Filter out parameters that are shown on the node card
        node_params = NODE_CARD_PARAMS.get(self._current_reader, [])
        # Also filter common first param variations
        node_params_lower = [p.lower() for p in node_params]
        
        self._all_params = [
            p for p in all_params 
            if p["name"].lower() not in node_params_lower
            and p["name"] not in ("self", "cls")
        ]
        
        self._info_label.setText(
            f"Showing {len(self._all_params)} additional parameters for pd.{self._current_reader}()\n"
            "✓ Enable parameters you want to use"
        )
        
        for param in self._all_params:
            row = self._create_param_row(param)
            self._params_layout.insertWidget(self._params_layout.count() - 1, row)

    def _create_param_row(self, param: dict) -> QFrame:
        """Create a row with checkbox + label + input widget for a parameter."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 40, 55, 120);
                border-radius: 6px;
                padding: 4px;
            }
            QFrame:hover {
                background-color: rgba(40, 55, 75, 150);
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        
        # Top row: checkbox + parameter name
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        
        checkbox = QCheckBox()
        checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid rgba(100, 120, 150, 180);
                background-color: rgba(30, 40, 55, 200);
            }
            QCheckBox::indicator:checked {
                background-color: rgba(70, 140, 200, 220);
                border-color: rgba(100, 160, 220, 200);
            }
        """)
        
        name_label = QLabel(param["name"])
        name_label.setStyleSheet("color: rgba(200, 220, 255, 230); font-weight: 500;")
        
        type_label = QLabel(f"({param['type']})")
        type_label.setStyleSheet("color: rgba(150, 170, 200, 150); font-size: 10px;")
        
        top_row.addWidget(checkbox)
        top_row.addWidget(name_label)
        top_row.addWidget(type_label)
        top_row.addStretch(1)
        
        layout.addLayout(top_row)
        
        # Input widget based on type
        input_widget = self._create_input_widget(param)
        input_widget.setEnabled(False)  # Disabled until checkbox is checked
        input_widget.setFocusPolicy(Qt.StrongFocus)  # Ensure widget can receive focus
        
        # Connect checkbox to enable/disable input
        # Using a helper to properly capture widget reference
        self._connect_checkbox_to_widget(checkbox, input_widget)
        
        layout.addWidget(input_widget)
        
        # Description
        if param.get("desc"):
            desc_label = QLabel(param["desc"])
            desc_label.setStyleSheet("color: rgba(150, 170, 200, 130); font-size: 10px;")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)
        
        self._param_widgets[param["name"]] = (checkbox, input_widget, frame)
        
        return frame

    def _create_input_widget(self, param: dict) -> QWidget:
        """Create appropriate input widget based on parameter type."""
        ptype = param["type"]
        default = param.get("default", "")
        
        widget_style = """
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QSlider {
                background-color: rgba(25, 35, 50, 200);
                border: 1px solid rgba(80, 100, 130, 150);
                border-radius: 4px;
                padding: 4px 8px;
                color: rgba(220, 230, 240, 220);
            }
            QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QSlider:disabled {
                background-color: rgba(20, 28, 38, 150);
                color: rgba(150, 160, 170, 100);
            }
        """
        
        if ptype == "bool":
            widget = QComboBox()
            widget.addItems(["True", "False"])
            if default is True:
                widget.setCurrentIndex(0)
            elif default is False:
                widget.setCurrentIndex(1)
            widget.setStyleSheet(widget_style)
            return widget
        
        if ptype == "int":
            widget = QSpinBox()
            min_val = param.get("min", -999999)
            max_val = param.get("max", 999999)
            widget.setRange(min_val if min_val is not None else -999999, 
                          max_val if max_val is not None else 999999)
            if isinstance(default, int):
                widget.setValue(default)
            widget.setStyleSheet(widget_style)
            return widget
        
        if ptype == "float":
            widget = QDoubleSpinBox()
            min_val = param.get("min", -999999.0)
            max_val = param.get("max", 999999.0)
            step = param.get("step", 0.1)
            widget.setRange(min_val if min_val is not None else -999999.0, 
                          max_val if max_val is not None else 999999.0)
            widget.setSingleStep(step if step is not None else 0.1)
            widget.setDecimals(4)
            if isinstance(default, (int, float)):
                widget.setValue(float(default))
            widget.setStyleSheet(widget_style)
            return widget
        
        if ptype == "list" and param.get("options"):
            widget = QComboBox()
            widget.addItems([str(o) for o in param["options"]])
            # Set default if specified
            if default and str(default) in [str(o) for o in param["options"]]:
                widget.setCurrentText(str(default))
            widget.setStyleSheet(widget_style)
            return widget
        
        # Default to text input
        widget = QLineEdit()
        if default not in (None, inspect.Parameter.empty, ""):
            widget.setText(str(default))
        widget.setPlaceholderText(f"Enter {param['name']}...")
        widget.setStyleSheet(widget_style)
        return widget

    def get_enabled_params(self) -> dict:
        """Get dictionary of enabled parameters and their values."""
        result = {}
        for name, (checkbox, widget, frame) in self._param_widgets.items():
            if checkbox.isChecked():
                if isinstance(widget, QComboBox):
                    value = widget.currentText()
                    if value in ("True", "False"):
                        value = value == "True"
                    result[name] = value
                elif isinstance(widget, QSpinBox):
                    result[name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    result[name] = widget.value()
                elif isinstance(widget, QLineEdit):
                    text = widget.text().strip()
                    if text:
                        result[name] = text
        return result


def _get_reader_params(reader_name: str) -> list[dict]:
    """Get all parameters for a pandas reader function."""
    func = getattr(pd, reader_name, None)
    if func is None:
        return []
    
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return []
    
    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        
        # Determine type from annotation or default value
        ptype = "str"
        default = param.default if param.default != inspect.Parameter.empty else None
        
        if param.annotation != inspect.Parameter.empty:
            ann = str(param.annotation)
            if "bool" in ann.lower():
                ptype = "bool"
            elif "int" in ann.lower():
                ptype = "int"
            elif "float" in ann.lower():
                ptype = "float"
            elif "list" in ann.lower() or "sequence" in ann.lower():
                ptype = "list"
        elif default is not None:
            if isinstance(default, bool):
                ptype = "bool"
            elif isinstance(default, int):
                ptype = "int"
            elif isinstance(default, float):
                ptype = "float"
        
        # Get description from common parameter names
        desc = _get_param_description(name)
        
        params.append({
            "name": name,
            "type": ptype,
            "default": default,
            "desc": desc,
        })
    
    return params


def _get_param_description(name: str) -> str:
    """Get description for common pandas reader parameters."""
    descriptions = {
        "filepath_or_buffer": "File path, URL, or file-like object",
        "sep": "Delimiter to use (default: ',')",
        "delimiter": "Alternative argument name for sep",
        "header": "Row number(s) to use as column names",
        "names": "List of column names to use",
        "index_col": "Column(s) to use as row labels",
        "usecols": "Columns to read from the file",
        "dtype": "Data type for columns",
        "engine": "Parser engine to use ('c', 'python', 'pyarrow')",
        "converters": "Dict of functions for converting values",
        "true_values": "Values to consider as True",
        "false_values": "Values to consider as False",
        "skipinitialspace": "Skip spaces after delimiter",
        "skiprows": "Line numbers to skip at start",
        "skipfooter": "Number of lines to skip at end",
        "nrows": "Number of rows to read",
        "na_values": "Additional strings to recognize as NA/NaN",
        "keep_default_na": "Include default NaN values",
        "na_filter": "Detect missing value markers",
        "verbose": "Indicate number of NA values",
        "skip_blank_lines": "Skip blank lines",
        "parse_dates": "Parse date columns",
        "infer_datetime_format": "Infer datetime format",
        "keep_date_col": "Keep original date columns",
        "date_parser": "Function for parsing dates",
        "dayfirst": "DD/MM format dates",
        "cache_dates": "Cache unique converted dates",
        "iterator": "Return TextFileReader for iteration",
        "chunksize": "Number of rows per chunk",
        "compression": "Compression type ('infer', 'gzip', etc.)",
        "thousands": "Thousands separator",
        "decimal": "Decimal point character",
        "lineterminator": "Line terminator character",
        "quotechar": "Quote character",
        "quoting": "Quoting behavior",
        "doublequote": "Handle quote inside field",
        "escapechar": "Escape character",
        "comment": "Comment character",
        "encoding": "File encoding (e.g., 'utf-8')",
        "encoding_errors": "How to handle encoding errors",
        "dialect": "CSV dialect",
        "on_bad_lines": "How to handle bad lines",
        "delim_whitespace": "Use whitespace as delimiter",
        "low_memory": "Process file in chunks",
        "memory_map": "Map file to memory",
        "float_precision": "Float conversion precision",
        "storage_options": "Storage backend options",
        "dtype_backend": "Backend for result dtypes",
        "sheet_name": "Excel sheet name or index",
        "io": "File path or ExcelFile object",
        "orient": "Expected JSON string format",
        "typ": "Type of object to recover",
        "lines": "Read JSON per line",
        "path": "File path or URL",
        "columns": "Columns to read",
        "filters": "Row filtering predicates",
    }
    return descriptions.get(name, "")
