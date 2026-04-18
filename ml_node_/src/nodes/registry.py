"""
Node Registry - Maps node names to their runtime implementations.
"""
from __future__ import annotations

# Data Ingestion Nodes
from nodes.io.dataset_loader_node import (
    DatasetLoaderNode, DataPreviewNode, DatasetMergerNode,
    ColumnSelectorNode, FilterNode, FinalOutputNode
)
from nodes.io.export_node import ExportModelNode, InferenceNode

# Preprocessing Nodes
from nodes.preprocess.missing_value_node import MissingValueNode, DataTypeConverterNode
from nodes.preprocess.scaling_node import ScalingNode, OutlierHandlerNode
from nodes.preprocess.encoding_node import EncodingNode
from nodes.preprocess.feature_select_node import FeatureSelectNode, TextPreprocessorNode

# Split Nodes
from nodes.split.train_test_split_node import (
    TrainTestSplitNode, TrainValTestSplitNode, CrossValidationSplitNode,
    BatchControllerNode, ConditionalRouterNode
)
from nodes.split.time_series_split_node import TimeSeriesSplitNode

# Model Nodes
from nodes.models.model_selector_node import (
    ModelSelectorNode, TrainingNode, HyperparameterTunerNode, EnsembleBuilderNode
)
from nodes.models.classification_node import ClassificationNode
from nodes.models.regression_node import RegressionNode
from nodes.models.clustering_node import ClusteringNode
from nodes.models.anomaly_node import AnomalyNode
from nodes.models.nn_node import NeuralNetNode

# Evaluation Nodes
from nodes.eval.metrics_node import MetricsNode
from nodes.eval.report_node import ReportNode, ModelExplainerNode
from nodes.eval.visualization_node import VisualizationNode

# System & Utility Nodes
from nodes.system.resource_control_node import (
    ResourceControlNode, DebugInspectorNode, DataLoggerNode,
    CheckpointNode, TimerNode, LoopControllerNode,
    NoteCommentNode, AIAdvisorNode
)


# ═══════════════════════════════════════════════════════════════
# NODE REGISTRY - Maps UI node names to runtime classes
# ═══════════════════════════════════════════════════════════════

NODE_REGISTRY = {
    # Data Ingestion
    "Dataset Loader": DatasetLoaderNode,
    "Data Preview": DataPreviewNode,
    "Dataset Merger": DatasetMergerNode,
    "Column Selector": ColumnSelectorNode,
    "Filter Node": FilterNode,
    
    # Export & Inference
    "Model Export": ExportModelNode,
    "Export Model": ExportModelNode,
    "Inference Node": InferenceNode,
    "Final Output": FinalOutputNode,
    
    # Preprocessing
    "Missing Value Handler": MissingValueNode,
    "Data Type Converter": DataTypeConverterNode,
    "Categorical Encoder": EncodingNode,
    "Encoding": EncodingNode,
    "Feature Scaler": ScalingNode,
    "Scaling": ScalingNode,
    "Feature Selector": FeatureSelectNode,
    "Outlier Handler": OutlierHandlerNode,
    "Text Preprocessor": TextPreprocessorNode,
    
    # Split & Flow Control
    "Train/Test Split": TrainTestSplitNode,
    "Train/Val/Test Split": TrainValTestSplitNode,
    "Cross Validation Split": CrossValidationSplitNode,
    "Time Series Split": TimeSeriesSplitNode,
    "Batch Controller": BatchControllerNode,
    "Conditional Router": ConditionalRouterNode,
    
    # Models
    "Model Selector": ModelSelectorNode,
    "Classification Model": ClassificationNode,
    "Regression Model": RegressionNode,
    "Clustering Model": ClusteringNode,
    "Anomaly Detector": AnomalyNode,
    "Anomaly Model": AnomalyNode,
    "Neural Network": NeuralNetNode,
    "Neural Net": NeuralNetNode,
    
    # Training & Optimization
    "Training Controller": TrainingNode,
    "Training Node": TrainingNode,
    "Hyperparameter Tuner": HyperparameterTunerNode,
    "Ensemble Builder": EnsembleBuilderNode,
    
    # Evaluation
    "Metrics Evaluator": MetricsNode,
    "Metrics": MetricsNode,
    "Visualization Node": VisualizationNode,
    "Visualization": VisualizationNode,
    "Report Generator": ReportNode,
    "Report": ReportNode,
    "Model Explainer": ModelExplainerNode,
    
    # System & Utility
    "Resource Manager": ResourceControlNode,
    "Resource Control": ResourceControlNode,
    "Debug Inspector": DebugInspectorNode,
    "Data Logger": DataLoggerNode,
    "Checkpoint Node": CheckpointNode,
    "Timer Node": TimerNode,
    "Loop Controller": LoopControllerNode,
    "Note/Comment": NoteCommentNode,
    "AI Advisor": AIAdvisorNode,
}


def get_node_runtime(node_name: str):
    """Get the runtime class for a node by name."""
    return NODE_REGISTRY.get(node_name)


def list_available_nodes() -> list[str]:
    """List all available node names."""
    return list(NODE_REGISTRY.keys())


def get_node_categories() -> dict[str, list[str]]:
    """Get nodes organized by category."""
    categories = {
        "Data": ["Dataset Loader", "Data Preview", "Dataset Merger", "Column Selector", "Filter Node"],
        "Preprocessing": ["Missing Value Handler", "Data Type Converter", "Categorical Encoder", 
                         "Feature Scaler", "Feature Selector", "Outlier Handler", "Text Preprocessor"],
        "Split": ["Train/Test Split", "Train/Val/Test Split", "Cross Validation Split", 
                 "Time Series Split", "Batch Controller", "Conditional Router"],
        "Model": ["Model Selector", "Classification Model", "Regression Model", 
                 "Clustering Model", "Anomaly Detector", "Neural Network"],
        "Training": ["Training Controller", "Hyperparameter Tuner", "Ensemble Builder"],
        "Evaluation": ["Metrics Evaluator", "Visualization Node", "Report Generator", "Model Explainer"],
        "Output": ["Model Export", "Inference Node", "Final Output"],
        "System": ["Resource Manager", "Debug Inspector", "Data Logger", 
                  "Checkpoint Node", "Timer Node", "Loop Controller"],
        "Utility": ["Note/Comment", "AI Advisor"],
    }
    return categories
