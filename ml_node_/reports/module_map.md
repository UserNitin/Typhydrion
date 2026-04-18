# Module Map

## src/main.py
- Lines: 21
- Classes: -
- Functions: main

## src/nodes/base/edge.py
- Lines: 346
- Classes: EdgeItem
- Functions: -

## src/nodes/base/link_model.py
- Lines: 261
- Classes: ColumnRole, LinkState, ColumnConfig, LinkModel
- Functions: -

## src/nodes/base/node_base.py
- Lines: 618
- Classes: NodeItem, ResizeHandle
- Functions: _rounded_rect_path, _top_rounded_path, _bottom_rounded_path

## src/nodes/base/node_graph_scene.py
- Lines: 63
- Classes: NodeGraphScene
- Functions: -

## src/nodes/base/node_runtime.py
- Lines: 155
- Classes: NodeResult, NodeContext, NodeRuntime
- Functions: ensure_dataframe, safe_column_select

## src/nodes/base/port.py
- Lines: 47
- Classes: PortItem
- Functions: -

## src/nodes/eval/metrics_node.py
- Lines: 135
- Classes: MetricsNode
- Functions: -

## src/nodes/eval/report_node.py
- Lines: 359
- Classes: ReportNode, ModelExplainerNode
- Functions: -

## src/nodes/eval/visualization_node.py
- Lines: 330
- Classes: VisualizationNode
- Functions: -

## src/nodes/io/dataset_loader_node.py
- Lines: 329
- Classes: DatasetLoaderNode, DataPreviewNode, DatasetMergerNode, ColumnSelectorNode, FilterNode, FinalOutputNode
- Functions: -

## src/nodes/io/export_node.py
- Lines: 183
- Classes: ExportModelNode, InferenceNode
- Functions: -

## src/nodes/models/anomaly_node.py
- Lines: 193
- Classes: AnomalyNode
- Functions: -

## src/nodes/models/classification_node.py
- Lines: 171
- Classes: ClassificationNode
- Functions: -

## src/nodes/models/clustering_node.py
- Lines: 143
- Classes: ClusteringNode
- Functions: -

## src/nodes/models/model_selector_node.py
- Lines: 361
- Classes: ModelSelectorNode, TrainingNode, HyperparameterTunerNode, EnsembleBuilderNode
- Functions: -

## src/nodes/models/nn_node.py
- Lines: 327
- Classes: NeuralNetNode
- Functions: -

## src/nodes/models/regression_node.py
- Lines: 158
- Classes: RegressionNode
- Functions: -

## src/nodes/preprocess/encoding_node.py
- Lines: 167
- Classes: EncodingNode
- Functions: -

## src/nodes/preprocess/feature_select_node.py
- Lines: 297
- Classes: FeatureSelectNode, TextPreprocessorNode
- Functions: -

## src/nodes/preprocess/missing_value_node.py
- Lines: 159
- Classes: MissingValueNode, DataTypeConverterNode
- Functions: -

## src/nodes/preprocess/scaling_node.py
- Lines: 281
- Classes: ScalingNode, OutlierHandlerNode
- Functions: -

## src/nodes/registry.py
- Lines: 152
- Classes: -
- Functions: get_node_runtime, list_available_nodes, get_node_categories

## src/nodes/split/time_series_split_node.py
- Lines: 153
- Classes: TimeSeriesSplitNode
- Functions: -

## src/nodes/split/train_test_split_node.py
- Lines: 423
- Classes: TrainTestSplitNode, TrainValTestSplitNode, CrossValidationSplitNode, BatchControllerNode, ConditionalRouterNode
- Functions: -

## src/nodes/system/resource_control_node.py
- Lines: 530
- Classes: ResourceControlNode, DebugInspectorNode, DataLoggerNode, CheckpointNode, TimerNode, LoopControllerNode, NoteCommentNode, AIAdvisorNode
- Functions: -

## src/ui/app_settings.py
- Lines: 207
- Classes: SettingKey, AppSettings
- Functions: -

## src/ui/main_window.py
- Lines: 1647
- Classes: _GradientBackground, _AnimatedCard, _GlowButton, MainWindow
- Functions: -

## src/ui/widgets/node_palette.py
- Lines: 176
- Classes: NodePaletteWidget
- Functions: -

## src/ui/widgets/status_bar.py
- Lines: 67
- Classes: NodeEditorStatusBar
- Functions: -

## src/ui/widgets/toolbar.py
- Lines: 129
- Classes: NodeEditorToolbar
- Functions: _tool_btn

## src/ui/windows/ai_advisor_window.py
- Lines: 16
- Classes: AIAdvisorWindow
- Functions: -

## src/ui/windows/data_profiler_window.py
- Lines: 290
- Classes: DataProfilerWindow
- Functions: -

## src/ui/windows/data_statistics_window.py
- Lines: 607
- Classes: DataStatisticsWindow
- Functions: -

## src/ui/windows/data_window.py
- Lines: 226
- Classes: DataPreviewWindow
- Functions: -

## src/ui/windows/model_output_window.py
- Lines: 17
- Classes: ModelOutputWindow
- Functions: -

## src/ui/windows/node_editor_window.py
- Lines: 4304
- Classes: NodeEditorWindow, NodeGraphView, NodeMenuDialog, ComboButton, DatasetLoaderConfigWidget, ColumnSelectionDialog, LinkInspectorDialog
- Functions: _build_node_catalog, _build_option_widget, _all_pandas_readers, _reader_signature, _reader_fields, _build_field_widget, _browse_file, _collect_reader_kwargs, _infer_port_type

## src/ui/windows/node_output_window.py
- Lines: 504
- Classes: NodeOutputWindow
- Functions: -

## src/ui/windows/node_properties_window.py
- Lines: 902
- Classes: NodePropertiesWindow
- Functions: _get_reader_params, _get_param_description

## src/ui/windows/settings_window.py
- Lines: 974
- Classes: SettingsWindow
- Functions: -
